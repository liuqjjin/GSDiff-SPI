"""3D-TV baseline — convex spatio-temporal total-variation dynamic reconstruction.

    min_{V>=0}  0.5||A(V) - y_n||^2 + lam_xy*(|D_x V| + |D_y V|) + lam_t*|D_t V|

No motion model: the full [T,H,W] video is solved jointly, frames coupled only by
temporal TV. Solver = Chambolle-Pock primal-dual on the stacked operator
K(V)=[A(V); D_xV; D_yV; D_tV] with exact adjoints and a nonnegativity projection —
the math is adapted verbatim from the user's ReCINR baselines/tv3d.py (copied,
not modified), with ONLY the measurement operator swapped for GSDiff's frame-routed
random-pattern forward (row-major) and the warm start swapped from S^-1 to DGI.

lam_xy, lam_t are selected GT-free on a held-out measurement residual; scored with
the same normalize_01+psnr_fn as every other method.
"""
import time
import numpy as np
import torch

from .common import dgi_image, evaluate_video, holdout_residual

LAM_XY_GRID = (3e-3, 3e-2, 3e-1)
LAM_T_GRID = (1e-3, 1e-2, 1e-1, 1e0)


# ── frame-routed forward operator y_k = <P_k, V[frame_idx[k]]> and exact adjoint ──
class _Op:
    def __init__(self, patterns, frame_idx, T, H, W, device):
        self.T, self.H, self.W = T, H, W
        self.HW = H * W
        self.dev = device
        self.A = torch.as_tensor(patterns, dtype=torch.float32, device=device).reshape(
            patterns.shape[0], self.HW)                    # [K,HW] row-major
        self.fidx = torch.as_tensor(frame_idx, dtype=torch.long, device=device)
        self.M = self.A.shape[0]

    def forward(self, V):                                  # V [T,H,W] -> y [K]
        Vf = V.reshape(self.T, self.HW)
        return (self.A * Vf[self.fidx]).sum(-1)

    def adjoint(self, r):                                  # r [K] -> [T,H,W]
        out = torch.zeros(self.T, self.HW, device=self.dev, dtype=self.A.dtype)
        out.index_add_(0, self.fidx, r.unsqueeze(1) * self.A)
        return out.reshape(self.T, self.H, self.W)


# ── anisotropic spatial + temporal finite differences (exact adjoints) ──
def _dx(X):  return X[:, :, 1:] - X[:, :, :-1]
def _dy(X):  return X[:, 1:, :] - X[:, :-1, :]
def _dt(X):  return X[1:] - X[:-1]


def _dxT(P, shape):
    o = torch.zeros(shape, device=P.device, dtype=P.dtype)
    o[:, :, 1:] += P; o[:, :, :-1] -= P; return o


def _dyT(P, shape):
    o = torch.zeros(shape, device=P.device, dtype=P.dtype)
    o[:, 1:, :] += P; o[:, :-1, :] -= P; return o


def _dtT(P, shape):
    o = torch.zeros(shape, device=P.device, dtype=P.dtype)
    o[1:] += P; o[:-1] -= P; return o


def _op_norm(op, n_iter=30, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    v = torch.randn(op.T, op.H, op.W, generator=g).to(op.dev)
    v = v / v.norm()
    lam = torch.tensor(1.0, device=op.dev)
    for _ in range(n_iter):
        w = (op.adjoint(op.forward(v)) + _dxT(_dx(v), v.shape)
             + _dyT(_dy(v), v.shape) + _dtT(_dt(v), v.shape))
        lam = w.norm(); v = w / lam
    return float(torch.sqrt(lam))


def _chambolle_pock(op, y, tmask, lam_xy, lam_t, iters, L, X0):
    tau = sigma = 1.0 / L
    T, H, W = op.T, op.H, op.W
    tm = tmask.float()
    X = X0.clone(); Xbar = X.clone()
    y_a = torch.zeros(op.M, device=op.dev)
    y_dx = torch.zeros(T, H, W - 1, device=op.dev)
    y_dy = torch.zeros(T, H - 1, W, device=op.dev)
    y_dt = torch.zeros(T - 1, H, W, device=op.dev)
    for _ in range(iters):
        y_a = ((y_a + sigma * op.forward(Xbar) - sigma * y) / (1.0 + sigma)) * tm
        y_dx = (y_dx + sigma * _dx(Xbar)).clamp(-lam_xy, lam_xy)
        y_dy = (y_dy + sigma * _dy(Xbar)).clamp(-lam_xy, lam_xy)
        y_dt = (y_dt + sigma * _dt(Xbar)).clamp(-lam_t, lam_t)
        KTy = (op.adjoint(y_a) + _dxT(y_dx, X.shape) + _dyT(y_dy, X.shape)
               + _dtT(y_dt, X.shape))
        Xn = (X - tau * KTy).clamp(min=0.0)
        Xbar = 2.0 * Xn - X; X = Xn
    return X


def tv3d(data, device="cuda", iters=500, lam_xy_grid=LAM_XY_GRID, lam_t_grid=LAM_T_GRID):
    """3D-TV-CS reconstruction. Returns (recon [T,H,W], info). No motion model."""
    H, W, T = data.H, data.W, data.T
    op = _Op(data.patterns, data.frame_idx, T, H, W, device)
    y = torch.as_tensor(data.measurements, dtype=torch.float32, device=device)
    b_scale = float(y.std()) + 1e-12
    yn = y / b_scale

    # holdout: prefer the non-invasive eval set; else k%10==7 internal
    if getattr(data, "eval_patterns", None) is not None:
        ep, em, ef = data.eval_patterns, data.eval_measurements, data.eval_frame_idx
        val_pack, tmask = (ep, em, ef), torch.ones(op.M, dtype=torch.bool, device=device)
    else:
        tmask = (torch.arange(op.M, device=device) % 10) != 7
        val_pack = None

    L = 1.01 * _op_norm(op)
    # DGI warm start tiled over T frames (motion-blur), with a least-squares scale
    # so A(X0)≈yn (correct DC/magnitude → CP converges from a good point).
    dgi = dgi_image(data.patterns, data.measurements).to(device)
    dgi = (dgi - dgi.min()).clamp(min=0) / (dgi.max() - dgi.min() + 1e-8)
    tiled = dgi.unsqueeze(0).repeat(T, 1, 1)
    a = op.forward(tiled)
    alpha = float((a @ yn) / (a @ a + 1e-12))
    X0 = (alpha * tiled).clamp(min=0)
    all_mask = torch.ones(op.M, dtype=torch.bool, device=device)

    best, best_recon, best_res = None, None, np.inf
    for gxy in lam_xy_grid:
        for gt in lam_t_grid:
            X = _chambolle_pock(op, yn, tmask, gxy, gt, iters, L, X0)  # fit on train mask
            res = holdout_residual(X.cpu().numpy(), *val_pack) if val_pack else \
                float(torch.sqrt(((op.forward(X) - yn)[~tmask] ** 2).mean()))
            if res < best_res:
                best, best_res = (gxy, gt), res
    # final fit on ALL measurements with selected weights
    X = _chambolle_pock(op, yn, all_mask, best[0], best[1], iters, L, X0)
    recon = X.cpu().numpy()
    psnrs, mean_p = evaluate_video(data.gt_frames, recon)
    return recon, {"method": "tv3d", "lam_xy": best[0], "lam_t": best[1],
                   "holdout": best_res, "mean_psnr": mean_p, "per_frame_psnr": psnrs,
                   "note": "convex 3D spatio-temporal TV, no motion model"}
