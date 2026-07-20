"""GIDC — untrained deep-image-prior + physics constraint (Wang et al., LSA 2022).

A randomly-initialized U-Net whose output is forced through the exact SPI forward
operator to match the measurements (z-score data loss + TV), with a fixed DGI
input. NO motion model, NO pretraining, NO external data.

- static_gidc     : one image from all K measurements → tiled over T (motion-blur
                    lower bound; the DIP analog of the freeze_motion static fit).
- dynamic_gidc3dtv: a shared 2D U-Net applied to all T frames (batch), the data
                    term routes each measurement to V[frame_idx[k]], and the ONLY
                    cross-frame coupling is temporal TV — the "strong deep prior +
                    3D TV but NO shared-SE(2) model" pole (recommend_include: YES).

Selection (ξ, ξ_t, early-stop step) is GT-free on a held-out measurement residual;
the fixed DGI input is built from TRAIN measurements only (no holdout leakage).
Scored with the SAME normalize_01+psnr_fn as GSDiff (never a more generous metric).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import build_operator, evaluate_video, holdout_residual


def _zscore(a):
    return (a - a.mean()) / (a.std() + 1e-8)          # unbiased std (matches GSDiff)


# ── GIDC U-Net (2D, 5-level, no-bias convs + BatchNorm + LeakyReLU, Sigmoid head) ──
class GIDCUNet2D(nn.Module):
    def __init__(self, base=16):
        super().__init__()
        c = [base, 2 * base, 4 * base, 8 * base, 16 * base]     # 16,32,64,128,256

        def blk(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 5, padding=2, bias=False), nn.BatchNorm2d(o),
                nn.LeakyReLU(0.2, inplace=True))

        self.in_conv = nn.Sequential(blk(1, c[0]), blk(c[0], c[0]))
        self.down = nn.ModuleList()
        for k in range(4):                                       # 4 stride-2 downsamples
            self.down.append(nn.ModuleList([
                nn.Sequential(nn.Conv2d(c[k], c[k + 1], 5, stride=2, padding=2, bias=False),
                              nn.BatchNorm2d(c[k + 1]), nn.LeakyReLU(0.2, inplace=True)),
                blk(c[k + 1], c[k + 1])]))
        self.up = nn.ModuleList()
        for k in range(3, -1, -1):                              # mirror upsamples
            self.up.append(nn.ModuleList([
                nn.ConvTranspose2d(c[k + 1], c[k], 4, stride=2, padding=1, bias=False),
                nn.Sequential(blk(2 * c[k], c[k]), blk(c[k], c[k]))]))
        self.head = nn.Sequential(nn.Conv2d(c[0], 1, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(1), nn.Sigmoid())

    def forward(self, x):
        x = self.in_conv(x)
        skips = [x]
        for dwn, ref in self.down:
            x = ref(dwn(x)); skips.append(x)
        skips.pop()                                            # bottleneck has no skip
        for (upc, ref), s in zip(self.up, reversed(skips)):
            x = upc(x)
            x = ref(torch.cat([x, s], 1))
        return self.head(x)


# ── measurement layer (autograd-differentiable; same as SPIForwardModel.measure) ──
def _measure_static(A, x):
    return A @ x.reshape(-1)                                    # [K]


def _measure_dynamic(A, V, frame_idx, T):
    """ŷ_k = <P_k, V[frame_idx[k]]>. A:[K,HW], V:[T,H,W] → [K]."""
    yhat = torch.empty(A.shape[0], device=A.device, dtype=A.dtype)
    Vf = V.reshape(T, -1)
    for f in range(T):
        m = frame_idx == f
        yhat[m] = A[m] @ Vf[f]
    return yhat


def _tv2d(x):                                                  # anisotropic, mean-reduced
    return (x[..., 1:, :] - x[..., :-1, :]).abs().mean() + \
           (x[..., :, 1:] - x[..., :, :-1]).abs().mean()


def _lr_at(step, lr0=0.05):
    return lr0 * 0.90 ** (step / 100.0)


def _train_val(data, device):
    pat = torch.as_tensor(data.patterns, dtype=torch.float32, device=device)
    y = torch.as_tensor(data.measurements, dtype=torch.float32, device=device)
    fidx = torch.as_tensor(data.frame_idx, dtype=torch.long, device=device)
    if getattr(data, "eval_patterns", None) is not None:
        ep, em, ef = data.eval_patterns, data.eval_measurements, data.eval_frame_idx
        return pat, y, fidx, ep, em, ef, None
    mask = (torch.arange(len(y), device=device) % 10) == 7
    # holdout carved from training → DGI input must use TRAIN only (no leakage)
    return (pat[~mask], y[~mask], fidx[~mask],
            data.patterns[mask.cpu().numpy()], data.measurements[mask.cpu().numpy()],
            data.frame_idx[mask.cpu().numpy()], mask)


# ── static GIDC ──────────────────────────────────────────────
def static_gidc(data, device="cpu", n_steps=1500, xi_grid=(0.0, 0.01, 0.1, 1.0),
                eval_every=25, seed=0):
    H, W, T = data.H, data.W, data.T
    pat, y, _, ep, em, ef, _ = _train_val(data, device)
    A = build_operator(pat).to(device)
    from .common import dgi_image
    x_in = _zscore(dgi_image(pat, y).to(device)).reshape(1, 1, H, W)
    zy = _zscore(y)

    def run(xi):
        torch.manual_seed(seed)
        net = GIDCUNet2D().to(device).train()
        opt = torch.optim.Adam(net.parameters(), lr=0.05, betas=(0.5, 0.9), eps=1e-8)
        best_res, best = np.inf, None
        for s in range(n_steps):
            for g in opt.param_groups:
                g["lr"] = _lr_at(s)
            opt.zero_grad()
            x = net(x_in)[0, 0]
            loss = F.mse_loss(_zscore(_measure_static(A, x)), zy) + xi * _tv2d(_zscore(x))
            loss.backward(); opt.step()
            if s % eval_every == 0 or s == n_steps - 1:
                with torch.no_grad():
                    vid = x.detach().unsqueeze(0).repeat(T, 1, 1).cpu().numpy()
                res = holdout_residual(vid, ep, em, ef)
                if res < best_res:
                    best_res, best = res, vid
        return best, best_res

    results = [(xi, *run(xi)) for xi in xi_grid]
    xi_best, recon, res = min(results, key=lambda r: r[2])
    psnrs, mean_p = evaluate_video(data.gt_frames, recon)
    return recon, {"method": "static_gidc", "xi": xi_best, "holdout": res,
                   "mean_psnr": mean_p, "per_frame_psnr": psnrs,
                   "note": "motion-blur lower bound (untrained deep prior, static)"}


# ── dynamic GIDC-3DTV (shared 2D U-Net over frames + temporal TV) ──
def dynamic_gidc3dtv(data, device="cpu", n_steps=2500,
                     xi_xy_grid=(3e-3, 3e-2, 3e-1), xi_t_grid=(1e-2, 1e-1),
                     eval_every=25, seed=0):
    H, W, T = data.H, data.W, data.T
    pat, y, fidx, ep, em, ef, _ = _train_val(data, device)
    A = build_operator(pat).to(device)
    from .common import dgi_image
    x_in = _zscore(dgi_image(pat, y).to(device)).reshape(1, 1, H, W).repeat(T, 1, 1, 1)
    zy = _zscore(y)

    def run(xi_xy, xi_t):
        torch.manual_seed(seed)
        net = GIDCUNet2D().to(device).train()
        opt = torch.optim.Adam(net.parameters(), lr=0.05, betas=(0.5, 0.9), eps=1e-8)
        best_res, best = np.inf, None
        for s in range(n_steps):
            for g in opt.param_groups:
                g["lr"] = _lr_at(s)
            opt.zero_grad()
            V = net(x_in)[:, 0]                                 # [T,H,W]
            data_loss = F.mse_loss(_zscore(_measure_dynamic(A, V, fidx, T)), zy)
            tv_xy = _tv2d(_zscore(V))
            tv_t = (V[1:] - V[:-1]).abs().mean()
            (data_loss + xi_xy * tv_xy + xi_t * tv_t).backward()
            opt.step()
            if s % eval_every == 0 or s == n_steps - 1:
                res = holdout_residual(V.detach().cpu().numpy(), ep, em, ef)
                if res < best_res:
                    best_res, best = res, V.detach().cpu().numpy()
        return best, best_res

    results = [(xy, t, *run(xy, t)) for xy in xi_xy_grid for t in xi_t_grid]
    xy_b, t_b, recon, res = min(results, key=lambda r: r[3])
    psnrs, mean_p = evaluate_video(data.gt_frames, recon)
    return recon, {"method": "dynamic_gidc3dtv", "xi_xy": xy_b, "xi_t": t_b,
                   "holdout": res, "mean_psnr": mean_p, "per_frame_psnr": psnrs,
                   "note": "untrained deep prior + 3D TV, NO shared-SE(2) model"}
