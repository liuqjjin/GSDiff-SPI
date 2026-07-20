"""Monin 2021 baseline: joint global-translation estimation + motion-compensated
TV-regularized linear reconstruction (Sci. Rep. 11:7712).

Translation-only by construction: it estimates one 2D shift per frame from DGI
previews, folds the shifts into a single motion-compensated forward operator,
solves for ONE canonical image (ADMM-TV), and back-warps it to a video. It has
no rotation DOF — on an SE(2) scene with ω>0 the rotation re-appears as blur,
which is the fair, quantifiable point for the SE(2)-aware method. Run on both an
ω=0 scene (where it is strong) and the ω>0 scene.

Motion is estimated from MEASUREMENTS ONLY (never GT). λ is selected GT-free.
"""
import numpy as np
import torch
from scipy.ndimage import shift as nd_shift, gaussian_filter

from .common import build_operator, admm_tv, evaluate_video, holdout_residual

LAMBDA_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]


# ── 1. per-frame DGI previews ────────────────────────────────
def _frame_previews(patterns, y, frame_idx, T, blur_sigma=1.0):
    """DGI preview per frame from its ~125-pattern block. [T,H,W] numpy."""
    P = patterns.cpu().numpy() if torch.is_tensor(patterns) else np.asarray(patterns)
    yy = y.cpu().numpy() if torch.is_tensor(y) else np.asarray(y)
    fi = frame_idx.cpu().numpy() if torch.is_tensor(frame_idx) else np.asarray(frame_idx)
    H, W = P.shape[1:]
    previews = np.zeros((T, H, W), np.float64)
    for f in range(T):
        m = fi == f
        Pf, yf = P[m].astype(np.float64), yy[m].astype(np.float64)
        # single-pixel covariance estimator (DGI / Eq. 1): E[prev]=σ_P²·I
        prev = ((yf - yf.mean())[:, None, None] * (Pf - Pf.mean(0, keepdims=True))).mean(0)
        if blur_sigma > 0:
            prev = gaussian_filter(prev, blur_sigma)
        previews[f] = prev
    return previews


# ── 2. trajectory estimation ─────────────────────────────────
def _ncc_shift(a, b, rng=12):
    """Object displacement a→b (b's content = a shifted by +d) via bounded
    normalized cross-correlation with parabolic sub-pixel refinement.

    A bounded integer search (|d|<=rng px) is used because FFT phase correlation
    locks onto W/2 aliasing peaks on the noisy 3%-sampled DGI previews; restricting
    to physically-plausible shifts is far more robust.
    """
    an = (a - a.mean()) / (a.std() + 1e-8)
    bn = (b - b.mean()) / (b.std() + 1e-8)
    corr = np.full((2 * rng + 1, 2 * rng + 1), -2.0)
    for iy, dy in enumerate(range(-rng, rng + 1)):
        for ix, dx in enumerate(range(-rng, rng + 1)):
            bs = nd_shift(bn, [-dy, -dx], order=1, mode="constant")   # undo candidate disp
            corr[iy, ix] = float((an * bs).mean())
    iy, ix = np.unravel_index(np.argmax(corr), corr.shape)
    dy, dx = iy - rng, ix - rng
    # parabolic sub-pixel refinement per axis (peak vs its two neighbours)
    def refine(i, axis):
        if i == 0 or i == 2 * rng:
            return 0.0
        c0 = corr[i - 1, ix] if axis == 0 else corr[iy, i - 1]
        c1 = corr[i, ix] if axis == 0 else corr[iy, i]
        c2 = corr[i + 1, ix] if axis == 0 else corr[iy, i + 1]
        denom = (c0 - 2 * c1 + c2)
        return 0.0 if abs(denom) < 1e-12 else 0.5 * (c0 - c2) / denom
    return np.array([dy + refine(iy, 0), dx + refine(ix, 1)], float)


def _estimate_trajectory(previews, t_grid, n_blocks=5, rng=12, poly_degree=1,
                         blur_sigma=1.5):
    """Return (r_smooth [T,2], v_est [2], r_nodes [n_blocks,2]).

    Robust to 3%-sampled preview noise: average previews into n_blocks temporal
    blocks (√N SNR boost; small intra-block motion blur), register each block-mean
    to block 0 by bounded NCC, then fit the known motion class (linear in t, +t²
    if poly_degree=2) through the block time-nodes.
    """
    T = previews.shape[0]
    tg = np.asarray(t_grid)
    h = np.hanning(previews.shape[1])
    win = np.outer(h, np.hanning(previews.shape[2]))   # taper borders (kills edge artifacts)
    idx = np.array_split(np.arange(T), n_blocks)
    means = [previews[i].mean(0) * win for i in idx]
    tnodes = np.array([tg[i].mean() for i in idx])
    r_nodes = np.stack([_ncc_shift(means[0], means[j], rng) for j in range(n_blocks)])
    tt = tnodes - tnodes[0]
    if poly_degree == 2:
        Bnodes = np.stack([tt, tt ** 2], 1)
        Bfull = np.stack([tg, tg ** 2], 1)
    else:
        Bnodes = tt[:, None]
        Bfull = tg[:, None]
    coef, *_ = np.linalg.lstsq(Bnodes, r_nodes, rcond=None)   # [deg,2]
    r_smooth = Bfull @ coef                                   # [T,2] r(t) relative to t=0
    v_est = coef[0] if poly_degree >= 1 else np.zeros(2)
    return r_smooth, np.asarray(v_est), r_nodes


# ── 3. compensated operator + back-warp ──────────────────────
def _shift_content(img, s, method="bilinear"):
    """Shift image CONTENT by s=[sy,sx]: out(p)=img(p-s)."""
    if method == "fft":
        H, W = img.shape
        # content shift out(p)=img(p-s) ⇔ multiply spectrum by exp(-2πi f·s)
        ramp = np.exp(-2j * np.pi * (s[0] * np.fft.fftfreq(H)[:, None]
                                     + s[1] * np.fft.fftfreq(W)[None, :]))
        return np.real(np.fft.ifft2(np.fft.fft2(img) * ramp))
    return nd_shift(img, s, order=1, mode="constant", cval=0.0)


def _build_Acomp(patterns, frame_idx, r, method="bilinear"):
    """Row k = shift(P_k, -r[frame_idx[k]]).reshape(-1). Dense [K,HW]; A.T = exact adjoint."""
    P = patterns.cpu().numpy() if torch.is_tensor(patterns) else np.asarray(patterns)
    fi = frame_idx.cpu().numpy() if torch.is_tensor(frame_idx) else np.asarray(frame_idx)
    K, H, W = P.shape
    rows = np.empty((K, H * W), np.float32)
    for k in range(K):
        rows[k] = _shift_content(P[k].astype(np.float64), -r[fi[k]], method).reshape(-1)
    return torch.from_numpy(rows)


def _backwarp_video(u_star, r, T, method="bilinear"):
    """recon_t = shift(u*, +r_t). u_star [H,W] → video [T,H,W]."""
    u = u_star.cpu().numpy() if torch.is_tensor(u_star) else np.asarray(u_star)
    return np.stack([_shift_content(u, r[t], method) for t in range(T)], 0)


# ── driver ───────────────────────────────────────────────────
def monin(data, device="cpu", rho=1.0, n_admm=150, lam_grid=LAMBDA_GRID,
          shift_method="bilinear", n_blocks=5, poly_degree=1):
    """Returns (recon[T,H,W], info). Motion from measurements only; λ GT-free."""
    H, W, T = data.H, data.W, data.T
    pat = torch.as_tensor(data.patterns, dtype=torch.float32)
    y = torch.as_tensor(data.measurements, dtype=torch.float32)
    fidx = torch.as_tensor(data.frame_idx, dtype=torch.long)

    # 1-2. estimate translation trajectory from DGI previews
    previews = _frame_previews(pat, y, fidx, T, blur_sigma=1.5)
    r, v_est, r_nodes = _estimate_trajectory(previews, data.t_grid,
                                             n_blocks=n_blocks, poly_degree=poly_degree)

    # 3. build compensated operator (once) on TRAIN measurements
    A_comp = _build_Acomp(pat, fidx, r, method=shift_method).to(device)
    y_dev = y.to(device)

    # holdout arrays for GT-free λ selection
    if getattr(data, "eval_patterns", None) is not None:
        ep, em, ef = data.eval_patterns, data.eval_measurements, data.eval_frame_idx
    else:
        mask = (np.arange(len(y)) % 10) == 7
        ep, em, ef = data.patterns[mask], data.measurements[mask], data.frame_idx[mask]

    best, best_video, best_res, table = None, None, np.inf, []
    for lam in lam_grid:
        u = admm_tv(A_comp, y_dev, H, W, lam, rho=rho, n_admm=n_admm)   # canonical [H,W]
        video = _backwarp_video(u, r, T, method=shift_method)          # [T,H,W]
        res = holdout_residual(video, ep, em, ef)
        table.append((lam, res))
        if res < best_res:
            best, best_video, best_res = lam, video, res

    psnrs, mean_p = evaluate_video(data.gt_frames, best_video)
    v_gt = np.asarray(data.gt_velocity, float)
    return best_video, {
        "method": "monin", "lambda": best, "mean_psnr": mean_p,
        "per_frame_psnr": psnrs, "est_velocity": v_est.tolist(),
        "gt_velocity": v_gt.tolist(),
        "velocity_error": np.abs(v_est - v_gt).tolist(),
        "selection_table": table,
        "note": "translation-only; rotation (if any) re-appears as blur"}
