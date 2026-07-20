"""Motion-free classical compressive-sensing lower bounds.

(A) static_tvcs   — all K measurements → ONE image, tiled across T frames.
                    The "ignore motion" bound (converges to a motion-blurred
                    average of the moving scene).
(B) perframe_tvcs — each frame reconstructed independently from its ~125
                    measurements. The "no cross-frame sharing" bound; near-total
                    failure at ~3% per-frame sampling IS the intended result.

Neither uses a motion model — that absence is the point. Both solve the convex
  min_{x>=0} 0.5||A x - y_n||^2 + lambda*TV(x)
by ADMM-TV (common.admm_tv), with lambda selected GT-free on a held-out
measurement residual and then refit on all measurements. Requires
time_assignment_mode == 'uniform' (per-frame model is exact only then).
"""
import numpy as np
import torch

from .common import (build_operator, admm_tv, evaluate_video,
                     holdout_residual, select_by_holdout)

LAMBDA_GRID = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1e0]


def _train_val_split(data, device):
    """Return (A_train, y_train, fidx_train) and the held-out eval arrays.

    Prefer the non-invasive eval set baked into SPIData; else carve k%10==7.
    The forward operator is always built from TRAIN patterns only.
    """
    pat = torch.as_tensor(data.patterns, dtype=torch.float32, device=device)
    y = torch.as_tensor(data.measurements, dtype=torch.float32, device=device)
    fidx = torch.as_tensor(data.frame_idx, dtype=torch.long, device=device)
    if getattr(data, 'eval_patterns', None) is not None:
        ep = torch.as_tensor(data.eval_patterns, dtype=torch.float32, device=device)
        em = torch.as_tensor(data.eval_measurements, dtype=torch.float32, device=device)
        ef = torch.as_tensor(data.eval_frame_idx, dtype=torch.long, device=device)
        return pat, y, fidx, ep, em, ef
    mask = (torch.arange(len(y), device=device) % 10) == 7
    return (pat[~mask], y[~mask], fidx[~mask], pat[mask], y[mask], fidx[mask])


def static_tvcs(data, device='cpu', rho=0.5, n_admm=150, lam_grid=LAMBDA_GRID):
    """(A) Static full-measurement TV-CS. Returns (recon[T,H,W], info)."""
    assert getattr(data, 'time_assignment_mode', 'uniform') == 'uniform' or \
        not hasattr(data, 'time_assignment_mode'), "per-frame/static CS need uniform assignment"
    H, W, T = data.H, data.W, data.T
    pat_tr, y_tr, _, ep, em, ef = _train_val_split(data, device)
    A = build_operator(pat_tr).to(device)

    def run(lam):
        x = admm_tv(A, y_tr, H, W, lam, rho=rho, n_admm=n_admm)   # [H,W]
        return x.unsqueeze(0).repeat(T, 1, 1)                     # tile → [T,H,W]

    best, best_recon, table = select_by_holdout(
        lam_grid, run, ep, em, ef, gt_frames=data.gt_frames)
    psnrs, mean_p = evaluate_video(data.gt_frames, best_recon)
    return best_recon.cpu().numpy(), {
        "method": "static_tvcs", "lambda": best, "mean_psnr": mean_p,
        "per_frame_psnr": psnrs, "selection_table": table,
        "note": "motion-blur lower bound (single image tiled over T)"}


def perframe_tvcs(data, device='cpu', rho=0.5, n_admm=120, lam_grid=LAMBDA_GRID):
    """(B) Per-frame independent TV-CS. Returns (recon[T,H,W], info).

    One shared lambda for all frames (selected on the pooled holdout) keeps it a
    single fair hyperparameter. Each frame is solved from its own 125 patterns.
    """
    H, W, T = data.H, data.W, data.T
    pat_tr, y_tr, fidx_tr, ep, em, ef = _train_val_split(data, device)

    # Assert the frame partition is clean on the TRAIN set (verify spec req).
    counts = torch.bincount(fidx_tr, minlength=T)
    assert (counts > 0).all(), f"some frame has no training patterns: {counts.tolist()}"

    A_by_frame, y_by_frame = [], []
    for f in range(T):
        m = fidx_tr == f
        A_by_frame.append(build_operator(pat_tr[m]).to(device))
        y_by_frame.append(y_tr[m])

    def run(lam):
        frames = [admm_tv(A_by_frame[f], y_by_frame[f], H, W, lam,
                          rho=rho, n_admm=n_admm) for f in range(T)]
        return torch.stack(frames, 0)                            # [T,H,W]

    best, best_recon, table = select_by_holdout(
        lam_grid, run, ep, em, ef, gt_frames=data.gt_frames)
    psnrs, mean_p = evaluate_video(data.gt_frames, best_recon)
    return best_recon.cpu().numpy(), {
        "method": "perframe_tvcs", "lambda": best, "mean_psnr": mean_p,
        "per_frame_psnr": psnrs, "selection_table": table,
        "note": "no-cross-frame-sharing bound (~3% per-frame sampling)"}
