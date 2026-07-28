"""Shared infrastructure for the classical/learned SPI baselines.

Linear forward operator, exact adjoint, ADMM-TV solver (reusing the repo's
Chambolle prox), DGI init, GT-free hyperparameter selection, and the
comparison-neutral per-frame evaluation.

Conventions (verified against the baseline specs, 2026-07):
- Patterns are flattened ROW-MAJOR (C-order, torch reshape(-1)) so that the
  materialized A and A.T are provably adjoint. This matches SPIForwardModel.measure.
- Measurements are conditioned by DIVIDING by their global std (NO mean removal;
  mean removal would break physical non-negativity and desync A from y).
- Compatibility evaluation is the explicitly named legacy per-frame min-max
  definition — identical to train.evaluate().
"""
import numpy as np
import torch

from ..prior.tv import TVPrior
from ..data.dgi import dgi_reconstruct
from ..experiments.objectives import heldout_normalized_l2


def __getattr__(name):
    if name != "evaluate_video":
        raise AttributeError(name)
    from ._evaluation import evaluate_video

    return evaluate_video


# ── Linear forward operator ──────────────────────────────────
def build_operator(patterns):
    """Dense forward matrix A [K, HW] from patterns [K,H,W] (C-order flatten).

    (A @ vec(x))_k = <P_k, x>. A.T is the exact adjoint by construction.
    Returns a torch.float32 tensor on the patterns' device.
    """
    if not torch.is_tensor(patterns):
        patterns = torch.as_tensor(np.asarray(patterns), dtype=torch.float32)
    K = patterns.shape[0]
    return patterns.reshape(K, -1).contiguous().float()


def apply_operator(A, x):
    """A @ vec(x). x: [H,W] or [HW] → [K]."""
    return A @ x.reshape(-1)


def adjoint(A, r):
    """A.T @ r → [HW] (reshape to [H,W] by caller). Exact adjoint of apply_operator."""
    return A.t() @ r


# ── ADMM-TV solver (scaled form, Boyd 2011; Chambolle z-step) ──
def admm_tv(A, y, H, W, lam, rho=0.5, n_admm=150, chambolle_iter=100,
            b_scale=None, nonneg=True):
    """Solve  min_{x>=0} 0.5||A x - y_n||^2 + lam * TV(x)  by ADMM-TV.

    A       : [K, HW] forward operator (raw, unnormalized)
    y       : [K] measurements
    b_scale : conditioning scale; default std(y). Pass 1.0 for the exactness test.
    Returns : x* [H,W] (torch, on A.device), the TV-regular non-negative iterate.

    x-step is the exact quadratic solve (A^T A + rho I) x = A^T y_n + rho(z-u)
    via a prefactored Cholesky (rho fixed). z-step is the repo's isotropic
    Chambolle prox with weight = lam/rho (the ADMM TV-weight scaling — NOT lam).
    """
    dev = A.device
    out_dtype = A.dtype
    K = A.shape[0]
    # Solve in float64: the Woodbury x-step subtracts two large near-equal terms on
    # the DC mode (AAᵀ has a ~1e7 DC eigenvalue for U[0,1] patterns while ρ~1), which
    # loses all precision in float32. Double gives ~8 stable digits; return float32.
    A = A.double()
    y = y.to(dev).double()
    if b_scale is None:
        b_scale = float(y.std()) + 1e-12
    y_n = y / b_scale

    # x-step solves (AᵀA + ρI) x = rhs via Woodbury (system in K-dim measurement
    # space, K≤HW; K=125 per-frame): S = ρ I_K + A Aᵀ prefactored once,
    # (ρI+AᵀA)⁻¹ b = (b - Aᵀ S⁻¹(A b))/ρ.
    S = rho * torch.eye(K, device=dev, dtype=torch.float64) + A @ A.t()
    L_S = torch.linalg.cholesky(S)

    def solve(rhs):                                # (ρI + AᵀA)⁻¹ rhs
        Ab = (A @ rhs).unsqueeze(1)
        return (rhs - A.t() @ torch.cholesky_solve(Ab, L_S).squeeze(1)) / rho

    ATb = A.t() @ y_n                             # [HW]
    x = solve(ATb)                                # warm start = ridge back-proj
    z = x.clamp(min=0) if nonneg else x.clone()
    u = torch.zeros_like(x)

    w = lam / (rho + 1e-12)
    for _ in range(n_admm):
        x = solve(ATb + rho * (z - u))            # exact x-step (Woodbury)
        zt = TVPrior._chambolle((x + u).reshape(H, W), w, chambolle_iter).reshape(-1)
        z = zt.clamp(min=0) if nonneg else zt
        u = u + x - z
    # Conditioning is strictly an internal numerical device.  The returned
    # image stays on the physical forward-model scale so callers can safely
    # compare it with raw measurements.
    return (z * float(b_scale)).reshape(H, W).to(out_dtype)


# ── DGI init ─────────────────────────────────────────────────
def dgi_image(patterns, measurements):
    """z-scored DGI reconstruction [H,W] (float32 torch)."""
    pat = patterns.cpu().numpy() if torch.is_tensor(patterns) else np.asarray(patterns)
    y = measurements.cpu().numpy() if torch.is_tensor(measurements) else np.asarray(measurements)
    return torch.as_tensor(dgi_reconstruct(pat, y), dtype=torch.float32)


def holdout_residual(recon, eval_patterns, eval_measurements, eval_frame_idx):
    """Compatibility alias for the locked physical held-out objective."""
    rc = recon.cpu().numpy() if torch.is_tensor(recon) else np.asarray(recon)
    P = eval_patterns.cpu().numpy() if torch.is_tensor(eval_patterns) else np.asarray(eval_patterns)
    y = eval_measurements.cpu().numpy() if torch.is_tensor(eval_measurements) else np.asarray(eval_measurements)
    fi = eval_frame_idx.cpu().numpy() if torch.is_tensor(eval_frame_idx) else np.asarray(eval_frame_idx)
    return heldout_normalized_l2(rc, P, y, fi).value


def select_by_holdout(candidates, run_fn, eval_patterns, eval_measurements,
                      eval_frame_idx):
    """Compatibility selection using only the locked raw held-out objective."""
    if eval_patterns is None or eval_measurements is None or eval_frame_idx is None:
        raise ValueError("a distinct holdout set is required for selection")
    best, best_recon, best_res, table = None, None, np.inf, []
    for c in candidates:
        recon = run_fn(c)
        objective = heldout_normalized_l2(
            recon.cpu().numpy() if torch.is_tensor(recon) else recon,
            eval_patterns.cpu().numpy() if torch.is_tensor(eval_patterns) else eval_patterns,
            eval_measurements.cpu().numpy() if torch.is_tensor(eval_measurements) else eval_measurements,
            eval_frame_idx.cpu().numpy() if torch.is_tensor(eval_frame_idx) else eval_frame_idx,
        )
        res = objective.value
        table.append((c, res))
        if res < best_res:
            best, best_recon, best_res = c, recon, res
    return best, best_recon, table
