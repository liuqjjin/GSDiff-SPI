"""Convex 3D-TV baseline with physical blind selection and all-data refit."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch

from gsdiff.experiments.objectives import heldout_normalized_l2

from .common import dgi_image, legacy_acquisition_view


class _Op:
    def __init__(self, patterns, frame_idx, T, H, W, device):
        self.T, self.H, self.W, self.HW, self.dev = T, H, W, H * W, device
        self.A = torch.tensor(patterns, dtype=torch.float32, device=device).reshape(len(patterns), self.HW)
        self.fidx = torch.tensor(frame_idx, dtype=torch.long, device=device)
        self.M = self.A.shape[0]

    def forward(self, video):
        return (self.A * video.reshape(self.T, self.HW)[self.fidx]).sum(-1)

    def adjoint(self, residual):
        output = torch.zeros(self.T, self.HW, device=self.dev, dtype=self.A.dtype)
        output.index_add_(0, self.fidx, residual.unsqueeze(1) * self.A)
        return output.reshape(self.T, self.H, self.W)


def _dx(value): return value[:, :, 1:] - value[:, :, :-1]
def _dy(value): return value[:, 1:, :] - value[:, :-1, :]
def _dt(value): return value[1:] - value[:-1]


def _dxT(value, shape):
    output = torch.zeros(shape, device=value.device, dtype=value.dtype)
    output[:, :, 1:] += value; output[:, :, :-1] -= value
    return output


def _dyT(value, shape):
    output = torch.zeros(shape, device=value.device, dtype=value.dtype)
    output[:, 1:, :] += value; output[:, :-1, :] -= value
    return output


def _dtT(value, shape):
    output = torch.zeros(shape, device=value.device, dtype=value.dtype)
    output[1:] += value; output[:-1] -= value
    return output


def _op_norm(op, n_iter=30, seed=0):
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    value = torch.randn(op.T, op.H, op.W, generator=generator).to(op.dev)
    value = value / value.norm().clamp_min(1e-12)
    eigenvalue = torch.tensor(1.0, device=op.dev)
    for _ in range(n_iter):
        update = op.adjoint(op.forward(value)) + _dxT(_dx(value), value.shape) + _dyT(_dy(value), value.shape) + _dtT(_dt(value), value.shape)
        eigenvalue = update.norm().clamp_min(1e-12)
        value = update / eigenvalue
    return float(torch.sqrt(eigenvalue))


def _chambolle_pock(
    op, y, tmask, lam_xy, lam_t, iters, L, X0,
    progress_callback=None,
):
    tau = sigma = 1.0 / max(float(L), 1e-12)
    T, H, W = op.T, op.H, op.W
    mask = tmask.float()
    X = X0.clone(); Xbar = X.clone()
    y_a = torch.zeros(op.M, device=op.dev)
    y_dx = torch.zeros(T, H, W - 1, device=op.dev)
    y_dy = torch.zeros(T, H - 1, W, device=op.dev)
    y_dt = torch.zeros(T - 1, H, W, device=op.dev)
    for iteration in range(iters):
        y_a = ((y_a + sigma * op.forward(Xbar) - sigma * y) / (1.0 + sigma)) * mask
        y_dx = (y_dx + sigma * _dx(Xbar)).clamp(-lam_xy, lam_xy)
        y_dy = (y_dy + sigma * _dy(Xbar)).clamp(-lam_xy, lam_xy)
        y_dt = (y_dt + sigma * _dt(Xbar)).clamp(-lam_t, lam_t)
        dual = op.adjoint(y_a) + _dxT(y_dx, X.shape) + _dyT(y_dy, X.shape) + _dtT(y_dt, X.shape)
        next_x = (X - tau * dual).clamp(min=0.0)
        Xbar = 2.0 * next_x - X
        X = next_x
        if progress_callback is not None:
            residual = (op.forward(X) - y) * mask
            progress_callback(
                iteration + 1,
                {"data_fidelity": float(0.5 * residual.square().sum())},
            )
    return X


def _require_holdout(acquisition):
    values = (acquisition.holdout_patterns, acquisition.holdout_measurements, acquisition.holdout_frame_indices)
    if any(value is None for value in values) or acquisition.holdout_K <= 0:
        raise ValueError("a distinct holdout set is required for selection")
    return values  # type: ignore[return-value]


def _initial(op, patterns, measurements, T, scale):
    dgi = dgi_image(patterns, measurements).to(op.dev)
    dgi = (dgi - dgi.min()).clamp(min=0) / (dgi.max() - dgi.min() + 1e-8)
    tiled = dgi.unsqueeze(0).repeat(T, 1, 1)
    predicted = op.forward(tiled)
    target = torch.tensor(measurements, dtype=torch.float32, device=op.dev)
    return ((predicted @ (target / scale)) / (predicted @ predicted + 1e-12) * tiled).clamp(min=0)


def run_tv3d(acquisition, semantic_config: Mapping[str, object], algorithm_seed, device: str):
    solver = semantic_config.get("solver")
    if not isinstance(solver, Mapping):
        raise ValueError("TV3D solver semantics are required")
    holdout = _require_holdout(acquisition)
    H, W, T = acquisition.H, acquisition.W, acquisition.T
    train_op = _Op(acquisition.patterns, acquisition.frame_indices, T, H, W, device)
    train_y = torch.tensor(acquisition.measurements, dtype=torch.float32, device=device)
    train_scale = float(train_y.std(unbiased=False)) + 1e-12
    train_y_normalized = train_y / train_scale
    train_initial = _initial(train_op, acquisition.patterns, acquisition.measurements, T, train_scale)
    train_norm = 1.01 * _op_norm(train_op, n_iter=int(solver["opnorm_iterations"]), seed=algorithm_seed.seed_u32)
    candidates = [{"lambda_xy": xy, "lambda_t": temporal} for xy in solver["lambda_xy"] for temporal in solver["lambda_t"]]
    rows = []
    selected = None
    selected_value = None
    for candidate in candidates:
        normalized = _chambolle_pock(train_op, train_y_normalized, torch.ones(train_op.M, dtype=torch.bool, device=device), candidate["lambda_xy"], candidate["lambda_t"], int(solver["iterations"]), train_norm, train_initial)
        physical = (normalized * train_scale).detach().cpu().numpy()
        objective = heldout_normalized_l2(physical, *holdout)
        rows.append({"candidate": candidate, "formula_id": objective.formula_id, "numerator": objective.numerator, "denominator": objective.denominator, "value": objective.value})
        if selected_value is None or objective.value < selected_value:
            selected, selected_value = candidate, objective.value
    assert selected is not None
    all_patterns = np.concatenate((acquisition.patterns, holdout[0]), axis=0)
    all_measurements = np.concatenate((acquisition.measurements, holdout[1]), axis=0)
    all_indices = np.concatenate((acquisition.frame_indices, holdout[2]), axis=0)
    all_op = _Op(all_patterns, all_indices, T, H, W, device)
    all_y = torch.tensor(all_measurements, dtype=torch.float32, device=device)
    all_scale = float(all_y.std(unbiased=False)) + 1e-12
    all_initial = _initial(all_op, all_patterns, all_measurements, T, all_scale)
    all_norm = 1.01 * _op_norm(all_op, n_iter=int(solver["opnorm_iterations"]), seed=algorithm_seed.seed_u32)
    history: list[dict[str, object]] = []

    def observe(iteration, metrics):
        history.append({"kind": "iteration", "iteration": iteration, **metrics})

    final = _chambolle_pock(
        all_op, all_y / all_scale,
        torch.ones(all_op.M, dtype=torch.bool, device=device),
        selected["lambda_xy"], selected["lambda_t"],
        int(solver["iterations"]), all_norm, all_initial,
        progress_callback=observe,
    )
    return (final * all_scale).detach().cpu().numpy(), {
        "selected_hyperparameters": selected,
        "selection": {"formula_id": "heldout-normalized-l2-v1", "candidate_grid": candidates, "selected_candidate": selected, "rows": rows},
        "history": tuple(history),
    }


def tv3d(data, device="cuda", iters=500, lam_xy_grid=(3e-3, 3e-2, 3e-1), lam_t_grid=(1e-3, 1e-2, 1e-1, 1e0)):
    from ._evaluation import evaluate_video
    reconstruction, details = run_tv3d(legacy_acquisition_view(data), {"solver": {"iterations": iters, "opnorm_iterations": 30, "lambda_xy": list(lam_xy_grid), "lambda_t": list(lam_t_grid)}}, type("Seed", (), {"seed_u32": 0})(), device)
    psnrs, mean = evaluate_video(data.gt_frames, reconstruction)
    return reconstruction, {"method": "tv3d", "lam_xy": details["selected_hyperparameters"]["lambda_xy"], "lam_t": details["selected_hyperparameters"]["lambda_t"], "mean_psnr": mean, "per_frame_psnr": psnrs}
