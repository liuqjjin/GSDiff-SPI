"""Motion-free TV-CS baselines with blind, physical-scale selection."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch

from gsdiff.experiments.objectives import heldout_normalized_l2

from .common import admm_tv, build_operator, legacy_acquisition_view


LAMBDA_GRID = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1e0]


def _holdout(acquisition):
    values = (
        acquisition.holdout_patterns,
        acquisition.holdout_measurements,
        acquisition.holdout_frame_indices,
    )
    if any(value is None for value in values) or acquisition.holdout_K <= 0:
        raise ValueError("a distinct holdout set is required for selection")
    return values  # type: ignore[return-value]


def _solver(semantic_config: Mapping[str, object]) -> Mapping[str, object]:
    solver = semantic_config.get("solver")
    if not isinstance(solver, Mapping):
        raise ValueError("CS solver semantics are required")
    return solver


def _selection(candidate_grid, candidate_reconstructions, holdout):
    rows: list[dict[str, object]] = []
    best_index = 0
    best_value: float | None = None
    for index, (candidate, reconstruction) in enumerate(zip(candidate_grid, candidate_reconstructions)):
        objective = heldout_normalized_l2(reconstruction, *holdout)
        rows.append({"candidate": candidate, "formula_id": objective.formula_id,
                     "numerator": objective.numerator, "denominator": objective.denominator,
                     "value": objective.value})
        if best_value is None or objective.value < best_value:
            best_index, best_value = index, objective.value
    selected = candidate_grid[best_index]
    return selected, {
        "formula_id": "heldout-normalized-l2-v1",
        "candidate_grid": list(candidate_grid),
        "selected_candidate": selected,
        "rows": rows,
    }


def run_static_cs(acquisition, semantic_config: Mapping[str, object], algorithm_seed, device: str):
    """Select λ on training rows, then refit the selected λ on all rows."""
    del algorithm_seed
    if acquisition.time_assignment_mode != "uniform":
        raise ValueError("static_cs requires uniform frame assignment")
    solver = _solver(semantic_config)
    holdout = _holdout(acquisition)
    H, W, T = acquisition.H, acquisition.W, acquisition.T
    train_patterns = torch.tensor(acquisition.patterns, dtype=torch.float32, device=device)
    train_measurements = torch.tensor(acquisition.measurements, dtype=torch.float32, device=device)
    A = build_operator(train_patterns).to(device)
    candidates = list(solver["lambda_grid"])
    candidate_reconstructions = []
    for lam in candidates:
        image = admm_tv(A, train_measurements, H, W, float(lam), rho=float(solver["rho"]),
                        n_admm=int(solver["n_admm"]), chambolle_iter=int(solver["chambolle_iter"]),
                        nonneg=bool(solver["nonnegative"]))
        candidate_reconstructions.append(image.unsqueeze(0).repeat(T, 1, 1).detach().cpu().numpy())
    selected, selection = _selection(candidates, candidate_reconstructions, holdout)
    all_patterns = np.concatenate((acquisition.patterns, holdout[0]), axis=0)
    all_measurements = np.concatenate((acquisition.measurements, holdout[1]), axis=0)
    A_all = build_operator(torch.as_tensor(all_patterns, dtype=torch.float32, device=device)).to(device)
    history: list[dict[str, object]] = []

    def observe(iteration, metrics):
        history.append({"kind": "iteration", "iteration": iteration, **metrics})

    final = admm_tv(A_all, torch.as_tensor(all_measurements, dtype=torch.float32, device=device), H, W,
                    float(selected), rho=float(solver["rho"]), n_admm=int(solver["n_admm"]),
                    chambolle_iter=int(solver["chambolle_iter"]), nonneg=bool(solver["nonnegative"]),
                    progress_callback=observe)
    return final.unsqueeze(0).repeat(T, 1, 1).detach().cpu().numpy(), {
        "selected_hyperparameters": {"lambda": selected}, "selection": selection,
        "history": tuple(history),
    }


def run_perframe_cs(acquisition, semantic_config: Mapping[str, object], algorithm_seed, device: str):
    """Fit candidate λ values per frame, then refit using train plus holdout rows."""
    del algorithm_seed
    if acquisition.time_assignment_mode != "uniform":
        raise ValueError("perframe_cs requires uniform frame assignment")
    solver = _solver(semantic_config)
    holdout = _holdout(acquisition)
    H, W, T = acquisition.H, acquisition.W, acquisition.T

    def solve(
        patterns: np.ndarray, measurements: np.ndarray, indices: np.ndarray,
        lam: float, *, observe: bool = False,
    ) -> tuple[np.ndarray, tuple[dict[str, object], ...]]:
        frames = []
        frame_histories: list[list[dict[str, float]]] = []
        for frame in range(T):
            mask = indices == frame
            if not np.any(mask):
                raise ValueError("each frame requires at least one measurement")
            A = build_operator(torch.as_tensor(patterns[mask], dtype=torch.float32, device=device)).to(device)
            y = torch.as_tensor(measurements[mask], dtype=torch.float32, device=device)
            frame_history: list[dict[str, float]] = []

            def record(_iteration, metrics):
                frame_history.append(metrics)

            frames.append(admm_tv(A, y, H, W, lam, rho=float(solver["rho"]),
                                  n_admm=int(solver["n_admm"]), chambolle_iter=int(solver["chambolle_iter"]),
                                  nonneg=bool(solver["nonnegative"]),
                                  progress_callback=record if observe else None))
            if observe:
                frame_histories.append(frame_history)
        history = ()
        if observe:
            history = tuple(
                {
                    "kind": "iteration",
                    "iteration": iteration + 1,
                    "data_fidelity": float(np.mean([
                        rows[iteration]["data_fidelity"]
                        for rows in frame_histories
                    ])),
                    "primal_residual": float(np.mean([
                        rows[iteration]["primal_residual"]
                        for rows in frame_histories
                    ])),
                    "dual_residual": float(np.mean([
                        rows[iteration]["dual_residual"]
                        for rows in frame_histories
                    ])),
                }
                for iteration in range(int(solver["n_admm"]))
            )
        return torch.stack(frames).detach().cpu().numpy(), history

    candidates = list(solver["lambda_grid"])
    candidate_reconstructions = [
        solve(
            acquisition.patterns, acquisition.measurements,
            acquisition.frame_indices, float(lam),
        )[0]
        for lam in candidates
    ]
    selected, selection = _selection(candidates, candidate_reconstructions, holdout)
    final, history = solve(
        np.concatenate((acquisition.patterns, holdout[0]), axis=0),
        np.concatenate((acquisition.measurements, holdout[1]), axis=0),
        np.concatenate((acquisition.frame_indices, holdout[2]), axis=0),
        float(selected),
        observe=True,
    )
    return final, {
        "selected_hyperparameters": {"lambda": selected},
        "selection": selection,
        "history": history,
    }


# Compatibility entry points are intentionally separate from strict dispatch.
def static_tvcs(data, device="cpu", rho=0.5, n_admm=150, lam_grid=LAMBDA_GRID):
    from ._evaluation import evaluate_video
    semantic = {"solver": {"rho": rho, "n_admm": n_admm, "chambolle_iter": 100,
                             "lambda_grid": list(lam_grid), "nonnegative": True}}
    reconstruction, details = run_static_cs(
        legacy_acquisition_view(data), semantic, None, device
    )
    psnrs, mean = evaluate_video(data.gt_frames, reconstruction)
    return reconstruction, {"method": "static_tvcs", "lambda": details["selected_hyperparameters"]["lambda"],
                            "mean_psnr": mean, "per_frame_psnr": psnrs,
                            "selection_table": details["selection"]["rows"]}


def perframe_tvcs(data, device="cpu", rho=0.5, n_admm=120, lam_grid=LAMBDA_GRID):
    from ._evaluation import evaluate_video
    semantic = {"solver": {"rho": rho, "n_admm": n_admm, "chambolle_iter": 100,
                             "lambda_grid": list(lam_grid), "nonnegative": True}}
    reconstruction, details = run_perframe_cs(
        legacy_acquisition_view(data), semantic, None, device
    )
    psnrs, mean = evaluate_video(data.gt_frames, reconstruction)
    return reconstruction, {"method": "perframe_tvcs", "lambda": details["selected_hyperparameters"]["lambda"],
                            "mean_psnr": mean, "per_frame_psnr": psnrs,
                            "selection_table": details["selection"]["rows"]}
