"""Blind held-out measurement objectives for method-side selection."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np


_FORMULA_ID = "heldout-normalized-l2-v1"


@dataclass(frozen=True)
class BlindObjective:
    formula_id: str
    numerator: float
    denominator: float
    value: float


def _exact_int(
    solver: Mapping[str, object],
    key: str,
    *,
    minimum: int,
) -> int:
    value = solver.get(key)
    if type(value) is not int or value < minimum:
        raise ValueError(f"{key} must be an integer at least {minimum}")
    return value


def _ordered_grid_values(
    solver: Mapping[str, object],
    key: str,
) -> Sequence[object]:
    values = solver.get(key)
    if (
        not isinstance(values, Sequence)
        or isinstance(values, (str, bytes))
        or not values
    ):
        raise ValueError(f"{key} must be a nonempty ordered sequence")
    return values


def gidc_snapshot_candidate_grid(
    solver: Mapping[str, object],
) -> list[dict[str, object]]:
    """Return the locked xi/snapshot grid in publication order."""
    if not isinstance(solver, Mapping):
        raise TypeError("solver must be a mapping")
    n_steps = _exact_int(solver, "n_steps", minimum=1)
    eval_every = _exact_int(solver, "eval_every", minimum=1)
    steps = list(range(eval_every, n_steps + 1, eval_every))
    if not steps or steps[-1] != n_steps:
        steps.append(n_steps)
    return [
        {"xi_xy": xi_xy, "xi_t": xi_t, "snapshot_step": step}
        for xi_xy in _ordered_grid_values(solver, "xi_xy")
        for xi_t in _ordered_grid_values(solver, "xi_t")
        for step in steps
    ]


def recinr_snapshot_candidate_grid(
    solver: Mapping[str, object],
) -> list[dict[str, object]]:
    """Return eligible 1-based global ReCINR snapshot steps."""
    if not isinstance(solver, Mapping):
        raise TypeError("solver must be a mapping")
    warm_steps = _exact_int(solver, "warm_steps", minimum=0)
    flow_steps = _exact_int(solver, "flow_steps", minimum=0)
    joint_steps = _exact_int(solver, "joint_steps", minimum=1)
    snapshot_every = _exact_int(solver, "snapshot_every", minimum=1)
    total_steps = flow_steps + joint_steps
    return [
        {"snapshot_step": warm_steps + loop_step + 1}
        for loop_step in range(total_steps)
        if loop_step >= flow_steps
        and (
            loop_step % snapshot_every == 0
            or loop_step == total_steps - 1
        )
    ]


def _finite_real_array(value: object, name: str, *, ndim: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}")
    if (
        np.issubdtype(array.dtype, np.bool_)
        or not np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.complexfloating)
        or not np.isfinite(array).all()
    ):
        raise ValueError(f"{name} must be finite real numeric values")
    return array


def heldout_normalized_l2(
    reconstruction: np.ndarray,
    patterns: np.ndarray,
    measurements: np.ndarray,
    frame_indices: np.ndarray,
) -> BlindObjective:
    """Return the locked raw physical held-out normalized L2 residual."""
    reconstruction_array = _finite_real_array(reconstruction, "reconstruction", ndim=3)
    patterns_array = _finite_real_array(patterns, "patterns", ndim=3)
    measurements_array = _finite_real_array(measurements, "measurements", ndim=1)
    indices = np.asarray(frame_indices)
    if indices.ndim != 1 or np.issubdtype(indices.dtype, np.bool_) or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("frame_indices must be a rank-one integer array")
    if not (patterns_array.shape[0] == measurements_array.shape[0] == indices.shape[0]):
        raise ValueError("patterns, measurements, and frame_indices lengths must match")
    if measurements_array.shape[0] == 0:
        raise ValueError("held-out arrays must contain at least one row")
    if patterns_array.shape[1:] != reconstruction_array.shape[1:]:
        raise ValueError("patterns dimensions must match reconstruction")
    if np.any(indices < 0) or np.any(indices >= reconstruction_array.shape[0]):
        raise ValueError("frame_indices values must be in reconstruction range")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            selected = reconstruction_array[indices].astype(np.float64, copy=False)
            physical_patterns = patterns_array.astype(np.float64, copy=False)
            predicted = np.einsum("khw,khw->k", physical_patterns, selected)
            target = measurements_array.astype(np.float64, copy=False)
            numerator = float(np.linalg.norm(predicted - target))
            denominator = max(float(np.linalg.norm(target)), 1e-12)
            value = numerator / denominator
    except FloatingPointError as error:
        raise ValueError("objective fields must be finite") from error
    if not all(np.isfinite(item) for item in (numerator, denominator, value)):
        raise ValueError("objective fields must be finite")
    return BlindObjective(_FORMULA_ID, numerator, denominator, value)


def select_by_heldout_normalized_l2(
    candidates: Sequence[object],
    run_candidate: Callable[[object], np.ndarray],
    *,
    patterns: np.ndarray,
    measurements: np.ndarray,
    frame_indices: np.ndarray,
) -> tuple[object, np.ndarray, tuple[Mapping[str, object], ...]]:
    """Run candidates and choose the first one with the lowest raw objective."""
    if not callable(run_candidate):
        raise TypeError("run_candidate must be callable")
    if not candidates:
        raise ValueError("candidates must not be empty")
    sentinel = object()
    selected_candidate: object = sentinel
    selected_reconstruction: np.ndarray | None = None
    selected_value: float | None = None
    history: list[Mapping[str, object]] = []
    for candidate in candidates:
        reconstruction = run_candidate(candidate)
        objective = heldout_normalized_l2(reconstruction, patterns, measurements, frame_indices)
        history.append({
            "candidate": candidate,
            "formula_id": objective.formula_id,
            "numerator": objective.numerator,
            "denominator": objective.denominator,
            "value": objective.value,
        })
        if selected_value is None or objective.value < selected_value:
            selected_candidate = candidate
            selected_reconstruction = reconstruction
            selected_value = objective.value
    assert selected_candidate is not sentinel and selected_reconstruction is not None
    return selected_candidate, selected_reconstruction, tuple(history)
