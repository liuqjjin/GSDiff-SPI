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
