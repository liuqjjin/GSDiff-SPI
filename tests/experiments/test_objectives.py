from __future__ import annotations

import numpy as np
import pytest

from gsdiff.experiments.objectives import (
    heldout_normalized_l2,
    select_by_heldout_normalized_l2,
)


def valid_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.ones((2, 2, 2), dtype=np.float32),
        np.ones((2, 2, 2), dtype=np.float32),
        np.ones(2, dtype=np.float32),
        np.array([0, 1], dtype=np.int64),
    )


def test_raw_objective_matches_physical_float64_formula() -> None:
    reconstruction = np.array([[[1, 2], [3, 4]], [[4, 3], [2, 1]]], dtype=np.float32)
    patterns = np.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]]], dtype=np.float32)
    measurements = np.array([6.0, 5.0], dtype=np.float32)
    result = heldout_normalized_l2(reconstruction, patterns, measurements, np.array([0, 1], dtype=np.int64))
    expected = np.array([5.0, 5.0], dtype=np.float64)
    numerator = np.linalg.norm(expected - measurements.astype(np.float64))
    denominator = max(np.linalg.norm(measurements.astype(np.float64)), 1e-12)
    assert result.formula_id == "heldout-normalized-l2-v1"
    assert result.numerator == pytest.approx(numerator)
    assert result.denominator == pytest.approx(denominator)
    assert result.value == pytest.approx(numerator / denominator)


def test_zero_measurement_norm_uses_locked_floor() -> None:
    result = heldout_normalized_l2(np.ones((1, 1, 1)), np.ones((1, 1, 1)), np.zeros(1), np.zeros(1, dtype=np.int64))
    assert result.denominator == 1e-12
    assert result.value == pytest.approx(1e12)


@pytest.mark.parametrize(
    ("arrays", "message"),
    [
        ((np.ones((2, 2)),) + valid_arrays()[1:], "reconstruction"),
        ((valid_arrays()[0], np.ones((2, 2), dtype=np.float32)) + valid_arrays()[2:], "patterns"),
        (valid_arrays()[:2] + (np.ones((2, 1)), valid_arrays()[3]), "measurements"),
        (valid_arrays()[:3] + (np.array([0.0, 1.0]),), "frame_indices"),
        (valid_arrays()[:3] + (np.array([0, 2], dtype=np.int64),), "frame_indices"),
        ((np.full((2, 2, 2), np.nan),) + valid_arrays()[1:], "finite"),
    ],
)
def test_objective_rejects_invalid_inputs(arrays: tuple[np.ndarray, ...], message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        heldout_normalized_l2(*arrays)


def test_selection_uses_raw_l2() -> None:
    candidates = ("first", "second")
    reconstructions = {
        "first": np.array([[[1.0]], [[3.0]]]),
        "second": np.array([[[2.0]], [[2.0]]]),
    }
    selected, reconstruction, history = select_by_heldout_normalized_l2(
        candidates, lambda candidate: reconstructions[candidate],
        patterns=np.ones((2, 1, 1)), measurements=np.array([2.0, 2.0]),
        frame_indices=np.array([0, 1], dtype=np.int64),
    )
    assert selected == "second"
    assert reconstruction is reconstructions["second"]
    assert [row["candidate"] for row in history] == ["first", "second"]
    assert history[0]["formula_id"] == "heldout-normalized-l2-v1"


def test_selection_preserves_declared_order_for_ties() -> None:
    reconstructions = {"first": np.ones((1, 1, 1)), "second": np.ones((1, 1, 1))}
    selected, _, _ = select_by_heldout_normalized_l2(
        ("first", "second"), lambda candidate: reconstructions[candidate],
        patterns=np.ones((1, 1, 1)), measurements=np.ones(1),
        frame_indices=np.zeros(1, dtype=np.int64),
    )
    assert selected == "first"
