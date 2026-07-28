from __future__ import annotations

import numpy as np
import pytest

import gsdiff.experiments.objectives as objectives
from gsdiff.experiments.methods import resolve_method_semantics
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


def _resolved_solver(method_id: str, profile: str):
    method_config_id = (
        "default"
        if profile == "publication-v1"
        else "smoke-default-v1"
    )
    method = resolve_method_semantics(
        method_id,
        method_config_id=method_config_id,
        base_config={},
        measurements_metadata={},
        execution_profile=profile,
    )
    return method.semantic_config["solver"]


def test_gidc_publication_snapshot_grid_is_600_rows_in_locked_order() -> None:
    publication = objectives.gidc_snapshot_candidate_grid(
        _resolved_solver("gidc3dtv", "publication-v1")
    )
    expected_steps = list(range(25, 2501, 25))
    expected = [
        {"xi_xy": xi_xy, "xi_t": xi_t, "snapshot_step": step}
        for xi_xy in (0.003, 0.03, 0.3)
        for xi_t in (0.01, 0.1)
        for step in expected_steps
    ]
    assert publication == expected
    assert len(publication) == 600
    assert [publication[index] for index in (0, 99, 100, 199, 200, 299)] == [
        {"xi_xy": 0.003, "xi_t": 0.01, "snapshot_step": 25},
        {"xi_xy": 0.003, "xi_t": 0.01, "snapshot_step": 2500},
        {"xi_xy": 0.003, "xi_t": 0.1, "snapshot_step": 25},
        {"xi_xy": 0.003, "xi_t": 0.1, "snapshot_step": 2500},
        {"xi_xy": 0.03, "xi_t": 0.01, "snapshot_step": 25},
        {"xi_xy": 0.03, "xi_t": 0.01, "snapshot_step": 2500},
    ]
    assert [publication[index] for index in (300, 399, 400, 499, 500, 599)] == [
        {"xi_xy": 0.03, "xi_t": 0.1, "snapshot_step": 25},
        {"xi_xy": 0.03, "xi_t": 0.1, "snapshot_step": 2500},
        {"xi_xy": 0.3, "xi_t": 0.01, "snapshot_step": 25},
        {"xi_xy": 0.3, "xi_t": 0.01, "snapshot_step": 2500},
        {"xi_xy": 0.3, "xi_t": 0.1, "snapshot_step": 25},
        {"xi_xy": 0.3, "xi_t": 0.1, "snapshot_step": 2500},
    ]
    for xi_xy in (0.003, 0.03, 0.3):
        for xi_t in (0.01, 0.1):
            assert publication.count(
                {"xi_xy": xi_xy, "xi_t": xi_t, "snapshot_step": 2500}
            ) == 1

    smoke = objectives.gidc_snapshot_candidate_grid(
        _resolved_solver("gidc3dtv", "controller-cpu-smoke-v1")
    )
    assert smoke == [{"xi_xy": 0.003, "xi_t": 0.01, "snapshot_step": 1}]


def test_recinr_publication_snapshot_grid_is_25_native_steps() -> None:
    publication = objectives.recinr_snapshot_candidate_grid(
        _resolved_solver("recinr", "publication-v1")
    )
    expected_steps = [701 + 50 * index for index in range(24)] + [1900]
    assert publication == [
        {"snapshot_step": step} for step in expected_steps
    ]
    assert len(publication) == 25

    smoke_solver = _resolved_solver(
        "recinr", "controller-cpu-smoke-v1"
    )
    assert objectives.recinr_snapshot_candidate_grid(smoke_solver) == [
        {"snapshot_step": 3}
    ]
    invalid_solver = dict(smoke_solver)
    invalid_solver["joint_steps"] = 0
    with pytest.raises(ValueError, match="joint_steps"):
        objectives.recinr_snapshot_candidate_grid(invalid_solver)


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
    ("reconstruction", "patterns", "measurements"),
    [
        (
            np.full((2, 1, 1), 1e308),
            np.ones((2, 1, 1)),
            np.full(2, 1e308),
        ),
        (
            np.full((2, 1, 1), 1e308),
            np.ones((2, 1, 1)),
            np.full(2, -1e308),
        ),
        (
            np.full((1, 1, 1), 1e308),
            np.full((1, 1, 1), 1e308),
            np.ones(1),
        ),
    ],
)
def test_objective_rejects_overflowed_recorded_fields(
    reconstruction: np.ndarray,
    patterns: np.ndarray,
    measurements: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="finite"):
        heldout_normalized_l2(
            reconstruction,
            patterns,
            measurements,
            np.arange(measurements.size, dtype=np.int64),
        )


@pytest.mark.parametrize(
    ("arrays", "message"),
    [
        ((np.ones((2, 2)),) + valid_arrays()[1:], "reconstruction"),
        ((valid_arrays()[0], np.ones((2, 2), dtype=np.float32)) + valid_arrays()[2:], "patterns"),
        (valid_arrays()[:2] + (np.ones((2, 1)), valid_arrays()[3]), "measurements"),
        (valid_arrays()[:3] + (np.array([0.0, 1.0]),), "frame_indices"),
        (valid_arrays()[:3] + (np.array([False, True]),), "frame_indices"),
        (valid_arrays()[:3] + (np.array([0 + 0j, 1 + 0j]),), "frame_indices"),
        (valid_arrays()[:3] + (np.array([0, 2], dtype=np.int64),), "frame_indices"),
        ((np.full((2, 2, 2), np.nan),) + valid_arrays()[1:], "finite"),
        (
            (
                np.empty((0, 2, 2)),
                np.empty((0, 2, 2)),
                np.empty(0),
                np.empty(0, dtype=np.int64),
            ),
            "at least one",
        ),
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


def test_raw_l2_ranking_reverses_zscore_ranking() -> None:
    candidates = ("raw-winner", "zscore-winner")
    reconstructions = {
        "raw-winner": np.array([[[10.0]], [[20.0]], [[29.0]]]),
        "zscore-winner": np.array([[[100.0]], [[200.0]], [[300.0]]]),
    }
    target = np.array([10.0, 20.0, 30.0])
    selected, _, history = select_by_heldout_normalized_l2(
        candidates,
        lambda candidate: reconstructions[candidate],
        patterns=np.ones((3, 1, 1)),
        measurements=target,
        frame_indices=np.arange(3, dtype=np.int64),
    )
    z_target = (target - target.mean()) / target.std()
    z_scores = {
        name: np.linalg.norm(
            (value[:, 0, 0] - value[:, 0, 0].mean())
            / value[:, 0, 0].std()
            - z_target
        )
        for name, value in reconstructions.items()
    }
    assert selected == "raw-winner"
    assert history[0]["value"] < history[1]["value"]
    assert z_scores["zscore-winner"] < z_scores["raw-winner"]


def test_selection_preserves_declared_order_for_ties() -> None:
    reconstructions = {"first": np.ones((1, 1, 1)), "second": np.ones((1, 1, 1))}
    selected, _, _ = select_by_heldout_normalized_l2(
        ("first", "second"), lambda candidate: reconstructions[candidate],
        patterns=np.ones((1, 1, 1)), measurements=np.ones(1),
        frame_indices=np.zeros(1, dtype=np.int64),
    )
    assert selected == "first"


def test_selector_accepts_none_candidate_and_rejects_empty_candidates() -> None:
    selected, reconstruction, history = select_by_heldout_normalized_l2(
        (None,),
        lambda candidate: np.ones((1, 1, 1)),
        patterns=np.ones((1, 1, 1)),
        measurements=np.ones(1),
        frame_indices=np.zeros(1, dtype=np.int64),
    )
    assert selected is None
    assert reconstruction.shape == (1, 1, 1)
    assert history[0]["candidate"] is None
    with pytest.raises(ValueError, match="candidates"):
        select_by_heldout_normalized_l2(
            (),
            lambda candidate: np.ones((1, 1, 1)),
            patterns=np.ones((1, 1, 1)),
            measurements=np.ones(1),
            frame_indices=np.zeros(1, dtype=np.int64),
        )
