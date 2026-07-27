import json

import numpy as np
import pytest

from gsdiff.baselines.common import evaluate_video
from gsdiff.evaluation.metrics import (
    apply_global_affine,
    evaluate_video_global_affine,
    evaluate_video_legacy_per_frame,
    fit_global_affine,
)


def test_global_affine_metric_recovers_one_known_video_transform():
    gt = np.linspace(0, 1, 2 * 8 * 9).reshape(2, 8, 9)
    recon = (gt - 0.2) / 1.7

    result = evaluate_video_global_affine(gt, recon)

    assert result["alignment"]["slope"] == pytest.approx(1.7)
    assert result["alignment"]["intercept"] == pytest.approx(0.2)
    assert result["psnr_global_affine"] == pytest.approx(120.0)
    assert result["nrmse_global_affine_l2"] < 1e-12


def test_per_frame_minmax_can_hide_different_frame_gains():
    frame = np.linspace(0.0, 1.0, 8 * 9).reshape(8, 9)
    gt = np.stack([frame, frame])
    recon = np.stack([0.5 * frame, 2.0 * frame])

    result = evaluate_video_global_affine(gt, recon)
    legacy = evaluate_video_legacy_per_frame(gt, recon)

    assert legacy["psnr_legacy_per_frame_minmax"] == pytest.approx(60.0)
    assert result["psnr_legacy_per_frame_minmax"] == pytest.approx(60.0)
    assert result["psnr_global_affine"] < 25.0


def test_constant_prediction_uses_mean_ground_truth_and_finite_metrics():
    gt = np.linspace(0.0, 1.0, 2 * 8 * 9).reshape(2, 8, 9)
    recon = np.full_like(gt, 4.25)

    result = evaluate_video_global_affine(gt, recon)

    assert result["alignment"]["slope"] == 0.0
    assert result["alignment"]["intercept"] == pytest.approx(float(np.mean(gt)))
    assert all(
        np.isfinite(result[key])
        for key in (
            "psnr_global_affine",
            "ssim_global_affine",
            "nrmse_global_affine_l2",
            "psnr_legacy_per_frame_minmax",
        )
    )


def test_negative_correlation_uses_nonnegative_boundary_and_finite_metrics():
    gt = np.linspace(0.0, 1.0, 2 * 8 * 9).reshape(2, 8, 9)
    recon = 1.0 - gt

    result = evaluate_video_global_affine(gt, recon)

    assert result["alignment"]["slope"] == 0.0
    assert result["alignment"]["intercept"] == pytest.approx(float(np.mean(gt)))
    assert all(
        np.isfinite(result[key])
        for key in (
            "psnr_global_affine",
            "ssim_global_affine",
            "nrmse_global_affine_l2",
            "psnr_legacy_per_frame_minmax",
        )
    )


def test_negative_slope_can_be_requested_explicitly():
    gt = np.linspace(0.0, 1.0, 2 * 8 * 9).reshape(2, 8, 9)
    recon = 1.0 - gt

    slope, intercept = fit_global_affine(
        gt, recon, nonnegative_slope=False
    )

    assert slope == pytest.approx(-1.0)
    assert intercept == pytest.approx(1.0)


def test_apply_global_affine_clips_once_and_preserves_float64():
    recon = np.array([[[-1.0, 0.25], [0.75, 2.0]]], dtype=np.float32)

    aligned = apply_global_affine(recon, 2.0, -0.25)

    np.testing.assert_array_equal(
        aligned, np.array([[[0.0, 0.25], [1.0, 1.0]]], dtype=np.float64)
    )
    assert aligned.dtype == np.float64


def test_metrics_v1_declares_numerical_and_ssim_policies():
    gt = np.linspace(0.0, 1.0, 2 * 8 * 9).reshape(2, 8, 9)

    result = evaluate_video_global_affine(gt, gt)

    assert result["definition_version"] == "metrics-v1"
    assert result["metric_definition"]["psnr_mse_floor"] == 1e-12
    assert result["metric_definition"]["psnr_cap_db"] == 120.0
    assert result["metric_definition"]["ssim_data_range"] == 1.0
    assert result["metric_definition"]["ssim_win_size"] == 7
    assert result["metric_definition"]["ssim_win_size_policy"] == (
        "fixed-7; requires H and W >= 7"
    )


@pytest.mark.parametrize(
    ("gt", "recon", "match"),
    [
        (
            np.zeros((2, 8, 9)),
            np.zeros((2, 8, 8)),
            "same shape",
        ),
        (
            np.zeros((8, 9)),
            np.zeros((8, 9)),
            r"\[T,H,W\]",
        ),
        (
            np.zeros((0, 8, 9)),
            np.zeros((0, 8, 9)),
            "at least one frame",
        ),
        (
            np.full((2, 8, 9), np.nan),
            np.zeros((2, 8, 9)),
            "finite",
        ),
        (
            np.zeros((2, 8, 9)),
            np.full((2, 8, 9), np.inf),
            "finite",
        ),
        (
            np.full((2, 8, 9), "not-numeric"),
            np.zeros((2, 8, 9)),
            "real numeric",
        ),
    ],
)
def test_pair_input_contract_rejects_malformed_arrays(gt, recon, match):
    with pytest.raises((TypeError, ValueError), match=match):
        fit_global_affine(gt, recon)


def test_pair_input_contract_requires_numpy_arrays():
    values = [[[0.0] * 9] * 8] * 2

    with pytest.raises(TypeError, match="numpy arrays"):
        fit_global_affine(values, np.zeros((2, 8, 9)))


@pytest.mark.parametrize("shape", [(1, 0, 8), (1, 8, 0)])
def test_video_contract_rejects_empty_spatial_dimensions(shape):
    empty = np.zeros(shape)
    match = "spatial dimensions H and W must be positive"

    with pytest.raises(ValueError, match=match):
        fit_global_affine(empty, empty)
    with pytest.raises(ValueError, match=match):
        apply_global_affine(empty, 1.0, 0.0)
    with pytest.raises(ValueError, match=match):
        evaluate_video_legacy_per_frame(empty, empty)
    with pytest.raises(ValueError, match=match):
        evaluate_video_global_affine(empty, empty)


@pytest.mark.parametrize("shape", [(2, 6, 9), (2, 8, 6)])
def test_global_evaluator_rejects_spatial_dimensions_too_small_for_ssim(shape):
    gt = np.zeros(shape)

    with pytest.raises(ValueError, match="SSIM requires spatial dimensions"):
        evaluate_video_global_affine(gt, gt)


@pytest.mark.parametrize(
    ("recon", "slope", "intercept", "match"),
    [
        (np.zeros((8, 9)), 1.0, 0.0, r"\[T,H,W\]"),
        (np.full((2, 8, 9), np.nan), 1.0, 0.0, "finite"),
        (np.zeros((2, 8, 9)), np.inf, 0.0, "finite slope"),
        (np.zeros((2, 8, 9)), 1.0, np.nan, "finite intercept"),
    ],
)
def test_apply_global_affine_rejects_invalid_inputs(
    recon, slope, intercept, match
):
    with pytest.raises((TypeError, ValueError), match=match):
        apply_global_affine(recon, slope, intercept)


def test_metrics_v1_payload_is_strict_json_native():
    gt = np.linspace(0.0, 1.0, 2 * 8 * 9).reshape(2, 8, 9)

    payload = evaluate_video_global_affine(gt, gt)

    encoded = json.dumps(payload, allow_nan=False)
    assert json.loads(encoded) == payload


def test_common_evaluate_video_remains_legacy_compatible():
    frame = np.linspace(0.0, 1.0, 8 * 9).reshape(8, 9)
    gt = np.stack([frame, frame])
    recon = np.stack([0.5 * frame, 2.0 * frame])

    per_frame, mean_psnr = evaluate_video(gt, recon)

    assert per_frame == pytest.approx([60.0, 60.0])
    assert mean_psnr == pytest.approx(60.0)


def test_train_writes_separate_primary_and_labelled_compatibility_payloads(
    tmp_path,
):
    from train import _write_metrics_json, _write_results_json

    gt = np.linspace(0.0, 1.0, 2 * 8 * 9).reshape(2, 8, 9)
    recon = (gt - 0.2) / 1.7
    metrics_path = tmp_path / "metrics.json"
    results_path = tmp_path / "results.json"
    compatibility = {
        "solver": "admm",
        "mean_psnr": 34.5,
        "per_frame_psnr": [34.0, 35.0],
        "dgi_psnr": 12.25,
    }

    _write_metrics_json(metrics_path, gt, recon)
    _write_results_json(results_path, compatibility, [{"loss_data": 1.0}])

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    results = json.loads(results_path.read_text(encoding="utf-8"))
    assert metrics["definition_version"] == "metrics-v1"
    assert {
        "psnr_global_affine",
        "ssim_global_affine",
        "nrmse_global_affine_l2",
        "psnr_legacy_per_frame_minmax",
        "alignment",
        "metric_definition",
    } <= metrics.keys()
    assert "mean_psnr" not in metrics
    assert results["metric_definition_version"] == (
        "legacy-per-frame-minmax-v1"
    )
    assert results["mean_psnr"] == 34.5
    assert results["per_frame_psnr"] == [34.0, 35.0]
    assert results["dgi_psnr"] == 12.25
    assert results["mean_psnr_legacy_per_frame_minmax"] == 34.5
    assert results["per_frame_psnr_legacy_per_frame_minmax"] == [34.0, 35.0]
    assert results["dgi_psnr_legacy_canonical_minmax_60db"] == 12.25
    assert "psnr_global_affine" not in results
    assert metrics_path.read_bytes() != results_path.read_bytes()


def test_real_baseline_writer_labels_root_and_legacy_row_aliases(tmp_path):
    from scripts.run_baselines import _write_baselines_json

    path = tmp_path / "baselines.json"
    summary = {
        "name": "fixture",
        "baselines": {
            "static_cs": {
                "mean_psnr": 22.5,
                "per_frame_psnr": [22.0, 23.0],
            }
        },
    }

    _write_baselines_json(path, summary)

    payload = json.loads(path.read_text(encoding="utf-8"))
    row = payload["baselines"]["static_cs"]
    assert payload["metric_definition_version"] == (
        "legacy-per-frame-minmax-v1"
    )
    assert row["mean_psnr"] == 22.5
    assert row["per_frame_psnr"] == [22.0, 23.0]
    assert row["mean_psnr_legacy_per_frame_minmax"] == 22.5
    assert row["per_frame_psnr_legacy_per_frame_minmax"] == [22.0, 23.0]
    assert "psnr_global_affine" not in row
