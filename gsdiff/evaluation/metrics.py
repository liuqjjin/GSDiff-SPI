"""Video evaluation metrics with explicit definition versions.

``metrics-v1`` fits one affine calibration over every pixel in the complete
video. The fitted slope is constrained to be nonnegative by selecting the
boundary solution ``slope=0, intercept=mean(gt)`` when the unconstrained slope
is negative. The same boundary is selected for numerically constant
predictions, defined deterministically as::

    variance(recon) <= eps64 * max(1, max(abs(recon - mean(recon))))**2

The aligned video is clipped to [0, 1] once and that exact float64 array is used
for PSNR, SSIM, and normalized RMSE. Legacy per-frame min-max PSNR remains an
explicitly named compatibility metric with its historical 60-dB sentinel.
"""

import numpy as np
from skimage.metrics import structural_similarity

from ..utils import normalize_01_legacy_minmax, psnr_legacy_60db


_FLOAT64_EPS = float(np.finfo(np.float64).eps)
_FLOAT64_MAX = float(np.finfo(np.float64).max)
_PSNR_MSE_FLOOR = 1e-12
_PSNR_CAP_DB = 120.0
_SSIM_WIN_SIZE = 7


def _safe_input_abs_bound(size: int) -> float:
    """Bound values so fourth-order image-metric arithmetic stays finite."""
    return float((_FLOAT64_MAX / (16.0 * size)) ** 0.25)


def _as_float64_video(array: np.ndarray, name: str) -> np.ndarray:
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} and its counterpart must be numpy arrays")
    if array.ndim != 3:
        raise ValueError(f"{name} must have shape [T,H,W]")
    if array.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one frame")
    if array.shape[1] == 0 or array.shape[2] == 0:
        raise ValueError(
            f"{name} spatial dimensions H and W must be positive"
        )
    if not (
        np.issubdtype(array.dtype, np.number)
        and not np.issubdtype(array.dtype, np.complexfloating)
    ):
        raise TypeError(f"{name} must be a real numeric numpy array")
    result = array.astype(np.float64, copy=False)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _validated_pair(
    gt: np.ndarray, recon: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    gt64 = _as_float64_video(gt, "gt")
    recon64 = _as_float64_video(recon, "recon")
    if gt64.shape != recon64.shape:
        raise ValueError(
            f"gt and recon must have the same shape [T,H,W], got "
            f"{gt64.shape} and {recon64.shape}"
        )
    return gt64, recon64


def _require_primary_evaluable(array: np.ndarray, name: str) -> None:
    safe_bound = _safe_input_abs_bound(array.size)
    if float(np.max(np.abs(array))) > safe_bound:
        raise ValueError(
            f"{name} magnitude exceeds the safe evaluability bound "
            f"{safe_bound:.17g}"
        )


def _validated_primary_pair(
    gt: np.ndarray, recon: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    gt64, recon64 = _validated_pair(gt, recon)
    _require_primary_evaluable(gt64, "gt")
    _require_primary_evaluable(recon64, "recon")
    return gt64, recon64


def _centered_values(
    values: np.ndarray,
) -> tuple[float, np.ndarray, float]:
    """Center without summing the large common offset."""
    reference = float(values.flat[0])
    mean = reference + float(np.mean(values - reference))
    centered = values - mean
    scale = float(np.max(np.abs(centered)))
    return mean, centered, scale


def _scaled_variance(centered: np.ndarray, scale: float) -> float:
    if scale == 0.0:
        return 0.0
    unit = centered / scale
    return float(scale * scale * np.mean(unit * unit))


def _prediction_variance_threshold(recon64: np.ndarray) -> float:
    _, _, centered_scale = _centered_values(recon64)
    return float(
        _FLOAT64_EPS * max(1.0, centered_scale) ** 2
    )


def _require_finite_numbers(value: object, path: str = "result") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _require_finite_numbers(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _require_finite_numbers(child, f"{path}[{index}]")
    elif isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            raise ValueError(f"{path} is not finite")


def fit_global_affine(
    gt: np.ndarray,
    recon: np.ndarray,
    *,
    nonnegative_slope: bool = True,
) -> tuple[float, float]:
    """Fit one least-squares ``slope * recon + intercept`` for a full video."""
    gt64, recon64 = _validated_primary_pair(gt, recon)
    if not isinstance(nonnegative_slope, (bool, np.bool_)):
        raise TypeError("nonnegative_slope must be a boolean")

    gt_mean, gt_centered, gt_scale = _centered_values(gt64)
    recon_mean, recon_centered, recon_scale = _centered_values(recon64)
    prediction_variance = _scaled_variance(
        recon_centered, recon_scale
    )
    if prediction_variance <= _prediction_variance_threshold(recon64):
        return 0.0, gt_mean

    recon_unit = recon_centered / recon_scale
    recon_unit_variance = float(np.mean(recon_unit * recon_unit))
    if gt_scale == 0.0:
        slope = 0.0
    else:
        gt_unit = gt_centered / gt_scale
        scaled_covariance = float(np.mean(recon_unit * gt_unit))
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            slope = (
                (gt_scale / recon_scale)
                * (scaled_covariance / recon_unit_variance)
            )
    if not np.isfinite(slope):
        raise ValueError("global affine slope cannot be represented finitely")
    if nonnegative_slope and slope < 0.0:
        return 0.0, gt_mean
    with np.errstate(over="ignore", invalid="ignore"):
        intercept = gt_mean - slope * recon_mean
    if not np.isfinite(intercept):
        raise ValueError(
            "global affine intercept cannot be represented finitely"
        )
    return float(slope), float(intercept)


def apply_global_affine(
    recon: np.ndarray, slope: float, intercept: float
) -> np.ndarray:
    """Apply and clip a global affine calibration, returning float64."""
    recon64 = _as_float64_video(recon, "recon")
    _require_primary_evaluable(recon64, "recon")
    if not np.isscalar(slope) or not np.isreal(slope):
        raise TypeError("slope must be a real scalar")
    if not np.isscalar(intercept) or not np.isreal(intercept):
        raise TypeError("intercept must be a real scalar")
    slope = float(slope)
    intercept = float(intercept)
    if not np.isfinite(slope):
        raise ValueError("slope must be a finite slope")
    if not np.isfinite(intercept):
        raise ValueError("intercept must be a finite intercept")
    with np.errstate(over="ignore", invalid="ignore"):
        transformed = slope * recon64 + intercept
    if not np.all(np.isfinite(transformed)):
        raise ValueError("global affine application produced nonfinite values")
    return np.clip(transformed, 0.0, 1.0)


def evaluate_video_legacy_per_frame(
    gt: np.ndarray, recon: np.ndarray
) -> dict[str, object]:
    """Evaluate the historical per-frame min-max PSNR compatibility metric."""
    gt64, recon64 = _validated_pair(gt, recon)
    per_frame = [
        float(
            psnr_legacy_60db(
                normalize_01_legacy_minmax(np.clip(recon64[t], 0.0, None)),
                normalize_01_legacy_minmax(gt64[t]),
            )
        )
        for t in range(gt64.shape[0])
    ]
    payload = {
        "definition_version": "legacy-per-frame-minmax-v1",
        "psnr_legacy_per_frame_minmax": float(np.mean(per_frame)),
        "per_frame_psnr_legacy_per_frame_minmax": per_frame,
        "metric_definition": {
            "reconstruction_preclip_min": 0.0,
            "normalization_scope": "independent-per-frame-minmax",
            "constant_range_threshold": 1e-8,
            "psnr_exact_recovery_sentinel_db": 60.0,
            "psnr_mse_sentinel_threshold": 1e-12,
        },
    }
    _require_finite_numbers(payload)
    return payload


def evaluate_video_global_affine(
    gt: np.ndarray, recon: np.ndarray
) -> dict[str, object]:
    """Evaluate ``metrics-v1`` primary metrics and the labelled legacy PSNR."""
    gt64, recon64 = _validated_primary_pair(gt, recon)
    if gt64.shape[1] < _SSIM_WIN_SIZE or gt64.shape[2] < _SSIM_WIN_SIZE:
        raise ValueError(
            "SSIM requires spatial dimensions H and W >= 7; "
            f"got H={gt64.shape[1]}, W={gt64.shape[2]}"
        )

    slope, intercept = fit_global_affine(gt64, recon64)
    aligned = apply_global_affine(recon64, slope, intercept)
    error = aligned - gt64
    mse = float(np.mean(np.square(error)))
    psnr = float(-10.0 * np.log10(max(mse, _PSNR_MSE_FLOOR)))
    per_frame_ssim = [
        float(
            structural_similarity(
                gt64[t],
                aligned[t],
                data_range=1.0,
                win_size=_SSIM_WIN_SIZE,
            )
        )
        for t in range(gt64.shape[0])
    ]
    nrmse = float(
        np.linalg.norm(error)
        / max(float(np.linalg.norm(gt64)), _FLOAT64_EPS)
    )
    legacy = evaluate_video_legacy_per_frame(gt64, recon64)

    payload = {
        "psnr_global_affine": psnr,
        "ssim_global_affine": float(np.mean(per_frame_ssim)),
        "nrmse_global_affine_l2": nrmse,
        "psnr_legacy_per_frame_minmax": legacy[
            "psnr_legacy_per_frame_minmax"
        ],
        "alignment": {
            "slope": slope,
            "intercept": intercept,
        },
        "definition_version": "metrics-v1",
        "metric_definition": {
            "alignment_scope": "one-affine-fit-over-all-video-pixels",
            "nonnegative_slope": True,
            "prediction_variance_threshold": _prediction_variance_threshold(
                recon64
            ),
            "prediction_variance_threshold_policy": (
                "variance <= eps64 * "
                "max(1, max(abs(recon - mean(recon))))**2"
            ),
            "input_abs_evaluability_bound": _safe_input_abs_bound(
                recon64.size
            ),
            "input_abs_evaluability_bound_policy": (
                "max(abs(input)) <= "
                "(float64_max / (16 * number_of_video_values))**0.25"
            ),
            "alignment_output_clip": [0.0, 1.0],
            "psnr_data_range": 1.0,
            "psnr_mse_floor": _PSNR_MSE_FLOOR,
            "psnr_cap_db": _PSNR_CAP_DB,
            "ssim_data_range": 1.0,
            "ssim_win_size": _SSIM_WIN_SIZE,
            "ssim_win_size_policy": "fixed-7; requires H and W >= 7",
            "nrmse_norm": "l2",
            "nrmse_denominator_epsilon": _FLOAT64_EPS,
        },
    }
    _require_finite_numbers(payload)
    return payload


def validate_metrics_v1_payload(
    value: object,
    reconstruction: np.ndarray,
) -> None:
    """Validate truth-independent metrics-v1 structure and policy constants."""
    if type(value) is not dict or set(value) != {
        "psnr_global_affine",
        "ssim_global_affine",
        "nrmse_global_affine_l2",
        "psnr_legacy_per_frame_minmax",
        "alignment",
        "definition_version",
        "metric_definition",
    }:
        raise ValueError("metrics-v1 top-level shape is invalid")
    if value["definition_version"] != "metrics-v1":
        raise ValueError("metrics-v1 definition version is invalid")
    alignment = value["alignment"]
    if type(alignment) is not dict or set(alignment) != {"slope", "intercept"}:
        raise ValueError("metrics-v1 alignment shape is invalid")
    for field in (
        "psnr_global_affine",
        "ssim_global_affine",
        "nrmse_global_affine_l2",
        "psnr_legacy_per_frame_minmax",
    ):
        if type(value[field]) not in (int, float) or not np.isfinite(value[field]):
            raise ValueError(f"metrics-v1 {field} must be finite")
    for field in ("slope", "intercept"):
        if type(alignment[field]) not in (int, float) or not np.isfinite(
            alignment[field]
        ):
            raise ValueError(f"metrics-v1 alignment {field} must be finite")
    if alignment["slope"] < 0:
        raise ValueError("metrics-v1 alignment slope must be nonnegative")
    if value["nrmse_global_affine_l2"] < 0:
        raise ValueError("metrics-v1 NRMSE must be nonnegative")
    if not -1.0 <= value["ssim_global_affine"] <= 1.0:
        raise ValueError("metrics-v1 SSIM must be in [-1,1]")
    recon64 = _as_float64_video(reconstruction, "reconstruction")
    expected_definition = {
        "alignment_scope": "one-affine-fit-over-all-video-pixels",
        "nonnegative_slope": True,
        "prediction_variance_threshold": _prediction_variance_threshold(recon64),
        "prediction_variance_threshold_policy": (
            "variance <= eps64 * max(1, max(abs(recon - mean(recon))))**2"
        ),
        "input_abs_evaluability_bound": _safe_input_abs_bound(recon64.size),
        "input_abs_evaluability_bound_policy": (
            "max(abs(input)) <= "
            "(float64_max / (16 * number_of_video_values))**0.25"
        ),
        "alignment_output_clip": [0.0, 1.0],
        "psnr_data_range": 1.0,
        "psnr_mse_floor": _PSNR_MSE_FLOOR,
        "psnr_cap_db": _PSNR_CAP_DB,
        "ssim_data_range": 1.0,
        "ssim_win_size": _SSIM_WIN_SIZE,
        "ssim_win_size_policy": "fixed-7; requires H and W >= 7",
        "nrmse_norm": "l2",
        "nrmse_denominator_epsilon": _FLOAT64_EPS,
    }
    if value["metric_definition"] != expected_definition:
        raise ValueError("metrics-v1 policy definition is invalid")
    _require_finite_numbers(value)
