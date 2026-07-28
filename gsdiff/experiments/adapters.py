"""Capability-safe in-process adapters for the seven baseline methods."""
from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
import hashlib
import random

import numpy as np
import torch

from gsdiff.data._artifact_models import SPIAcquisitionData

from .child_outputs import MethodChildResult
from .methods import AlgorithmSeed, ResolvedMethod


BASELINE_METHOD_IDS = (
    "dgi", "static_cs", "perframe_cs", "tv3d", "monin", "gidc3dtv", "recinr",
)


@contextmanager
def _algorithm_rng(seed: AlgorithmSeed) -> Iterator[None]:
    """Seed a method without leaking a changed caller RNG state."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        random.seed(seed.seed_u32)
        np.random.seed(seed.seed_u32)
        torch.manual_seed(seed.seed_u32)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed.seed_u32)
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _require_positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_real_array(
    value: object, name: str, expected_shape: tuple[int, ...]
) -> np.ndarray:
    if type(value) is not np.ndarray or value.shape != expected_shape:
        raise ValueError(f"{name} shape must be {expected_shape}")
    if (
        np.issubdtype(value.dtype, np.bool_)
        or not np.issubdtype(value.dtype, np.number)
        or np.iscomplexobj(value)
        or not np.isfinite(value).all()
    ):
        raise ValueError(f"{name} must be finite real numeric")
    return value


def _validate_indices(
    value: object, name: str, expected_shape: tuple[int, ...], T: int
) -> np.ndarray:
    if (
        type(value) is not np.ndarray
        or value.shape != expected_shape
        or np.issubdtype(value.dtype, np.bool_)
        or not np.issubdtype(value.dtype, np.integer)
        or np.any(value < 0)
        or np.any(value >= T)
    ):
        raise ValueError(f"{name} must be integer frame indices in [0,T)")
    return value


def _triple_keys(
    patterns: np.ndarray, measurements: np.ndarray, indices: np.ndarray
) -> set[bytes]:
    return {
        hashlib.sha256(
            np.ascontiguousarray(pattern, dtype=np.float64).tobytes()
            + float(measurement).hex().encode("ascii")
            + int(index).to_bytes(8, "big", signed=True)
        ).digest()
        for pattern, measurement, index in zip(patterns, measurements, indices)
    }


def _validate_acquisition(
    acquisition: SPIAcquisitionData, *, require_holdout: bool
) -> None:
    H = _require_positive_int(acquisition.H, "H")
    W = _require_positive_int(acquisition.W, "W")
    T = _require_positive_int(acquisition.T, "T")
    K = _require_positive_int(acquisition.K, "K")
    if type(acquisition.patterns) is not np.ndarray or acquisition.patterns.ndim != 3:
        raise ValueError("patterns must have rank 3")
    if acquisition.patterns.shape[0] != K:
        raise ValueError("K does not match the training row count")
    patterns = _validate_real_array(
        acquisition.patterns, "patterns", (K, H, W)
    )
    measurements = _validate_real_array(
        acquisition.measurements, "measurements", (K,)
    )
    indices = _validate_indices(
        acquisition.frame_indices, "frame_indices", (K,), T
    )
    time_grid = _validate_real_array(
        acquisition.time_grid, "time_grid", (T,)
    )
    if T > 1 and not np.all(time_grid[1:] > time_grid[:-1]):
        raise ValueError("time_grid must be strictly increasing")

    holdout = (
        acquisition.holdout_patterns,
        acquisition.holdout_measurements,
        acquisition.holdout_frame_indices,
    )
    present = tuple(value is not None for value in holdout)
    if not any(present):
        if acquisition.holdout_K != 0:
            raise ValueError("holdout_K must be zero when holdout arrays are absent")
        if require_holdout:
            raise ValueError("a distinct holdout set is required for selection")
        return
    if not all(present):
        raise ValueError("holdout arrays must be all present or all absent")
    if type(acquisition.holdout_K) is not int or acquisition.holdout_K <= 0:
        raise ValueError("holdout_K must be positive for a present holdout")
    holdout_K = acquisition.holdout_K
    if type(holdout[0]) is not np.ndarray or holdout[0].ndim != 3:
        raise ValueError("holdout_patterns shape must have rank 3")
    if holdout[0].shape[0] != holdout_K:
        raise ValueError("holdout_K does not match the holdout row count")
    holdout_patterns = _validate_real_array(
        holdout[0], "holdout_patterns", (holdout_K, H, W)
    )
    holdout_measurements = _validate_real_array(
        holdout[1], "holdout_measurements", (holdout_K,)
    )
    holdout_indices = _validate_indices(
        holdout[2], "holdout_frame_indices", (holdout_K,), T
    )
    if (
        np.shares_memory(patterns, holdout_patterns)
        or np.shares_memory(measurements, holdout_measurements)
        or np.shares_memory(indices, holdout_indices)
    ):
        raise ValueError("holdout arrays must be distinct from training arrays")
    if _triple_keys(patterns, measurements, indices) & _triple_keys(
        holdout_patterns, holdout_measurements, holdout_indices
    ):
        raise ValueError("holdout triples reuse training data")


def _base_info(
    method: ResolvedMethod, *, parameter_count: int, unit: str, budget: int,
    selected_hyperparameters, selection, native_motion_model: str = "none",
) -> dict[str, object]:
    return {
        "parameter_count": parameter_count,
        "native_iteration_unit": unit,
        "native_iteration_budget": budget,
        "convergence_status": method.convergence_status,
        "selected_hyperparameters": selected_hyperparameters,
        "selection": selection,
        "checkpoint_hashes": [],
        "native_motion_model": native_motion_model,
    }


def _result(
    method: ResolvedMethod, reconstruction, *, dgi=None, motion=None,
    info: dict[str, object], history=(),
) -> MethodChildResult:
    return MethodChildResult(
        method_id=method.method_id,
        reconstruction=np.asarray(reconstruction),
        estimated_motion_trajectory=(
            None if motion is None else np.asarray(motion)
        ),
        dgi=None if dgi is None else np.asarray(dgi),
        info=info,
        history=tuple(history),
    )


def run_dgi(
    acquisition: SPIAcquisitionData, semantic_config,
    algorithm_seed: AlgorithmSeed, device: str,
):
    del semantic_config, algorithm_seed, device
    from gsdiff.baselines.common import (
        calibrate_reconstruction_physical,
        dgi_image,
    )
    conditioned = dgi_image(
        acquisition.patterns, acquisition.measurements
    ).numpy()
    video = np.repeat(conditioned[None], acquisition.T, axis=0)
    physical = calibrate_reconstruction_physical(
        video,
        acquisition.patterns,
        acquisition.measurements,
        acquisition.frame_indices,
    )
    return physical, physical[0], None, {
        "selected_hyperparameters": None,
        "selection": None,
        "history": (),
    }


def _run_dgi_adapter(method, acquisition, algorithm_seed, device):
    reconstruction, dgi, motion, details = run_dgi(
        acquisition, method.semantic_config, algorithm_seed, device
    )
    info = _base_info(
        method,
        parameter_count=0,
        unit=str(method.semantic_config["native_unit"]),
        budget=int(method.semantic_config["native_budget"]),
        selected_hyperparameters=None,
        selection=None,
    )
    return _result(
        method, reconstruction, dgi=dgi, motion=motion, info=info,
        history=details["history"],
    )


def _run_static_cs_adapter(method, acquisition, algorithm_seed, device):
    from gsdiff.baselines.cs import run_static_cs
    reconstruction, details = run_static_cs(
        acquisition, method.semantic_config, algorithm_seed, device
    )
    info = _base_info(
        method, parameter_count=acquisition.H * acquisition.W,
        unit="admm-iteration",
        budget=int(method.semantic_config["solver"]["n_admm"]),
        selected_hyperparameters=details["selected_hyperparameters"],
        selection=details["selection"],
    )
    return _result(
        method, reconstruction, info=info, history=details.get("history", ())
    )


def _run_perframe_cs_adapter(method, acquisition, algorithm_seed, device):
    from gsdiff.baselines.cs import run_perframe_cs
    reconstruction, details = run_perframe_cs(
        acquisition, method.semantic_config, algorithm_seed, device
    )
    info = _base_info(
        method, parameter_count=acquisition.T * acquisition.H * acquisition.W,
        unit="admm-iteration",
        budget=int(method.semantic_config["solver"]["n_admm"]),
        selected_hyperparameters=details["selected_hyperparameters"],
        selection=details["selection"],
    )
    return _result(
        method, reconstruction, info=info, history=details.get("history", ())
    )


def _run_tv3d_adapter(method, acquisition, algorithm_seed, device):
    from gsdiff.baselines.tv3d import run_tv3d
    reconstruction, details = run_tv3d(
        acquisition, method.semantic_config, algorithm_seed, device
    )
    info = _base_info(
        method, parameter_count=acquisition.T * acquisition.H * acquisition.W,
        unit="primal-dual-iteration",
        budget=int(method.semantic_config["solver"]["iterations"]),
        selected_hyperparameters=details["selected_hyperparameters"],
        selection=details["selection"],
    )
    return _result(
        method, reconstruction, info=info, history=details.get("history", ())
    )


def _run_monin_adapter(method, acquisition, algorithm_seed, device):
    from gsdiff.baselines.monin import run_monin
    reconstruction, motion, details = run_monin(
        acquisition, method.semantic_config, algorithm_seed, device
    )
    degree = int(method.semantic_config["solver"]["polynomial_degree"])
    info = _base_info(
        method,
        parameter_count=acquisition.H * acquisition.W + 2 * (degree + 1),
        unit="admm-iteration",
        budget=int(method.semantic_config["solver"]["n_admm"]),
        selected_hyperparameters=details["selected_hyperparameters"],
        selection=details["selection"],
        native_motion_model="translation-polynomial",
    )
    return _result(
        method, reconstruction, motion=motion, info=info,
        history=details.get("history", ()),
    )


def _run_gidc_adapter(method, acquisition, algorithm_seed, device):
    from gsdiff.baselines.gidc import run_gidc3dtv
    reconstruction, details = run_gidc3dtv(
        acquisition, method.semantic_config, algorithm_seed, device
    )
    info = _base_info(
        method, parameter_count=int(details["parameter_count"]),
        unit="adam-step",
        budget=int(method.semantic_config["solver"]["n_steps"]),
        selected_hyperparameters=details["selected_hyperparameters"],
        selection=details["selection"],
    )
    return _result(
        method, reconstruction, info=info, history=details.get("history", ())
    )


def _run_recinr_adapter(method, acquisition, algorithm_seed, device):
    from gsdiff.baselines.recinr import run_recinr
    reconstruction, details = run_recinr(
        acquisition, method.semantic_config, algorithm_seed, device
    )
    solver = method.semantic_config["solver"]
    info = _base_info(
        method, parameter_count=int(details["parameter_count"]),
        unit="optimization-step",
        budget=int(
            solver["warm_steps"] + solver["flow_steps"] + solver["joint_steps"]
        ),
        selected_hyperparameters=None,
        selection=None,
    )
    return _result(
        method, reconstruction, info=info, history=details.get("history", ())
    )


_Runner = Callable[
    [ResolvedMethod, SPIAcquisitionData, AlgorithmSeed, str],
    MethodChildResult,
]
_RUNNER_TABLE: dict[str, _Runner] = {
    "dgi": _run_dgi_adapter,
    "static_cs": _run_static_cs_adapter,
    "perframe_cs": _run_perframe_cs_adapter,
    "tv3d": _run_tv3d_adapter,
    "monin": _run_monin_adapter,
    "gidc3dtv": _run_gidc_adapter,
    "recinr": _run_recinr_adapter,
}
if tuple(_RUNNER_TABLE) != BASELINE_METHOD_IDS:
    raise RuntimeError("baseline runner table does not match canonical IDs")


def run_baseline_method(
    method: ResolvedMethod,
    acquisition: SPIAcquisitionData,
    *,
    algorithm_seed: AlgorithmSeed,
    device: str,
) -> MethodChildResult:
    """Run one canonical baseline without evaluator or truth capabilities."""
    if method.method_id not in BASELINE_METHOD_IDS:
        raise ValueError(f"unsupported baseline method: {method.method_id}")
    if method.execution_family != "baseline":
        raise ValueError("baseline adapter requires the baseline execution family")
    if type(acquisition) is not SPIAcquisitionData:
        raise TypeError("acquisition must be SPIAcquisitionData")
    if type(algorithm_seed) is not AlgorithmSeed:
        raise TypeError("algorithm_seed must be AlgorithmSeed")
    _validate_acquisition(
        acquisition, require_holdout=method.method_id != "dgi"
    )
    runner = _RUNNER_TABLE[method.method_id]
    with _algorithm_rng(algorithm_seed):
        return runner(method, acquisition, algorithm_seed, device)
