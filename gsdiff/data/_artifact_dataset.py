"""Acquisition/truth identity construction and schema codecs."""

import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from ._artifact_identity import (
    ArtifactValidationError,
    acquisition_identity_spec,
    array_descriptor,
    canonical_json_bytes,
    optional_readonly_array,
    readonly_array,
    sha256_bytes,
    validate_acquisition_identity_spec,
    validate_array_descriptor,
    validate_exact_json_native,
    validate_exact_keys,
    validate_exact_int,
    validate_generation_config,
    validate_index_array,
    validate_real_finite_array,
    validate_sha256,
    validate_time_grid,
    validate_path_free_opaque_id,
)
from ._artifact_io import (
    METADATA_MEMBER,
    decode_metadata,
    load_array_member,
    npz_bytes,
    read_npz_members,
    read_npz_members_bytes,
    write_npz,
)
from ._artifact_models import EvaluationTruth, SPIAcquisitionData


ACQUISITION_SCHEMA = "measurements-blind-v1"
_LEGACY_IDENTITY_SCHEMA = "measurements-v1"
BLIND_ACQUISITION_SPEC_SCHEMA = "blind-acquisition-spec-v1"
_ACQUISITION_KEYS = {
    "pattern_family",
    "pattern_values",
    "pattern_order",
    "time_assignment",
    "holdout_pattern_family",
    "noise_convention",
    "noise_sigma_absolute",
}
_BLIND_PATTERN_FAMILIES = frozenset(
    {
        "bernoulli",
        "gaussian",
        "random",
        "hadamard",
        "hadamard_cc",
        "hadamard_walsh",
        "hadamard_natural",
        "fourier",
        "s_matrix",
        "s_matrix_m",
    }
)
_ACQUISITION_MEMBER_ALLOWLIST = {
    METADATA_MEMBER,
    "patterns.npy",
    "measurements.npy",
    "frame_indices.npy",
    "time_grid.npy",
    "holdout_patterns.npy",
    "holdout_measurements.npy",
    "holdout_frame_indices.npy",
}


def _acquisition_arrays(
    data: SPIAcquisitionData,
) -> Mapping[str, np.ndarray | None]:
    return {
        "patterns": data.patterns,
        "measurements": data.measurements,
        "frame_indices": data.frame_indices,
        "time_grid": data.time_grid,
        "holdout_patterns": data.holdout_patterns,
        "holdout_measurements": data.holdout_measurements,
        "holdout_frame_indices": data.holdout_frame_indices,
    }


def _validate_acquisition_shapes(data: SPIAcquisitionData) -> None:
    for name in ("H", "W", "T", "K"):
        validate_exact_int(getattr(data, name), name, minimum=1)
    validate_exact_int(data.holdout_K, "holdout_K", minimum=0)
    if min(data.H, data.W, data.T, data.K) <= 0:
        raise ArtifactValidationError("H, W, T, and K must be positive")
    expected_shapes = {
        "patterns": (data.K, data.H, data.W),
        "measurements": (data.K,),
        "frame_indices": (data.K,),
        "time_grid": (data.T,),
    }
    for name, shape in expected_shapes.items():
        if getattr(data, name).shape != shape:
            raise ArtifactValidationError(
                f"{name} shape must be {shape}, got {getattr(data, name).shape}"
            )
    validate_real_finite_array(
        data.patterns, "patterns", shape=expected_shapes["patterns"]
    )
    validate_real_finite_array(
        data.measurements,
        "measurements",
        shape=expected_shapes["measurements"],
    )
    validate_index_array(
        data.frame_indices,
        "frame_indices",
        shape=expected_shapes["frame_indices"],
        upper_bound=data.T,
    )
    validate_time_grid(data.time_grid, "time_grid", length=data.T)
    optional = (
        data.holdout_patterns,
        data.holdout_measurements,
        data.holdout_frame_indices,
    )
    if any(value is None for value in optional) and not all(
        value is None for value in optional
    ):
        raise ArtifactValidationError(
            "all holdout arrays must be present or all must be absent"
        )
    if optional[0] is not None:
        count = optional[0].shape[0]
        if (
            optional[0].shape != (count, data.H, data.W)
            or optional[1].shape != (count,)
            or optional[2].shape != (count,)
        ):
            raise ArtifactValidationError("holdout array shapes do not agree")
        validate_real_finite_array(
            data.holdout_patterns,
            "holdout_patterns",
            shape=(count, data.H, data.W),
        )
        validate_real_finite_array(
            data.holdout_measurements,
            "holdout_measurements",
            shape=(count,),
        )
        validate_index_array(
            data.holdout_frame_indices,
            "holdout_frame_indices",
            shape=(count,),
            upper_bound=data.T,
        )
    else:
        count = 0
    if data.holdout_K != count:
        raise ArtifactValidationError(
            "holdout_K disagrees with holdout array presence or count"
        )


def _validate_blind_acquisition(value: object) -> Mapping[str, object]:
    if type(value) not in (dict, MappingProxyType) or any(
        type(key) is not str for key in value
    ):
        raise ArtifactValidationError("acquisition must be an exact JSON object")
    validate_exact_keys(value, _ACQUISITION_KEYS, "acquisition")
    for name in (
        "pattern_family",
        "pattern_order",
        "time_assignment",
        "holdout_pattern_family",
        "noise_convention",
    ):
        if type(value[name]) is not str or not value[name]:
            raise ArtifactValidationError(
                f"acquisition.{name} must be a nonempty exact string"
            )
    if value["pattern_family"] not in _BLIND_PATTERN_FAMILIES:
        raise ArtifactValidationError(
            "acquisition.pattern_family is unsupported"
        )
    if value["pattern_order"] not in {"sequential", "stratified", "random"}:
        raise ArtifactValidationError(
            "acquisition.pattern_order is unsupported"
        )
    if value["time_assignment"] not in {"uniform", "interpolation"}:
        raise ArtifactValidationError(
            "acquisition.time_assignment is unsupported"
        )
    if value["holdout_pattern_family"] != "uniform-random":
        raise ArtifactValidationError(
            "acquisition.holdout_pattern_family is unsupported"
        )
    if value["noise_convention"] not in {
        "ac-variance-snr",
        "detector-absolute",
    }:
        raise ArtifactValidationError(
            "acquisition.noise_convention is unsupported"
        )
    pattern_values = value["pattern_values"]
    if type(pattern_values) not in (list, tuple) or not pattern_values:
        raise ArtifactValidationError(
            "acquisition.pattern_values must be a nonempty sequence"
        )
    if value["pattern_family"] == "gaussian":
        valid_pattern_values = (
            len(pattern_values) == 1 and type(pattern_values[0]) is str
        )
    else:
        valid_pattern_values = all(
            type(item) in (int, float) and math.isfinite(item)
            for item in pattern_values
        )
    if not valid_pattern_values:
        raise ArtifactValidationError(
            "acquisition.pattern_values contain invalid native values"
        )
    if list(pattern_values) != _pattern_values(value["pattern_family"]):
        raise ArtifactValidationError(
            "acquisition.pattern_values disagree with pattern_family"
        )
    sigma = value["noise_sigma_absolute"]
    if (
        type(sigma) not in (int, float)
        or not math.isfinite(sigma)
        or sigma < 0
    ):
        raise ArtifactValidationError(
            "acquisition.noise_sigma_absolute must be a finite nonnegative number"
        )
    canonical_json_bytes(value)
    return value


def _validate_blind_acquisition_spec(
    value: object,
) -> Mapping[str, object]:
    if type(value) not in (dict, MappingProxyType):
        raise ArtifactValidationError(
            "expected acquisition spec must be an exact object"
        )
    validate_exact_json_native(value, "expected acquisition spec")
    validate_exact_keys(
        value,
        {"schema_version", "dimensions", "acquisition"},
        "expected acquisition spec",
    )
    if value["schema_version"] != BLIND_ACQUISITION_SPEC_SCHEMA:
        raise ArtifactValidationError(
            "expected acquisition spec schema mismatch"
        )
    dimensions = value["dimensions"]
    if type(dimensions) not in (dict, MappingProxyType):
        raise ArtifactValidationError(
            "expected acquisition dimensions must be an exact object"
        )
    validate_exact_keys(
        dimensions,
        {"H", "W", "T", "K", "holdout_K"},
        "expected acquisition dimensions",
    )
    for name in ("H", "W", "T", "K"):
        validate_exact_int(
            dimensions[name],
            f"expected acquisition dimensions.{name}",
            minimum=1,
        )
    validate_exact_int(
        dimensions["holdout_K"],
        "expected acquisition dimensions.holdout_K",
        minimum=0,
    )
    try:
        _validate_blind_acquisition(value["acquisition"])
    except (ArtifactValidationError, TypeError) as error:
        raise ArtifactValidationError(
            "expected acquisition spec is invalid"
        ) from error
    return value


def _validate_acquisition_identity(data: SPIAcquisitionData) -> None:
    validate_sha256(data.dataset_identity_sha256, "dataset identity")
    _validate_acquisition_shapes(data)
    _validate_blind_acquisition(data.acquisition)
    arrays = {
        name: value
        for name, value in _acquisition_arrays(data).items()
        if value is not None
    }
    if not isinstance(data.array_descriptors, Mapping):
        raise ArtifactValidationError("array_descriptors must be an object")
    validate_exact_keys(
        data.array_descriptors,
        set(arrays),
        "acquisition array descriptors",
    )
    for name, array in arrays.items():
        validate_array_descriptor(
            name, array, data.array_descriptors[name]
        )


def blind_acquisition_spec(
    data: SPIAcquisitionData,
) -> Mapping[str, object]:
    _validate_acquisition_identity(data)
    return {
        "schema_version": BLIND_ACQUISITION_SPEC_SCHEMA,
        "dimensions": {
            "H": data.H,
            "W": data.W,
            "T": data.T,
            "K": data.K,
            "holdout_K": data.holdout_K,
        },
        "acquisition": data.acquisition,
    }


def _pattern_values(pattern_family: str) -> list[object]:
    if pattern_family.startswith("hadamard"):
        return [-1, 1]
    if pattern_family == "gaussian":
        return ["real"]
    return [0, 1]


def split_spi_data(
    data: Any,
    *,
    resolved_generation_config: Mapping[str, object],
    generator_code_version: str,
    target_asset_sha256: str,
) -> tuple[SPIAcquisitionData, EvaluationTruth]:
    config = validate_generation_config(resolved_generation_config)
    validate_sha256(target_asset_sha256, "target asset hash")
    validate_path_free_opaque_id(
        generator_code_version, "generator_code_version"
    )
    source_dimensions = []
    for name in ("H", "W", "T", "K"):
        value = getattr(data, name)
        if type(value) is not int or value <= 0:
            raise ArtifactValidationError(
                f"generated SPIData {name} must be an exact positive integer"
            )
        source_dimensions.append(value)
    dimensions = tuple(config[name] for name in ("H", "W", "T", "K"))
    if dimensions != tuple(source_dimensions):
        raise ArtifactValidationError(
            "resolved dimensions do not match generated SPIData"
        )
    pattern = config["pattern"]
    time_assignment = config["time_assignment"]
    noise = config["noise"]
    motion = config["motion"]
    motion_parameters = motion["parameters"]
    velocity = np.asarray(motion_parameters["velocity"], dtype=np.float64)
    acceleration = np.asarray(
        motion_parameters["acceleration"], dtype=np.float64
    )
    if velocity.shape != (2,) or acceleration.shape != (2,):
        raise ArtifactValidationError(
            "motion velocity and acceleration must each have shape [2]"
        )
    source_velocity = validate_real_finite_array(
        data.gt_velocity, "generated SPIData gt_velocity", shape=(2,)
    )
    if not np.issubdtype(source_velocity.dtype, np.floating):
        raise ArtifactValidationError(
            "generated SPIData gt_velocity must use a real floating dtype"
        )
    expected_source_velocity = np.asarray(
        motion_parameters["velocity"], dtype=source_velocity.dtype
    )
    if not np.array_equal(source_velocity, expected_source_velocity):
        raise ArtifactValidationError(
            "generated SPIData gt_velocity disagrees with resolved motion"
        )
    source_omega = validate_real_finite_array(
        data.gt_omega, "generated SPIData gt_omega", shape=()
    )
    if not np.issubdtype(source_omega.dtype, np.floating):
        raise ArtifactValidationError(
            "generated SPIData gt_omega must use a real floating dtype"
        )
    expected_source_omega = np.asarray(
        motion_parameters["omega"], dtype=source_omega.dtype
    )
    if not np.array_equal(source_omega, expected_source_omega):
        raise ArtifactValidationError(
            "generated SPIData gt_omega disagrees with resolved motion"
        )

    arrays = {
        "patterns": readonly_array(data.patterns, "patterns"),
        "measurements": readonly_array(data.measurements, "measurements"),
        "frame_indices": readonly_array(data.frame_idx, "frame_indices"),
        "time_grid": readonly_array(data.t_grid, "time_grid"),
        "holdout_patterns": optional_readonly_array(
            data.eval_patterns, "holdout_patterns"
        ),
        "holdout_measurements": optional_readonly_array(
            data.eval_measurements, "holdout_measurements"
        ),
        "holdout_frame_indices": optional_readonly_array(
            data.eval_frame_idx, "holdout_frame_indices"
        ),
    }
    holdout = config["holdout"]
    actual_holdout_present = all(
        arrays[name] is not None
        for name in (
            "holdout_patterns",
            "holdout_measurements",
            "holdout_frame_indices",
        )
    )
    actual_holdout_count = (
        arrays["holdout_patterns"].shape[0]
        if arrays["holdout_patterns"] is not None
        else 0
    )
    if (
        holdout["present"] != actual_holdout_present
        or holdout["count"] != actual_holdout_count
    ):
        raise ArtifactValidationError(
            "holdout config disagrees with generated array presence or count"
        )
    identity_spec = acquisition_identity_spec(
        arrays=arrays,
        H=data.H,
        W=data.W,
        T=data.T,
        K=data.K,
        resolved_generation_config=config,
        generator_code_version=generator_code_version,
        target_asset_sha256=target_asset_sha256,
        seed=config["seed"],
        pattern_family=pattern["family"],
        pattern_order=pattern["order"],
        time_assignment_mode=time_assignment["mode"],
        noise_convention=noise["convention"],
        noise_parameters=noise["parameters"],
        motion_model=motion["model"],
        motion_parameters=motion_parameters,
        schema=_LEGACY_IDENTITY_SCHEMA,
    )
    dataset_identity_sha256 = sha256_bytes(canonical_json_bytes(identity_spec))
    sigma_absolute = noise["parameters"]["sigma_abs"]
    if sigma_absolute is None:
        raise ArtifactValidationError(
            "blind acquisition artifacts require an absolute noise sigma"
        )
    acquisition = SPIAcquisitionData(
        dataset_identity_sha256=dataset_identity_sha256,
        **arrays,
        H=config["H"],
        W=config["W"],
        T=config["T"],
        K=config["K"],
        holdout_K=actual_holdout_count,
        acquisition={
            "pattern_family": pattern["family"],
            "pattern_values": _pattern_values(pattern["family"]),
            "pattern_order": pattern["order"],
            "time_assignment": time_assignment["mode"],
            "holdout_pattern_family": holdout["pattern_family"],
            "noise_convention": noise["convention"],
            "noise_sigma_absolute": sigma_absolute,
        },
        array_descriptors={
            name: array_descriptor(value)
            for name, value in arrays.items()
            if value is not None
        },
    )
    time_grid = np.asarray(data.t_grid, dtype=np.float64)
    omega = float(motion_parameters["omega"])
    beta = float(motion_parameters["beta"])
    truth = EvaluationTruth(
        dataset_identity_sha256=dataset_identity_sha256,
        dataset_identity_spec=identity_spec,
        canonical_image=data.canonical,
        gt_frames=data.gt_frames,
        translation_trajectory=(
            time_grid[:, None] * velocity
            + time_grid[:, None] ** 2 * acceleration
        ).astype(np.float32),
        rotation_trajectory=(time_grid * omega + time_grid**2 * beta).astype(
            np.float32
        ),
        gt_velocity=velocity.astype(np.float32),
        gt_acceleration=acceleration.astype(np.float32),
        gt_omega=omega,
        gt_beta=beta,
        motion_model=motion["model"],
        H=config["H"],
        W=config["W"],
        T=config["T"],
        evaluator_metadata={},
    )
    _validate_acquisition_identity(acquisition)
    from ._artifact_truth import _validate_truth

    _validate_truth(truth)
    return acquisition, truth


_ACQUISITION_METADATA_KEYS = {
    "array_descriptors",
    "acquisition",
    "dataset_identity_sha256",
    "dimensions",
    "optional_arrays",
    "schema_version",
}


def _acquisition_metadata(
    data: SPIAcquisitionData,
) -> tuple[Mapping[str, np.ndarray], Mapping[str, object]]:
    _validate_acquisition_identity(data)
    all_arrays = _acquisition_arrays(data)
    arrays = {name: value for name, value in all_arrays.items() if value is not None}
    optional_arrays = {
        name: all_arrays[name] is not None
        for name in (
            "holdout_frame_indices",
            "holdout_measurements",
            "holdout_patterns",
        )
    }
    metadata = {
        "schema_version": ACQUISITION_SCHEMA,
        "dataset_identity_sha256": data.dataset_identity_sha256,
        "dimensions": {
            "H": data.H,
            "W": data.W,
            "T": data.T,
            "K": data.K,
            "holdout_K": data.holdout_K,
        },
        "acquisition": data.acquisition,
        "optional_arrays": optional_arrays,
        "array_descriptors": {
            name: array_descriptor(value) for name, value in arrays.items()
        },
    }
    return arrays, metadata


def acquisition_npz_bytes(data: SPIAcquisitionData) -> bytes:
    arrays, metadata = _acquisition_metadata(data)
    return npz_bytes(arrays=arrays, metadata=metadata)


def save_acquisition_data(data: SPIAcquisitionData, path: Path) -> str:
    arrays, metadata = _acquisition_metadata(data)
    return write_npz(Path(path), arrays=arrays, metadata=metadata)


def load_acquisition_data(
    path: Path,
    *,
    expected_dataset_identity_sha256: str,
    expected_acquisition_spec: Mapping[str, object] | None = None,
) -> SPIAcquisitionData:
    return _load_acquisition_members(
        read_npz_members(
            Path(path),
            allowed_members=_ACQUISITION_MEMBER_ALLOWLIST,
        ),
        expected_dataset_identity_sha256=(
            expected_dataset_identity_sha256
        ),
        expected_acquisition_spec=expected_acquisition_spec,
    )


def load_acquisition_data_bytes(
    payload: bytes,
    *,
    expected_dataset_identity_sha256: str,
    expected_acquisition_spec: Mapping[str, object] | None = None,
) -> SPIAcquisitionData:
    return _load_acquisition_members(
        read_npz_members_bytes(
            payload,
            allowed_members=_ACQUISITION_MEMBER_ALLOWLIST,
        ),
        expected_dataset_identity_sha256=(
            expected_dataset_identity_sha256
        ),
        expected_acquisition_spec=expected_acquisition_spec,
    )


def _load_acquisition_members(
    members: Mapping[str, bytes],
    *,
    expected_dataset_identity_sha256: str,
    expected_acquisition_spec: Mapping[str, object] | None,
) -> SPIAcquisitionData:
    if type(expected_dataset_identity_sha256) is not str:
        raise ArtifactValidationError(
            "expected dataset identity must be an exact string"
        )
    validate_sha256(
        expected_dataset_identity_sha256, "expected dataset identity"
    )
    metadata = decode_metadata(members)
    validate_exact_keys(
        metadata, _ACQUISITION_METADATA_KEYS, "acquisition metadata"
    )
    if metadata["schema_version"] != ACQUISITION_SCHEMA:
        raise ArtifactValidationError("acquisition schema mismatch")
    validate_sha256(metadata["dataset_identity_sha256"], "dataset identity")
    if (
        metadata["dataset_identity_sha256"]
        != expected_dataset_identity_sha256
    ):
        raise ArtifactValidationError("acquisition dataset identity mismatch")
    optional = metadata["optional_arrays"]
    if not isinstance(optional, Mapping):
        raise ArtifactValidationError("optional_arrays must be an object")
    validate_exact_keys(
        optional,
        {
            "holdout_frame_indices",
            "holdout_measurements",
            "holdout_patterns",
        },
        "optional arrays",
    )
    if any(not isinstance(present, bool) for present in optional.values()):
        raise ArtifactValidationError("optional array flags must be boolean")
    required_array_names = {
        "patterns",
        "measurements",
        "frame_indices",
        "time_grid",
    } | {name for name, present in optional.items() if present}
    expected_members = {METADATA_MEMBER} | {
        f"{name}.npy" for name in required_array_names
    }
    if set(members) != expected_members:
        raise ArtifactValidationError("missing or extra acquisition ZIP member")
    arrays = {
        name: load_array_member(members, name)
        for name in required_array_names
    }
    descriptors = metadata["array_descriptors"]
    if not isinstance(descriptors, Mapping):
        raise ArtifactValidationError("array_descriptors must be an object")
    validate_exact_keys(
        descriptors, required_array_names, "acquisition array descriptors"
    )
    for name, array in arrays.items():
        validate_array_descriptor(name, array, descriptors[name])
    dimensions = metadata["dimensions"]
    if not isinstance(dimensions, Mapping):
        raise ArtifactValidationError("dimensions must be an object")
    validate_exact_keys(
        dimensions, {"H", "W", "T", "K", "holdout_K"}, "dimensions"
    )
    for name in ("H", "W", "T", "K"):
        validate_exact_int(dimensions[name], f"dimensions.{name}", minimum=1)
    validate_exact_int(
        dimensions["holdout_K"], "dimensions.holdout_K", minimum=0
    )
    acquisition = _validate_blind_acquisition(metadata["acquisition"])
    if expected_acquisition_spec is not None:
        expected_acquisition_spec = _validate_blind_acquisition_spec(
            expected_acquisition_spec
        )
        stored_spec = {
            "schema_version": BLIND_ACQUISITION_SPEC_SCHEMA,
            "dimensions": dimensions,
            "acquisition": acquisition,
        }
        if canonical_json_bytes(stored_spec) != canonical_json_bytes(
            expected_acquisition_spec
        ):
            raise ArtifactValidationError(
                "stored acquisition does not match expected acquisition spec"
            )
    data = SPIAcquisitionData(
        dataset_identity_sha256=metadata["dataset_identity_sha256"],
        patterns=arrays["patterns"],
        measurements=arrays["measurements"],
        frame_indices=arrays["frame_indices"],
        time_grid=arrays["time_grid"],
        holdout_patterns=arrays.get("holdout_patterns"),
        holdout_measurements=arrays.get("holdout_measurements"),
        holdout_frame_indices=arrays.get("holdout_frame_indices"),
        H=dimensions["H"],
        W=dimensions["W"],
        T=dimensions["T"],
        K=dimensions["K"],
        holdout_K=dimensions["holdout_K"],
        acquisition=acquisition,
        array_descriptors=descriptors,
    )
    _validate_acquisition_identity(data)
    return data
