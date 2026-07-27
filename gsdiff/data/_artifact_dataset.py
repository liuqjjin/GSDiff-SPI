"""Acquisition/truth identity construction and schema codecs."""

from pathlib import Path
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
    validate_exact_keys,
    validate_exact_int,
    validate_finite_number,
    validate_generation_config,
    validate_sha256,
)
from ._artifact_io import (
    METADATA_MEMBER,
    decode_metadata,
    load_array_member,
    read_npz_members,
    write_npz,
)
from ._artifact_models import EvaluationTruth, SPIAcquisitionData


ACQUISITION_SCHEMA = "measurements-v1"
TRUTH_SCHEMA = "evaluation-truth-v1"


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


def _validate_acquisition_identity(data: SPIAcquisitionData) -> None:
    validate_sha256(data.dataset_identity_sha256, "dataset identity")
    validate_sha256(data.target_asset_sha256, "target asset hash")
    if type(data.generator_code_version) is not str or not (
        data.generator_code_version.strip()
    ):
        raise ArtifactValidationError(
            "generator_code_version must be a non-empty string"
        )
    validate_exact_int(data.seed, "seed")
    for name in (
        "pattern_family",
        "pattern_order",
        "time_assignment_mode",
        "noise_convention",
        "motion_model",
    ):
        value = getattr(data, name)
        if type(value) is not str or not value.strip():
            raise ArtifactValidationError(f"{name} must be a non-empty string")
    _validate_acquisition_shapes(data)
    config = validate_generation_config(data.resolved_generation_config)
    if canonical_json_bytes(config) != canonical_json_bytes(
        data.resolved_generation_config
    ):
        raise ArtifactValidationError(
            "resolved generation config is not canonical"
        )
    identity_config = validate_acquisition_identity_spec(
        data.dataset_identity_spec
    )
    if canonical_json_bytes(identity_config) != canonical_json_bytes(config):
        raise ArtifactValidationError(
            "identity config disagrees with acquisition config"
        )
    redundant_fields = {
        "H": data.H,
        "W": data.W,
        "T": data.T,
        "K": data.K,
        "seed": data.seed,
        "pattern_family": data.pattern_family,
        "pattern_order": data.pattern_order,
        "time_assignment_mode": data.time_assignment_mode,
        "noise_convention": data.noise_convention,
        "motion_model": data.motion_model,
    }
    expected_fields = {
        "H": config["H"],
        "W": config["W"],
        "T": config["T"],
        "K": config["K"],
        "seed": config["seed"],
        "pattern_family": config["pattern"]["family"],
        "pattern_order": config["pattern"]["order"],
        "time_assignment_mode": config["time_assignment"]["mode"],
        "noise_convention": config["noise"]["convention"],
        "motion_model": config["motion"]["model"],
    }
    if redundant_fields != expected_fields:
        raise ArtifactValidationError(
            "redundant acquisition fields disagree with resolved generation config"
        )
    if canonical_json_bytes(data.noise_parameters) != canonical_json_bytes(
        config["noise"]["parameters"]
    ):
        raise ArtifactValidationError(
            "noise parameters disagree with resolved generation config"
        )
    if canonical_json_bytes(data.motion_parameters) != canonical_json_bytes(
        config["motion"]["parameters"]
    ):
        raise ArtifactValidationError(
            "motion parameters disagree with resolved generation config"
        )
    holdout_arrays = (
        data.holdout_patterns,
        data.holdout_measurements,
        data.holdout_frame_indices,
    )
    holdout_present = all(value is not None for value in holdout_arrays)
    holdout_count = (
        data.holdout_patterns.shape[0] if data.holdout_patterns is not None else 0
    )
    if (
        config["holdout"]["present"] != holdout_present
        or config["holdout"]["count"] != holdout_count
    ):
        raise ArtifactValidationError(
            "holdout config disagrees with array presence or count"
        )
    expected_spec = acquisition_identity_spec(
        arrays=_acquisition_arrays(data),
        H=data.H,
        W=data.W,
        T=data.T,
        K=data.K,
        resolved_generation_config=data.resolved_generation_config,
        generator_code_version=data.generator_code_version,
        target_asset_sha256=data.target_asset_sha256,
        seed=data.seed,
        pattern_family=data.pattern_family,
        pattern_order=data.pattern_order,
        time_assignment_mode=data.time_assignment_mode,
        noise_convention=data.noise_convention,
        noise_parameters=data.noise_parameters,
        motion_model=data.motion_model,
        motion_parameters=data.motion_parameters,
        schema=ACQUISITION_SCHEMA,
    )
    if canonical_json_bytes(data.dataset_identity_spec) != canonical_json_bytes(
        expected_spec
    ):
        raise ArtifactValidationError(
            "dataset identity spec does not match acquisition fields"
        )
    expected_identity = sha256_bytes(canonical_json_bytes(expected_spec))
    if data.dataset_identity_sha256 != expected_identity:
        raise ArtifactValidationError("dataset identity mismatch")


def split_spi_data(
    data: Any,
    *,
    resolved_generation_config: Mapping[str, object],
    generator_code_version: str,
    target_asset_sha256: str,
) -> tuple[SPIAcquisitionData, EvaluationTruth]:
    config = validate_generation_config(resolved_generation_config)
    validate_sha256(target_asset_sha256, "target asset hash")
    if type(generator_code_version) is not str or not generator_code_version.strip():
        raise ArtifactValidationError(
            "generator_code_version must be a non-empty string"
        )
    dimensions = tuple(config[name] for name in ("H", "W", "T", "K"))
    if dimensions != (int(data.H), int(data.W), int(data.T), int(data.K)):
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
    if not np.array_equal(
        np.asarray(data.gt_velocity, dtype=np.float64), velocity
    ) or not np.isclose(float(data.gt_omega), float(motion_parameters["omega"])):
        raise ArtifactValidationError(
            "resolved motion parameters do not match generated SPIData"
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
        schema=ACQUISITION_SCHEMA,
    )
    dataset_identity_sha256 = sha256_bytes(canonical_json_bytes(identity_spec))
    acquisition = SPIAcquisitionData(
        dataset_identity_sha256=dataset_identity_sha256,
        dataset_identity_spec=identity_spec,
        **arrays,
        H=config["H"],
        W=config["W"],
        T=config["T"],
        K=config["K"],
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
    _validate_truth(truth)
    return acquisition, truth


_ACQUISITION_METADATA_KEYS = {
    "array_descriptors",
    "dataset_identity_sha256",
    "dataset_identity_spec",
    "dimensions",
    "generator_code_version",
    "motion_model",
    "motion_parameters",
    "noise_convention",
    "noise_parameters",
    "optional_arrays",
    "pattern_family",
    "pattern_order",
    "resolved_generation_config",
    "schema",
    "seed",
    "target_asset_sha256",
    "time_assignment_mode",
}


def save_acquisition_data(data: SPIAcquisitionData, path: Path) -> str:
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
        "schema": ACQUISITION_SCHEMA,
        "dataset_identity_sha256": data.dataset_identity_sha256,
        "dataset_identity_spec": data.dataset_identity_spec,
        "dimensions": {"H": data.H, "W": data.W, "T": data.T, "K": data.K},
        "resolved_generation_config": data.resolved_generation_config,
        "generator_code_version": data.generator_code_version,
        "target_asset_sha256": data.target_asset_sha256,
        "seed": data.seed,
        "pattern_family": data.pattern_family,
        "pattern_order": data.pattern_order,
        "time_assignment_mode": data.time_assignment_mode,
        "noise_convention": data.noise_convention,
        "noise_parameters": data.noise_parameters,
        "motion_model": data.motion_model,
        "motion_parameters": data.motion_parameters,
        "optional_arrays": optional_arrays,
        "array_descriptors": {
            name: array_descriptor(value) for name, value in arrays.items()
        },
    }
    return write_npz(Path(path), arrays=arrays, metadata=metadata)


def load_acquisition_data(
    path: Path,
    *,
    expected_spec: Mapping[str, object] | None = None,
) -> SPIAcquisitionData:
    members = read_npz_members(Path(path))
    metadata = decode_metadata(members)
    validate_exact_keys(
        metadata, _ACQUISITION_METADATA_KEYS, "acquisition metadata"
    )
    if metadata["schema"] != ACQUISITION_SCHEMA:
        raise ArtifactValidationError("acquisition schema mismatch")
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
    validate_exact_keys(dimensions, {"H", "W", "T", "K"}, "dimensions")
    for name in ("H", "W", "T", "K"):
        validate_exact_int(dimensions[name], f"dimensions.{name}", minimum=1)
    config = validate_generation_config(metadata["resolved_generation_config"])
    if expected_spec is not None:
        try:
            validated_expected_spec = validate_generation_config(expected_spec)
        except ArtifactValidationError as exc:
            raise ArtifactValidationError(
                f"expected spec is invalid: {exc}"
            ) from exc
        if canonical_json_bytes(config) != canonical_json_bytes(
            validated_expected_spec
        ):
            raise ArtifactValidationError(
                "stored acquisition does not match expected spec"
            )
    data = SPIAcquisitionData(
        dataset_identity_sha256=metadata["dataset_identity_sha256"],
        dataset_identity_spec=metadata["dataset_identity_spec"],
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
        resolved_generation_config=config,
        generator_code_version=metadata["generator_code_version"],
        target_asset_sha256=metadata["target_asset_sha256"],
        seed=metadata["seed"],
        pattern_family=metadata["pattern_family"],
        pattern_order=metadata["pattern_order"],
        time_assignment_mode=metadata["time_assignment_mode"],
        noise_convention=metadata["noise_convention"],
        noise_parameters=metadata["noise_parameters"],
        motion_model=metadata["motion_model"],
        motion_parameters=metadata["motion_parameters"],
    )
    _validate_acquisition_identity(data)
    return data


_TRUTH_ARRAY_NAMES = {
    "canonical_image",
    "gt_frames",
    "translation_trajectory",
    "rotation_trajectory",
    "gt_velocity",
    "gt_acceleration",
}
_TRUTH_METADATA_KEYS = {
    "array_descriptors",
    "dataset_identity_sha256",
    "dataset_identity_spec",
    "dimensions",
    "evaluator_metadata",
    "gt_beta",
    "gt_omega",
    "motion_model",
    "schema",
}


def _truth_arrays(data: EvaluationTruth) -> Mapping[str, np.ndarray]:
    return {name: getattr(data, name) for name in _TRUTH_ARRAY_NAMES}


def _validate_truth(data: EvaluationTruth) -> None:
    validate_sha256(data.dataset_identity_sha256, "dataset identity")
    expected_identity = sha256_bytes(canonical_json_bytes(data.dataset_identity_spec))
    if data.dataset_identity_sha256 != expected_identity:
        raise ArtifactValidationError("dataset identity mismatch")
    config = validate_acquisition_identity_spec(data.dataset_identity_spec)
    for name in ("H", "W", "T"):
        validate_exact_int(getattr(data, name), f"truth {name}", minimum=1)
    if (data.H, data.W, data.T) != (
        config["H"],
        config["W"],
        config["T"],
    ):
        raise ArtifactValidationError(
            "truth dimensions disagree with dataset identity"
        )
    expected_shapes = {
        "canonical_image": (data.H, data.W),
        "gt_frames": (data.T, data.H, data.W),
        "translation_trajectory": (data.T, 2),
        "rotation_trajectory": (data.T,),
        "gt_velocity": (2,),
        "gt_acceleration": (2,),
    }
    for name, shape in expected_shapes.items():
        array = getattr(data, name)
        if array.shape != shape:
            raise ArtifactValidationError(
                f"{name} shape must be {shape}, got {array.shape}"
            )
        if not np.all(np.isfinite(array)):
            raise ArtifactValidationError(f"{name} must contain finite values")

    motion = config["motion"]
    motion_parameters = motion["parameters"]
    if data.motion_model != motion["model"]:
        raise ArtifactValidationError(
            "truth motion model disagrees with dataset identity"
        )
    gt_omega = validate_finite_number(data.gt_omega, "truth gt_omega")
    gt_beta = validate_finite_number(data.gt_beta, "truth gt_beta")
    if (
        gt_omega != motion_parameters["omega"]
        or gt_beta != motion_parameters["beta"]
    ):
        raise ArtifactValidationError(
            "truth angular motion disagrees with dataset identity"
        )
    expected_velocity = np.asarray(
        motion_parameters["velocity"], dtype=data.gt_velocity.dtype
    )
    expected_acceleration = np.asarray(
        motion_parameters["acceleration"], dtype=data.gt_acceleration.dtype
    )
    if not np.array_equal(data.gt_velocity, expected_velocity) or not (
        np.array_equal(data.gt_acceleration, expected_acceleration)
    ):
        raise ArtifactValidationError(
            "truth linear motion disagrees with dataset identity"
        )
    time_grid = np.linspace(0.0, 1.0, data.T).astype(np.float32).astype(
        np.float64
    )
    expected_translation = (
        time_grid[:, None]
        * np.asarray(motion_parameters["velocity"], dtype=np.float64)
        + time_grid[:, None] ** 2
        * np.asarray(motion_parameters["acceleration"], dtype=np.float64)
    ).astype(data.translation_trajectory.dtype)
    expected_rotation = (
        time_grid * float(motion_parameters["omega"])
        + time_grid**2 * float(motion_parameters["beta"])
    ).astype(data.rotation_trajectory.dtype)
    if not np.array_equal(
        data.translation_trajectory, expected_translation
    ) or not np.array_equal(data.rotation_trajectory, expected_rotation):
        raise ArtifactValidationError(
            "truth trajectories disagree with dataset identity"
        )
    canonical_hash = sha256_bytes(
        np.ascontiguousarray(data.canonical_image).tobytes(order="C")
    )
    if canonical_hash != data.dataset_identity_spec["target_asset_sha256"]:
        raise ArtifactValidationError(
            "truth canonical image disagrees with target asset identity"
        )


def save_evaluation_truth(data: EvaluationTruth, path: Path) -> str:
    _validate_truth(data)
    arrays = _truth_arrays(data)
    metadata = {
        "schema": TRUTH_SCHEMA,
        "dataset_identity_sha256": data.dataset_identity_sha256,
        "dataset_identity_spec": data.dataset_identity_spec,
        "dimensions": {"H": data.H, "W": data.W, "T": data.T},
        "gt_omega": data.gt_omega,
        "gt_beta": data.gt_beta,
        "motion_model": data.motion_model,
        "evaluator_metadata": data.evaluator_metadata,
        "array_descriptors": {
            name: array_descriptor(array) for name, array in arrays.items()
        },
    }
    return write_npz(Path(path), arrays=arrays, metadata=metadata)


def load_evaluation_truth(
    path: Path,
    *,
    expected_dataset_identity_sha256: str,
) -> EvaluationTruth:
    validate_sha256(
        expected_dataset_identity_sha256, "expected dataset identity"
    )
    members = read_npz_members(Path(path))
    metadata = decode_metadata(members)
    validate_exact_keys(metadata, _TRUTH_METADATA_KEYS, "truth metadata")
    if metadata["schema"] != TRUTH_SCHEMA:
        raise ArtifactValidationError("truth schema mismatch")
    if metadata["dataset_identity_sha256"] != expected_dataset_identity_sha256:
        raise ArtifactValidationError("truth dataset identity mismatch")
    expected_members = {METADATA_MEMBER} | {
        f"{name}.npy" for name in _TRUTH_ARRAY_NAMES
    }
    if set(members) != expected_members:
        raise ArtifactValidationError("missing or extra truth ZIP member")
    arrays = {
        name: load_array_member(members, name)
        for name in _TRUTH_ARRAY_NAMES
    }
    descriptors = metadata["array_descriptors"]
    if not isinstance(descriptors, Mapping):
        raise ArtifactValidationError("array_descriptors must be an object")
    validate_exact_keys(descriptors, _TRUTH_ARRAY_NAMES, "truth descriptors")
    for name, array in arrays.items():
        validate_array_descriptor(name, array, descriptors[name])
    dimensions = metadata["dimensions"]
    if not isinstance(dimensions, Mapping):
        raise ArtifactValidationError("dimensions must be an object")
    validate_exact_keys(dimensions, {"H", "W", "T"}, "truth dimensions")
    for name in ("H", "W", "T"):
        validate_exact_int(dimensions[name], f"truth dimensions.{name}", minimum=1)
    validate_finite_number(metadata["gt_omega"], "truth gt_omega")
    validate_finite_number(metadata["gt_beta"], "truth gt_beta")
    data = EvaluationTruth(
        dataset_identity_sha256=metadata["dataset_identity_sha256"],
        dataset_identity_spec=metadata["dataset_identity_spec"],
        canonical_image=arrays["canonical_image"],
        gt_frames=arrays["gt_frames"],
        translation_trajectory=arrays["translation_trajectory"],
        rotation_trajectory=arrays["rotation_trajectory"],
        gt_velocity=arrays["gt_velocity"],
        gt_acceleration=arrays["gt_acceleration"],
        gt_omega=metadata["gt_omega"],
        gt_beta=metadata["gt_beta"],
        motion_model=metadata["motion_model"],
        H=dimensions["H"],
        W=dimensions["W"],
        T=dimensions["T"],
        evaluator_metadata=metadata["evaluator_metadata"],
    )
    _validate_truth(data)
    return data
