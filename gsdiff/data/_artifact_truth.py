"""Evaluator-only truth artifact validation and schema codecs."""

from pathlib import Path
from typing import Mapping

import numpy as np

from ._artifact_identity import (
    ArtifactValidationError,
    array_descriptor,
    canonical_json_bytes,
    sha256_bytes,
    validate_acquisition_identity_spec,
    validate_array_descriptor,
    validate_exact_keys,
    validate_exact_int,
    validate_finite_number,
    validate_real_finite_array,
    validate_sha256,
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
from ._artifact_models import EvaluationTruth


TRUTH_SCHEMA = "evaluation-truth-v1"
CORRECTED_TRUTH_SCHEMA = "evaluation-truth-v2"
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
_TRUTH_MEMBER_ALLOWLIST = {METADATA_MEMBER} | {
    f"{name}.npy" for name in _TRUTH_ARRAY_NAMES
}


def _truth_arrays(data: EvaluationTruth) -> Mapping[str, np.ndarray]:
    return {name: getattr(data, name) for name in _TRUTH_ARRAY_NAMES}


def _validate_truth(data: EvaluationTruth) -> None:
    validate_sha256(data.dataset_identity_sha256, "dataset identity")
    if (
        isinstance(data.dataset_identity_spec, Mapping)
        and data.dataset_identity_spec.get("schema_version")
        == "dataset-identity-v1"
    ):
        from ._corrected_generation import validate_corrected_truth

        validate_corrected_truth(data)
        return
    expected_identity = sha256_bytes(
        canonical_json_bytes(data.dataset_identity_spec)
    )
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
        validate_real_finite_array(array, name, shape=shape)

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


def save_evaluation_truth(data: EvaluationTruth, path: Path) -> str:
    arrays, metadata = _truth_metadata(data)
    return write_npz(Path(path), arrays=arrays, metadata=metadata)


def evaluation_truth_npz_bytes(data: EvaluationTruth) -> bytes:
    arrays, metadata = _truth_metadata(data)
    return npz_bytes(arrays=arrays, metadata=metadata)


def _truth_metadata(
    data: EvaluationTruth,
) -> tuple[Mapping[str, np.ndarray], Mapping[str, object]]:
    _validate_truth(data)
    arrays = _truth_arrays(data)
    schema = (
        CORRECTED_TRUTH_SCHEMA
        if data.dataset_identity_spec.get("schema_version")
        == "dataset-identity-v1"
        else TRUTH_SCHEMA
    )
    metadata = {
        "schema": schema,
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
    return arrays, metadata


def load_evaluation_truth(
    path: Path,
    *,
    expected_dataset_identity_sha256: str,
) -> EvaluationTruth:
    return _load_evaluation_truth_members(
        read_npz_members(
            Path(path), allowed_members=_TRUTH_MEMBER_ALLOWLIST
        ),
        expected_dataset_identity_sha256=(
            expected_dataset_identity_sha256
        ),
    )


def load_evaluation_truth_bytes(
    payload: bytes,
    *,
    expected_dataset_identity_sha256: str,
) -> EvaluationTruth:
    return _load_evaluation_truth_members(
        read_npz_members_bytes(
            payload, allowed_members=_TRUTH_MEMBER_ALLOWLIST
        ),
        expected_dataset_identity_sha256=(
            expected_dataset_identity_sha256
        ),
    )


def _load_evaluation_truth_members(
    members: Mapping[str, bytes],
    *,
    expected_dataset_identity_sha256: str,
) -> EvaluationTruth:
    validate_sha256(
        expected_dataset_identity_sha256, "expected dataset identity"
    )
    metadata = decode_metadata(members)
    validate_exact_keys(metadata, _TRUTH_METADATA_KEYS, "truth metadata")
    expected_schema = (
        CORRECTED_TRUTH_SCHEMA
        if (
            isinstance(metadata["dataset_identity_spec"], Mapping)
            and metadata["dataset_identity_spec"].get("schema_version")
            == "dataset-identity-v1"
        )
        else TRUTH_SCHEMA
    )
    if metadata["schema"] != expected_schema:
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
        validate_exact_int(
            dimensions[name], f"truth dimensions.{name}", minimum=1
        )
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
