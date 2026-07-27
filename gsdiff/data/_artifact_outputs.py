"""Raw reconstruction outputs and execution-capability policy."""

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ._artifact_dataset import _validate_acquisition_identity
from ._artifact_identity import (
    ArtifactValidationError,
    array_descriptor,
    canonical_json_bytes,
    json_native,
    validate_array_descriptor,
    validate_exact_keys,
    validate_index_array,
    validate_real_finite_array,
    validate_sha256,
    validate_time_grid,
)
from ._artifact_io import (
    METADATA_MEMBER,
    atomic_write_bytes,
    decode_metadata,
    load_array_member,
    read_npz_members,
    write_npz,
)
from ._artifact_models import (
    EvaluationTruth,
    MethodExecutionPolicy,
    ReconstructionOutput,
    SPIAcquisitionData,
)


RECONSTRUCTION_SCHEMA = "reconstruction-v1"
METHOD_INFO_SCHEMA = "method-info-v1"


def method_execution_policy(*, truth_path: Path | None) -> MethodExecutionPolicy:
    if truth_path is None:
        return MethodExecutionPolicy(
            execution_class="blind_method_child",
            truth_access="unavailable",
            promotion_eligible=True,
        )
    return MethodExecutionPolicy(
        execution_class="compatibility_unblinded",
        truth_access="child_visible",
        promotion_eligible=False,
    )


def require_promotion_eligible(policy: MethodExecutionPolicy) -> None:
    if (
        policy.execution_class != "blind_method_child"
        or policy.truth_access != "unavailable"
        or policy.promotion_eligible is not True
    ):
        raise ArtifactValidationError(
            "execution policy is not eligible for promotion"
        )


def validate_evaluation_inputs(
    reconstruction: ReconstructionOutput,
    acquisition: SPIAcquisitionData,
    truth: EvaluationTruth,
) -> None:
    from ._artifact_truth import _validate_truth

    _validate_reconstruction(reconstruction)
    _validate_acquisition_identity(acquisition)
    _validate_truth(truth)
    identities = {
        reconstruction.dataset_identity_sha256,
        acquisition.dataset_identity_sha256,
        truth.dataset_identity_sha256,
    }
    if len(identities) != 1:
        raise ArtifactValidationError(
            "reconstruction/acquisition/truth dataset identity mismatch"
        )
    expected_shape = (acquisition.T, acquisition.H, acquisition.W)
    if reconstruction.reconstruction.shape != expected_shape:
        raise ArtifactValidationError(
            "reconstruction dimensions disagree with acquisition"
        )
    if (truth.T, truth.H, truth.W) != expected_shape:
        raise ArtifactValidationError(
            "truth dimensions disagree with acquisition"
        )
    if (
        reconstruction.time_grid.dtype != acquisition.time_grid.dtype
        or not np.array_equal(
            reconstruction.time_grid, acquisition.time_grid
        )
    ):
        raise ArtifactValidationError(
            "reconstruction time_grid must exactly match acquisition time_grid"
        )
    canonical_frame_indices = reconstruction.frame_indices.astype(
        np.int64, copy=False
    )
    expected_frame_indices = np.arange(acquisition.T, dtype=np.int64)
    if not np.array_equal(
        canonical_frame_indices, expected_frame_indices
    ):
        raise ArtifactValidationError(
            "reconstruction frame_indices must equal arange(T)"
        )


def _validate_reconstruction(data: ReconstructionOutput) -> None:
    validate_sha256(data.dataset_identity_sha256, "dataset identity")
    if data.reconstruction.ndim != 3:
        raise ArtifactValidationError("reconstruction must have shape [T,H,W]")
    T, H, W = data.reconstruction.shape
    if min(T, H, W) <= 0:
        raise ArtifactValidationError("reconstruction dimensions must be positive")
    validate_real_finite_array(
        data.reconstruction,
        "reconstruction",
        shape=(T, H, W),
    )
    if data.dgi is not None and data.dgi.shape != (H, W):
        raise ArtifactValidationError("DGI shape must match reconstruction H,W")
    if data.dgi is not None:
        validate_real_finite_array(data.dgi, "dgi", shape=(H, W))
    if data.estimated_motion_trajectory.shape != (T, 3):
        raise ArtifactValidationError(
            "estimated motion trajectory must have shape [T,3]"
        )
    validate_real_finite_array(
        data.estimated_motion_trajectory,
        "estimated_motion_trajectory",
        shape=(T, 3),
    )
    validate_index_array(
        data.frame_indices,
        "frame_indices",
        shape=(T,),
        upper_bound=T,
    )
    validate_time_grid(data.time_grid, "time_grid", length=T)
    if not data.method_name:
        raise ArtifactValidationError("method_name cannot be empty")
    policy = data.execution_policy
    if policy.execution_class not in {
        "blind_method_child",
        "compatibility_unblinded",
    }:
        raise ArtifactValidationError("unknown execution class")
    if policy.truth_access not in {"unavailable", "child_visible"}:
        raise ArtifactValidationError("unknown truth access policy")
    if (
        policy.execution_class == "blind_method_child"
        and (
            policy.truth_access != "unavailable"
            or policy.promotion_eligible is not True
        )
    ):
        raise ArtifactValidationError("inconsistent blind execution policy")
    if (
        policy.execution_class == "compatibility_unblinded"
        and (
            policy.truth_access != "child_visible"
            or policy.promotion_eligible is not False
        )
    ):
        raise ArtifactValidationError(
            "inconsistent compatibility execution policy"
        )


_RECONSTRUCTION_ARRAY_NAMES = {
    "reconstruction",
    "estimated_motion_trajectory",
    "frame_indices",
    "time_grid",
}
_RECONSTRUCTION_METADATA_KEYS = {
    "array_descriptors",
    "dataset_identity_sha256",
    "execution_class",
    "method_metadata",
    "method_name",
    "optional_arrays",
    "promotion_eligible",
    "schema",
    "truth_access",
}


def _save_reconstruction_output(data: ReconstructionOutput, path: Path) -> str:
    _validate_reconstruction(data)
    arrays = {
        name: getattr(data, name) for name in _RECONSTRUCTION_ARRAY_NAMES
    }
    if data.dgi is not None:
        arrays["dgi"] = data.dgi
    policy = data.execution_policy
    metadata = {
        "schema": RECONSTRUCTION_SCHEMA,
        "dataset_identity_sha256": data.dataset_identity_sha256,
        "method_name": data.method_name,
        "method_metadata": data.method_metadata,
        "execution_class": policy.execution_class,
        "truth_access": policy.truth_access,
        "promotion_eligible": policy.promotion_eligible,
        "optional_arrays": {"dgi": data.dgi is not None},
        "array_descriptors": {
            name: array_descriptor(array) for name, array in arrays.items()
        },
    }
    return write_npz(Path(path), arrays=arrays, metadata=metadata)


def load_reconstruction_output(path: Path) -> ReconstructionOutput:
    members = read_npz_members(Path(path))
    metadata = decode_metadata(members)
    validate_exact_keys(
        metadata, _RECONSTRUCTION_METADATA_KEYS, "reconstruction metadata"
    )
    if metadata["schema"] != RECONSTRUCTION_SCHEMA:
        raise ArtifactValidationError("reconstruction schema mismatch")
    optional = metadata["optional_arrays"]
    if not isinstance(optional, Mapping):
        raise ArtifactValidationError("optional_arrays must be an object")
    validate_exact_keys(optional, {"dgi"}, "reconstruction optional arrays")
    if not isinstance(optional["dgi"], bool):
        raise ArtifactValidationError("DGI optional flag must be boolean")
    array_names = set(_RECONSTRUCTION_ARRAY_NAMES)
    if optional["dgi"]:
        array_names.add("dgi")
    expected_members = {METADATA_MEMBER} | {
        f"{name}.npy" for name in array_names
    }
    if set(members) != expected_members:
        raise ArtifactValidationError(
            "missing or extra reconstruction ZIP member"
        )
    arrays = {
        name: load_array_member(members, name)
        for name in array_names
    }
    descriptors = metadata["array_descriptors"]
    if not isinstance(descriptors, Mapping):
        raise ArtifactValidationError("array_descriptors must be an object")
    validate_exact_keys(
        descriptors, array_names, "reconstruction descriptors"
    )
    for name, array in arrays.items():
        validate_array_descriptor(name, array, descriptors[name])
    data = ReconstructionOutput(
        dataset_identity_sha256=metadata["dataset_identity_sha256"],
        reconstruction=arrays["reconstruction"],
        dgi=arrays.get("dgi"),
        estimated_motion_trajectory=arrays["estimated_motion_trajectory"],
        frame_indices=arrays["frame_indices"],
        time_grid=arrays["time_grid"],
        method_name=metadata["method_name"],
        method_metadata=metadata["method_metadata"],
        execution_policy=MethodExecutionPolicy(
            execution_class=metadata["execution_class"],
            truth_access=metadata["truth_access"],
            promotion_eligible=metadata["promotion_eligible"],
        ),
    )
    _validate_reconstruction(data)
    return data


def _reject_forbidden_output_keys(value: object, location: str = "") -> None:
    forbidden = (
        "canonical",
        "ground_truth",
        "gt_",
        "gt-",
        "psnr",
        "ssim",
        "nrmse",
        "metric",
        "evaluator",
        "display",
        "normalized",
    )
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_lower = str(key).lower()
            if any(token in key_lower for token in forbidden):
                raise ArtifactValidationError(
                    f"forbidden child output field: {location}{key}"
                )
            _reject_forbidden_output_keys(item, f"{location}{key}.")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_forbidden_output_keys(item, f"{location}{index}.")


def write_method_child_outputs(
    output_dir: Path,
    output: ReconstructionOutput,
    *,
    history: Sequence[Mapping[str, object]],
) -> Mapping[str, str]:
    _validate_reconstruction(output)
    _reject_forbidden_output_keys(output.method_metadata)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    allowed_names = {
        "reconstruction.npz",
        "iteration-history.jsonl",
        "method-info.json",
    }
    unexpected = {
        entry.name for entry in output_dir.iterdir() if entry.name not in allowed_names
    }
    if unexpected:
        raise ArtifactValidationError(
            f"method-child output directory is not isolated: {sorted(unexpected)}"
        )
    reconstruction_path = output_dir / "reconstruction.npz"
    reconstruction_hash = _save_reconstruction_output(
        output, reconstruction_path
    )
    history_rows = []
    for row in history:
        scalar_row = {key: value for key, value in row.items() if key != "video"}
        native = json_native(scalar_row)
        _reject_forbidden_output_keys(native)
        history_rows.append(canonical_json_bytes(native))
    history_payload = b"".join(row + b"\n" for row in history_rows)
    history_path = output_dir / "iteration-history.jsonl"
    history_hash = atomic_write_bytes(history_path, history_payload)
    policy = output.execution_policy
    method_info = {
        "schema": METHOD_INFO_SCHEMA,
        "dataset_identity_sha256": output.dataset_identity_sha256,
        "method_name": output.method_name,
        "method_metadata": output.method_metadata,
        "execution_class": policy.execution_class,
        "truth_access": policy.truth_access,
        "promotion_eligible": policy.promotion_eligible,
    }
    method_info_path = output_dir / "method-info.json"
    method_info_hash = atomic_write_bytes(
        method_info_path, canonical_json_bytes(method_info) + b"\n"
    )
    return {
        "reconstruction.npz": reconstruction_hash,
        "iteration-history.jsonl": history_hash,
        "method-info.json": method_info_hash,
    }
