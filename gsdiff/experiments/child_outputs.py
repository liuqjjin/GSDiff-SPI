"""Strict v2 artifacts owned by a blind method child.

This module deliberately depends only on the staged acquisition and method
request.  It neither imports evaluator code nor accepts evaluation inputs.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import re

import numpy as np
from jsonschema import Draft202012Validator

from gsdiff.data._artifact_identity import (
    ArtifactValidationError,
    array_descriptor,
    canonical_json_bytes,
    validate_array_descriptor,
    validate_exact_json_native,
    validate_exact_keys,
)
from gsdiff.data._artifact_io import (
    METADATA_MEMBER,
    atomic_write_bytes,
    capture_directory_inventory,
    decode_metadata,
    load_array_member,
    read_npz_members,
    read_safe_file_snapshot,
    verify_safe_file_snapshot,
    write_npz,
)
from gsdiff.data._artifact_models import SPIAcquisitionData

from .methods import AlgorithmSeed, ResolvedMethod, thaw_json


RECONSTRUCTION_V2_SCHEMA = "reconstruction-v2"
METHOD_INFO_V2_SCHEMA = "method-info-v2"
_OUTPUT_FILES = frozenset({"reconstruction.npz", "method-info.json"})
_SHA256_RE = re.compile(r"[0-9a-f]{64}", re.ASCII)
_UTC_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z", re.ASCII)


@dataclass(frozen=True, kw_only=True)
class MethodChildResult:
    method_id: str
    reconstruction: np.ndarray
    estimated_motion_trajectory: np.ndarray | None
    dgi: np.ndarray | None
    info: Mapping[str, object]
    history: tuple[Mapping[str, object], ...]


@dataclass(frozen=True)
class ReconstructionV2:
    dataset_identity_sha256: str
    method_id: str
    reconstruction: np.ndarray
    frame_indices: np.ndarray
    time_grid: np.ndarray
    dgi: np.ndarray | None
    estimated_motion_trajectory: np.ndarray | None
    array_descriptors: Mapping[str, object]


def _json_object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ArtifactValidationError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> object:
    raise ArtifactValidationError(f"non-finite JSON number rejected: {value}")


def _parse_json_file(path: Path) -> tuple[Mapping[str, object], str]:
    snapshot = read_safe_file_snapshot(path, max_bytes=16 * 1024 * 1024, noun="method info")
    try:
        loaded = json.loads(snapshot.raw.decode("utf-8", errors="strict"), object_pairs_hook=_json_object_pairs, parse_constant=_reject_json_constant)
    except ArtifactValidationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArtifactValidationError("malformed method info JSON") from error
    if type(loaded) is not dict:
        raise ArtifactValidationError("method info JSON must be an object")
    validate_exact_json_native(loaded, "method info")
    if canonical_json_bytes(loaded) != snapshot.raw:
        raise ArtifactValidationError("method info JSON must use canonical bytes")
    verify_safe_file_snapshot(snapshot)
    return loaded, snapshot.sha256


def _strict_output_names(output_dir: Path) -> set[str]:
    """Return a link-safe, flat inventory for the child-owned directory."""
    inventory = capture_directory_inventory(output_dir)
    names: set[str] = set()
    for relative, _ in inventory._entries:
        if "/" in relative:
            raise ArtifactValidationError("child output inventory must be flat")
        names.add(relative)
    return names


def _require_sha256(value: object, field: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ArtifactValidationError(f"{field} must be a SHA-256")
    return value


def _require_utc(value: object, field: str) -> datetime:
    if type(value) is not str or _UTC_RE.fullmatch(value) is None:
        raise ArtifactValidationError(f"{field} must be RFC 3339 UTC")
    try:
        return datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ArtifactValidationError(f"{field} must be RFC 3339 UTC") from error


def _validate_result(result: MethodChildResult, acquisition: SPIAcquisitionData, method: ResolvedMethod) -> None:
    if type(result) is not MethodChildResult:
        raise TypeError("result must be a MethodChildResult")
    if result.method_id != method.method_id:
        raise ArtifactValidationError("result method_id does not match method")
    reconstruction = np.asarray(result.reconstruction)
    if reconstruction.shape != (acquisition.T, acquisition.H, acquisition.W):
        raise ArtifactValidationError("reconstruction shape disagrees with acquisition")
    if not np.issubdtype(reconstruction.dtype, np.number) or np.iscomplexobj(reconstruction) or not np.isfinite(reconstruction).all():
        raise ArtifactValidationError("reconstruction must be finite real numeric")
    if result.dgi is not None:
        dgi = np.asarray(result.dgi)
        if dgi.shape != (acquisition.H, acquisition.W) or not np.issubdtype(dgi.dtype, np.number) or np.iscomplexobj(dgi) or not np.isfinite(dgi).all():
            raise ArtifactValidationError("dgi shape or values are invalid")
    if result.estimated_motion_trajectory is not None:
        motion = np.asarray(result.estimated_motion_trajectory)
        if motion.ndim != 2 or motion.shape[0] != acquisition.T or not np.issubdtype(motion.dtype, np.number) or np.iscomplexobj(motion) or not np.isfinite(motion).all():
            raise ArtifactValidationError("estimated_motion_trajectory shape or values are invalid")
    if not isinstance(result.info, Mapping):
        raise ArtifactValidationError("result info must be an object")
    allowed = {"parameter_count", "native_iteration_unit", "native_iteration_budget", "convergence_status", "selected_hyperparameters", "selection", "checkpoint_hashes", "native_motion_model"}
    required = allowed - {"native_motion_model"}
    if set(result.info) != required and set(result.info) != allowed:
        raise ArtifactValidationError("result info keys are invalid")
    if type(result.info["parameter_count"]) is not int or result.info["parameter_count"] < 0:
        raise ArtifactValidationError("parameter_count must be nonnegative integer")
    if type(result.info["native_iteration_unit"]) is not str or not result.info["native_iteration_unit"]:
        raise ArtifactValidationError("native_iteration_unit must be nonempty")
    if type(result.info["native_iteration_budget"]) is not int or result.info["native_iteration_budget"] < 0:
        raise ArtifactValidationError("native_iteration_budget must be nonnegative integer")
    if result.info["convergence_status"] != method.convergence_status:
        raise ArtifactValidationError("convergence_status does not match method profile")
    if not isinstance(result.info["checkpoint_hashes"], list):
        raise ArtifactValidationError("checkpoint_hashes must be a list")
    if (
        type(result.info.get("native_motion_model", "none")) is not str
        or not result.info.get("native_motion_model", "none")
    ):
        raise ArtifactValidationError("native_motion_model must be a nonempty string")
    if result.info["checkpoint_hashes"] != _expected_checkpoints(method):
        raise ArtifactValidationError("checkpoint_hashes do not match method requirements")
    validate_exact_json_native(dict(result.info), "result info")
    if type(result.history) is not tuple or any(not isinstance(row, Mapping) for row in result.history):
        raise ArtifactValidationError("history must be a tuple of objects")
    for index, row in enumerate(result.history):
        validate_exact_json_native(dict(row), f"history[{index}]")


def _expected_checkpoints(method: ResolvedMethod) -> list[dict[str, str]]:
    return [
        {"logical_id": requirement.logical_id, "sha256": requirement.sha256}
        for requirement in method.checkpoint_requirements
    ]


def _warmup_counts(method: ResolvedMethod) -> dict[str, int]:
    solver = method.semantic_config.get("solver")
    if not isinstance(solver, Mapping):
        return {"splitting": 0, "motion": 0}
    splitting = solver.get("splitting_warmup_outer", 0)
    motion = solver.get("motion_warmup_outer", solver.get("motion_warmup_steps", 0))
    if type(splitting) is not int or type(motion) is not int or splitting < 0 or motion < 0:
        raise ArtifactValidationError("method warmup counts must be nonnegative integers")
    return {"splitting": splitting, "motion": motion}


def _sample_history(history: tuple[Mapping[str, object], ...]) -> tuple[list[object], str]:
    observed = len(history)
    if observed <= 20:
        indices = range(observed)
        policy = "all-observations"
    else:
        indices = (observed - 1) * np.arange(21, dtype=np.int64) // 20
        policy = "floor(i*(observed_count-1)/20), i=0..20"
    return [dict(history[int(index)]) for index in indices], policy


def _reconstruction_metadata(*, method_id: str, acquisition: SPIAcquisitionData, arrays: Mapping[str, np.ndarray]) -> dict[str, object]:
    return {
        "schema": RECONSTRUCTION_V2_SCHEMA,
        "dataset_identity_sha256": acquisition.dataset_identity_sha256,
        "method_id": method_id,
        "optional_arrays": {
            "dgi": "dgi" in arrays,
            "estimated_motion_trajectory": "estimated_motion_trajectory" in arrays,
        },
        "array_descriptors": {name: array_descriptor(array) for name, array in arrays.items()},
    }


def write_method_child_outputs_v2(
    output_dir: Path,
    *,
    method: ResolvedMethod,
    acquisition: SPIAcquisitionData,
    measurements_file_sha256: str,
    algorithm_seed: AlgorithmSeed,
    result: MethodChildResult,
    child_started_at_utc: str,
    child_finished_at_utc: str,
) -> Mapping[str, str]:
    """Atomically write the two v2 child-owned artifacts into a clean directory."""
    if type(method) is not ResolvedMethod or type(acquisition) is not SPIAcquisitionData or type(algorithm_seed) is not AlgorithmSeed:
        raise TypeError("method, acquisition, and algorithm_seed must be resolved values")
    _require_sha256(measurements_file_sha256, "measurements_file_sha256")
    started, finished = _require_utc(child_started_at_utc, "child_started_at_utc"), _require_utc(child_finished_at_utc, "child_finished_at_utc")
    if finished < started:
        raise ArtifactValidationError("child timing cannot be negative")
    _validate_result(result, acquisition, method)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if _strict_output_names(output_dir):
        raise ArtifactValidationError("method-child output directory is not isolated")
    arrays: dict[str, np.ndarray] = {
        "reconstruction": np.ascontiguousarray(result.reconstruction),
        "frame_indices": np.arange(acquisition.T, dtype=np.int64),
        "time_grid": np.ascontiguousarray(acquisition.time_grid),
    }
    if result.dgi is not None:
        arrays["dgi"] = np.ascontiguousarray(result.dgi)
    if result.estimated_motion_trajectory is not None:
        arrays["estimated_motion_trajectory"] = np.ascontiguousarray(result.estimated_motion_trajectory)
    reconstruction_path = output_dir / "reconstruction.npz"
    reconstruction_hash = write_npz(reconstruction_path, arrays=arrays, metadata=_reconstruction_metadata(method_id=method.method_id, acquisition=acquisition, arrays=arrays))
    samples, sampling_policy = _sample_history(result.history)
    status = result.info["convergence_status"]
    if status == "convergence-required" and len(result.history) < 21:
        raise ArtifactValidationError("convergence-required profiles need 21 observations")
    info = {
        "schema": METHOD_INFO_V2_SCHEMA,
        "method_id": method.method_id,
        "method_config_id": method.method_config_id,
        "execution_family": method.execution_family,
        "execution_profile": method.execution_profile,
        "dataset_identity_sha256": acquisition.dataset_identity_sha256,
        "measurements_file_sha256": measurements_file_sha256,
        "method_config_sha256": method.method_config_sha256,
        "semantic_config": thaw_json(method.semantic_config),
        "algorithm_seed": {"domain": "algorithm-seed-v1", "derivation_sha256": algorithm_seed.derivation_sha256, "seed_u32": algorithm_seed.seed_u32},
        "parameter_count": result.info["parameter_count"],
        "native_iteration": {"unit": result.info["native_iteration_unit"], "budget": result.info["native_iteration_budget"]},
        "warmup": _warmup_counts(method),
        "selected_hyperparameters": result.info["selected_hyperparameters"],
        "selection": result.info["selection"],
        "convergence": {"status": status, "sampling_policy": sampling_policy, "observed_count": len(result.history), "serialized_count": len(samples), "history": samples},
        "checkpoints": result.info["checkpoint_hashes"],
        "motion_estimate": {"present": result.estimated_motion_trajectory is not None, "native_model": result.info.get("native_motion_model", "none")},
        "reconstruction": {"sha256": reconstruction_hash, "array_descriptors": _reconstruction_metadata(method_id=method.method_id, acquisition=acquisition, arrays=arrays)["array_descriptors"]},
        "child_timing": {"started_at": child_started_at_utc, "finished_at": child_finished_at_utc, "elapsed_seconds": (finished - started).total_seconds()},
    }
    method_info_hash = atomic_write_bytes(output_dir / "method-info.json", canonical_json_bytes(info))
    return {"reconstruction.npz": reconstruction_hash, "method-info.json": method_info_hash}


def load_reconstruction_v2(path: Path) -> ReconstructionV2:
    members = read_npz_members(Path(path))
    metadata = decode_metadata(members)
    validate_exact_keys(metadata, {"schema", "dataset_identity_sha256", "method_id", "optional_arrays", "array_descriptors"}, "reconstruction-v2 metadata")
    if metadata["schema"] != RECONSTRUCTION_V2_SCHEMA:
        raise ArtifactValidationError("reconstruction schema mismatch")
    _require_sha256(metadata["dataset_identity_sha256"], "dataset_identity_sha256")
    if type(metadata["method_id"]) is not str or not metadata["method_id"]:
        raise ArtifactValidationError("method_id must be nonempty")
    optional = metadata["optional_arrays"]
    if not isinstance(optional, Mapping):
        raise ArtifactValidationError("optional_arrays must be an object")
    validate_exact_keys(optional, {"dgi", "estimated_motion_trajectory"}, "optional_arrays")
    if any(type(optional[name]) is not bool for name in optional):
        raise ArtifactValidationError("optional array flags must be booleans")
    names = {"reconstruction", "frame_indices", "time_grid"}
    names.update(name for name in optional if optional[name])
    expected_members = {METADATA_MEMBER} | {f"{name}.npy" for name in names}
    if set(members) != expected_members:
        raise ArtifactValidationError("missing or extra reconstruction ZIP member")
    arrays = {name: load_array_member(members, name) for name in names}
    descriptors = metadata["array_descriptors"]
    if not isinstance(descriptors, Mapping):
        raise ArtifactValidationError("array_descriptors must be an object")
    validate_exact_keys(descriptors, names, "reconstruction array descriptors")
    for name, array in arrays.items():
        validate_array_descriptor(name, array, descriptors[name])
    reconstruction = arrays["reconstruction"]
    if reconstruction.ndim != 3:
        raise ArtifactValidationError("reconstruction must have shape (T,H,W)")
    t, h, w = reconstruction.shape
    indices = arrays["frame_indices"]
    if not np.issubdtype(indices.dtype, np.integer) or indices.shape != (t,) or not np.array_equal(indices, np.arange(t, dtype=indices.dtype)):
        raise ArtifactValidationError("frame_indices must equal arange(T)")
    time_grid = arrays["time_grid"]
    if time_grid.shape != (t,) or not np.issubdtype(time_grid.dtype, np.number) or not np.isfinite(time_grid).all() or (t > 1 and not np.all(time_grid[1:] > time_grid[:-1])):
        raise ArtifactValidationError("time_grid must be finite and strictly increasing")
    if "dgi" in arrays and arrays["dgi"].shape != (h, w):
        raise ArtifactValidationError("dgi shape must equal (H,W)")
    if "estimated_motion_trajectory" in arrays and (arrays["estimated_motion_trajectory"].ndim != 2 or arrays["estimated_motion_trajectory"].shape[0] != t):
        raise ArtifactValidationError("estimated_motion_trajectory shape must begin with T")
    return ReconstructionV2(metadata["dataset_identity_sha256"], metadata["method_id"], reconstruction, indices, time_grid, arrays.get("dgi"), arrays.get("estimated_motion_trajectory"), descriptors)


def _load_schema() -> Mapping[str, object]:
    schema_path = Path(__file__).resolve().parents[2] / "schemas" / "method-info-v2.schema.json"
    with schema_path.open("r", encoding="utf-8") as stream:
        return json.load(stream, object_pairs_hook=_json_object_pairs, parse_constant=_reject_json_constant)


def validate_method_child_outputs_v2(
    output_dir: Path,
    *,
    expected_method: ResolvedMethod,
    expected_dataset_identity_sha256: str,
    expected_measurements_file_sha256: str,
    expected_algorithm_seed: AlgorithmSeed,
) -> Mapping[str, str]:
    """Fail closed on malformed, linked, incomplete, or mismatched child output."""
    if type(expected_method) is not ResolvedMethod or type(expected_algorithm_seed) is not AlgorithmSeed:
        raise TypeError("expected method and seed must be resolved values")
    _require_sha256(expected_dataset_identity_sha256, "expected_dataset_identity_sha256")
    _require_sha256(expected_measurements_file_sha256, "expected_measurements_file_sha256")
    output_dir = Path(output_dir)
    names = _strict_output_names(output_dir)
    if names != _OUTPUT_FILES:
        raise ArtifactValidationError("child output inventory must contain exactly two files")
    reconstruction_path, info_path = output_dir / "reconstruction.npz", output_dir / "method-info.json"
    reconstruction = load_reconstruction_v2(reconstruction_path)
    info, info_hash = _parse_json_file(info_path)
    validator = Draft202012Validator(_load_schema())
    errors = sorted(validator.iter_errors(info), key=lambda error: list(error.path))
    if errors:
        raise ArtifactValidationError(f"method-info-v2 schema validation failed: {errors[0].message}")
    timing = info["child_timing"]
    started = _require_utc(timing["started_at"], "child_timing.started_at")
    finished = _require_utc(timing["finished_at"], "child_timing.finished_at")
    if finished < started or timing["elapsed_seconds"] != (finished - started).total_seconds():
        raise ArtifactValidationError("child timing is inconsistent")
    expected = {
        "method_id": expected_method.method_id,
        "method_config_id": expected_method.method_config_id,
        "execution_family": expected_method.execution_family,
        "execution_profile": expected_method.execution_profile,
        "dataset_identity_sha256": expected_dataset_identity_sha256,
        "measurements_file_sha256": expected_measurements_file_sha256,
        "method_config_sha256": expected_method.method_config_sha256,
        "semantic_config": thaw_json(expected_method.semantic_config),
        "algorithm_seed": {"domain": "algorithm-seed-v1", "derivation_sha256": expected_algorithm_seed.derivation_sha256, "seed_u32": expected_algorithm_seed.seed_u32},
    }
    for key, value in expected.items():
        if info[key] != value:
            raise ArtifactValidationError(f"method info {key} does not match parent request")
    if info["convergence"]["status"] != expected_method.convergence_status:
        raise ArtifactValidationError("convergence status does not match parent profile")
    if info["checkpoints"] != _expected_checkpoints(expected_method):
        raise ArtifactValidationError("checkpoints do not match parent request")
    if info["warmup"] != _warmup_counts(expected_method):
        raise ArtifactValidationError("warmup does not match parent method")
    if reconstruction.dataset_identity_sha256 != expected_dataset_identity_sha256 or reconstruction.method_id != expected_method.method_id:
        raise ArtifactValidationError("reconstruction metadata does not match parent request")
    snapshot = read_safe_file_snapshot(reconstruction_path, max_bytes=1024 * 1024 * 1024, noun="reconstruction")
    try:
        reconstruction_hash = snapshot.sha256
    finally:
        verify_safe_file_snapshot(snapshot)
    if info["reconstruction"]["sha256"] != reconstruction_hash or info["reconstruction"]["array_descriptors"] != reconstruction.array_descriptors:
        raise ArtifactValidationError("reconstruction hash or descriptors disagree")
    if info["motion_estimate"]["present"] != (reconstruction.estimated_motion_trajectory is not None):
        raise ArtifactValidationError("motion estimate presence disagrees with reconstruction")
    convergence = info["convergence"]
    if convergence["serialized_count"] != len(convergence["history"]) or convergence["observed_count"] < convergence["serialized_count"] or convergence["serialized_count"] > 21:
        raise ArtifactValidationError("convergence history counts are invalid")
    if convergence["status"] == "convergence-required" and convergence["observed_count"] < 21:
        raise ArtifactValidationError("convergence-required profiles need 21 observations")
    return {"reconstruction.npz": reconstruction_hash, "method-info.json": info_hash}
