"""Strict v2 artifacts owned by a blind method child.

This module deliberately depends only on the staged acquisition and method
request.  It neither imports evaluator code nor accepts evaluation inputs.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
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
    DirectoryInventory,
    METADATA_MEMBER,
    atomic_write_bytes,
    capture_directory_inventory,
    decode_metadata,
    load_array_member,
    npz_bytes,
    read_npz_members,
    read_safe_file_snapshot,
    verify_directory_inventory,
    verify_safe_file_snapshot,
)
from gsdiff.data._artifact_models import SPIAcquisitionData
from gsdiff.data._artifact_dataset import (
    _validate_blind_acquisition_spec,
)

from .methods import AlgorithmSeed, ResolvedMethod, thaw_json
from .parameter_counts import (
    _expected_trainable_parameter_count_for_dimensions,
    expected_trainable_parameter_count,
)
from .objectives import (
    gidc_snapshot_candidate_grid,
    recinr_snapshot_candidate_grid,
)


RECONSTRUCTION_V2_SCHEMA = "reconstruction-v2"
METHOD_INFO_V2_SCHEMA = "method-info-v2"
_OUTPUT_FILES = frozenset({"reconstruction.npz", "method-info.json"})
_SHA256_RE = re.compile(r"[0-9a-f]{64}", re.ASCII)
_UTC_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z",
    re.ASCII,
)


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


@dataclass(frozen=True)
class _CandidateSpec:
    candidate_grid: list[object]
    selected_hyperparameter_keys: tuple[str, ...] | None
    has_snapshot_state: bool


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


def _strict_output_inventory(
    output_dir: Path,
) -> tuple[DirectoryInventory, set[str]]:
    """Return a retained link-safe, flat child-output inventory."""
    inventory = capture_directory_inventory(output_dir)
    names: set[str] = set()
    for relative, _ in inventory._entries:
        if "/" in relative:
            raise ArtifactValidationError("child output inventory must be flat")
        names.add(relative)
    return inventory, names


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


def _require_exact_int(
    value: object,
    field: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if type(value) is not int or value < minimum:
        raise ArtifactValidationError(
            f"{field} must be an exact integer at least {minimum}"
        )
    if maximum is not None and value > maximum:
        raise ArtifactValidationError(
            f"{field} must be at most {maximum}"
        )
    return value


def _require_finite_number(
    value: object,
    field: str,
    *,
    minimum: float | None = None,
) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ArtifactValidationError(f"{field} must be a finite number")
    number = float(value)
    if minimum is not None and number < minimum:
        raise ArtifactValidationError(
            f"{field} must be at least {minimum}"
        )
    return number


def _validate_algorithm_seed(seed: AlgorithmSeed, field: str) -> None:
    if type(seed) is not AlgorithmSeed:
        raise TypeError(f"{field} must be an AlgorithmSeed")
    _require_sha256(seed.derivation_sha256, f"{field}.derivation_sha256")
    _require_exact_int(
        seed.seed_u32,
        f"{field}.seed_u32",
        maximum=2**32 - 1,
    )


def _validate_acquisition_contract(
    acquisition: SPIAcquisitionData,
    field: str,
) -> None:
    if type(acquisition) is not SPIAcquisitionData:
        raise TypeError(f"{field} must be an SPIAcquisitionData")
    _require_sha256(
        acquisition.dataset_identity_sha256,
        f"{field}.dataset_identity_sha256",
    )
    for name in ("H", "W", "T", "K"):
        _require_exact_int(
            getattr(acquisition, name), f"{field}.{name}", minimum=1
        )
    _require_exact_int(
        acquisition.holdout_K, f"{field}.holdout_K"
    )
    expected_shapes = {
        "patterns": (acquisition.K, acquisition.H, acquisition.W),
        "measurements": (acquisition.K,),
        "frame_indices": (acquisition.K,),
        "time_grid": (acquisition.T,),
    }
    arrays = {
        name: getattr(acquisition, name) for name in expected_shapes
    }
    for name, expected_shape in expected_shapes.items():
        array = arrays[name]
        if type(array) is not np.ndarray or array.shape != expected_shape:
            raise ArtifactValidationError(
                f"{field}.{name} shape is inconsistent"
            )
        if (
            np.iscomplexobj(array)
            or not np.issubdtype(array.dtype, np.number)
            or not np.isfinite(array).all()
        ):
            raise ArtifactValidationError(
                f"{field}.{name} must be finite real numeric"
            )
    frame_indices = acquisition.frame_indices
    if (
        np.issubdtype(frame_indices.dtype, np.bool_)
        or not np.issubdtype(frame_indices.dtype, np.integer)
        or np.any(frame_indices < 0)
        or np.any(frame_indices >= acquisition.T)
    ):
        raise ArtifactValidationError(
            f"{field}.frame_indices must be integer values in [0,T)"
        )
    time_grid = acquisition.time_grid
    if acquisition.T > 1 and not np.all(time_grid[1:] > time_grid[:-1]):
        raise ArtifactValidationError(
            f"{field}.time_grid must be strictly increasing"
        )
    for name in (
        "holdout_patterns",
        "holdout_measurements",
        "holdout_frame_indices",
    ):
        value = getattr(acquisition, name)
        if value is not None:
            arrays[name] = value
    if not isinstance(acquisition.array_descriptors, Mapping):
        raise ArtifactValidationError(
            f"{field}.array_descriptors must be an object"
        )
    validate_exact_keys(
        acquisition.array_descriptors,
        set(arrays),
        f"{field}.array_descriptors",
    )
    for name, array in arrays.items():
        validate_array_descriptor(
            name, array, acquisition.array_descriptors[name]
        )


def _native_iteration(method: ResolvedMethod) -> dict[str, object]:
    semantics = method.semantic_config
    solver = semantics.get("solver")
    if method.method_id == "dgi":
        return {
            "unit": semantics["native_unit"],
            "budget": semantics["native_budget"],
        }
    if not isinstance(solver, Mapping):
        raise ArtifactValidationError("method solver semantics are missing")
    if method.method_id in {"static_cs", "perframe_cs", "monin"}:
        return {"unit": "admm-iteration", "budget": solver["n_admm"]}
    if method.method_id == "tv3d":
        return {
            "unit": "primal-dual-iteration",
            "budget": solver["iterations"],
        }
    if method.method_id == "gidc3dtv":
        return {"unit": "adam-step", "budget": solver["n_steps"]}
    if method.method_id == "recinr":
        return {
            "unit": "optimization-step",
            "budget": (
                solver["warm_steps"]
                + solver["flow_steps"]
                + solver["joint_steps"]
            ),
        }
    if method.method_id in {"siren", "recinr_se2"}:
        return {"unit": "sgd-step", "budget": solver["sgd_steps"]}
    if method.method_id in {"gsdiff_tv", "gsdiff_diffusion"}:
        return {
            "unit": "outer-iteration",
            "budget": solver["outer_iterations"],
        }
    raise ArtifactValidationError("unknown native method iteration binding")


def _classical_parameter_count(
    method: ResolvedMethod,
    acquisition: SPIAcquisitionData,
) -> int:
    return _expected_trainable_parameter_count_for_dimensions(
        method,
        h=acquisition.H,
        w=acquisition.W,
        t=acquisition.T,
    )


def _candidate_spec(
    method: ResolvedMethod,
) -> _CandidateSpec | None:
    solver = method.semantic_config.get("solver")
    if not isinstance(solver, Mapping):
        return None
    if method.method_id == "gidc3dtv":
        return _CandidateSpec(
            candidate_grid=gidc_snapshot_candidate_grid(solver),
            selected_hyperparameter_keys=("xi_xy", "xi_t"),
            has_snapshot_state=True,
        )
    if method.method_id == "recinr":
        return _CandidateSpec(
            candidate_grid=recinr_snapshot_candidate_grid(solver),
            selected_hyperparameter_keys=None,
            has_snapshot_state=True,
        )
    if "lambda_grid" in solver:
        return _CandidateSpec(
            candidate_grid=list(solver["lambda_grid"]),
            selected_hyperparameter_keys=("lambda",),
            has_snapshot_state=False,
        )
    if "lambda_xy" in solver and "lambda_t" in solver:
        return _CandidateSpec(
            candidate_grid=[
                {"lambda_xy": xy, "lambda_t": temporal}
                for xy in solver["lambda_xy"]
                for temporal in solver["lambda_t"]
            ],
            selected_hyperparameter_keys=("lambda_xy", "lambda_t"),
            has_snapshot_state=False,
        )
    return None


def _validate_selection_spec(
    spec: _CandidateSpec | None,
    selected_hyperparameters: object,
    selection: object,
) -> None:
    if spec is None:
        if selected_hyperparameters is not None or selection is not None:
            raise ArtifactValidationError(
                "method has no locked candidate selection"
            )
        return
    candidate_grid = spec.candidate_grid
    if not isinstance(selection, Mapping):
        raise ArtifactValidationError("selection must be an object")
    validate_exact_keys(
        selection,
        {"formula_id", "candidate_grid", "selected_candidate", "rows"},
        "selection",
    )
    if selection["formula_id"] != "heldout-normalized-l2-v1":
        raise ArtifactValidationError("selection formula_id is not locked")
    if canonical_json_bytes(selection["candidate_grid"]) != canonical_json_bytes(
        candidate_grid
    ):
        raise ArtifactValidationError(
            "selection candidate_grid does not match method semantics"
        )
    rows = selection["rows"]
    if type(rows) is not list or len(rows) != len(candidate_grid):
        raise ArtifactValidationError(
            "selection rows must cover the exact candidate grid"
        )
    values: list[float] = []
    for index, (row, candidate) in enumerate(zip(rows, candidate_grid)):
        if not isinstance(row, Mapping):
            raise ArtifactValidationError("selection row must be an object")
        validate_exact_keys(
            row,
            {"candidate", "formula_id", "numerator", "denominator", "value"},
            f"selection.rows[{index}]",
        )
        if canonical_json_bytes(row["candidate"]) != canonical_json_bytes(
            candidate
        ):
            raise ArtifactValidationError(
                "selection row candidate order is not locked"
            )
        if row["formula_id"] != "heldout-normalized-l2-v1":
            raise ArtifactValidationError(
                "selection row formula_id is not locked"
            )
        numerator = _require_finite_number(
            row["numerator"],
            f"selection.rows[{index}].numerator",
            minimum=0,
        )
        denominator = _require_finite_number(
            row["denominator"],
            f"selection.rows[{index}].denominator",
            minimum=0,
        )
        if denominator == 0:
            raise ArtifactValidationError(
                "selection row denominator must be positive"
            )
        value = _require_finite_number(
            row["value"],
            f"selection.rows[{index}].value",
            minimum=0,
        )
        if not math.isclose(
            value,
            numerator / denominator,
            rel_tol=1e-15,
            abs_tol=0.0,
        ):
            raise ArtifactValidationError(
                "selection row value does not equal numerator/denominator"
            )
        values.append(value)
    expected_selected = candidate_grid[
        min(range(len(values)), key=values.__getitem__)
    ]
    if canonical_json_bytes(
        selection["selected_candidate"]
    ) != canonical_json_bytes(expected_selected):
        raise ArtifactValidationError(
            "selection candidate/hyperparameter does not preserve "
            "first-minimum candidate order"
        )
    hyperparameter_keys = spec.selected_hyperparameter_keys
    if hyperparameter_keys is None:
        if selected_hyperparameters is not None:
            raise ArtifactValidationError(
                "selected_hyperparameters must be null for snapshot-only selection"
            )
        return
    if not isinstance(selected_hyperparameters, Mapping):
        raise ArtifactValidationError(
            "selected_hyperparameters must be an object"
        )
    validate_exact_keys(
        selected_hyperparameters,
        set(hyperparameter_keys),
        "selected_hyperparameters",
    )
    if len(hyperparameter_keys) == 1:
        expected_hyperparameters: object = {
            hyperparameter_keys[0]: expected_selected
        }
    else:
        if not isinstance(expected_selected, Mapping):
            raise ArtifactValidationError(
                "selected candidate cannot be projected to hyperparameters"
            )
        expected_hyperparameters = {
            key: expected_selected[key] for key in hyperparameter_keys
        }
    if canonical_json_bytes(
        selected_hyperparameters
    ) != canonical_json_bytes(expected_hyperparameters):
        raise ArtifactValidationError(
            "selected_hyperparameters do not match selected candidate projection"
        )


def _validate_selection(
    method: ResolvedMethod,
    selected_hyperparameters: object,
    selection: object,
) -> None:
    _validate_selection_spec(
        _candidate_spec(method),
        selected_hyperparameters,
        selection,
    )


_HISTORY_INDEX_FIELDS = {
    "iteration": "iteration",
    "step": "step",
    "outer-iteration": "outer_iteration",
    "pass": "pass",
}
_HISTORY_METRIC_FIELDS = {
    "objective",
    "loss",
    "data_fidelity",
    "primal_residual",
    "dual_residual",
    "learning_rate",
}


def _validate_history_row(row: Mapping[str, object], index: int) -> None:
    kind = row.get("kind")
    if type(kind) is not str or kind not in _HISTORY_INDEX_FIELDS:
        raise ArtifactValidationError(
            f"history[{index}].kind is not a supported native row kind"
        )
    index_field = _HISTORY_INDEX_FIELDS[kind]
    allowed = {"kind", index_field} | _HISTORY_METRIC_FIELDS
    if kind == "step":
        allowed.add("reconstruction_source")
    if not set(row).issubset(allowed):
        raise ArtifactValidationError(
            f"history[{index}] contains an unapproved field"
        )
    if index_field not in row:
        raise ArtifactValidationError(
            f"history[{index}] is missing {index_field}"
        )
    _require_exact_int(row[index_field], f"history[{index}].{index_field}")
    for field in set(row) & _HISTORY_METRIC_FIELDS:
        _require_finite_number(row[field], f"history[{index}].{field}")
    if (
        "reconstruction_source" in row
        and row["reconstruction_source"] is not True
    ):
        raise ArtifactValidationError(
            f"history[{index}].reconstruction_source must be true"
        )


def _validate_reconstruction_source_spec(
    spec: _CandidateSpec | None,
    selection: object,
    history: tuple[Mapping[str, object], ...] | list[object],
    *,
    field: str,
) -> None:
    source_rows = [
        (index, row)
        for index, row in enumerate(history)
        if isinstance(row, Mapping)
        and "reconstruction_source" in row
    ]
    if spec is None or not spec.has_snapshot_state:
        if source_rows:
            raise ArtifactValidationError(
                f"{field} contains reconstruction_source for a non-snapshot method"
            )
        return
    if len(source_rows) != 1:
        raise ArtifactValidationError(
            f"{field} must contain exactly one reconstruction_source row"
        )
    if not isinstance(selection, Mapping):
        raise ArtifactValidationError(
            "snapshot selection is required for reconstruction_source"
        )
    selected_candidate = selection["selected_candidate"]
    if not isinstance(selected_candidate, Mapping):
        raise ArtifactValidationError(
            "snapshot selected_candidate must be an object"
        )
    selected_step = _require_exact_int(
        selected_candidate.get("snapshot_step"),
        "selection.selected_candidate.snapshot_step",
        minimum=1,
    )
    index, source_row = source_rows[0]
    if source_row.get("kind") != "step":
        raise ArtifactValidationError(
            f"{field}[{index}] reconstruction_source must be a step row"
        )
    source_step = _require_exact_int(
        source_row.get("step"),
        f"{field}[{index}].step",
        minimum=1,
    )
    if source_step != selected_step:
        raise ArtifactValidationError(
            "reconstruction_source step does not match selected snapshot"
        )


def _validate_reconstruction_source(
    method: ResolvedMethod,
    selection: object,
    history: tuple[Mapping[str, object], ...] | list[object],
    *,
    field: str,
) -> None:
    _validate_reconstruction_source_spec(
        _candidate_spec(method),
        selection,
        history,
        field=field,
    )


def _validate_result(
    result: MethodChildResult,
    acquisition: SPIAcquisitionData,
    method: ResolvedMethod,
) -> None:
    if type(result) is not MethodChildResult:
        raise TypeError("result must be a MethodChildResult")
    if result.method_id != method.method_id:
        raise ArtifactValidationError("result method_id does not match method")
    if type(result.reconstruction) is not np.ndarray:
        raise ArtifactValidationError("reconstruction must be an ndarray")
    reconstruction = result.reconstruction
    if reconstruction.shape != (acquisition.T, acquisition.H, acquisition.W):
        raise ArtifactValidationError("reconstruction shape disagrees with acquisition")
    if not np.issubdtype(reconstruction.dtype, np.number) or np.iscomplexobj(reconstruction) or not np.isfinite(reconstruction).all():
        raise ArtifactValidationError("reconstruction must be finite real numeric")
    if result.dgi is not None:
        dgi = np.asarray(result.dgi)
        if dgi.shape != (acquisition.H, acquisition.W) or not np.issubdtype(dgi.dtype, np.number) or np.iscomplexobj(dgi) or not np.isfinite(dgi).all():
            raise ArtifactValidationError("dgi shape or values are invalid")
    _validate_auxiliary_arrays(
        method,
        reconstruction,
        result.dgi,
    )
    if result.estimated_motion_trajectory is not None:
        motion = np.asarray(result.estimated_motion_trajectory)
        if motion.shape != (acquisition.T, 3) or not np.issubdtype(motion.dtype, np.number) or np.iscomplexobj(motion) or not np.isfinite(motion).all():
            raise ArtifactValidationError("estimated_motion_trajectory shape or values are invalid")
    if not isinstance(result.info, Mapping):
        raise ArtifactValidationError("result info must be an object")
    allowed = {"parameter_count", "native_iteration_unit", "native_iteration_budget", "convergence_status", "selected_hyperparameters", "selection", "checkpoint_hashes", "native_motion_model"}
    required = allowed - {"native_motion_model"}
    if set(result.info) != required and set(result.info) != allowed:
        raise ArtifactValidationError("result info keys are invalid")
    parameter_count = _require_exact_int(
        result.info["parameter_count"], "parameter_count"
    )
    expected_parameter_count = _classical_parameter_count(method, acquisition)
    if parameter_count != expected_parameter_count:
        raise ArtifactValidationError(
            "parameter_count does not match the locked classical formula"
        )
    expected_iteration = _native_iteration(method)
    actual_iteration = {
        "unit": result.info["native_iteration_unit"],
        "budget": result.info["native_iteration_budget"],
    }
    if canonical_json_bytes(actual_iteration) != canonical_json_bytes(
        expected_iteration
    ):
        raise ArtifactValidationError(
            "native iteration unit/budget do not match method semantics"
        )
    if result.info["convergence_status"] != method.convergence_status:
        raise ArtifactValidationError("convergence_status does not match method profile")
    if not isinstance(result.info["checkpoint_hashes"], list):
        raise ArtifactValidationError("checkpoint_hashes must be a list")
    if (
        type(result.info.get("native_motion_model", "none")) is not str
        or not result.info.get("native_motion_model", "none")
    ):
        raise ArtifactValidationError("native_motion_model must be a nonempty string")
    native_motion_model = result.info.get("native_motion_model", "none")
    if (
        result.estimated_motion_trajectory is None
        and native_motion_model != "none"
    ) or (
        result.estimated_motion_trajectory is not None
        and native_motion_model == "none"
    ):
        raise ArtifactValidationError(
            "native_motion_model must agree with trajectory presence"
        )
    expected_motion = (
        method.method_id == "monin" or method.execution_family == "gsdiff"
    )
    if (result.estimated_motion_trajectory is not None) is not expected_motion:
        raise ArtifactValidationError(
            "motion trajectory presence disagrees with method contract"
        )
    if result.info["checkpoint_hashes"] != _expected_checkpoints(method):
        raise ArtifactValidationError("checkpoint_hashes do not match method requirements")
    _validate_selection(
        method,
        result.info["selected_hyperparameters"],
        result.info["selection"],
    )
    validate_exact_json_native(dict(result.info), "result info")
    if type(result.history) is not tuple or any(not isinstance(row, Mapping) for row in result.history):
        raise ArtifactValidationError("history must be a tuple of objects")
    for index, row in enumerate(result.history):
        validate_exact_json_native(dict(row), f"history[{index}]")
        _validate_history_row(row, index)
    _validate_full_history(result.history, _history_contract(method))
    _validate_reconstruction_source(
        method,
        result.info["selection"],
        result.history,
        field="history",
    )


def _expected_checkpoints(method: ResolvedMethod) -> list[dict[str, str]]:
    return [
        {"logical_id": requirement.logical_id, "sha256": requirement.sha256}
        for requirement in method.checkpoint_requirements
    ]


def _dgi_presence(method: ResolvedMethod) -> str:
    return (
        "required"
        if method.method_id
        in {"dgi", "siren", "recinr_se2", "gsdiff_tv", "gsdiff_diffusion"}
        else "forbidden"
    )


def _validate_auxiliary_arrays(
    method: ResolvedMethod,
    reconstruction: np.ndarray,
    dgi: np.ndarray | None,
) -> None:
    required = _dgi_presence(method) == "required"
    if (dgi is not None) is not required:
        raise ArtifactValidationError(
            "dgi auxiliary array presence disagrees with method contract"
        )
    if method.method_id == "dgi":
        assert dgi is not None
        if not np.array_equal(dgi, reconstruction[0]) or not np.all(
            reconstruction == reconstruction[0]
        ):
            raise ArtifactValidationError(
                "DGI reconstruction must repeat the exact dgi frame"
            )


def _warmup_counts(method: ResolvedMethod) -> dict[str, int]:
    solver = method.semantic_config.get("solver")
    if not isinstance(solver, Mapping):
        return {"splitting": 0, "motion": 0}
    splitting = solver.get("splitting_warmup_outer", 0)
    motion = solver.get("motion_warmup_outer", solver.get("motion_warmup_steps", 0))
    if type(splitting) is not int or type(motion) is not int or splitting < 0 or motion < 0:
        raise ArtifactValidationError("method warmup counts must be nonnegative integers")
    return {"splitting": splitting, "motion": motion}


_NATIVE_HISTORY_FIELDS = {
    "admm-iteration": ("iteration", "iteration"),
    "primal-dual-iteration": ("iteration", "iteration"),
    "adam-step": ("step", "step"),
    "optimization-step": ("step", "step"),
    "sgd-step": ("step", "step"),
    "outer-iteration": ("outer-iteration", "outer_iteration"),
}


def _history_contract(method: ResolvedMethod) -> dict[str, object]:
    native = _native_iteration(method)
    if method.method_id == "dgi":
        return {
            "kind": None,
            "index_field": None,
            "observed_count": 0,
            "allowed_metric_fields": [],
            "phases": [],
            "final_required_metric_fields": [],
        }
    binding = _NATIVE_HISTORY_FIELDS.get(native["unit"])
    if binding is None:
        raise ArtifactValidationError("native history unit is unsupported")
    kind, index_field = binding
    budget = native["budget"]
    assert type(budget) is int
    if method.method_id in {"static_cs", "perframe_cs", "monin"}:
        required = [
            "data_fidelity",
            "primal_residual",
            "dual_residual",
        ]
        phases = [_history_phase(1, budget, required)]
        allowed = required
        final_required: list[str] = []
    elif method.method_id == "tv3d":
        required = ["data_fidelity"]
        phases = [_history_phase(1, budget, required)]
        allowed = required
        final_required = []
    elif method.method_id == "gidc3dtv":
        required = ["loss", "data_fidelity", "learning_rate"]
        phases = [_history_phase(1, budget, required)]
        allowed = required
        final_required = []
    elif method.method_id == "recinr":
        solver = method.semantic_config["solver"]
        assert isinstance(solver, Mapping)
        warm_steps = solver["warm_steps"]
        assert type(warm_steps) is int and 0 <= warm_steps <= budget
        phases = []
        if warm_steps:
            phases.append(
                _history_phase(
                    1,
                    warm_steps,
                    ["loss", "learning_rate"],
                )
            )
        if warm_steps < budget:
            phases.append(
                _history_phase(
                    warm_steps + 1,
                    budget,
                    ["loss", "data_fidelity", "learning_rate"],
                )
            )
        allowed = ["loss", "data_fidelity", "learning_rate"]
        final_required = []
    elif method.method_id in {"siren", "recinr_se2"}:
        required = ["loss", "data_fidelity"]
        phases = [_history_phase(1, budget, required)]
        allowed = [*required, "objective"]
        final_required = ["objective"]
    elif method.method_id in {"gsdiff_tv", "gsdiff_diffusion"}:
        required = ["loss", "primal_residual", "dual_residual"]
        phases = [_history_phase(1, budget, required)]
        allowed = [*required, "objective"]
        final_required = ["objective"]
    else:
        raise ArtifactValidationError("unknown method history metric contract")
    return {
        "kind": kind,
        "index_field": index_field,
        "observed_count": budget,
        "allowed_metric_fields": allowed,
        "phases": phases,
        "final_required_metric_fields": final_required,
    }


def _history_phase(
    start: int,
    end: int,
    required_metric_fields: list[str],
) -> dict[str, object]:
    return {
        "start": start,
        "end": end,
        "required_metric_fields": required_metric_fields,
    }


def _validate_history_metrics(
    row: Mapping[str, object],
    policy: Mapping[str, object],
    native_index: int,
) -> None:
    allowed = policy["allowed_metric_fields"]
    phases = policy["phases"]
    final_required = policy["final_required_metric_fields"]
    observed = policy["observed_count"]
    if (
        type(allowed) is not list
        or any(field not in _HISTORY_METRIC_FIELDS for field in allowed)
        or len(allowed) != len(set(allowed))
        or type(phases) is not list
        or type(final_required) is not list
        or any(field not in allowed for field in final_required)
        or type(observed) is not int
    ):
        raise ArtifactValidationError("history metric contract is invalid")
    matching = [
        phase
        for phase in phases
        if isinstance(phase, Mapping)
        and type(phase.get("start")) is int
        and type(phase.get("end")) is int
        and phase["start"] <= native_index <= phase["end"]
    ]
    if len(matching) != 1:
        raise ArtifactValidationError("history metric phase contract is invalid")
    phase = matching[0]
    if set(phase) != {"start", "end", "required_metric_fields"}:
        raise ArtifactValidationError("history metric phase contract is invalid")
    required = phase["required_metric_fields"]
    if (
        type(required) is not list
        or any(field not in allowed for field in required)
        or len(required) != len(set(required))
    ):
        raise ArtifactValidationError("history metric phase contract is invalid")
    present = set(row) & _HISTORY_METRIC_FIELDS
    if not set(required).issubset(present) or not present.issubset(set(allowed)):
        raise ArtifactValidationError(
            "history row metrics disagree with the method contract"
        )
    if native_index == observed:
        if not set(final_required).issubset(present):
            raise ArtifactValidationError(
                "final history row metrics disagree with the method contract"
            )
    elif "objective" in present:
        raise ArtifactValidationError(
            "objective is allowed only on the final method history row"
        )


def _validate_full_history(
    history: tuple[Mapping[str, object], ...],
    policy: Mapping[str, object],
) -> None:
    observed = policy["observed_count"]
    if type(observed) is not int or observed < 0 or len(history) != observed:
        raise ArtifactValidationError(
            "history does not cover the exact native iteration budget"
        )
    kind = policy["kind"]
    index_field = policy["index_field"]
    for index, row in enumerate(history):
        _validate_history_row(row, index)
        if (
            row["kind"] != kind
            or type(index_field) is not str
            or row[index_field] != index + 1
        ):
            raise ArtifactValidationError(
                "history kind/index does not match native iteration order"
            )
        _validate_history_metrics(row, policy, index + 1)


def _expected_sampled_history_indices(
    policy: Mapping[str, object],
    selection_contract: Mapping[str, object],
    selection: object,
) -> list[int]:
    observed = policy["observed_count"]
    assert type(observed) is int
    if observed < 21:
        return list(range(1, observed + 1))
    zero_based = [
        (observed - 1) * index // 20 for index in range(21)
    ]
    if selection_contract["reconstruction_source"] == "required":
        if not isinstance(selection, Mapping):
            raise ArtifactValidationError(
                "snapshot history requires an identity-bound selection"
            )
        selected = selection["selected_candidate"]
        if not isinstance(selected, Mapping):
            raise ArtifactValidationError(
                "snapshot history selected candidate is invalid"
            )
        selected_step = _require_exact_int(
            selected.get("snapshot_step"),
            "selection.selected_candidate.snapshot_step",
            minimum=1,
            maximum=observed,
        )
        source_index = selected_step - 1
        if source_index not in zero_based:
            replacement = min(
                range(1, 20),
                key=lambda index: (
                    abs(zero_based[index] - source_index),
                    index,
                ),
            )
            zero_based[replacement] = source_index
            zero_based.sort()
    return [index + 1 for index in zero_based]


def _validate_serialized_history_against_method(
    convergence: Mapping[str, object],
    method: ResolvedMethod,
    selection: object,
) -> None:
    history_policy = _history_contract(method)
    candidate = _candidate_spec(method)
    selection_contract = {
        "reconstruction_source": (
            "required"
            if candidate is not None and candidate.has_snapshot_state
            else "forbidden"
        )
    }
    expected_indices = _expected_sampled_history_indices(
        history_policy,
        selection_contract,
        selection,
    )
    observed = history_policy["observed_count"]
    assert type(observed) is int
    expected_policy = (
        "all-observations"
        if observed < 21
        else "floor-grid-21-with-reconstruction-source-v1"
        if selection_contract["reconstruction_source"] == "required"
        else "floor(i*(observed_count-1)/20), i=0..20"
    )
    history = convergence["history"]
    if (
        convergence["observed_count"] != observed
        or convergence["serialized_count"] != len(expected_indices)
        or convergence["sampling_policy"] != expected_policy
        or type(history) is not list
        or len(history) != len(expected_indices)
    ):
        raise ArtifactValidationError(
            "method info history disagrees with parent native iteration contract"
        )
    kind = history_policy["kind"]
    index_field = history_policy["index_field"]
    for row, expected_index in zip(history, expected_indices, strict=True):
        if (
            not isinstance(row, Mapping)
            or row["kind"] != kind
            or type(index_field) is not str
            or row[index_field] != expected_index
        ):
            raise ArtifactValidationError(
                "method info sampled history index disagrees with parent contract"
            )
        _validate_history_metrics(row, history_policy, expected_index)


def build_method_info_contract_v1(
    method: ResolvedMethod,
    expected_acquisition_spec: Mapping[str, object],
) -> dict[str, object]:
    """Build the path-free, parent-authoritative method-info contract."""
    if type(method) is not ResolvedMethod:
        raise TypeError("method must be a ResolvedMethod")
    acquisition_spec = _validate_blind_acquisition_spec(
        expected_acquisition_spec
    )
    dimensions = acquisition_spec["dimensions"]
    assert isinstance(dimensions, Mapping)
    parameter_count = expected_trainable_parameter_count(
        method,
        acquisition_spec,
    )
    candidate = _candidate_spec(method)
    selection_contract: dict[str, object]
    if candidate is None:
        selection_contract = {
            "formula_id": None,
            "candidate_grid": None,
            "selected_hyperparameter_keys": None,
            "reconstruction_source": "forbidden",
        }
    else:
        selection_contract = {
            "formula_id": "heldout-normalized-l2-v1",
            "candidate_grid": candidate.candidate_grid,
            "selected_hyperparameter_keys": (
                None
                if candidate.selected_hyperparameter_keys is None
                else list(candidate.selected_hyperparameter_keys)
            ),
            "reconstruction_source": (
                "required" if candidate.has_snapshot_state else "forbidden"
            ),
        }
    if method.method_id == "monin":
        motion_policy = {
            "presence": "required",
            "native_model": "translation-polynomial",
        }
    elif method.execution_family == "gsdiff":
        motion_policy = {
            "presence": "required",
            "native_model": "se2-polynomial",
        }
    else:
        motion_policy = {"presence": "forbidden", "native_model": "none"}
    contract = {
        "schema": "method-info-contract-v1",
        "method_id": method.method_id,
        "method_config_id": method.method_config_id,
        "execution_family": method.execution_family,
        "execution_profile": method.execution_profile,
        "method_config_sha256": method.method_config_sha256,
        "semantic_config": thaw_json(method.semantic_config),
        "native_iteration": _native_iteration(method),
        "warmup": _warmup_counts(method),
        "checkpoints": _expected_checkpoints(method),
        "convergence_status": method.convergence_status,
        "selection": selection_contract,
        "motion_estimate": motion_policy,
        "expected_parameter_count": parameter_count,
        "history": _history_contract(method),
        "auxiliary_arrays": {"dgi": _dgi_presence(method)},
    }
    validate_exact_json_native(contract, "method info contract")
    return contract


def validate_method_info_contract_v1(
    info: Mapping[str, object],
    contract: Mapping[str, object],
) -> None:
    """Cross-lock method-info evidence to its identity-bound contract."""
    validate_exact_keys(
        contract,
        {
            "schema",
            "method_id",
            "method_config_id",
            "execution_family",
            "execution_profile",
            "method_config_sha256",
            "semantic_config",
            "native_iteration",
            "warmup",
            "checkpoints",
            "convergence_status",
            "selection",
            "motion_estimate",
            "expected_parameter_count",
            "history",
            "auxiliary_arrays",
        },
        "method info contract",
    )
    validate_exact_json_native(contract, "method info contract")
    if contract["schema"] != "method-info-contract-v1":
        raise ArtifactValidationError("method info contract schema is invalid")
    auxiliary = contract["auxiliary_arrays"]
    if (
        type(auxiliary) is not dict
        or set(auxiliary) != {"dgi"}
        or auxiliary["dgi"] not in {"required", "forbidden"}
    ):
        raise ArtifactValidationError(
            "method info auxiliary array contract is invalid"
        )
    direct_fields = (
        "method_id",
        "method_config_id",
        "execution_family",
        "execution_profile",
        "method_config_sha256",
        "semantic_config",
        "native_iteration",
        "warmup",
        "checkpoints",
    )
    for field in direct_fields:
        if canonical_json_bytes(info[field]) != canonical_json_bytes(
            contract[field]
        ):
            raise ArtifactValidationError(
                f"method info {field} disagrees with identity contract"
            )
    convergence = info["convergence"]
    if (
        not isinstance(convergence, Mapping)
        or convergence["status"] != contract["convergence_status"]
    ):
        raise ArtifactValidationError(
            "method info convergence disagrees with identity contract"
        )
    expected_parameter_count = contract["expected_parameter_count"]
    if info["parameter_count"] != expected_parameter_count:
        raise ArtifactValidationError(
            "method info parameter_count disagrees with identity contract"
        )
    selection_contract = contract["selection"]
    assert isinstance(selection_contract, Mapping)
    selection = info["selection"]
    selected = info["selected_hyperparameters"]
    if selection_contract["formula_id"] is None:
        candidate_spec = None
    else:
        candidate_grid = selection_contract["candidate_grid"]
        key_values = selection_contract["selected_hyperparameter_keys"]
        if (
            selection_contract["formula_id"]
            != "heldout-normalized-l2-v1"
            or type(candidate_grid) is not list
            or (
                key_values is not None
                and (
                    type(key_values) is not list
                    or any(type(item) is not str for item in key_values)
                )
            )
            or selection_contract["reconstruction_source"]
            not in {"required", "forbidden"}
        ):
            raise ArtifactValidationError(
                "method info selection contract is invalid"
            )
        candidate_spec = _CandidateSpec(
            list(candidate_grid),
            None if key_values is None else tuple(key_values),
            selection_contract["reconstruction_source"] == "required",
        )
    _validate_selection_spec(candidate_spec, selected, selection)
    convergence_history = convergence["history"]
    assert type(convergence_history) is list
    _validate_reconstruction_source_spec(
        candidate_spec,
        selection,
        convergence_history,
        field="convergence.history",
    )
    history_policy = contract["history"]
    if not isinstance(history_policy, Mapping):
        raise ArtifactValidationError("method info history contract is invalid")
    validate_exact_keys(
        history_policy,
        {
            "kind",
            "index_field",
            "observed_count",
            "allowed_metric_fields",
            "phases",
            "final_required_metric_fields",
        },
        "method info history contract",
    )
    observed_count = history_policy["observed_count"]
    if (
        type(observed_count) is not int
        or observed_count < 0
        or convergence["observed_count"] != observed_count
    ):
        raise ArtifactValidationError(
            "method info history count disagrees with identity contract"
        )
    expected_indices = _expected_sampled_history_indices(
        history_policy,
        selection_contract,
        selection,
    )
    expected_policy = (
        "all-observations"
        if observed_count < 21
        else "floor-grid-21-with-reconstruction-source-v1"
        if selection_contract["reconstruction_source"] == "required"
        else "floor(i*(observed_count-1)/20), i=0..20"
    )
    if (
        convergence["serialized_count"] != len(expected_indices)
        or convergence["sampling_policy"] != expected_policy
        or len(convergence_history) != len(expected_indices)
    ):
        raise ArtifactValidationError(
            "method info sampled history disagrees with identity contract"
        )
    kind = history_policy["kind"]
    index_field = history_policy["index_field"]
    for row_index, (row, expected_index) in enumerate(
        zip(convergence_history, expected_indices, strict=True)
    ):
        assert isinstance(row, Mapping)
        if (
            row["kind"] != kind
            or type(index_field) is not str
            or row[index_field] != expected_index
        ):
            raise ArtifactValidationError(
                "method info sampled history kind/index disagrees with identity contract"
            )
        _validate_history_metrics(row, history_policy, expected_index)
    motion = info["motion_estimate"]
    assert isinstance(motion, Mapping)
    motion_contract = contract["motion_estimate"]
    assert isinstance(motion_contract, Mapping)
    expected_present = motion_contract["presence"] == "required"
    if (
        motion["present"] is not expected_present
        or motion["native_model"] != motion_contract["native_model"]
    ):
        raise ArtifactValidationError(
            "method info motion estimate disagrees with identity contract"
        )


def _sample_history(history: tuple[Mapping[str, object], ...]) -> tuple[list[object], str]:
    observed = len(history)
    if observed < 21:
        indices: list[int] = list(range(observed))
        policy = "all-observations"
    else:
        indices = [
            int(index)
            for index in (
                (observed - 1) * np.arange(21, dtype=np.int64) // 20
            )
        ]
        source_indices = [
            index
            for index, row in enumerate(history)
            if "reconstruction_source" in row
        ]
        if source_indices:
            source_index = source_indices[0]
            if source_index not in indices:
                replacement = min(
                    range(1, 20),
                    key=lambda index: (
                        abs(indices[index] - source_index),
                        index,
                    ),
                )
                indices[replacement] = source_index
                indices.sort()
            policy = "floor-grid-21-with-reconstruction-source-v1"
        else:
            policy = "floor(i*(observed_count-1)/20), i=0..20"
    return [dict(history[index]) for index in indices], policy


def _validate_method_info_schema(info: Mapping[str, object]) -> None:
    errors = sorted(
        Draft202012Validator(_load_schema()).iter_errors(info),
        key=lambda error: [str(item) for item in error.path],
    )
    if errors:
        raise ArtifactValidationError(
            "method-info-v2 schema validation failed: "
            f"{errors[0].message}"
        )


def _validate_convergence(
    convergence: object,
    *,
    method: ResolvedMethod,
    selection: object,
    expected_status: str,
) -> None:
    if not isinstance(convergence, Mapping):
        raise ArtifactValidationError("convergence must be an object")
    validate_exact_keys(
        convergence,
        {
            "status",
            "sampling_policy",
            "observed_count",
            "serialized_count",
            "history",
        },
        "convergence",
    )
    if convergence["status"] != expected_status:
        raise ArtifactValidationError(
            "convergence status does not match parent profile"
        )
    observed = _require_exact_int(
        convergence["observed_count"], "convergence.observed_count"
    )
    serialized = _require_exact_int(
        convergence["serialized_count"],
        "convergence.serialized_count",
        maximum=21,
    )
    expected_serialized = observed if observed < 21 else 21
    spec = _candidate_spec(method)
    has_snapshot_state = spec is not None and spec.has_snapshot_state
    if observed < 21:
        expected_policy = "all-observations"
    elif has_snapshot_state:
        expected_policy = "floor-grid-21-with-reconstruction-source-v1"
    else:
        expected_policy = "floor(i*(observed_count-1)/20), i=0..20"
    if serialized != expected_serialized:
        raise ArtifactValidationError(
            "convergence serialized_count is inconsistent"
        )
    if convergence["sampling_policy"] != expected_policy:
        raise ArtifactValidationError(
            "convergence sampling_policy is inconsistent"
        )
    history = convergence["history"]
    if type(history) is not list or len(history) != serialized:
        raise ArtifactValidationError(
            "convergence history length is inconsistent"
        )
    for index, row in enumerate(history):
        if not isinstance(row, Mapping):
            raise ArtifactValidationError(
                f"convergence.history[{index}] must be an object"
            )
        _validate_history_row(row, index)
    _validate_reconstruction_source(
        method,
        selection,
        history,
        field="convergence.history",
    )
    if expected_status == "convergence-required" and observed < 21:
        raise ArtifactValidationError(
            "convergence-required profiles need 21 observations"
        )


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
    if type(method) is not ResolvedMethod:
        raise TypeError("method must be a ResolvedMethod")
    _validate_acquisition_contract(acquisition, "acquisition")
    _validate_algorithm_seed(algorithm_seed, "algorithm_seed")
    _require_sha256(measurements_file_sha256, "measurements_file_sha256")
    started = _require_utc(
        child_started_at_utc, "child_started_at_utc"
    )
    finished = _require_utc(
        child_finished_at_utc, "child_finished_at_utc"
    )
    if finished < started:
        raise ArtifactValidationError("child timing cannot be negative")
    _validate_result(result, acquisition, method)
    samples, sampling_policy = _sample_history(result.history)
    status = result.info["convergence_status"]
    convergence = {
        "status": status,
        "sampling_policy": sampling_policy,
        "observed_count": len(result.history),
        "serialized_count": len(samples),
        "history": samples,
    }
    _validate_convergence(
        convergence,
        method=method,
        selection=result.info["selection"],
        expected_status=method.convergence_status,
    )
    arrays: dict[str, np.ndarray] = {
        "reconstruction": np.ascontiguousarray(result.reconstruction),
        "frame_indices": np.arange(acquisition.T, dtype=np.int64),
        "time_grid": np.ascontiguousarray(acquisition.time_grid),
    }
    if result.dgi is not None:
        arrays["dgi"] = np.ascontiguousarray(result.dgi)
    if result.estimated_motion_trajectory is not None:
        arrays["estimated_motion_trajectory"] = np.ascontiguousarray(result.estimated_motion_trajectory)
    reconstruction_metadata = _reconstruction_metadata(
        method_id=method.method_id,
        acquisition=acquisition,
        arrays=arrays,
    )
    reconstruction_payload = npz_bytes(
        arrays=arrays,
        metadata=reconstruction_metadata,
    )
    reconstruction_hash = hashlib.sha256(
        reconstruction_payload
    ).hexdigest()
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
        "native_iteration": _native_iteration(method),
        "warmup": _warmup_counts(method),
        "selected_hyperparameters": result.info["selected_hyperparameters"],
        "selection": result.info["selection"],
        "convergence": convergence,
        "checkpoints": result.info["checkpoint_hashes"],
        "motion_estimate": {"present": result.estimated_motion_trajectory is not None, "native_model": result.info.get("native_motion_model", "none")},
        "reconstruction": {
            "sha256": reconstruction_hash,
            "array_descriptors": reconstruction_metadata[
                "array_descriptors"
            ],
        },
        "child_timing": {"started_at": child_started_at_utc, "finished_at": child_finished_at_utc, "elapsed_seconds": (finished - started).total_seconds()},
    }
    _validate_method_info_schema(info)
    method_info_payload = canonical_json_bytes(info)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    initial_inventory, names = _strict_output_inventory(output_dir)
    if names:
        raise ArtifactValidationError(
            "method-child output directory is not isolated"
        )
    verify_directory_inventory(initial_inventory)
    reconstruction_path = output_dir / "reconstruction.npz"
    written_reconstruction_hash = atomic_write_bytes(
        reconstruction_path, reconstruction_payload
    )
    if written_reconstruction_hash != reconstruction_hash:
        raise ArtifactValidationError(
            "reconstruction hash changed during atomic write"
        )
    method_info_hash = atomic_write_bytes(
        output_dir / "method-info.json", method_info_payload
    )
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
    if indices.dtype != np.dtype(np.int64) or indices.shape != (t,) or not np.array_equal(indices, np.arange(t, dtype=np.int64)):
        raise ArtifactValidationError("frame_indices must equal arange(T)")
    time_grid = arrays["time_grid"]
    if time_grid.shape != (t,) or not np.issubdtype(time_grid.dtype, np.number) or not np.isfinite(time_grid).all() or (t > 1 and not np.all(time_grid[1:] > time_grid[:-1])):
        raise ArtifactValidationError("time_grid must be finite and strictly increasing")
    if "dgi" in arrays and arrays["dgi"].shape != (h, w):
        raise ArtifactValidationError("dgi shape must equal (H,W)")
    if (
        "estimated_motion_trajectory" in arrays
        and arrays["estimated_motion_trajectory"].shape != (t, 3)
    ):
        raise ArtifactValidationError(
            "estimated_motion_trajectory shape must equal (T,3)"
        )
    return ReconstructionV2(metadata["dataset_identity_sha256"], metadata["method_id"], reconstruction, indices, time_grid, arrays.get("dgi"), arrays.get("estimated_motion_trajectory"), descriptors)


def load_method_info_v2(path: Path) -> Mapping[str, object]:
    """Load canonical method-info-v2 with intrinsic consistency checks."""
    info, _digest = _parse_json_file(Path(path))
    _validate_method_info_schema(info)
    timing = info["child_timing"]
    assert isinstance(timing, Mapping)
    started = _require_utc(timing["started_at"], "child_timing.started_at")
    finished = _require_utc(timing["finished_at"], "child_timing.finished_at")
    elapsed = _require_finite_number(
        timing["elapsed_seconds"],
        "child_timing.elapsed_seconds",
        minimum=0.0,
    )
    if finished < started or elapsed != (finished - started).total_seconds():
        raise ArtifactValidationError("method info child timing is inconsistent")
    convergence = info["convergence"]
    assert isinstance(convergence, Mapping)
    observed = convergence["observed_count"]
    serialized = convergence["serialized_count"]
    history = convergence["history"]
    if (
        type(observed) is not int
        or type(serialized) is not int
        or type(history) is not list
        or serialized != len(history)
        or serialized != (observed if observed < 21 else 21)
    ):
        raise ArtifactValidationError("method info convergence counts are inconsistent")
    for index, row in enumerate(history):
        if not isinstance(row, Mapping):
            raise ArtifactValidationError("method info convergence row is invalid")
        _validate_history_row(row, index)
    checkpoints = info["checkpoints"]
    assert type(checkpoints) is list
    logical_ids = [item["logical_id"] for item in checkpoints]
    if len(logical_ids) != len(set(logical_ids)):
        raise ArtifactValidationError("method info checkpoints are duplicated")
    return info


def _load_schema() -> Mapping[str, object]:
    schema_path = Path(__file__).resolve().parents[2] / "schemas" / "method-info-v2.schema.json"
    with schema_path.open("r", encoding="utf-8") as stream:
        return json.load(stream, object_pairs_hook=_json_object_pairs, parse_constant=_reject_json_constant)


def validate_method_child_outputs_v2(
    output_dir: Path,
    *,
    expected_method: ResolvedMethod,
    expected_acquisition: SPIAcquisitionData,
    expected_dataset_identity_sha256: str,
    expected_measurements_file_sha256: str,
    expected_algorithm_seed: AlgorithmSeed,
) -> Mapping[str, str]:
    """Fail closed on malformed, linked, incomplete, or mismatched child output."""
    if type(expected_method) is not ResolvedMethod:
        raise TypeError("expected_method must be a ResolvedMethod")
    _validate_acquisition_contract(
        expected_acquisition, "expected_acquisition"
    )
    _validate_algorithm_seed(
        expected_algorithm_seed, "expected_algorithm_seed"
    )
    _require_sha256(expected_dataset_identity_sha256, "expected_dataset_identity_sha256")
    _require_sha256(expected_measurements_file_sha256, "expected_measurements_file_sha256")
    if (
        expected_acquisition.dataset_identity_sha256
        != expected_dataset_identity_sha256
    ):
        raise ArtifactValidationError(
            "expected acquisition dataset identity mismatch"
        )
    output_dir = Path(output_dir)
    inventory, names = _strict_output_inventory(output_dir)
    if names != _OUTPUT_FILES:
        raise ArtifactValidationError("child output inventory must contain exactly two files")
    reconstruction_path, info_path = output_dir / "reconstruction.npz", output_dir / "method-info.json"
    reconstruction = load_reconstruction_v2(reconstruction_path)
    info, info_hash = _parse_json_file(info_path)
    _validate_method_info_schema(info)
    timing = info["child_timing"]
    assert isinstance(timing, Mapping)
    started = _require_utc(timing["started_at"], "child_timing.started_at")
    finished = _require_utc(timing["finished_at"], "child_timing.finished_at")
    elapsed = _require_finite_number(
        timing["elapsed_seconds"],
        "child_timing.elapsed_seconds",
        minimum=0,
    )
    if finished < started or elapsed != (finished - started).total_seconds():
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
        if canonical_json_bytes(info[key]) != canonical_json_bytes(value):
            raise ArtifactValidationError(f"method info {key} does not match parent request")
    algorithm_seed_info = info["algorithm_seed"]
    assert isinstance(algorithm_seed_info, Mapping)
    _require_exact_int(
        algorithm_seed_info["seed_u32"],
        "algorithm_seed.seed_u32",
        maximum=2**32 - 1,
    )
    parameter_count = _require_exact_int(
        info["parameter_count"], "parameter_count"
    )
    expected_parameter_count = _classical_parameter_count(
        expected_method, expected_acquisition
    )
    if parameter_count != expected_parameter_count:
        raise ArtifactValidationError(
            "parameter_count does not match the locked classical formula"
        )
    if canonical_json_bytes(info["native_iteration"]) != canonical_json_bytes(
        _native_iteration(expected_method)
    ):
        raise ArtifactValidationError(
            "native iteration metadata does not match parent method"
        )
    native_iteration = info["native_iteration"]
    assert isinstance(native_iteration, Mapping)
    _require_exact_int(
        native_iteration["budget"], "native_iteration.budget"
    )
    if canonical_json_bytes(info["checkpoints"]) != canonical_json_bytes(
        _expected_checkpoints(expected_method)
    ):
        raise ArtifactValidationError("checkpoints do not match parent request")
    if canonical_json_bytes(info["warmup"]) != canonical_json_bytes(
        _warmup_counts(expected_method)
    ):
        raise ArtifactValidationError("warmup does not match parent method")
    warmup = info["warmup"]
    assert isinstance(warmup, Mapping)
    _require_exact_int(warmup["splitting"], "warmup.splitting")
    _require_exact_int(warmup["motion"], "warmup.motion")
    _validate_selection(
        expected_method,
        info["selected_hyperparameters"],
        info["selection"],
    )
    _validate_convergence(
        info["convergence"],
        method=expected_method,
        selection=info["selection"],
        expected_status=expected_method.convergence_status,
    )
    convergence = info["convergence"]
    assert isinstance(convergence, Mapping)
    _validate_serialized_history_against_method(
        convergence,
        expected_method,
        info["selection"],
    )
    if reconstruction.dataset_identity_sha256 != expected_dataset_identity_sha256 or reconstruction.method_id != expected_method.method_id:
        raise ArtifactValidationError("reconstruction metadata does not match parent request")
    expected_shape = (
        expected_acquisition.T,
        expected_acquisition.H,
        expected_acquisition.W,
    )
    if reconstruction.reconstruction.shape != expected_shape:
        raise ArtifactValidationError(
            "reconstruction shape disagrees with expected acquisition"
        )
    _validate_auxiliary_arrays(
        expected_method,
        reconstruction.reconstruction,
        reconstruction.dgi,
    )
    expected_frame_indices = np.arange(
        expected_acquisition.T, dtype=np.int64
    )
    if (
        reconstruction.frame_indices.dtype != expected_frame_indices.dtype
        or not np.array_equal(
            reconstruction.frame_indices, expected_frame_indices
        )
    ):
        raise ArtifactValidationError(
            "frame_indices disagree with expected acquisition"
        )
    if (
        reconstruction.time_grid.dtype
        != expected_acquisition.time_grid.dtype
        or not np.array_equal(
            reconstruction.time_grid, expected_acquisition.time_grid
        )
    ):
        raise ArtifactValidationError(
            "time_grid disagrees with expected acquisition"
        )
    if (
        reconstruction.dgi is not None
        and reconstruction.dgi.shape
        != (expected_acquisition.H, expected_acquisition.W)
    ):
        raise ArtifactValidationError(
            "dgi shape disagrees with expected acquisition"
        )
    if (
        reconstruction.estimated_motion_trajectory is not None
        and reconstruction.estimated_motion_trajectory.shape
        != (expected_acquisition.T, 3)
    ):
        raise ArtifactValidationError(
            "motion trajectory disagrees with expected acquisition"
        )
    snapshot = read_safe_file_snapshot(reconstruction_path, max_bytes=1024 * 1024 * 1024, noun="reconstruction")
    try:
        reconstruction_hash = snapshot.sha256
    finally:
        verify_safe_file_snapshot(snapshot)
    reconstruction_info = info["reconstruction"]
    assert isinstance(reconstruction_info, Mapping)
    if (
        reconstruction_info["sha256"] != reconstruction_hash
        or canonical_json_bytes(
            reconstruction_info["array_descriptors"]
        )
        != canonical_json_bytes(reconstruction.array_descriptors)
    ):
        raise ArtifactValidationError("reconstruction hash or descriptors disagree")
    motion_estimate = info["motion_estimate"]
    assert isinstance(motion_estimate, Mapping)
    if motion_estimate["present"] != (reconstruction.estimated_motion_trajectory is not None):
        raise ArtifactValidationError("motion estimate presence disagrees with reconstruction")
    if (
        motion_estimate["present"] is False
        and motion_estimate["native_model"] != "none"
    ) or (
        motion_estimate["present"] is True
        and motion_estimate["native_model"] == "none"
    ):
        raise ArtifactValidationError(
            "motion native_model disagrees with trajectory presence"
        )
    verify_directory_inventory(inventory)
    return {"reconstruction.npz": reconstruction_hash, "method-info.json": info_hash}
