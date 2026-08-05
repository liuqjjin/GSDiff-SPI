"""Immutable, content-addressed experiment manifest validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any

from jsonschema import Draft202012Validator
import numpy as np

from gsdiff.evaluation.metrics import (
    evaluate_video_global_affine,
    validate_metrics_v1_payload,
)
from gsdiff.data.artifacts import (
    discover_dataset_directories,
    verify_canonical_dataset_directory_discovery,
    verify_dataset_directory,
)
from gsdiff.data._artifact_dataset import _validate_blind_acquisition_spec

from .identity import (
    RunIdentity,
    _authoritative_python_executable_evidence,
    _authoritative_runtime_projection,
    build_run_identity,
    canonical_json_bytes,
    sha256_bytes,
    verify_environment_requirements,
)
from .audit import validate_audit_log
from .dataset_binding import (
    build_dataset_input_contract,
    dataset_measurement_record,
    validate_dataset_protocol_binding,
)
from .child_outputs import (
    load_method_info_v2,
    load_reconstruction_v2,
    validate_method_info_contract_v1,
)
from .methods import (
    METHODS_REGISTRY_PROTOCOL_SHA256,
    derive_algorithm_seed,
)
from .source_snapshot import (
    _load_canonical_manifest as _load_source_snapshot_manifest,
)
from ._windows_paths import windows_component_collision_key


_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_REQUIREMENTS_LOCK = _ROOT / "requirements-lock.txt"
_DEFAULT_ENVIRONMENT_LOCK = _ROOT / "docs" / "reproducibility" / "environment-lock.json"
_RUNTIME_DEVICE = re.compile(
    r"(?:cpu|cuda:(?:0|[1-9][0-9]*))\Z",
    re.ASCII,
)
_MANIFEST_SCHEMA = json.loads(
    (_ROOT / "schemas" / "experiment-manifest-v1.schema.json").read_text("utf-8")
)
_AGGREGATE_SCHEMA = json.loads(
    (_ROOT / "schemas" / "experiment-aggregate-v1.schema.json").read_text("utf-8")
)
_MANIFEST_VALIDATOR = Draft202012Validator(_MANIFEST_SCHEMA)
_AGGREGATE_VALIDATOR = Draft202012Validator(_AGGREGATE_SCHEMA)
_SHA256 = __import__("re").compile(r"^[0-9a-f]{64}$", __import__("re").ASCII)
_RUNNER_ARTIFACT_CONTRACT = (
    ("reconstruction", "outputs/reconstruction.npz", "reconstruction-v2"),
    ("method-info", "outputs/method-info.json", "method-info-v2"),
    ("stdout", "outputs/stdout.log", "text-v1"),
    ("stderr", "outputs/stderr.log", "text-v1"),
    ("resolved-config", "resolved-config.json", "resolved-run-config-v1"),
    ("lifecycle", "lifecycle.json", "run-lifecycle-v1"),
    ("audit", "evidence/audit.jsonl", "validated-method-audit-log-v1"),
    (
        "audit-validation",
        "evidence/audit-validation.json",
        "audit-validation-v1",
    ),
    (
        "materialization-logical",
        "evidence/materialization-logical.json",
        "materialized-method-execution-v1",
    ),
    (
        "resource-sampling",
        "evidence/resource-sampling.json",
        "run-resource-sampling-v1",
    ),
    (
        "source-snapshot",
        "evidence/source-snapshot.json",
        "source-snapshot-v1",
    ),
)
_VRAM_SAMPLING_INTERVAL_MS = 250
_StatSnapshot = tuple[int, int, int, int, int, int]


@dataclass(frozen=True)
class _SafeRead:
    raw: bytes
    link_snapshot: _StatSnapshot
    path_snapshot: _StatSnapshot
    handle_snapshot: _StatSnapshot


def build_manifest(
    *,
    status: str,
    identity: RunIdentity,
    config_resolved: Mapping[str, object],
    inputs: Mapping[str, object],
    runtime: Mapping[str, object],
    execution: Mapping[str, object],
    measurement: Mapping[str, object],
    metrics: Mapping[str, object] | None,
    artifacts: Sequence[Mapping[str, object]],
    failure: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build a manifest exclusively from a constructed blind-run identity."""
    if type(identity) is not RunIdentity:
        raise TypeError("identity must be a RunIdentity")
    _require_exact_mapping(
        "inputs", inputs,
        {"measurements_file_sha256", "evaluation_truth_file_sha256", "dataset_manifest_sha256"},
    )
    _require_exact_mapping("runtime", runtime, {"python", "pytorch", "cuda", "gpu", "os"})
    _require_exact_mapping(
        "execution", execution,
        {"command", "started_at_utc", "ended_at_utc", "return_code", "runtime_seconds", "peak_vram_bytes"},
    )
    _require_exact_mapping(
        "measurement", measurement,
        {"train_count", "holdout_count", "pattern_family", "requested_snr_db", "noise_calibration_id", "noise_calibration_sha256", "noise_sigma_absolute", "realized_train_snr_db", "realized_holdout_snr_db"},
    )
    if status == "complete":
        if metrics is None:
            raise ValueError("complete manifests require metrics")
        _require_exact_mapping("metrics", metrics, {"version", "path", "sha256"})
    elif metrics is not None:
        _require_exact_mapping("metrics", metrics, {"version", "path", "sha256"})
    if type(config_resolved) is not dict:
        raise TypeError("config_resolved must be an exact dict")
    _validate_exact_json(config_resolved)
    if type(artifacts) is not list:
        raise TypeError("artifacts must be an exact list")
    for artifact in artifacts:
        _require_exact_mapping(
            "artifact", artifact,
            {"role", "path", "sha256", "size_bytes", "schema_version", "required"},
        )
    if status == "failed":
        if failure is None:
            raise ValueError("failed manifests require failure diagnostics")
        _require_exact_mapping("failure", failure, {"stderr_tail", "diagnostic_paths"})
    elif failure is not None:
        raise ValueError("only failed manifests may carry failure diagnostics")
    payload = identity.payload()
    manifest: dict[str, object] = {
        "schema_version": "experiment-manifest-v1",
        "status": status,
        "execution_class": payload["execution_class"],
        "metric_version": payload["metric_version"],
        "run_id": identity.run_id,
        "identity_sha256": identity.identity_sha256,
        "protocol": {
            "scientific_contract_id": payload["scientific_contract_id"],
            "scientific_contract_sha256": payload["scientific_contract_sha256"],
            "target": payload["target_id"],
            "motion": payload["motion_id"],
            "seed": payload["seed"],
            "method": payload["method_id"],
        },
        "code": {
            "git_commit": payload["code_commit"],
            "dirty_worktree": payload["dirty_worktree"],
            "source_tree_sha256": payload["source_tree_hash"],
        },
        "config": {
            "resolved": _copy_exact_json(config_resolved),
            "sha256": payload["config_sha256"],
        },
        "inputs": {
            "dataset_identity_sha256": payload["dataset_identity_sha256"],
            "measurements_file_sha256": inputs.get("measurements_file_sha256"),
            "evaluation_truth_file_sha256": inputs.get("evaluation_truth_file_sha256"),
            "dataset_manifest_sha256": inputs.get("dataset_manifest_sha256"),
            "assets": _plain_json(payload["assets_sha256"]),
            "checkpoints": _plain_json(payload["checkpoints_sha256"]),
        },
        "runtime": {
            "python": runtime.get("python"),
            "pytorch": runtime.get("pytorch"),
            "cuda": runtime.get("cuda"),
            "gpu": runtime.get("gpu"),
            "os": runtime.get("os"),
            "dependencies_sha256": payload["dependencies_sha256"],
            "environment_lock_sha256": payload["environment_lock_sha256"],
        },
        "execution": _copy_exact_json(execution),
        "measurement": _copy_exact_json(measurement),
        "metrics": _copy_exact_json(metrics),
        "artifacts": [_copy_exact_json(item) for item in artifacts],
    }
    if failure is not None:
        manifest["failure"] = _copy_exact_json(failure)
    validate_manifest(manifest, identity=identity)
    return manifest


def validate_manifest(value: Mapping[str, object], *, identity: RunIdentity | None = None) -> None:
    """Reject malformed, mutable, or identity-inconsistent manifest data."""
    _validate_exact_json(value)
    _raise_schema_errors(_MANIFEST_VALIDATOR, value, "manifest")
    manifest = value
    _validate_manifest_semantics(manifest, identity=identity)


def validate_aggregate_index(
    value: Mapping[str, object],
    *,
    manifest_paths: Mapping[str, Path] | None = None,
    artifact_root: Path | None = None,
    requirements_lock: Path | None = None,
    environment_lock: Path | None = None,
    live_fingerprint: Mapping[str, object] | None = None,
) -> None:
    """Validate index structure; publication eligibility requires manifest_paths."""
    _validate_exact_json(value)
    _raise_schema_errors(_AGGREGATE_VALIDATOR, value, "aggregate index")
    expected = value["expected_identity_sha256s"]
    records = value["run_manifests"]
    assert type(expected) is list and type(records) is list
    if expected != sorted(expected) or len(expected) != len(set(expected)):
        raise ValueError("expected identities must be sorted and unique")
    record_ids = [record["identity_sha256"] for record in records]
    if record_ids != sorted(record_ids) or len(record_ids) != len(set(record_ids)):
        raise ValueError("run manifest identities must be sorted and unique")
    if record_ids != expected:
        raise ValueError("campaign index coverage must exactly equal expected identities")
    if manifest_paths is not None:
        if not isinstance(artifact_root, Path):
            raise TypeError(
                "artifact_root must be a Path for physical aggregate validation"
            )
        environment_hashes = _verify_environment_evidence(
            requirements_lock,
            environment_lock,
            live_fingerprint,
        )
        for run_identity in record_ids:
            path = manifest_paths.get(run_identity)
            if not isinstance(path, Path):
                raise ValueError("campaign index is missing a manifest path")
            verified = _load_complete_manifest_raw(
                path,
                artifact_root=artifact_root,
                expected_identity_sha256=run_identity,
                requirements_lock=requirements_lock,
                environment_lock=environment_lock,
                live_fingerprint=live_fingerprint,
                verified_environment_hashes=environment_hashes,
            )
            if verified is None:
                raise ValueError("campaign index requires a complete manifest")
            raw, manifest = verified
            record = next(item for item in records if item["identity_sha256"] == run_identity)
            if record["manifest_sha256"] != sha256_bytes(raw):
                raise ValueError("campaign index manifest hash does not match physical manifest bytes")
            protocol = manifest["protocol"]
            assert type(protocol) is dict
            if (
                protocol["scientific_contract_id"] != value["scientific_contract_id"]
                or protocol["scientific_contract_sha256"] != value["scientific_contract_sha256"]
                or manifest["metric_version"] != value["metric_version"]
            ):
                raise ValueError("campaign index protocol or metric version disagrees with manifest")


def build_aggregate_index(
    *,
    campaign_id: str,
    campaign_sha256: str,
    protocol_sha256: str,
    scientific_contract_id: str,
    scientific_contract_sha256: str,
    metric_version: str,
    expected_identity_sha256s: Sequence[str],
    manifest_paths: Mapping[str, Path],
    artifact_root: Path,
    requirements_lock: Path | None = None,
    environment_lock: Path | None = None,
    live_fingerprint: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build a campaign-local reference index for immutable clean manifests."""
    if type(expected_identity_sha256s) is not list:
        raise TypeError("expected identities must be an exact list")
    expected = list(expected_identity_sha256s)
    records: list[dict[str, str]] = []
    environment_hashes = _verify_environment_evidence(
        requirements_lock,
        environment_lock,
        live_fingerprint,
    )
    for run_identity in expected:
        path = manifest_paths.get(run_identity)
        if not isinstance(path, Path):
            raise ValueError("expected identity has no manifest path")
        verified = _load_complete_manifest_raw(
            path,
            artifact_root=artifact_root,
            expected_identity_sha256=run_identity,
            requirements_lock=requirements_lock, environment_lock=environment_lock,
            live_fingerprint=live_fingerprint,
            verified_environment_hashes=environment_hashes,
        )
        if verified is None:
            raise ValueError("campaign index requires a complete manifest")
        raw, manifest = verified
        protocol = manifest["protocol"]
        assert type(protocol) is dict
        if (
            protocol["scientific_contract_id"] != scientific_contract_id
            or protocol["scientific_contract_sha256"]
            != scientific_contract_sha256
            or manifest["metric_version"] != metric_version
        ):
            raise ValueError(
                "campaign index protocol or metric version disagrees with manifest"
            )
        records.append(
            {
                "identity_sha256": run_identity,
                "manifest_sha256": sha256_bytes(raw),
            }
        )
    index: dict[str, object] = {
        "schema_version": "experiment-aggregate-v1",
        "document_kind": "campaign-index",
        "campaign_id": campaign_id,
        "campaign_sha256": campaign_sha256,
        "protocol_sha256": protocol_sha256,
        "scientific_contract_id": scientific_contract_id,
        "scientific_contract_sha256": scientific_contract_sha256,
        "metric_version": metric_version,
        "expected_identity_sha256s": sorted(expected),
        "run_manifests": sorted(records, key=lambda item: item["identity_sha256"]),
    }
    validate_aggregate_index(index)
    return index


def load_complete_manifest(
    path: Path,
    *,
    artifact_root: Path,
    expected_identity_sha256: str,
    requirements_lock: Path | None = None,
    environment_lock: Path | None = None,
    live_fingerprint: Mapping[str, object] | None = None,
) -> dict[str, object] | None:
    """Load a complete manifest; only this public API returns reusable data."""
    verified = _load_complete_manifest_raw(
        path,
        artifact_root=artifact_root,
        expected_identity_sha256=expected_identity_sha256,
        requirements_lock=requirements_lock, environment_lock=environment_lock,
        live_fingerprint=live_fingerprint,
    )
    return None if verified is None else verified[1]


def _load_complete_manifest_raw(
    path: Path,
    *,
    artifact_root: Path,
    expected_identity_sha256: str,
    requirements_lock: Path | None = None,
    environment_lock: Path | None = None,
    live_fingerprint: Mapping[str, object] | None = None,
    verified_environment_hashes: Mapping[str, str] | None = None,
) -> tuple[bytes, dict[str, object]] | None:
    """Load a physical clean complete run directory, otherwise fail closed."""
    if not isinstance(artifact_root, Path):
        raise TypeError("artifact_root must be a Path")
    canonical_root = artifact_root.absolute()
    manifest_path = Path(path)
    if (
        type(expected_identity_sha256) is not str
        or _SHA256.fullmatch(expected_identity_sha256) is None
    ):
        raise ValueError("expected manifest identity must be a SHA-256")
    expected_manifest_path = (
        canonical_root
        / "runs"
        / expected_identity_sha256
        / "manifest.json"
    )
    if manifest_path.absolute() != expected_manifest_path:
        raise ValueError(
            "complete manifest must be under the explicit canonical artifact root"
        )
    manifest_read, value = _load_unique_json(manifest_path)
    raw = manifest_read.raw
    if type(value) is not dict:
        raise ValueError("manifest JSON must be an object")
    validate_manifest(value)
    if value["status"] != "complete":
        return None
    if value["code"]["dirty_worktree"]:
        raise ValueError("dirty complete manifests are diagnostic and not reusable")
    if value["identity_sha256"] != expected_identity_sha256:
        raise ValueError("manifest identity does not match expected identity")
    if manifest_path.parent.name != value["identity_sha256"]:
        raise ValueError("complete manifest directory must be the full identity SHA-256")
    if raw != canonical_json_bytes(value):
        raise ValueError("manifest file is not canonical JSON bytes")
    initial_inventory = _directory_inventory(manifest_path.parent)
    if (
        manifest_path.name,
        manifest_read.link_snapshot,
    ) not in initial_inventory:
        raise ValueError("manifest changed before run-directory verification")
    hashes = (
        verified_environment_hashes
        if verified_environment_hashes is not None
        else _verify_environment_evidence(
            requirements_lock,
            environment_lock,
            live_fingerprint,
        )
    )
    runtime = value["runtime"]
    assert type(runtime) is dict
    if (
        runtime["dependencies_sha256"] != hashes["dependencies_sha256"]
        or runtime["environment_lock_sha256"] != hashes["environment_lock_sha256"]
    ):
        raise ValueError("manifest runtime lock hashes do not match the strict environment")
    _verify_complete_outputs(
        manifest_path,
        value,
        initial_inventory=initial_inventory,
    )
    _validate_complete_runner_contract(
        manifest_path.parent,
        value,
        artifact_root=canonical_root,
    )
    _verify_safe_read_path_unchanged(manifest_path, manifest_read, "manifest")
    # Task 5's atomic promotion is the writer-side boundary after this final
    # pathname check; this loader cannot lock out a mutation after it returns.
    return raw, value


def _verify_environment_evidence(
    requirements_lock: Path | None,
    environment_lock: Path | None,
    live_fingerprint: Mapping[str, object] | None,
) -> dict[str, str]:
    if live_fingerprint is not None:
        raise ValueError(
            "reusable manifest validation cannot trust a caller fingerprint"
        )
    if requirements_lock is None and environment_lock is None:
        requirements_lock = _DEFAULT_REQUIREMENTS_LOCK
        environment_lock = _DEFAULT_ENVIRONMENT_LOCK
    elif requirements_lock is None or environment_lock is None:
        raise ValueError(
            "requirements_lock and environment_lock must be provided together"
        )
    return verify_environment_requirements(
        requirements_lock,
        environment_lock,
    )


def _validate_complete_runner_contract(
    run_dir: Path,
    manifest: Mapping[str, object],
    *,
    artifact_root: Path,
) -> None:
    metrics_descriptor = manifest["metrics"]
    artifacts = manifest["artifacts"]
    assert type(metrics_descriptor) is dict and type(artifacts) is list
    if metrics_descriptor["path"] != "outputs/metrics.json":
        raise ValueError("runner metrics path violates the exact contract")
    if len(artifacts) != len(_RUNNER_ARTIFACT_CONTRACT):
        raise ValueError("runner artifact inventory violates the exact contract")
    for descriptor, (role, relative, schema) in zip(
        artifacts,
        _RUNNER_ARTIFACT_CONTRACT,
        strict=True,
    ):
        if (
            descriptor["role"],
            descriptor["path"],
            descriptor["schema_version"],
            descriptor["required"],
        ) != (role, relative, schema, True):
            raise ValueError("runner artifact descriptor violates the exact contract")

    config = _read_canonical_output_json(
        run_dir / "resolved-config.json",
        noun="resolved config evidence",
    )
    manifest_config = manifest["config"]
    assert type(manifest_config) is dict
    if canonical_json_bytes(config) != canonical_json_bytes(
        manifest_config["resolved"]
    ):
        raise ValueError("resolved config evidence disagrees with the manifest")
    if type(config) is not dict:
        raise ValueError("resolved config evidence must be an object")
    runner_config = config.get("runner_execution")
    if type(runner_config) is not dict or set(runner_config) != {
        "schema",
        "requested_runtime_device",
        "source_snapshot_sha256",
        "source_projection_sha256",
        "compute_cap",
        "materialization_logical_sha256",
        "method_info_contract",
        "dataset_input_contract",
        "runtime_contract",
        "python_executable_sha256",
    }:
        raise ValueError("resolved config lacks identity-bound runner evidence")
    if (
        runner_config["schema"] != "runner-execution-identity-v1"
        or type(runner_config["requested_runtime_device"]) is not str
        or _RUNTIME_DEVICE.fullmatch(
            runner_config["requested_runtime_device"]
        ) is None
        or type(runner_config["source_snapshot_sha256"]) is not str
        or _SHA256.fullmatch(runner_config["source_snapshot_sha256"]) is None
        or type(runner_config["source_projection_sha256"]) is not str
        or _SHA256.fullmatch(runner_config["source_projection_sha256"]) is None
        or type(runner_config["materialization_logical_sha256"]) is not str
        or _SHA256.fullmatch(
            runner_config["materialization_logical_sha256"]
        ) is None
        or type(runner_config["python_executable_sha256"]) is not str
        or _SHA256.fullmatch(runner_config["python_executable_sha256"]) is None
    ):
        raise ValueError("identity-bound runner config is invalid")
    if (
        type(config.get("method_config_sha256")) is not str
        or _SHA256.fullmatch(config["method_config_sha256"]) is None
    ):
        raise ValueError("identity-bound method config digest is invalid")
    runtime_contract = runner_config["runtime_contract"]
    runtime = manifest["runtime"]
    assert type(runtime) is dict
    authoritative_runtime, runtime_hashes = _authoritative_runtime_projection(
        _DEFAULT_REQUIREMENTS_LOCK,
        _DEFAULT_ENVIRONMENT_LOCK,
        runner_config["requested_runtime_device"],
    )
    _python_path, python_sha256, _python_signature = (
        _authoritative_python_executable_evidence()
    )
    if (
        type(runtime_contract) is not dict
        or set(runtime_contract) != {"python", "pytorch", "cuda", "gpu", "os"}
        or runtime_contract != authoritative_runtime
        or authoritative_runtime
        != {name: runtime[name] for name in ("python", "pytorch", "cuda", "gpu", "os")}
        or runtime["dependencies_sha256"]
        != runtime_hashes["dependencies_sha256"]
        or runtime["environment_lock_sha256"]
        != runtime_hashes["environment_lock_sha256"]
        or runner_config["python_executable_sha256"] != python_sha256
    ):
        raise ValueError(
            "manifest runtime disagrees with live authoritative identity contract"
        )

    lifecycle = _read_canonical_output_json(
        run_dir / "lifecycle.json",
        noun="lifecycle evidence",
    )
    if type(lifecycle) is not dict or set(lifecycle) != {
        "schema",
        "state",
        "identity_sha256",
        "owner_token",
        "fence",
    }:
        raise ValueError("lifecycle evidence shape is invalid")
    owner_token = lifecycle["owner_token"]
    if (
        lifecycle["schema"] != "run-lifecycle-v1"
        or lifecycle["state"] != "complete"
        or lifecycle["identity_sha256"] != manifest["identity_sha256"]
        or type(owner_token) is not str
        or __import__("re").fullmatch(r"[0-9a-f]{32}", owner_token, __import__("re").ASCII)
        is None
        or type(lifecycle["fence"]) is not int
        or lifecycle["fence"] <= 0
    ):
        raise ValueError("lifecycle evidence violates the complete-run contract")

    audit_validation = _read_canonical_output_json(
        run_dir / "evidence/audit-validation.json",
        noun="audit validation evidence",
    )
    if type(audit_validation) is not dict or set(audit_validation) != {
        "schema",
        "policy_sha256",
        "audit_log_sha256",
        "event_count",
        "terminal_status",
    }:
        raise ValueError("audit validation evidence shape is invalid")
    if (
        audit_validation["schema"] != "validated-method-audit-log-v1"
        or audit_validation["terminal_status"] != "success"
        or type(audit_validation["event_count"]) is not int
        or audit_validation["event_count"] < 2
    ):
        raise ValueError("audit validation evidence is not terminal-success")
    independently_validated_audit = validate_audit_log(
        run_dir / "evidence/audit.jsonl",
        expected_policy_sha256=audit_validation["policy_sha256"],
    )
    if canonical_json_bytes(dict(independently_validated_audit)) != canonical_json_bytes(
        audit_validation
    ):
        raise ValueError("audit validation evidence disagrees with the audit log")

    source_raw = _read_regular_file(
        run_dir / "evidence/source-snapshot.json",
        "source snapshot evidence",
    )
    source_manifest = _load_source_snapshot_manifest(source_raw)
    code = manifest["code"]
    assert type(code) is dict
    if source_manifest["commit"] != code["git_commit"]:
        raise ValueError("source snapshot evidence disagrees with the code commit")
    if source_manifest["snapshot_sha256"] != runner_config["source_snapshot_sha256"]:
        raise ValueError("source snapshot evidence disagrees with resolved config")

    logical = _read_canonical_output_json(
        run_dir / "evidence/materialization-logical.json",
        noun="materialization evidence",
    )
    if type(logical) is not dict or set(logical) != {
        "schema",
        "method_id",
        "method_config_id",
        "execution_profile",
        "method_config_sha256",
        "methods_registry_protocol_sha256",
        "semantic_sha256",
        "materialized_config_sha256",
        "dataset_identity_sha256",
        "measurements_file_sha256",
        "expected_acquisition_spec",
        "algorithm_seed",
        "checkpoint_sha256",
        "source_inventory",
        "source_snapshot_sha256",
        "requested_runtime_device",
        "child_runtime_device",
        "entrypoint",
        "command_template",
    } or logical.get("schema") != "materialized-method-execution-v1":
        raise ValueError("materialization evidence schema is invalid")
    if hashlib.sha256(canonical_json_bytes(logical)).hexdigest() != (
        runner_config["materialization_logical_sha256"]
    ):
        raise ValueError("materialization evidence is not identity-bound")
    protocol = manifest["protocol"]
    inputs = manifest["inputs"]
    execution = manifest["execution"]
    assert all(type(item) is dict for item in (protocol, inputs, execution))
    if (
        logical.get("method_id") != protocol["method"]
        or logical.get("method_config_sha256")
        != config.get("method_config_sha256")
        or logical.get("dataset_identity_sha256")
        != inputs["dataset_identity_sha256"]
        or logical.get("measurements_file_sha256")
        != inputs["measurements_file_sha256"]
        or logical.get("checkpoint_sha256") != inputs["checkpoints"]
        or logical.get("command_template") != execution["command"]
        or logical.get("method_config_sha256")
        != config.get("method_config_sha256")
    ):
        raise ValueError("materialization evidence disagrees with the manifest")
    source_inventory = source_manifest["inventory"]
    assert type(source_inventory) is list
    full_sources = {
        item["path"]: item["sha256"]
        for item in source_inventory
        if type(item) is dict
    }
    selected_sources = logical.get("source_inventory")
    if type(selected_sources) is not list or not selected_sources:
        raise ValueError("materialization source inventory is invalid")
    for item in selected_sources:
        if (
            type(item) is not dict
            or set(item) != {"path", "sha256"}
            or full_sources.get(item["path"]) != item["sha256"]
        ):
            raise ValueError("materialization source inventory is not in the snapshot")
    selected_digest = sha256_bytes(canonical_json_bytes(selected_sources))
    if logical.get("source_snapshot_sha256") != selected_digest:
        raise ValueError("materialization source projection hash is invalid")
    if (
        logical.get("requested_runtime_device")
        != runner_config["requested_runtime_device"]
        or selected_digest != runner_config["source_projection_sha256"]
        or logical.get("child_runtime_device")
        != ("cpu" if logical.get("requested_runtime_device") == "cpu" else "cuda:0")
    ):
        raise ValueError("materialization evidence disagrees with resolved config")

    resource = _read_canonical_output_json(
        run_dir / "evidence/resource-sampling.json",
        noun="resource sampling evidence",
    )
    if type(resource) is not dict or set(resource) != {
        "schema",
        "status",
        "backend",
        "sampling_interval_ms",
        "sample_count",
        "peak_vram_bytes",
        "runtime_seconds",
        "compute_cap",
        "requested_runtime_device",
    }:
        raise ValueError("resource sampling evidence shape is invalid")
    _validate_exact_json(resource)
    requested_device = logical.get("requested_runtime_device")
    is_cuda = (
        type(requested_device) is str
        and _RUNTIME_DEVICE.fullmatch(requested_device) is not None
        and requested_device.startswith("cuda:")
    )
    expected_backend = (
        "windows-gpu-process-memory-dedicated-usage-v1"
        if is_cuda
        else "cpu-no-vram-sampling-v1"
        if requested_device == "cpu"
        else None
    )
    expected_interval = _VRAM_SAMPLING_INTERVAL_MS if is_cuda else 0
    if (
        resource["schema"] != "run-resource-sampling-v1"
        or resource["status"] != "complete"
        or resource["backend"] != expected_backend
        or resource["sampling_interval_ms"] != expected_interval
        or resource["requested_runtime_device"] != requested_device
        or resource["peak_vram_bytes"] != execution["peak_vram_bytes"]
        or resource["runtime_seconds"] != execution["runtime_seconds"]
        or type(resource["sample_count"]) is not int
        or (
            resource["sample_count"] < 1
            if is_cuda
            else resource["sample_count"] != 0
        )
        or type(resource["peak_vram_bytes"]) is not int
        or resource["peak_vram_bytes"] < 0
        or (requested_device == "cpu" and resource["peak_vram_bytes"] != 0)
        or type(resource["runtime_seconds"]) not in (int, float)
        or not math.isfinite(resource["runtime_seconds"])
        or resource["runtime_seconds"] < 0
    ):
        raise ValueError("resource sampling evidence disagrees with execution")
    compute_cap = resource["compute_cap"]
    if (
        type(compute_cap) is not dict
        or set(compute_cap) != {
            "wall_time_seconds",
            "peak_vram_bytes",
            "on_exceed",
        }
        or type(compute_cap["wall_time_seconds"]) is not int
        or compute_cap["wall_time_seconds"] <= 0
        or type(compute_cap["peak_vram_bytes"]) is not int
        or compute_cap["peak_vram_bytes"] <= 0
        or compute_cap["on_exceed"] != "ineligible-retain-artifacts"
        or compute_cap != runner_config["compute_cap"]
        or resource["runtime_seconds"] > compute_cap["wall_time_seconds"]
        or resource["peak_vram_bytes"] > compute_cap["peak_vram_bytes"]
    ):
        raise ValueError("resource sampling compute cap is invalid")

    metrics = _read_canonical_output_json(
        run_dir / "outputs/metrics.json",
        noun="metrics evidence",
    )
    _validate_exact_json(metrics)
    if (
        type(metrics) is not dict
        or metrics.get("definition_version") != manifest["metric_version"]
    ):
        raise ValueError("metrics evidence version disagrees with the manifest")
    reconstruction = load_reconstruction_v2(
        run_dir / "outputs/reconstruction.npz"
    )
    method_info = load_method_info_v2(run_dir / "outputs/method-info.json")
    method_info_contract = runner_config["method_info_contract"]
    if type(method_info_contract) is not dict:
        raise ValueError("identity-bound method info contract is invalid")
    validate_method_info_contract_v1(method_info, method_info_contract)
    auxiliary_contract = method_info_contract["auxiliary_arrays"]
    assert type(auxiliary_contract) is dict
    expected_dgi = auxiliary_contract["dgi"] == "required"
    if (reconstruction.dgi is not None) is not expected_dgi:
        raise ValueError(
            "reconstruction auxiliary arrays disagree with identity contract"
        )
    if protocol["method"] == "dgi":
        assert reconstruction.dgi is not None
        if not np.array_equal(
            reconstruction.dgi,
            reconstruction.reconstruction[0],
        ) or not np.all(
            reconstruction.reconstruction
            == reconstruction.reconstruction[0]
        ):
            raise ValueError("DGI reconstruction auxiliary evidence is invalid")
    reconstruction_descriptor = next(
        item for item in artifacts if item["role"] == "reconstruction"
    )
    if (
        reconstruction.method_id != protocol["method"]
        or reconstruction.dataset_identity_sha256
        != inputs["dataset_identity_sha256"]
        or method_info["method_id"] != protocol["method"]
        or method_info["dataset_identity_sha256"]
        != inputs["dataset_identity_sha256"]
        or method_info["measurements_file_sha256"]
        != inputs["measurements_file_sha256"]
        or method_info["method_config_sha256"]
        != config["method_config_sha256"]
        or method_info["reconstruction"]["sha256"]
        != reconstruction_descriptor["sha256"]
        or method_info["reconstruction"]["array_descriptors"]
        != reconstruction.array_descriptors
    ):
        raise ValueError("core method outputs disagree with run provenance")
    checkpoint_records = method_info["checkpoints"]
    checkpoint_mapping = {
        item["logical_id"]: item["sha256"] for item in checkpoint_records
    }
    if checkpoint_mapping != inputs["checkpoints"]:
        raise ValueError("method output checkpoints disagree with run inputs")
    expected_seed = derive_algorithm_seed(
        cell_seed=protocol["seed"],
        dataset_identity_sha256=inputs["dataset_identity_sha256"],
        method_id=protocol["method"],
        method_config_sha256=config["method_config_sha256"],
    )
    expected_seed_record = {
        "derivation_sha256": expected_seed.derivation_sha256,
        "seed_u32": expected_seed.seed_u32,
    }
    info_seed = dict(method_info["algorithm_seed"])
    info_seed.pop("domain", None)
    expected_acquisition = logical["expected_acquisition_spec"]
    _validate_blind_acquisition_spec(expected_acquisition)
    dimensions = expected_acquisition["dimensions"]
    acquisition = expected_acquisition["acquisition"]
    if (
        logical["methods_registry_protocol_sha256"]
        != METHODS_REGISTRY_PROTOCOL_SHA256
        or type(logical["semantic_sha256"]) is not str
        or _SHA256.fullmatch(logical["semantic_sha256"]) is None
        or type(logical["materialized_config_sha256"]) is not str
        or _SHA256.fullmatch(logical["materialized_config_sha256"]) is None
        or logical["method_config_id"] != method_info["method_config_id"]
        or logical["execution_profile"] != method_info["execution_profile"]
        or logical["algorithm_seed"] != expected_seed_record
        or info_seed != expected_seed_record
        or logical["checkpoint_sha256"] != inputs["checkpoints"]
        or logical["entrypoint"] != logical["command_template"][1]
        or dimensions["T"] != reconstruction.reconstruction.shape[0]
        or dimensions["H"] != reconstruction.reconstruction.shape[1]
        or dimensions["W"] != reconstruction.reconstruction.shape[2]
        or dimensions["K"] != manifest["measurement"]["train_count"]
        or dimensions["holdout_K"]
        != manifest["measurement"]["holdout_count"]
        or acquisition["pattern_family"]
        != manifest["measurement"]["pattern_family"]
        or acquisition["noise_sigma_absolute"]
        != manifest["measurement"]["noise_sigma_absolute"]
        or method_info["semantic_config"].get("compute_cap")
        != runner_config["compute_cap"]
    ):
        raise ValueError("materialization identity disagrees with core evidence")
    validate_metrics_v1_payload(metrics, reconstruction.reconstruction)
    verified_dataset = _verify_complete_dataset_root(
        artifact_root,
        run_dir,
        manifest,
        runner_config=runner_config,
    )
    expected_metrics = evaluate_video_global_affine(
        verified_dataset.truth.gt_frames,
        reconstruction.reconstruction,
    )
    if canonical_json_bytes(metrics) != canonical_json_bytes(expected_metrics):
        raise ValueError(
            "metrics evidence disagrees with independent dataset-truth evaluation"
        )


def _verify_complete_dataset_root(
    artifact_root: Path,
    run_dir: Path,
    manifest: Mapping[str, object],
    *,
    runner_config: Mapping[str, object],
):
    if not isinstance(artifact_root, Path):
        raise TypeError("artifact_root must be a Path")
    root = artifact_root.absolute()
    identity = manifest["identity_sha256"]
    if run_dir.absolute() != root / "runs" / identity:
        raise ValueError("complete run is outside the explicit artifact root")
    inputs = manifest["inputs"]
    assert type(inputs) is dict
    dataset_identity = inputs["dataset_identity_sha256"]
    dataset_dir = root / "datasets" / dataset_identity
    discovery = discover_dataset_directories(root)
    if dataset_dir not in discovery.canonical_directories:
        raise ValueError(
            "complete run dataset is not a canonical artifact-root dataset"
        )
    verified = verify_dataset_directory(
        dataset_dir,
        expected_dataset_identity_sha256=dataset_identity,
        expected_dataset_manifest_sha256=inputs["dataset_manifest_sha256"],
    )
    dataset_input_contract = runner_config["dataset_input_contract"]
    expected_dataset_input_contract = build_dataset_input_contract(verified)
    if (
        type(dataset_input_contract) is not dict
        or dataset_input_contract != expected_dataset_input_contract
        or dataset_input_contract != {
            "dataset_manifest_sha256": inputs[
                "dataset_manifest_sha256"
            ],
            "measurements_file_sha256": inputs[
                "measurements_file_sha256"
            ],
            "evaluation_truth_file_sha256": inputs[
                "evaluation_truth_file_sha256"
            ],
            "measurement": manifest["measurement"],
        }
    ):
        raise ValueError(
            "identity-bound dataset input contract disagrees with dataset"
        )
    protocol = manifest["protocol"]
    assert type(protocol) is dict
    validate_dataset_protocol_binding(
        verified.manifest,
        scientific_contract_id=protocol["scientific_contract_id"],
        scientific_contract_sha256=protocol[
            "scientific_contract_sha256"
        ],
        target_id=protocol["target"],
        motion_id=protocol["motion"],
        seed=protocol["seed"],
        assets_sha256=inputs["assets"],
    )
    if manifest["measurement"] != dataset_measurement_record(
        verified.manifest
    ):
        raise ValueError(
            "run measurement evidence disagrees with canonical dataset"
        )
    payload_evidence = verified.payload_evidence
    if (
        payload_evidence["measurements.npz"].sha256
        != inputs["measurements_file_sha256"]
        or payload_evidence["evaluation-truth.npz"].sha256
        != inputs["evaluation_truth_file_sha256"]
    ):
        raise ValueError(
            "complete run dataset payload hashes disagree with manifest inputs"
        )
    verify_canonical_dataset_directory_discovery(discovery)
    return verified


def _read_canonical_output_json(path: Path, *, noun: str) -> object:
    raw = _read_regular_file(path, noun)
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {token}")
            ),
        )
    except (UnicodeError, ValueError) as error:
        raise ValueError(f"{noun} is not canonical JSON") from error
    if raw != canonical_json_bytes(value):
        raise ValueError(f"{noun} is not canonical JSON")
    return value


def _validate_manifest_semantics(manifest: Mapping[str, object], *, identity: RunIdentity | None) -> None:
    config = manifest["config"]
    protocol = manifest["protocol"]
    code = manifest["code"]
    inputs = manifest["inputs"]
    runtime = manifest["runtime"]
    execution = manifest["execution"]
    measurement = manifest["measurement"]
    metrics = manifest["metrics"]
    artifacts = manifest["artifacts"]
    assert all(type(item) is dict for item in (config, protocol, code, inputs, runtime, execution, measurement))
    assert type(artifacts) is list
    if sha256_bytes(canonical_json_bytes(config["resolved"])) != config["sha256"]:
        raise ValueError("config.sha256 does not hash config.resolved")
    if measurement["noise_calibration_id"] != "detector-absolute-v1":
        raise ValueError("noise calibration must be detector-absolute-v1")
    if manifest["status"] == "complete":
        if execution["return_code"] != 0 or type(execution["ended_at_utc"]) is not str:
            raise ValueError("complete manifest requires a completed successful execution")
    elif manifest["status"] == "running":
        if execution["return_code"] is not None or execution["ended_at_utc"] is not None:
            raise ValueError("running manifest must not claim an execution result")
        if "failure" in manifest:
            raise ValueError("running manifest must not carry failure diagnostics")
    else:
        if (
            type(execution["return_code"]) is not int
            or execution["return_code"] == 0
            or type(execution["ended_at_utc"]) is not str
        ):
            raise ValueError("failed manifest requires a nonzero return code")
        failure = manifest.get("failure")
        if type(failure) is not dict:
            raise ValueError("failed manifest requires stderr_tail and diagnostic_paths")
        for path in failure["diagnostic_paths"]:
            _validate_relative_path(path)
    if manifest["status"] == "complete" and "failure" in manifest:
        raise ValueError("complete manifest must not carry failure diagnostics")
    if code["dirty_worktree"] and code["source_tree_sha256"] is None:
        raise ValueError("dirty manifest requires a deterministic source-tree hash")
    if not code["dirty_worktree"] and code["source_tree_sha256"] is not None:
        raise ValueError("clean manifest must not carry a source-tree hash")
    if manifest["status"] == "complete" and type(metrics) is not dict:
        raise ValueError("complete manifest requires metrics")
    metric_paths = [] if metrics is None else [metrics["path"]]
    for path in [*metric_paths, *(artifact["path"] for artifact in artifacts)]:
        _validate_relative_path(path)
    paths = [*metric_paths, *(artifact["path"] for artifact in artifacts)]
    folded = [path.casefold() for path in paths]
    if len(folded) != len(set(folded)):
        raise ValueError("metric and artifact paths must not collide case-insensitively")
    roles = [artifact["role"] for artifact in artifacts]
    if len(roles) != len(set(roles)):
        raise ValueError("artifact roles must be unique")
    for artifact in artifacts:
        if artifact["size_bytes"] < 0:
            raise ValueError("artifact size must be nonnegative")
    expected = identity or _identity_from_manifest(manifest)
    if manifest["identity_sha256"] != expected.identity_sha256 or manifest["run_id"] != expected.run_id:
        raise ValueError("manifest identity fields do not match identity-bearing content")
    if identity is not None and _identity_from_manifest(manifest).identity_sha256 != identity.identity_sha256:
        raise ValueError("manifest fields do not match the supplied identity")
    if manifest["metric_version"] != expected.payload()["metric_version"]:
        raise ValueError("metrics version does not match identity")
    if metrics is not None and metrics["version"] != manifest["metric_version"]:
        raise ValueError("metrics version does not match manifest metric version")


def _identity_from_manifest(manifest: Mapping[str, object]) -> RunIdentity:
    protocol = manifest["protocol"]
    code = manifest["code"]
    config = manifest["config"]
    inputs = manifest["inputs"]
    runtime = manifest["runtime"]
    metrics = manifest["metrics"]
    assert all(type(item) is dict for item in (protocol, code, config, inputs, runtime))
    return build_run_identity(
        execution_class=manifest["execution_class"],
        scientific_contract_id=protocol["scientific_contract_id"],
        scientific_contract_sha256=protocol["scientific_contract_sha256"],
        method_id=protocol["method"],
        target_id=protocol["target"],
        motion_id=protocol["motion"],
        seed=protocol["seed"],
        config_sha256=config["sha256"],
        dataset_identity_sha256=inputs["dataset_identity_sha256"],
        assets_sha256=inputs["assets"],
        checkpoints_sha256=inputs["checkpoints"],
        code_commit=code["git_commit"],
        dirty_worktree=code["dirty_worktree"],
        source_tree_hash=code["source_tree_sha256"],
        dependencies_sha256=runtime["dependencies_sha256"],
        environment_lock_sha256=runtime["environment_lock_sha256"],
        metric_version=manifest["metric_version"],
    )


def _validate_exact_json(value: object, path: str = "$") -> None:
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number")
        return
    if type(value) is list:
        for index, child in enumerate(value):
            _validate_exact_json(child, f"{path}[{index}]")
        return
    if type(value) is dict:
        for key, child in value.items():
            if type(key) is not str:
                raise TypeError(f"{path} has a non-string object key")
            _validate_exact_json(child, f"{path}.{key}")
        return
    raise TypeError(f"{path} uses unsupported JSON type {type(value).__name__}")


def _plain_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(child) for key, child in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain_json(child) for child in value]
    return value


def _copy_exact_json(value: object) -> object:
    _validate_exact_json(value)
    if type(value) is dict:
        return {key: _copy_exact_json(child) for key, child in value.items()}
    if type(value) is list:
        return [_copy_exact_json(child) for child in value]
    return value


def _require_exact_mapping(name: str, value: object, keys: set[str]) -> None:
    if type(value) is not dict:
        raise TypeError(f"{name} must be an exact dict")
    actual = set(value)
    if actual != keys:
        raise ValueError(
            f"{name} keys do not match: missing={sorted(keys - actual)}, unknown={sorted(actual - keys)}"
        )
    _validate_exact_json(value)


def _raise_schema_errors(validator: Draft202012Validator, value: object, noun: str) -> None:
    errors = sorted(validator.iter_errors(value), key=str)
    if errors:
        raise ValueError(f"invalid {noun}: " + "; ".join(error.message for error in errors))


def _validate_relative_path(value: object) -> None:
    if type(value) is not str or not value or "\x00" in value or "\\" in value:
        raise ValueError("output path must be a nonempty POSIX-relative path")
    if value.startswith(("/", "//")) or len(value) >= 2 and value[1] == ":":
        raise ValueError("output path must not be absolute, UNC, or drive-qualified")
    raw_parts = value.split("/")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in raw_parts):
        raise ValueError("output path contains an unsafe component")
    for part in path.parts:
        windows_component_collision_key(part)


def _load_unique_json(path: Path) -> tuple[_SafeRead, object]:
    safe_read = _read_regular_file_snapshot(path, "manifest")
    try:
        text = safe_read.raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError("manifest is not strict UTF-8") from error
    try:
        return safe_read, json.loads(text, object_pairs_hook=_unique_object)
    except json.JSONDecodeError as error:
        raise ValueError("manifest is not valid JSON") from error


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _snapshot(info: os.stat_result) -> _StatSnapshot:
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
        info.st_nlink,
    )


def _read_regular_file(path: Path, noun: str) -> bytes:
    return _read_regular_file_snapshot(path, noun).raw


def _open_regular_file(
    path: Path,
    noun: str,
) -> tuple[int, os.stat_result, os.stat_result, os.stat_result]:
    _reject_linked_path(path)
    try:
        before_link = os.lstat(path)
        before = os.stat(path)
        if (
            not stat.S_ISREG(before_link.st_mode)
            or not stat.S_ISREG(before.st_mode)
            or before_link.st_nlink != 1
            or before.st_nlink != 1
        ):
            raise ValueError(f"{noun} is not a regular file: {path}")
        flags = (
            os.O_RDONLY
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
        )
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"cannot safely open {noun}: {path}") from error
    try:
        opened = os.fstat(descriptor)
        # Windows may expose different timestamp precision for lstat versus a
        # file-handle fstat even without mutation. Bind both pathname checks to
        # the opened file identity; then require full stable handle metadata.
        # Task 5's private temporary directory and atomic promotion provide the
        # stronger writer-side isolation needed against hostile concurrent writes.
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or _snapshot(before_link)[:2] != _snapshot(opened)[:2]
            or _snapshot(before)[:2] != _snapshot(opened)[:2]
        ):
            raise ValueError(f"{noun} changed or is not a regular file: {path}")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor, before_link, before, opened


def _verify_opened_file_unchanged(
    descriptor: int,
    opened: os.stat_result,
    path: Path,
    noun: str,
) -> os.stat_result:
    after = os.fstat(descriptor)
    if _snapshot(opened) != _snapshot(after):
        raise ValueError(f"{noun} changed while being verified: {path}")
    return after


def _read_regular_file_snapshot(path: Path, noun: str) -> _SafeRead:
    descriptor, before_link, before, opened = _open_regular_file(path, noun)
    try:
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after = _verify_opened_file_unchanged(
            descriptor,
            opened,
            path,
            noun,
        )
    finally:
        os.close(descriptor)
    return _SafeRead(
        raw=b"".join(chunks),
        link_snapshot=_snapshot(before_link),
        path_snapshot=_snapshot(before),
        handle_snapshot=_snapshot(after),
    )


def _verify_safe_read_path_unchanged(
    path: Path,
    safe_read: _SafeRead,
    noun: str,
) -> None:
    _reject_linked_path(path)
    try:
        final_link = os.lstat(path)
        final_path = os.stat(path)
    except OSError as error:
        raise ValueError(f"cannot restat {noun}: {path}") from error
    if (
        _snapshot(final_link) != safe_read.link_snapshot
        or _snapshot(final_path) != safe_read.path_snapshot
    ):
        raise ValueError(f"{noun} path changed during verification: {path}")


def _verify_complete_outputs(
    manifest_path: Path,
    manifest: Mapping[str, object],
    *,
    initial_inventory: frozenset[tuple[str, _StatSnapshot]],
) -> None:
    root = manifest_path.parent
    for ancestor in (root, *root.parents):
        _reject_linked_path(ancestor)
    metrics = manifest["metrics"]
    artifacts = manifest["artifacts"]
    assert type(metrics) is dict and type(artifacts) is list
    expected: dict[str, tuple[str, int | None]] = {
        metrics["path"]: (metrics["sha256"], None)
    }
    for artifact in artifacts:
        expected[artifact["path"]] = (artifact["sha256"], artifact["size_bytes"])
    found: set[str] = set()
    for current, directories, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        _reject_linked_path(current_path)
        for directory in directories:
            directory_path = current_path / directory
            _reject_linked_path(directory_path)
            relative_directory = directory_path.relative_to(root).as_posix()
            if not any(path.startswith(f"{relative_directory}/") for path in expected):
                raise ValueError(f"unlisted output directory: {relative_directory}")
        for filename in files:
            path = current_path / filename
            relative = path.relative_to(root).as_posix()
            if relative == manifest_path.name:
                continue
            found.add(relative)
            if relative not in expected:
                raise ValueError(f"unlisted output file: {relative}")
            expected_hash, expected_size = expected[relative]
            actual_hash, actual_size = _hash_regular_file(path)
            if actual_hash != expected_hash or expected_size is not None and actual_size != expected_size:
                raise ValueError(f"output hash or size mismatch: {relative}")
    if found != set(expected):
        raise ValueError("complete manifest is missing a declared output")
    if _directory_inventory(root) != initial_inventory:
        raise ValueError("run directory changed while outputs were being verified")


def _directory_inventory(root: Path) -> frozenset[tuple[str, _StatSnapshot]]:
    entries: set[tuple[str, _StatSnapshot]] = set()
    for current, directories, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        _reject_linked_path(current_path)
        for name in [*directories, *files]:
            path = current_path / name
            _reject_linked_path(path)
            info = os.lstat(path)
            if stat.S_ISREG(info.st_mode) and info.st_nlink != 1:
                raise ValueError(f"output file has a hardlink alias: {path}")
            entries.add((path.relative_to(root).as_posix(), _snapshot(info)))
    return frozenset(entries)


def _reject_linked_path(path: Path) -> None:
    try:
        info = os.lstat(path)
    except OSError as error:
        raise ValueError(f"cannot stat output path: {path}") from error
    attributes = getattr(info, "st_file_attributes", 0)
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    if stat.S_ISLNK(info.st_mode) or attributes & reparse:
        raise ValueError(f"output path contains a symlink or reparse point: {path}")


def _hash_regular_file(path: Path) -> tuple[str, int]:
    descriptor, _before_link, _before, opened = _open_regular_file(
        path,
        "output",
    )
    digest = hashlib.sha256()
    size = 0
    try:
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
            size += len(block)
        _verify_opened_file_unchanged(
            descriptor,
            opened,
            path,
            "output",
        )
    finally:
        os.close(descriptor)
    return digest.hexdigest(), size
