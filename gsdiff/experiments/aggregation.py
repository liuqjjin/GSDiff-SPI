"""Strict loading and atomic publication for phase-local metric records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
from types import MappingProxyType

from .identity import canonical_json_bytes
from .manifest import load_complete_manifest
from .versioned_json import (
    validate_canonical_versioned_json_bytes,
    validate_versioned_json,
)


_SHA256 = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_COMMIT = re.compile(r"[0-9a-f]{40}\Z", re.ASCII)
_ID = re.compile(r"[a-z0-9][a-z0-9_-]*\Z", re.ASCII)
_METRIC_FIELDS = (
    "nrmse_global_affine_l2",
    "psnr_global_affine",
    "psnr_legacy_per_frame_minmax",
    "ssim_global_affine",
)
_KEY_FIELDS = {
    "phase_id",
    "acquisition_config_id",
    "method_config_id",
    "method_id",
    "target_id",
    "motion_id",
    "seed",
}
_AGGREGATE_FIELDS = {
    "schema_version",
    "status",
    "phase_id",
    "phase_sha256",
    "phase_evidence_contract_sha256",
    "statistics_contract_sha256",
    "metric_version",
    "records",
}
_SERIALIZED_RECORD_FIELDS = {
    "key",
    "scientific_contract_id",
    "scientific_contract_sha256",
    "method_config_sha256",
    "checkpoints_sha256",
    "dataset_identity_sha256",
    "run_identity_sha256",
    "manifest_sha256",
    "metrics_sha256",
    "metric_version",
    "code_commit",
    "dependencies_sha256",
    "environment_lock_sha256",
    "source_snapshot_sha256",
    "source_projection_sha256",
    "requested_runtime_device",
    "execution_profile",
    "metrics",
}
_SERIALIZED_SIGNATURE_FIELDS = (
    "scientific_contract_id",
    "scientific_contract_sha256",
    "metric_version",
    "code_commit",
    "dependencies_sha256",
    "environment_lock_sha256",
    "source_snapshot_sha256",
    "source_projection_sha256",
    "requested_runtime_device",
    "execution_profile",
)


class AggregationIntegrityError(ValueError):
    """Raised when selected phase evidence is contradictory or malformed."""


class IncompletePhaseError(RuntimeError):
    """Raised after collecting every expected key without a complete run."""

    def __init__(self, missing_keys: Sequence["LogicalRunKey"]) -> None:
        self.missing_keys = tuple(sorted(missing_keys))
        super().__init__(
            f"phase is incomplete: {len(self.missing_keys)} expected runs missing"
        )


@dataclass(frozen=True, order=True)
class LogicalRunKey:
    phase_id: str
    acquisition_config_id: str
    method_config_id: str
    method_id: str
    target_id: str
    motion_id: str
    seed: int

    def __post_init__(self) -> None:
        for name in (
            "phase_id",
            "acquisition_config_id",
            "method_config_id",
            "method_id",
            "target_id",
            "motion_id",
        ):
            _require_string(name, getattr(self, name))
        if type(self.seed) is not int:
            raise TypeError("seed must be an exact integer")


@dataclass(frozen=True)
class CompleteMetricRecord:
    key: LogicalRunKey
    scientific_contract_id: str
    scientific_contract_sha256: str
    method_config_sha256: str
    checkpoints_sha256: Mapping[str, str]
    dataset_identity_sha256: str
    run_identity_sha256: str
    manifest_sha256: str
    metrics_sha256: str
    metric_version: str
    code_commit: str
    dependencies_sha256: str
    environment_lock_sha256: str
    source_snapshot_sha256: str
    source_projection_sha256: str
    requested_runtime_device: str
    execution_profile: str
    metrics: Mapping[str, float]


def load_complete_records(
    artifact_root: Path,
    *,
    phase_id: str,
    expected_identities: Mapping[LogicalRunKey, str],
    requirements_lock: Path | None = None,
    environment_lock: Path | None = None,
) -> tuple[CompleteMetricRecord, ...]:
    """Load exactly the clean complete identities declared for one phase."""
    if not isinstance(artifact_root, Path):
        raise TypeError("artifact_root must be a Path")
    _require_string("phase_id", phase_id)
    if not isinstance(expected_identities, Mapping):
        raise TypeError("expected_identities must be a mapping")
    items = list(expected_identities.items())
    if not items:
        raise AggregationIntegrityError("expected identities must not be empty")

    keys: list[LogicalRunKey] = []
    identities: list[str] = []
    for key, identity_sha256 in items:
        if type(key) is not LogicalRunKey:
            raise TypeError("expected identity keys must be LogicalRunKey values")
        if key.phase_id != phase_id:
            raise AggregationIntegrityError(
                "expected logical key belongs to the wrong phase"
            )
        keys.append(key)
        identities.append(_require_sha256("expected identity", identity_sha256))
    if len(keys) != len(set(keys)):
        raise AggregationIntegrityError("duplicate logical key in expectations")
    if len(identities) != len(set(identities)):
        raise AggregationIntegrityError("duplicate run identity in expectations")

    missing: list[LogicalRunKey] = []
    records: list[CompleteMetricRecord] = []
    common_signature: tuple[str, ...] | None = None
    canonical_root = artifact_root.absolute()
    for key, identity_sha256 in sorted(
        zip(keys, identities, strict=True),
        key=lambda item: item[0],
    ):
        manifest_path = (
            canonical_root / "runs" / identity_sha256 / "manifest.json"
        )
        try:
            manifest_mode = manifest_path.lstat().st_mode
        except FileNotFoundError:
            missing.append(key)
            continue
        if not stat.S_ISREG(manifest_mode):
            raise AggregationIntegrityError(
                "expected manifest node is not a regular file"
            )
        manifest = load_complete_manifest(
            manifest_path,
            artifact_root=artifact_root,
            expected_identity_sha256=identity_sha256,
            requirements_lock=requirements_lock,
            environment_lock=environment_lock,
        )
        if manifest is None:
            missing.append(key)
            continue
        record = _record_from_manifest(
            key=key,
            expected_identity_sha256=identity_sha256,
            manifest_path=manifest_path,
            manifest=manifest,
        )
        signature = _common_signature(record)
        if common_signature is None:
            common_signature = signature
        elif signature != common_signature:
            raise AggregationIntegrityError("phase contains mixed complete evidence")
        records.append(record)

    _require_complete_record_dataset_consistency(records)
    _require_complete_record_method_provenance(records)
    if missing:
        raise IncompletePhaseError(missing)
    return tuple(records)


def build_partial_report(
    phase_id: str,
    expected_count: int,
    missing_keys: Sequence[LogicalRunKey],
) -> dict[str, object]:
    """Build a deterministic diagnostic report without computing statistics."""
    _require_string("phase_id", phase_id)
    if type(expected_count) is not int or expected_count < 0:
        raise TypeError("expected_count must be a nonnegative exact integer")
    if not isinstance(missing_keys, Sequence) or isinstance(
        missing_keys, (str, bytes, bytearray)
    ):
        raise TypeError("missing_keys must be a sequence")
    missing = tuple(sorted(missing_keys))
    if not missing:
        raise AggregationIntegrityError("a partial report requires missing keys")
    if len(missing) != len(set(missing)):
        raise AggregationIntegrityError("partial report contains duplicate keys")
    for key in missing:
        if type(key) is not LogicalRunKey:
            raise TypeError("missing keys must be LogicalRunKey values")
        if key.phase_id != phase_id:
            raise AggregationIntegrityError("partial report contains a wrong phase")
    if len(missing) > expected_count:
        raise AggregationIntegrityError("missing count exceeds expected count")
    return {
        "schema_version": "experiment-partial-report-v1",
        "status": "partial",
        "phase_id": phase_id,
        "expected_count": expected_count,
        "available_complete": expected_count - len(missing),
        "missing": [_logical_key_document(key) for key in missing],
    }


def publish_json_atomic(
    path: Path,
    document: Mapping[str, object],
    *,
    schema_version: str | None = None,
) -> str:
    """Atomically replace one explicit path with canonical JSON bytes."""
    if not isinstance(path, Path):
        raise TypeError("path must be a Path")
    if not isinstance(document, Mapping):
        raise TypeError("document must be a mapping")
    normalized = dict(document)
    if schema_version is not None:
        normalized = validate_versioned_json(normalized, schema_version)
    payload = canonical_json_bytes(normalized)
    digest = hashlib.sha256(payload).hexdigest()
    descriptor, raw_temporary = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(raw_temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary_payload = temporary.read_bytes()
        if temporary_payload != payload:
            raise AggregationIntegrityError(
                "atomic JSON temporary file failed exact readback"
            )
        if schema_version is not None:
            validate_canonical_versioned_json_bytes(
                temporary_payload,
                schema_version,
                noun="atomic JSON temporary file",
            )
        os.replace(temporary, path)
        published = path.read_bytes()
        if published != payload or hashlib.sha256(published).hexdigest() != digest:
            raise AggregationIntegrityError(
                "atomic JSON destination failed exact readback"
            )
        if schema_version is not None:
            validate_canonical_versioned_json_bytes(
                published,
                schema_version,
                noun="atomic JSON destination",
            )
    finally:
        if temporary.exists():
            temporary.unlink()
    return digest


def _merge_record_union(
    existing: Mapping[str, object],
    incoming: Mapping[str, object],
) -> dict[str, object]:
    """Build a validated source-record union for public aggregate finalization."""
    existing_value = _plain_json_document(existing, "existing aggregate")
    incoming_value = _plain_json_document(incoming, "incoming aggregate")
    _validate_aggregate_shape(existing_value, "existing aggregate")
    _validate_aggregate_shape(incoming_value, "incoming aggregate")
    if existing_value.get("status") != "complete" or incoming_value.get(
        "status"
    ) != "complete":
        raise AggregationIntegrityError("partial aggregate cannot be merged")
    existing_records = existing_value.get("records")
    incoming_records = incoming_value.get("records")
    if type(existing_records) is not list or type(incoming_records) is not list:
        raise AggregationIntegrityError("aggregate records must be exact arrays")

    existing_header = {
        key: value
        for key, value in existing_value.items()
        if key not in {"records", "summary"}
    }
    incoming_header = {
        key: value
        for key, value in incoming_value.items()
        if key not in {"records", "summary"}
    }
    if canonical_json_bytes(existing_header) != canonical_json_bytes(
        incoming_header
    ):
        raise AggregationIntegrityError("aggregate header mismatch")

    phase_id = existing_header.get("phase_id")
    _require_string("aggregate phase_id", phase_id)
    metric_version = existing_header.get("metric_version")
    _require_string("aggregate metric version", metric_version)
    records = _records_by_key(existing_records, phase_id, "existing aggregate")
    incoming_by_key = _records_by_key(
        incoming_records,
        phase_id,
        "incoming aggregate",
    )
    for key, record in incoming_by_key.items():
        prior = records.get(key)
        if prior is None:
            records[key] = record
        elif canonical_json_bytes(prior) != canonical_json_bytes(record):
            raise AggregationIntegrityError(
                "conflicting aggregate record for one logical key"
            )
    _validate_merged_record_evidence(records, phase_id, metric_version)
    return {
        **existing_header,
        "records": [records[key] for key in sorted(records)],
    }


def merge_aggregate(
    existing: Mapping[str, object],
    incoming: Mapping[str, object],
    *,
    phase_evidence_contract: object,
    statistics_contract: object,
) -> dict[str, object]:
    """Merge complete source records and independently rebuild their summary."""
    from .contracts import PhaseEvidenceContract, StatisticsContract
    from .statistics import aggregate_seed_metrics

    if type(phase_evidence_contract) is not PhaseEvidenceContract:
        raise TypeError("phase evidence contract has an invalid type")
    if type(statistics_contract) is not StatisticsContract:
        raise TypeError("statistics contract has an invalid type")
    merged = _merge_record_union(existing, incoming)
    if merged["phase_id"] != phase_evidence_contract.phase_id:
        raise AggregationIntegrityError(
            "merged aggregate phase disagrees with phase evidence"
        )
    if merged["phase_sha256"] != phase_evidence_contract.phase_sha256:
        raise AggregationIntegrityError(
            "merged aggregate phase hash disagrees with phase evidence"
        )
    if merged["phase_evidence_contract_sha256"] != (
        phase_evidence_contract.canonical_sha256
    ):
        raise AggregationIntegrityError(
            "merged aggregate binds another phase evidence contract"
        )
    if merged["statistics_contract_sha256"] != (
        statistics_contract.canonical_sha256
    ):
        raise AggregationIntegrityError(
            "merged aggregate binds another statistics contract"
        )
    if phase_evidence_contract.statistics_contract_sha256 != (
        statistics_contract.canonical_sha256
    ):
        raise AggregationIntegrityError(
            "phase evidence binds another statistics contract"
        )
    if (
        statistics_contract.phase_id != phase_evidence_contract.phase_id
        or statistics_contract.phase_sha256
        != phase_evidence_contract.phase_sha256
        or merged["metric_version"] != statistics_contract.metric_version
    ):
        raise AggregationIntegrityError(
            "merged aggregate disagrees with statistics authority"
        )
    records = merged["records"]
    if type(records) is not list:
        raise AggregationIntegrityError("merged records must be an exact array")
    observed_identities: dict[LogicalRunKey, str] = {}
    observed_scientific_contracts: dict[
        LogicalRunKey,
        tuple[str, str],
    ] = {}
    for record in records:
        if type(record) is not dict:
            raise AggregationIntegrityError(
                "merged record must be an exact object"
            )
        key = _logical_key_from_document(record["key"])
        identity = _require_sha256(
            "merged run identity",
            record["run_identity_sha256"],
        )
        observed_identities[key] = identity
        observed_scientific_contracts[key] = (
            _require_string(
                "merged scientific contract ID",
                record["scientific_contract_id"],
            ),
            _require_sha256(
                "merged scientific contract hash",
                record["scientific_contract_sha256"],
            ),
        )
    if len(records) != phase_evidence_contract.expected_record_count or (
        observed_identities
        != dict(phase_evidence_contract.expected_identities)
    ):
        raise AggregationIntegrityError(
            "merged records disagree with phase evidence identities"
        )
    if observed_scientific_contracts != dict(
        phase_evidence_contract.expected_scientific_contracts
    ):
        raise AggregationIntegrityError(
            "merged records disagree with phase scientific contracts"
        )
    summary = aggregate_seed_metrics(
        records,
        required_seeds=statistics_contract.required_seeds,
        comparisons=statistics_contract.comparisons,
        n_bootstrap=statistics_contract.n_bootstrap,
        bootstrap_seed=statistics_contract.bootstrap_seed,
    )
    return validate_versioned_json(
        {**merged, "summary": summary},
        "experiment-phase-aggregate-v1",
    )


def _validate_merged_record_evidence(
    records: Mapping[LogicalRunKey, dict[str, object]],
    phase_id: str,
    metric_version: str,
) -> None:
    identities: set[str] = set()
    datasets: dict[tuple[str, str, str, str, int], str] = {}
    method_configs: dict[tuple[str, str], str] = {}
    checkpoint_maps: dict[tuple[str, str], tuple[tuple[str, str], ...]] = {}
    common_signature: tuple[str, ...] | None = None
    for expected_key, record in records.items():
        if set(record) != _SERIALIZED_RECORD_FIELDS:
            raise AggregationIntegrityError("aggregate record shape is invalid")
        key = _logical_key_from_document(record["key"])
        if key != expected_key or key.phase_id != phase_id:
            raise AggregationIntegrityError(
                "aggregate record logical key is inconsistent"
            )
        _require_string(
            "aggregate scientific contract id",
            record["scientific_contract_id"],
        )
        _require_string("aggregate metric version", record["metric_version"])
        if record["metric_version"] != metric_version:
            raise AggregationIntegrityError(
                "aggregate record metric version disagrees with header"
            )
        _require_string(
            "aggregate requested runtime device",
            record["requested_runtime_device"],
        )
        _require_string(
            "aggregate execution profile", record["execution_profile"]
        )
        for field in (
            "scientific_contract_sha256",
            "method_config_sha256",
            "dataset_identity_sha256",
            "run_identity_sha256",
            "manifest_sha256",
            "metrics_sha256",
            "dependencies_sha256",
            "environment_lock_sha256",
            "source_snapshot_sha256",
            "source_projection_sha256",
        ):
            _require_sha256(f"aggregate record {field}", record[field])
        checkpoints = tuple(
            _require_checkpoint_mapping(
                "aggregate record checkpoints_sha256",
                record["checkpoints_sha256"],
            ).items()
        )
        _require_commit("aggregate record code commit", record["code_commit"])
        metrics = record["metrics"]
        if type(metrics) is not dict or set(metrics) != set(_METRIC_FIELDS):
            raise AggregationIntegrityError(
                "aggregate record metrics shape is invalid"
            )
        for metric, value in metrics.items():
            if type(value) not in (int, float) or not math.isfinite(value):
                raise AggregationIntegrityError(
                    f"aggregate record metric {metric} is not finite"
                )
        identity = str(record["run_identity_sha256"])
        if identity in identities:
            raise AggregationIntegrityError(
                "one run identity cannot satisfy two logical keys"
            )
        identities.add(identity)
        dataset_grain = (
            str(record["scientific_contract_sha256"]),
            key.acquisition_config_id,
            key.target_id,
            key.motion_id,
            key.seed,
        )
        dataset_identity = str(record["dataset_identity_sha256"])
        prior_dataset = datasets.get(dataset_grain)
        if prior_dataset is None:
            datasets[dataset_grain] = dataset_identity
        elif prior_dataset != dataset_identity:
            raise AggregationIntegrityError(
                "one scientific cell contains mixed dataset identities"
            )
        method_grain = (key.method_id, key.method_config_id)
        method_config_sha256 = str(record["method_config_sha256"])
        prior_method_config = method_configs.get(method_grain)
        if prior_method_config is None:
            method_configs[method_grain] = method_config_sha256
        elif prior_method_config != method_config_sha256:
            raise AggregationIntegrityError(
                "one method config contains mixed method config provenance"
            )
        prior_checkpoints = checkpoint_maps.get(method_grain)
        if prior_checkpoints is None:
            checkpoint_maps[method_grain] = checkpoints
        elif prior_checkpoints != checkpoints:
            raise AggregationIntegrityError(
                "one method config contains mixed checkpoints provenance"
            )
        signature = tuple(
            str(record[field]) for field in _SERIALIZED_SIGNATURE_FIELDS
        )
        if common_signature is None:
            common_signature = signature
        elif signature != common_signature:
            raise AggregationIntegrityError(
                "merged aggregate contains mixed complete evidence"
            )


def _validate_aggregate_shape(value: dict[str, object], noun: str) -> None:
    fields = set(value)
    if fields not in (_AGGREGATE_FIELDS, _AGGREGATE_FIELDS | {"summary"}):
        raise AggregationIntegrityError(f"{noun} header shape is invalid")
    if value["schema_version"] != "experiment-phase-aggregate-v1":
        raise AggregationIntegrityError(f"{noun} schema is invalid")
    _require_string(f"{noun} phase_id", value["phase_id"])
    _require_sha256(f"{noun} phase hash", value["phase_sha256"])
    _require_sha256(
        f"{noun} phase evidence contract hash",
        value["phase_evidence_contract_sha256"],
    )
    _require_sha256(
        f"{noun} statistics contract hash",
        value["statistics_contract_sha256"],
    )
    _require_string(f"{noun} metric version", value["metric_version"])


def _record_from_manifest(
    *,
    key: LogicalRunKey,
    expected_identity_sha256: str,
    manifest_path: Path,
    manifest: Mapping[str, object],
) -> CompleteMetricRecord:
    if type(manifest) is not dict:
        raise AggregationIntegrityError("complete manifest must be an exact object")
    if manifest.get("status") != "complete":
        raise AggregationIntegrityError("selected manifest is not complete")
    if manifest.get("execution_class") != "blind_method_child":
        raise AggregationIntegrityError("selected manifest is not blind")
    if manifest.get("identity_sha256") != expected_identity_sha256:
        raise AggregationIntegrityError("selected manifest identity mismatch")

    manifest_raw = manifest_path.read_bytes()
    try:
        expected_manifest_raw = canonical_json_bytes(manifest)
    except (TypeError, ValueError) as error:
        raise AggregationIntegrityError("manifest is not canonical JSON") from error
    if manifest_raw != expected_manifest_raw:
        raise AggregationIntegrityError("manifest bytes changed after strict load")

    protocol = _require_mapping("manifest protocol", manifest.get("protocol"))
    if (
        protocol.get("method") != key.method_id
        or protocol.get("target") != key.target_id
        or protocol.get("motion") != key.motion_id
        or type(protocol.get("seed")) is not int
        or protocol.get("seed") != key.seed
    ):
        raise AggregationIntegrityError("manifest protocol disagrees with logical key")
    scientific_contract_id = _require_string(
        "scientific contract id",
        protocol.get("scientific_contract_id"),
    )
    scientific_contract_sha256 = _require_sha256(
        "scientific contract hash",
        protocol.get("scientific_contract_sha256"),
    )

    code = _require_mapping("manifest code", manifest.get("code"))
    if code.get("dirty_worktree") is not False:
        raise AggregationIntegrityError("dirty complete evidence is not aggregatable")
    code_commit = _require_commit("code commit", code.get("git_commit"))

    config = _require_mapping("manifest config", manifest.get("config"))
    resolved = _require_mapping("resolved config", config.get("resolved"))
    if resolved.get("phase_id") != key.phase_id:
        raise AggregationIntegrityError(
            "resolved phase id disagrees with logical key"
        )
    if resolved.get("acquisition_config_id") != key.acquisition_config_id:
        raise AggregationIntegrityError(
            "resolved acquisition config id disagrees with logical key"
        )
    method_config_sha256 = _require_sha256(
        "resolved method config hash",
        resolved.get("method_config_sha256"),
    )
    runner = _require_mapping(
        "runner execution evidence",
        resolved.get("runner_execution"),
    )
    method_contract = _require_mapping(
        "method info contract",
        runner.get("method_info_contract"),
    )
    if method_contract.get("method_id") != key.method_id:
        raise AggregationIntegrityError("method contract disagrees with method id")
    if method_contract.get("method_config_id") != key.method_config_id:
        raise AggregationIntegrityError("method config id disagrees with logical key")
    if (
        _require_sha256(
            "method contract config hash",
            method_contract.get("method_config_sha256"),
        )
        != method_config_sha256
    ):
        raise AggregationIntegrityError("method config hash evidence disagrees")
    execution_profile = _require_string(
        "execution profile",
        method_contract.get("execution_profile"),
    )
    source_snapshot_sha256 = _require_sha256(
        "source snapshot hash",
        runner.get("source_snapshot_sha256"),
    )
    source_projection_sha256 = _require_sha256(
        "source projection hash",
        runner.get("source_projection_sha256"),
    )
    requested_runtime_device = _require_string(
        "requested runtime device",
        runner.get("requested_runtime_device"),
    )

    inputs = _require_mapping("manifest inputs", manifest.get("inputs"))
    dataset_identity_sha256 = _require_sha256(
        "dataset identity",
        inputs.get("dataset_identity_sha256"),
    )
    checkpoints_sha256 = _require_checkpoint_mapping(
        "manifest checkpoint hashes",
        inputs.get("checkpoints"),
    )
    runtime = _require_mapping("manifest runtime", manifest.get("runtime"))
    dependencies_sha256 = _require_sha256(
        "dependencies hash",
        runtime.get("dependencies_sha256"),
    )
    environment_lock_sha256 = _require_sha256(
        "environment lock hash",
        runtime.get("environment_lock_sha256"),
    )
    metric_version = _require_string(
        "metric version",
        manifest.get("metric_version"),
    )
    metrics_descriptor = _require_mapping(
        "manifest metrics descriptor",
        manifest.get("metrics"),
    )
    if (
        metrics_descriptor.get("version") != metric_version
        or metrics_descriptor.get("path") != "outputs/metrics.json"
    ):
        raise AggregationIntegrityError("manifest metrics descriptor mismatch")
    metrics_sha256 = _require_sha256(
        "metrics hash",
        metrics_descriptor.get("sha256"),
    )
    metrics_path = manifest_path.parent / "outputs" / "metrics.json"
    metrics_raw = metrics_path.read_bytes()
    if hashlib.sha256(metrics_raw).hexdigest() != metrics_sha256:
        raise AggregationIntegrityError("metrics file hash mismatch")
    metrics_document = _load_canonical_json(metrics_raw, "metrics")
    if type(metrics_document) is not dict:
        raise AggregationIntegrityError("metrics JSON must be an exact object")
    if metrics_document.get("definition_version") != metric_version:
        raise AggregationIntegrityError("metrics definition version mismatch")
    metric_values: dict[str, float] = {}
    for field in _METRIC_FIELDS:
        value = metrics_document.get(field)
        if type(value) not in (int, float) or not math.isfinite(value):
            raise AggregationIntegrityError(
                f"metric {field} must be an exact finite number"
            )
        normalized = float(value)
        metric_values[field] = 0.0 if normalized == 0.0 else normalized

    return CompleteMetricRecord(
        key=key,
        scientific_contract_id=scientific_contract_id,
        scientific_contract_sha256=scientific_contract_sha256,
        method_config_sha256=method_config_sha256,
        checkpoints_sha256=checkpoints_sha256,
        dataset_identity_sha256=dataset_identity_sha256,
        run_identity_sha256=expected_identity_sha256,
        manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),
        metrics_sha256=metrics_sha256,
        metric_version=metric_version,
        code_commit=code_commit,
        dependencies_sha256=dependencies_sha256,
        environment_lock_sha256=environment_lock_sha256,
        source_snapshot_sha256=source_snapshot_sha256,
        source_projection_sha256=source_projection_sha256,
        requested_runtime_device=requested_runtime_device,
        execution_profile=execution_profile,
        metrics=MappingProxyType(metric_values),
    )


def _common_signature(record: CompleteMetricRecord) -> tuple[str, ...]:
    return (
        record.scientific_contract_id,
        record.scientific_contract_sha256,
        record.metric_version,
        record.code_commit,
        record.dependencies_sha256,
        record.environment_lock_sha256,
        record.source_snapshot_sha256,
        record.source_projection_sha256,
        record.requested_runtime_device,
        record.execution_profile,
    )


def _require_complete_record_dataset_consistency(
    records: Sequence[CompleteMetricRecord],
) -> None:
    datasets: dict[tuple[str, str, str, str, int], str] = {}
    for record in records:
        key = record.key
        grain = (
            record.scientific_contract_sha256,
            key.acquisition_config_id,
            key.target_id,
            key.motion_id,
            key.seed,
        )
        prior = datasets.get(grain)
        if prior is None:
            datasets[grain] = record.dataset_identity_sha256
        elif prior != record.dataset_identity_sha256:
            raise AggregationIntegrityError(
                "one scientific cell contains mixed dataset identities"
            )


def _require_complete_record_method_provenance(
    records: Sequence[CompleteMetricRecord],
) -> None:
    method_configs: dict[tuple[str, str], str] = {}
    checkpoint_maps: dict[tuple[str, str], tuple[tuple[str, str], ...]] = {}
    for record in records:
        grain = (record.key.method_id, record.key.method_config_id)
        prior_method_config = method_configs.get(grain)
        if prior_method_config is None:
            method_configs[grain] = record.method_config_sha256
        elif prior_method_config != record.method_config_sha256:
            raise AggregationIntegrityError(
                "one method config contains mixed method config provenance"
            )
        checkpoints = tuple(record.checkpoints_sha256.items())
        prior_checkpoints = checkpoint_maps.get(grain)
        if prior_checkpoints is None:
            checkpoint_maps[grain] = checkpoints
        elif prior_checkpoints != checkpoints:
            raise AggregationIntegrityError(
                "one method config contains mixed checkpoints provenance"
            )


def _load_canonical_json(payload: bytes, noun: str) -> object:
    try:
        value = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {token}")
            ),
        )
        canonical = canonical_json_bytes(value)
    except (UnicodeError, TypeError, ValueError) as error:
        raise AggregationIntegrityError(f"{noun} is not canonical JSON") from error
    if payload != canonical:
        raise AggregationIntegrityError(f"{noun} is not canonical JSON")
    return value


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, child in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key!r}")
        value[key] = child
    return value


def _records_by_key(
    records: list[object],
    phase_id: str,
    noun: str,
) -> dict[LogicalRunKey, dict[str, object]]:
    result: dict[LogicalRunKey, dict[str, object]] = {}
    for record in records:
        if type(record) is not dict:
            raise AggregationIntegrityError(f"{noun} record must be an exact object")
        key = _logical_key_from_document(record.get("key"))
        if key.phase_id != phase_id:
            raise AggregationIntegrityError(f"{noun} contains a wrong-phase record")
        if key in result:
            raise AggregationIntegrityError(f"{noun} contains a duplicate record key")
        result[key] = record
    return result


def _logical_key_from_document(value: object) -> LogicalRunKey:
    if type(value) is not dict or set(value) != _KEY_FIELDS:
        raise AggregationIntegrityError("aggregate logical key shape is invalid")
    try:
        return LogicalRunKey(
            phase_id=value["phase_id"],
            acquisition_config_id=value["acquisition_config_id"],
            method_config_id=value["method_config_id"],
            method_id=value["method_id"],
            target_id=value["target_id"],
            motion_id=value["motion_id"],
            seed=value["seed"],
        )
    except (TypeError, ValueError) as error:
        raise AggregationIntegrityError("aggregate logical key is invalid") from error


def _logical_key_document(key: LogicalRunKey) -> dict[str, object]:
    return {
        "phase_id": key.phase_id,
        "acquisition_config_id": key.acquisition_config_id,
        "method_config_id": key.method_config_id,
        "method_id": key.method_id,
        "target_id": key.target_id,
        "motion_id": key.motion_id,
        "seed": key.seed,
    }


def _plain_json_document(
    value: Mapping[str, object],
    noun: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{noun} must be a mapping")
    try:
        loaded = json.loads(canonical_json_bytes(dict(value)))
    except (TypeError, ValueError) as error:
        raise AggregationIntegrityError(f"{noun} is not finite canonical JSON") from error
    if type(loaded) is not dict:
        raise AggregationIntegrityError(f"{noun} must be an exact object")
    return loaded


def _require_mapping(noun: str, value: object) -> dict[str, object]:
    if type(value) is not dict:
        raise AggregationIntegrityError(f"{noun} must be an exact object")
    return value


def _require_checkpoint_mapping(
    noun: str,
    value: object,
) -> Mapping[str, str]:
    if type(value) is not dict:
        raise AggregationIntegrityError(f"{noun} must be an exact object")
    normalized: dict[str, str] = {}
    for logical_id in sorted(value):
        if type(logical_id) is not str or _ID.fullmatch(logical_id) is None:
            raise AggregationIntegrityError(
                f"{noun} keys must be canonical logical IDs"
            )
        normalized[logical_id] = _require_sha256(
            f"{noun}[{logical_id!r}]",
            value[logical_id],
        )
    return MappingProxyType(normalized)


def _require_string(noun: str, value: object) -> str:
    if type(value) is not str or not value:
        raise TypeError(f"{noun} must be a nonempty exact string")
    return value


def _require_sha256(noun: str, value: object) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise AggregationIntegrityError(f"{noun} must be a lowercase SHA-256")
    return value


def _require_commit(noun: str, value: object) -> str:
    if type(value) is not str or _COMMIT.fullmatch(value) is None:
        raise AggregationIntegrityError(f"{noun} must be a lowercase Git commit")
    return value


__all__ = [
    "AggregationIntegrityError",
    "CompleteMetricRecord",
    "IncompletePhaseError",
    "LogicalRunKey",
    "build_partial_report",
    "load_complete_records",
    "merge_aggregate",
    "publish_json_atomic",
]
