"""Verify a canonical phase aggregate against independent physical contracts."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import math
from pathlib import Path
import re
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    repository_root = str(REPO_ROOT)
    if repository_root not in sys.path:
        sys.path.insert(0, repository_root)

from gsdiff.experiments.aggregation import LogicalRunKey, load_complete_records
from gsdiff.experiments.contracts import (
    PhaseEvidenceContract,
    StatisticsContract,
    load_phase_evidence_contract,
    load_statistics_contract,
)
from gsdiff.experiments.identity import canonical_json_bytes
from gsdiff.experiments.phases import (
    PhasePlan,
    VerifiedPhaseCompletion,
    _verified_phase_completion_from_verified_aggregate_claims,
)
from gsdiff.experiments.statistics import aggregate_seed_metrics
from gsdiff.experiments.versioned_json import (
    load_canonical_versioned_json,
    validate_versioned_json,
)
from scripts.experiments.aggregate_campaign import (
    _record_document,
    materialize_authoritative_phase,
)


_SHA256 = re.compile(r"^[0-9a-f]{64}$", re.ASCII)
_COMMIT = re.compile(r"^[0-9a-f]{40}$", re.ASCII)
_AGGREGATE_FIELDS = {
    "schema_version",
    "status",
    "phase_id",
    "phase_sha256",
    "phase_evidence_contract_sha256",
    "statistics_contract_sha256",
    "metric_version",
    "records",
    "summary",
}
_KEY_FIELDS = {
    "phase_id",
    "acquisition_config_id",
    "method_config_id",
    "method_id",
    "target_id",
    "motion_id",
    "seed",
}
_RECORD_FIELDS = {
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
_METRICS = {
    "psnr_global_affine",
    "ssim_global_affine",
    "nrmse_global_affine_l2",
    "psnr_legacy_per_frame_minmax",
}
_SIGNATURE_FIELDS = (
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--phase-id", required=True)
    parser.add_argument("--phase-evidence-contract", type=Path, required=True)
    parser.add_argument("--statistics-contract", type=Path, required=True)
    return parser


def verify_aggregate_document(
    value: object,
    *,
    statistics_contract: StatisticsContract,
    expected_phase_evidence_contract_sha256: str,
) -> dict[str, object]:
    """Recompute one aggregate using controls not sourced from its summary."""
    if type(statistics_contract) is not StatisticsContract:
        raise TypeError("statistics contract has an invalid type")
    phase_evidence_sha256 = _require_sha256(
        "expected phase evidence contract SHA-256",
        expected_phase_evidence_contract_sha256,
    )
    document = validate_versioned_json(
        value,
        "experiment-phase-aggregate-v1",
    )
    if set(document) != _AGGREGATE_FIELDS:
        raise ValueError("aggregate top-level shape is invalid")
    if document["schema_version"] != "experiment-phase-aggregate-v1":
        raise ValueError("aggregate schema version is invalid")
    if document["status"] != "complete":
        raise ValueError("aggregate status is not complete")
    phase_id = _require_string("aggregate phase_id", document["phase_id"])
    phase_sha256 = _require_sha256(
        "aggregate phase_sha256", document["phase_sha256"]
    )
    if phase_id != statistics_contract.phase_id:
        raise ValueError("aggregate phase disagrees with statistics contract")
    if phase_sha256 != statistics_contract.phase_sha256:
        raise ValueError("aggregate phase hash disagrees with statistics contract")
    if (
        document["statistics_contract_sha256"]
        != statistics_contract.canonical_sha256
    ):
        raise ValueError("aggregate binds another statistics contract")
    if document["phase_evidence_contract_sha256"] != phase_evidence_sha256:
        raise ValueError("aggregate binds another phase evidence contract")
    metric_version = _require_string(
        "aggregate metric_version", document["metric_version"]
    )
    if metric_version != statistics_contract.metric_version:
        raise ValueError("aggregate metric version disagrees with statistics contract")
    records = document["records"]
    if type(records) is not list or not records:
        raise ValueError("aggregate records must be a nonempty exact array")

    keys: list[LogicalRunKey] = []
    identities: list[str] = []
    datasets_by_cell: dict[tuple[str, str, str, str, int], str] = {}
    method_provenance: dict[
        tuple[str, str], tuple[str, tuple[tuple[str, str], ...]]
    ] = {}
    common_signature: tuple[str, ...] | None = None
    for record in records:
        key, identity, signature, provenance = _validate_record(
            record,
            phase_id=phase_id,
            metric_version=metric_version,
        )
        keys.append(key)
        identities.append(identity)
        cell = (
            str(record["scientific_contract_sha256"]),
            key.acquisition_config_id,
            key.target_id,
            key.motion_id,
            key.seed,
        )
        dataset_identity = str(record["dataset_identity_sha256"])
        prior_dataset = datasets_by_cell.get(cell)
        if prior_dataset is None:
            datasets_by_cell[cell] = dataset_identity
        elif prior_dataset != dataset_identity:
            raise ValueError("aggregate cell contains mixed dataset identities")
        method_grain = (key.method_id, key.method_config_id)
        prior_provenance = method_provenance.get(method_grain)
        if prior_provenance is None:
            method_provenance[method_grain] = provenance
        elif prior_provenance != provenance:
            raise ValueError("aggregate contains mixed method provenance")
        if common_signature is None:
            common_signature = signature
        elif signature != common_signature:
            raise ValueError("aggregate contains mixed record evidence")
    if keys != sorted(keys):
        raise ValueError("aggregate records are not canonically sorted")
    if len(keys) != len(set(keys)):
        raise ValueError("aggregate contains a duplicate logical key")
    if len(identities) != len(set(identities)):
        raise ValueError("aggregate contains a duplicate run identity")

    summary = validate_versioned_json(
        document["summary"],
        "experiment-statistics-v1",
    )
    rebuilt = validate_versioned_json(
        aggregate_seed_metrics(
            records,
            required_seeds=statistics_contract.required_seeds,
            comparisons=statistics_contract.comparisons,
            n_bootstrap=statistics_contract.n_bootstrap,
            bootstrap_seed=statistics_contract.bootstrap_seed,
        ),
        "experiment-statistics-v1",
    )
    if canonical_json_bytes(rebuilt) != canonical_json_bytes(summary):
        raise ValueError("aggregate summary disagrees with statistics contract")
    assert common_signature is not None
    return {
        "phase_id": phase_id,
        "phase_sha256": phase_sha256,
        "phase_evidence_contract_sha256": phase_evidence_sha256,
        "statistics_contract_sha256": statistics_contract.canonical_sha256,
        "record_count": len(records),
        "code_commit": common_signature[3],
        "execution_profile": common_signature[9],
    }


def load_canonical_document(
    path: Path, noun: str = "aggregate"
) -> tuple[dict[str, object], bytes]:
    return load_canonical_versioned_json(
        path,
        "experiment-phase-aggregate-v1",
        noun=noun,
    )


def verify_physical_aggregate_document(
    value: object,
    *,
    artifact_root: Path,
    phase_evidence_contract: PhaseEvidenceContract,
    statistics_contract: StatisticsContract,
) -> dict[str, object]:
    """Verify statistics and exact records against plan-bound physical evidence."""
    if type(phase_evidence_contract) is not PhaseEvidenceContract:
        raise TypeError("phase evidence contract has an invalid type")
    completion = verify_aggregate_document(
        value,
        statistics_contract=statistics_contract,
        expected_phase_evidence_contract_sha256=(
            phase_evidence_contract.canonical_sha256
        ),
    )
    if completion["phase_id"] != phase_evidence_contract.phase_id:
        raise ValueError("aggregate phase disagrees with phase evidence contract")
    if completion["phase_sha256"] != phase_evidence_contract.phase_sha256:
        raise ValueError("aggregate phase hash disagrees with phase evidence contract")
    if completion["record_count"] != phase_evidence_contract.expected_record_count:
        raise ValueError("aggregate count disagrees with phase evidence contract")
    if (
        phase_evidence_contract.statistics_contract_sha256
        != statistics_contract.canonical_sha256
    ):
        raise ValueError("phase evidence binds another statistics contract")
    aggregate_document = validate_versioned_json(
        value,
        "experiment-phase-aggregate-v1",
    )
    aggregate_scientific_contracts = {
        _logical_key(record["key"]): (
            record["scientific_contract_id"],
            record["scientific_contract_sha256"],
        )
        for record in aggregate_document["records"]
    }
    if aggregate_scientific_contracts != dict(
        phase_evidence_contract.expected_scientific_contracts
    ):
        raise ValueError(
            "aggregate records disagree with phase evidence scientific contracts"
        )
    physical_records = load_complete_records(
        artifact_root,
        phase_id=phase_evidence_contract.phase_id,
        expected_identities=phase_evidence_contract.expected_identities,
    )
    physical_scientific_contracts = {
        record.key: (
            record.scientific_contract_id,
            record.scientific_contract_sha256,
        )
        for record in physical_records
    }
    if physical_scientific_contracts != dict(
        phase_evidence_contract.expected_scientific_contracts
    ):
        raise ValueError(
            "physical records disagree with phase evidence scientific contracts"
        )
    expected_records = [
        _record_document(record)
        for record in sorted(physical_records, key=lambda item: item.key)
    ]
    document = validate_versioned_json(
        value,
        "experiment-phase-aggregate-v1",
    )
    if canonical_json_bytes(document["records"]) != canonical_json_bytes(
        expected_records
    ):
        raise ValueError("aggregate records disagree with physical evidence")
    return completion


def load_verified_phase_completion(
    aggregate_path: Path,
    *,
    artifact_root: Path,
    expected_plan: PhasePlan,
    source_protocol: Mapping[str, object],
    phase_evidence_contract_path: Path,
    statistics_contract_path: Path,
) -> VerifiedPhaseCompletion:
    """Mint a completion only after independent plan and physical verification."""
    statistics_contract = load_statistics_contract(
        statistics_contract_path,
        expected_plan=expected_plan,
        source_protocol=source_protocol,
    )
    phase_evidence_contract = load_phase_evidence_contract(
        phase_evidence_contract_path,
        expected_plan=expected_plan,
        expected_statistics_contract_sha256=(
            statistics_contract.canonical_sha256
        ),
    )
    document, payload = load_canonical_document(aggregate_path)
    completion = verify_physical_aggregate_document(
        document,
        artifact_root=artifact_root,
        phase_evidence_contract=phase_evidence_contract,
        statistics_contract=statistics_contract,
    )
    return _verified_phase_completion_from_verified_aggregate_claims(
        phase_id=str(completion["phase_id"]),
        phase_sha256=str(completion["phase_sha256"]),
        complete_count=int(completion["record_count"]),
        publication_experiment_commit=str(completion["code_commit"]),
        aggregate_sha256=hashlib.sha256(payload).hexdigest(),
    )


def main(argv: list[str] | None = None) -> int:
    try:
        arguments = _parser().parse_args(argv)
        plan, source_protocol = materialize_authoritative_phase(
            arguments.phase_id
        )
        statistics_contract = load_statistics_contract(
            arguments.statistics_contract,
            expected_plan=plan,
            source_protocol=source_protocol,
        )
        phase_evidence_contract = load_phase_evidence_contract(
            arguments.phase_evidence_contract,
            expected_plan=plan,
            expected_statistics_contract_sha256=(
                statistics_contract.canonical_sha256
            ),
        )
        document, _payload = load_canonical_document(arguments.aggregate)
        completion = verify_physical_aggregate_document(
            document,
            artifact_root=arguments.artifact_root,
            phase_evidence_contract=phase_evidence_contract,
            statistics_contract=statistics_contract,
        )
        sys.stdout.buffer.write(canonical_json_bytes(completion) + b"\n")
        return 0
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"verification refused: {type(error).__name__}", file=sys.stderr)
        return 1


def _validate_record(
    value: object,
    *,
    phase_id: str,
    metric_version: str,
) -> tuple[
    LogicalRunKey,
    str,
    tuple[str, ...],
    tuple[str, tuple[tuple[str, str], ...]],
]:
    if type(value) is not dict or set(value) != _RECORD_FIELDS:
        raise ValueError("aggregate record shape is invalid")
    key = _logical_key(value["key"])
    if key.phase_id != phase_id:
        raise ValueError("aggregate record belongs to the wrong phase")
    for field in (
        "scientific_contract_id",
        "metric_version",
        "requested_runtime_device",
        "execution_profile",
    ):
        _require_string(f"record {field}", value[field])
    if value["metric_version"] != metric_version:
        raise ValueError("aggregate record metric version disagrees with header")
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
        _require_sha256(f"record {field}", value[field])
    checkpoints = _checkpoint_mapping(value["checkpoints_sha256"])
    _require_commit("record code_commit", value["code_commit"])
    metrics = value["metrics"]
    if type(metrics) is not dict or set(metrics) != _METRICS:
        raise ValueError("aggregate record metrics shape is invalid")
    for metric, metric_value in metrics.items():
        if type(metric_value) not in (int, float) or not math.isfinite(metric_value):
            raise ValueError(f"aggregate record metric {metric} is not finite")
        if metric_value == 0.0 and math.copysign(1.0, metric_value) < 0.0:
            raise ValueError("aggregate record metric contains negative zero")
    signature = tuple(str(value[field]) for field in _SIGNATURE_FIELDS)
    provenance = (str(value["method_config_sha256"]), checkpoints)
    return key, str(value["run_identity_sha256"]), signature, provenance


def _checkpoint_mapping(value: object) -> tuple[tuple[str, str], ...]:
    if type(value) is not dict:
        raise ValueError("record checkpoints_sha256 must be an exact object")
    checkpoints: list[tuple[str, str]] = []
    for name, digest in value.items():
        checkpoints.append(
            (
                _require_string("checkpoint name", name),
                _require_sha256("checkpoint SHA-256", digest),
            )
        )
    return tuple(sorted(checkpoints))


def _logical_key(value: object) -> LogicalRunKey:
    if type(value) is not dict or set(value) != _KEY_FIELDS:
        raise ValueError("aggregate logical key shape is invalid")
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
        raise ValueError("aggregate logical key is invalid") from error


def _require_string(noun: str, value: object) -> str:
    if type(value) is not str or not value:
        raise TypeError(f"{noun} must be a nonempty exact string")
    return value


def _require_sha256(noun: str, value: object) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{noun} must be a lowercase SHA-256")
    return value


def _require_commit(noun: str, value: object) -> str:
    if type(value) is not str or _COMMIT.fullmatch(value) is None:
        raise ValueError(f"{noun} must be a lowercase Git commit")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
