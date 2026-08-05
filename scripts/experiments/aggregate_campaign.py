"""Build one plan-bound phase aggregate from physical complete runs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    repository_root = str(REPO_ROOT)
    if repository_root not in sys.path:
        sys.path.insert(0, repository_root)

from gsdiff.experiments.aggregation import (
    CompleteMetricRecord,
    IncompletePhaseError,
    LogicalRunKey,
    build_partial_report,
    load_complete_records,
    publish_json_atomic,
)
from gsdiff.experiments.contracts import (
    PhaseEvidenceContract,
    StatisticsContract,
    load_phase_evidence_contract,
    load_statistics_contract,
)
from gsdiff.experiments.phases import PhasePlan, materialize_phase
from gsdiff.experiments.path_roles import PathRole, require_disjoint_path_roles
from gsdiff.experiments.protocol import load_protocol
from gsdiff.experiments.statistics import aggregate_seed_metrics
from gsdiff.experiments.versioned_json import validate_versioned_json


_DIRECTLY_MATERIALIZABLE_PHASES = {
    "selection-decision-v1",
    "selection-replay-v1",
    "primary-selection-v1",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--phase-id", required=True)
    parser.add_argument("--phase-evidence-contract", type=Path, required=True)
    parser.add_argument("--statistics-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--partial-report", type=Path, required=True)
    return parser


def load_expectations(
    path: Path,
    *,
    expected_plan: PhasePlan,
    expected_statistics_contract_sha256: str,
) -> PhaseEvidenceContract:
    """Compatibility name for the now strictly plan-bound evidence loader."""
    return load_phase_evidence_contract(
        path,
        expected_plan=expected_plan,
        expected_statistics_contract_sha256=(
            expected_statistics_contract_sha256
        ),
    )


def materialize_authoritative_phase(
    phase_id: str,
) -> tuple[PhasePlan, dict[str, object]]:
    """Materialize only phases whose authority is complete in tracked inputs."""
    if phase_id not in _DIRECTLY_MATERIALIZABLE_PHASES:
        raise ValueError(
            "phase cannot yet be materialized from validated repository inputs"
        )
    ablation_protocol = load_protocol(
        REPO_ROOT / "configs" / "protocols" / "ablations-v1.yaml"
    )
    primary_protocol = load_protocol(
        REPO_ROOT / "configs" / "protocols" / "primary-v1.yaml"
    )
    plan = materialize_phase(
        phase_id,
        ablation_protocol=ablation_protocol,
        primary_protocol=primary_protocol,
    )
    source_protocol = (
        ablation_protocol
        if phase_id in {"selection-decision-v1", "selection-replay-v1"}
        else primary_protocol
    )
    return plan, source_protocol


def build_aggregate_document(
    records: tuple[CompleteMetricRecord, ...],
    *,
    phase_evidence_contract: PhaseEvidenceContract,
    statistics_contract: StatisticsContract,
) -> dict[str, object]:
    """Build a complete aggregate using only independently loaded controls."""
    if type(phase_evidence_contract) is not PhaseEvidenceContract:
        raise TypeError("phase evidence contract has an invalid type")
    if type(statistics_contract) is not StatisticsContract:
        raise TypeError("statistics contract has an invalid type")
    if not records:
        raise ValueError("complete records must not be empty")
    if phase_evidence_contract.phase_id != statistics_contract.phase_id:
        raise ValueError("phase and statistics contracts disagree on phase ID")
    if phase_evidence_contract.phase_sha256 != statistics_contract.phase_sha256:
        raise ValueError("phase and statistics contracts disagree on phase hash")
    if (
        phase_evidence_contract.statistics_contract_sha256
        != statistics_contract.canonical_sha256
    ):
        raise ValueError("phase evidence binds another statistics contract")
    if len(records) != phase_evidence_contract.expected_record_count:
        raise ValueError("record count disagrees with phase evidence contract")
    record_identities = {
        record.key: record.run_identity_sha256 for record in records
    }
    if len(record_identities) != len(records):
        raise ValueError("complete records contain a duplicate logical key")
    if record_identities != dict(phase_evidence_contract.expected_identities):
        raise ValueError("complete records disagree with phase evidence identities")
    record_scientific_contracts = {
        record.key: (
            record.scientific_contract_id,
            record.scientific_contract_sha256,
        )
        for record in records
    }
    if record_scientific_contracts != dict(
        phase_evidence_contract.expected_scientific_contracts
    ):
        raise ValueError(
            "complete records disagree with phase evidence scientific contracts"
        )
    metric_versions = {record.metric_version for record in records}
    if metric_versions != {statistics_contract.metric_version}:
        raise ValueError("complete records disagree with statistics metric version")
    _require_shared_dataset_identities(records)
    summary = validate_versioned_json(
        aggregate_seed_metrics(
            records,
            required_seeds=statistics_contract.required_seeds,
            comparisons=statistics_contract.comparisons,
            n_bootstrap=statistics_contract.n_bootstrap,
            bootstrap_seed=statistics_contract.bootstrap_seed,
        ),
        "experiment-statistics-v1",
    )
    document = {
        "schema_version": "experiment-phase-aggregate-v1",
        "status": "complete",
        "phase_id": phase_evidence_contract.phase_id,
        "phase_sha256": phase_evidence_contract.phase_sha256,
        "phase_evidence_contract_sha256": (
            phase_evidence_contract.canonical_sha256
        ),
        "statistics_contract_sha256": statistics_contract.canonical_sha256,
        "metric_version": statistics_contract.metric_version,
        "records": [
            _record_document(record)
            for record in sorted(records, key=lambda item: item.key)
        ],
        "summary": summary,
    }
    return validate_versioned_json(document, "experiment-phase-aggregate-v1")


def main(argv: list[str] | None = None) -> int:
    try:
        arguments = _parser().parse_args(argv)
        path_roles = _aggregate_path_roles(arguments)
        require_disjoint_path_roles(*path_roles)
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
        try:
            records = load_complete_records(
                arguments.artifact_root,
                phase_id=plan.phase_id,
                expected_identities=(
                    phase_evidence_contract.expected_identities
                ),
            )
        except IncompletePhaseError as error:
            partial = build_partial_report(
                plan.phase_id,
                phase_evidence_contract.expected_record_count,
                error.missing_keys,
            )
            partial = validate_versioned_json(
                partial,
                "experiment-partial-report-v1",
            )
            require_disjoint_path_roles(*path_roles)
            publish_json_atomic(
                arguments.partial_report,
                partial,
                schema_version="experiment-partial-report-v1",
            )
            return 1
        document = build_aggregate_document(
            records,
            phase_evidence_contract=phase_evidence_contract,
            statistics_contract=statistics_contract,
        )
        require_disjoint_path_roles(*path_roles)
        publish_json_atomic(
            arguments.output,
            document,
            schema_version="experiment-phase-aggregate-v1",
        )
        return 0
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"aggregate refused: {type(error).__name__}", file=sys.stderr)
        return 1


def _aggregate_path_roles(arguments: argparse.Namespace) -> tuple[PathRole, ...]:
    return (
        PathRole("artifact root", arguments.artifact_root, "read"),
        PathRole(
            "ablation protocol",
            REPO_ROOT / "configs" / "protocols" / "ablations-v1.yaml",
            "read",
        ),
        PathRole(
            "primary protocol",
            REPO_ROOT / "configs" / "protocols" / "primary-v1.yaml",
            "read",
        ),
        PathRole("schema directory", REPO_ROOT / "schemas", "read"),
        PathRole(
            "requirements lock",
            REPO_ROOT / "requirements-lock.txt",
            "read",
        ),
        PathRole(
            "environment lock",
            REPO_ROOT / "docs" / "reproducibility" / "environment-lock.json",
            "read",
        ),
        PathRole("runtime environment", Path(sys.prefix), "read"),
        PathRole(
            "phase evidence contract",
            arguments.phase_evidence_contract,
            "read",
        ),
        PathRole(
            "statistics contract",
            arguments.statistics_contract,
            "read",
        ),
        PathRole("complete aggregate", arguments.output, "write"),
        PathRole("partial report", arguments.partial_report, "write"),
    )


def _require_shared_dataset_identities(
    records: tuple[CompleteMetricRecord, ...],
) -> None:
    datasets_by_cell: dict[tuple[str, str, str, str, int], str] = {}
    for record in records:
        cell = (
            record.scientific_contract_sha256,
            record.key.acquisition_config_id,
            record.key.target_id,
            record.key.motion_id,
            record.key.seed,
        )
        prior = datasets_by_cell.get(cell)
        if prior is None:
            datasets_by_cell[cell] = record.dataset_identity_sha256
        elif prior != record.dataset_identity_sha256:
            raise ValueError("phase cell contains mixed dataset identities")


def _record_document(record: CompleteMetricRecord) -> dict[str, object]:
    if type(record) is not CompleteMetricRecord:
        raise TypeError("complete record has an invalid type")
    return {
        "key": _key_document(record.key),
        "scientific_contract_id": record.scientific_contract_id,
        "scientific_contract_sha256": record.scientific_contract_sha256,
        "method_config_sha256": record.method_config_sha256,
        "checkpoints_sha256": dict(record.checkpoints_sha256),
        "dataset_identity_sha256": record.dataset_identity_sha256,
        "run_identity_sha256": record.run_identity_sha256,
        "manifest_sha256": record.manifest_sha256,
        "metrics_sha256": record.metrics_sha256,
        "metric_version": record.metric_version,
        "code_commit": record.code_commit,
        "dependencies_sha256": record.dependencies_sha256,
        "environment_lock_sha256": record.environment_lock_sha256,
        "source_snapshot_sha256": record.source_snapshot_sha256,
        "source_projection_sha256": record.source_projection_sha256,
        "requested_runtime_device": record.requested_runtime_device,
        "execution_profile": record.execution_profile,
        "metrics": dict(record.metrics),
    }


def _key_document(key: LogicalRunKey) -> dict[str, object]:
    return {
        "phase_id": key.phase_id,
        "acquisition_config_id": key.acquisition_config_id,
        "method_config_id": key.method_config_id,
        "method_id": key.method_id,
        "target_id": key.target_id,
        "motion_id": key.motion_id,
        "seed": key.seed,
    }


if __name__ == "__main__":
    raise SystemExit(main())
