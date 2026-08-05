"""Canonical authority contracts for phase aggregation and publication locks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import re
from types import MappingProxyType

from .aggregation import LogicalRunKey
from .identity import canonical_json_bytes, sha256_bytes
from .phases import DatasetPlanKey, PhasePlan
from .protocol import ExperimentCell, validate_protocol
from .statistics import PairedComparison
from .versioned_json import (
    load_canonical_versioned_json,
    validate_versioned_json,
)


_SHA256 = re.compile(r"^[0-9a-f]{64}$", re.ASCII)
_PHASE_EVIDENCE_FIELDS = {
    "schema_version",
    "phase_id",
    "phase_sha256",
    "expected_record_count",
    "statistics_contract_sha256",
    "expected",
}
_EXPECTED_ENTRY_FIELDS = {
    "key",
    "identity_sha256",
    "scientific_contract_id",
    "scientific_contract_sha256",
}
_LOGICAL_KEY_FIELDS = {
    "phase_id",
    "acquisition_config_id",
    "method_config_id",
    "method_id",
    "target_id",
    "motion_id",
    "seed",
}
_STATISTICS_FIELDS = {
    "schema_version",
    "phase_id",
    "phase_sha256",
    "metric_version",
    "required_seeds",
    "comparisons",
    "n_bootstrap",
    "bootstrap_seed",
}
_COMPARISON_FIELDS = {
    "comparison_id",
    "method_id",
    "comparator_id",
    "metric",
    "method_config_id",
    "comparator_config_id",
}
_PUBLICATION_LOCK_FIELDS = {
    "schema_version",
    "contract_id",
    "required_phases",
    "required_total_records",
}
_PUBLICATION_PHASE_FIELDS = {"phase_id", "expected_record_count"}
_PUBLICATION_PHASE_COUNTS = (
    ("selection-replay-v1", 207),
    ("selection-stress-v1", 126),
    ("primary-selection-v1", 297),
    ("primary-confirmatory-v1", 198),
    ("supplement-grid-v1", 231),
    ("ood-v1", 198),
    ("failure-v1", 180),
)


@dataclass(frozen=True)
class PhaseEvidenceContract:
    phase_id: str
    phase_sha256: str
    expected_record_count: int
    statistics_contract_sha256: str
    expected_identities: Mapping[LogicalRunKey, str]
    expected_scientific_contracts: Mapping[LogicalRunKey, tuple[str, str]]
    canonical_sha256: str


@dataclass(frozen=True)
class StatisticsContract:
    phase_id: str
    phase_sha256: str
    metric_version: str
    required_seeds: tuple[int, ...]
    comparisons: tuple[PairedComparison, ...]
    n_bootstrap: int
    bootstrap_seed: int
    canonical_sha256: str


@dataclass(frozen=True)
class PublicationLockContract:
    contract_id: str
    required_phase_counts: Mapping[str, int]
    required_total_records: int
    canonical_sha256: str


def build_phase_evidence_contract(
    plan: PhasePlan,
    *,
    expected_identities: Mapping[LogicalRunKey, str],
    statistics_contract_sha256: str,
) -> dict[str, object]:
    """Build the exact identity allow-list for one already-materialized plan."""
    plan_keys = _logical_keys_from_plan(plan)
    scientific_contracts = _scientific_contracts_from_plan(plan)
    statistics_sha256 = _require_sha256(
        "statistics contract SHA-256", statistics_contract_sha256
    )
    if not isinstance(expected_identities, Mapping):
        raise TypeError("expected identities must be a mapping")
    pairs: list[tuple[LogicalRunKey, str]] = []
    for key, identity in expected_identities.items():
        if type(key) is not LogicalRunKey:
            raise TypeError("expected identity key must be a LogicalRunKey")
        pairs.append(
            (key, _require_sha256("expected run identity SHA-256", identity))
        )
    if len(pairs) != len(plan_keys) or {key for key, _ in pairs} != set(plan_keys):
        raise ValueError("expected identities do not exactly match the phase plan")
    identities = [identity for _key, identity in pairs]
    if len(identities) != len(set(identities)):
        raise ValueError("expected identities contain a duplicate identity")
    by_key = dict(pairs)
    document: dict[str, object] = {
        "schema_version": "phase-evidence-contract-v1",
        "phase_id": plan.phase_id,
        "phase_sha256": plan.phase_sha256,
        "expected_record_count": plan.expected_runs,
        "statistics_contract_sha256": statistics_sha256,
        "expected": [
            {
                "key": _logical_key_document(key),
                "identity_sha256": by_key[key],
                "scientific_contract_id": scientific_contracts[key][0],
                "scientific_contract_sha256": scientific_contracts[key][1],
            }
            for key in plan_keys
        ],
    }
    return validate_versioned_json(document, "phase-evidence-contract-v1")


def load_phase_evidence_contract(
    path: Path,
    *,
    expected_plan: PhasePlan,
    expected_statistics_contract_sha256: str,
) -> PhaseEvidenceContract:
    """Load phase evidence only after independently materializing its plan."""
    _logical_keys_from_plan(expected_plan)
    statistics_sha256 = _require_sha256(
        "expected statistics contract SHA-256",
        expected_statistics_contract_sha256,
    )
    document, payload = load_canonical_versioned_json(
        path,
        "phase-evidence-contract-v1",
        noun="phase evidence contract",
    )
    if set(document) != _PHASE_EVIDENCE_FIELDS:
        raise ValueError("phase evidence contract top-level shape is invalid")
    if document["schema_version"] != "phase-evidence-contract-v1":
        raise ValueError("phase evidence contract schema version is invalid")
    if document["phase_id"] != expected_plan.phase_id:
        raise ValueError("phase evidence contract phase ID disagrees with the plan")
    if document["phase_sha256"] != expected_plan.phase_sha256:
        raise ValueError("phase evidence contract phase hash disagrees with the plan")
    if document["expected_record_count"] != expected_plan.expected_runs:
        raise ValueError("phase evidence contract count disagrees with the plan")
    if document["statistics_contract_sha256"] != statistics_sha256:
        raise ValueError("phase evidence contract binds another statistics contract")
    expected = document["expected"]
    if type(expected) is not list:
        raise ValueError("phase evidence expected identities must be an exact array")
    pairs: list[tuple[LogicalRunKey, str]] = []
    scientific_contract_pairs: list[
        tuple[LogicalRunKey, tuple[str, str]]
    ] = []
    for entry in expected:
        if type(entry) is not dict or set(entry) != _EXPECTED_ENTRY_FIELDS:
            raise ValueError("phase evidence expected entry shape is invalid")
        key = _logical_key_from_document(entry["key"])
        pairs.append(
            (
                key,
                _require_sha256(
                    "phase evidence run identity SHA-256",
                    entry["identity_sha256"],
                ),
            )
        )
        scientific_contract_pairs.append(
            (
                key,
                (
                    _require_string(
                        "phase evidence scientific contract ID",
                        entry["scientific_contract_id"],
                    ),
                    _require_sha256(
                        "phase evidence scientific contract SHA-256",
                        entry["scientific_contract_sha256"],
                    ),
                ),
            )
        )
    keys = [key for key, _identity in pairs]
    identities = [identity for _key, identity in pairs]
    if keys != sorted(keys):
        raise ValueError("phase evidence entries are not canonically sorted")
    if len(keys) != len(set(keys)):
        raise ValueError("phase evidence contains a duplicate logical key")
    if len(identities) != len(set(identities)):
        raise ValueError("phase evidence contains a duplicate identity")
    scientific_contracts = dict(scientific_contract_pairs)
    if scientific_contracts != _scientific_contracts_from_plan(expected_plan):
        raise ValueError(
            "phase evidence scientific contracts disagree with the phase plan"
        )
    rebuilt = build_phase_evidence_contract(
        expected_plan,
        expected_identities=dict(pairs),
        statistics_contract_sha256=statistics_sha256,
    )
    if payload != canonical_json_bytes(rebuilt):
        raise ValueError("phase evidence contract disagrees with the phase plan")
    return PhaseEvidenceContract(
        phase_id=expected_plan.phase_id,
        phase_sha256=expected_plan.phase_sha256,
        expected_record_count=expected_plan.expected_runs,
        statistics_contract_sha256=statistics_sha256,
        expected_identities=MappingProxyType(dict(pairs)),
        expected_scientific_contracts=MappingProxyType(scientific_contracts),
        canonical_sha256=sha256_bytes(payload),
    )


def build_statistics_contract(
    plan: PhasePlan,
    *,
    source_protocol: Mapping[str, object],
) -> dict[str, object]:
    """Derive immutable aggregation controls from a plan and checked protocol."""
    plan_keys = _logical_keys_from_plan(plan)
    validate_protocol(source_protocol)
    metric_version = _require_string(
        "protocol metric_version", source_protocol.get("metric_version")
    )
    required_seeds = sorted({key.seed for key in plan_keys})
    comparisons: list[dict[str, object]] = []
    if plan.phase_id == "primary-confirmatory-v1":
        rule = _primary_confirmation_rule(plan, source_protocol)
        comparisons.append(
            {
                "comparison_id": rule["rule_id"],
                "method_id": rule["method"],
                "comparator_id": rule["comparator"],
                "metric": rule["metric"],
                "method_config_id": "default",
                "comparator_config_id": "default",
            }
        )
    else:
        _require_no_applicable_publication_comparison(plan, source_protocol)
    document: dict[str, object] = {
        "schema_version": "phase-statistics-contract-v1",
        "phase_id": plan.phase_id,
        "phase_sha256": plan.phase_sha256,
        "metric_version": metric_version,
        "required_seeds": required_seeds,
        "comparisons": comparisons,
        "n_bootstrap": 10_000,
        "bootstrap_seed": 20260727,
    }
    return validate_versioned_json(document, "phase-statistics-contract-v1")


def load_statistics_contract(
    path: Path,
    *,
    expected_plan: PhasePlan,
    source_protocol: Mapping[str, object],
) -> StatisticsContract:
    """Load aggregation controls by exact comparison with derived authority."""
    expected = build_statistics_contract(
        expected_plan,
        source_protocol=source_protocol,
    )
    document, payload = load_canonical_versioned_json(
        path,
        "phase-statistics-contract-v1",
        noun="statistics contract",
    )
    if set(document) != _STATISTICS_FIELDS:
        raise ValueError("statistics contract top-level shape is invalid")
    if document["schema_version"] != "phase-statistics-contract-v1":
        raise ValueError("statistics contract schema version is invalid")
    if payload != canonical_json_bytes(expected):
        raise ValueError("statistics contract disagrees with plan and protocol")
    comparisons_value = document["comparisons"]
    if type(comparisons_value) is not list:
        raise ValueError("statistics comparisons must be an exact array")
    comparisons: list[PairedComparison] = []
    for entry in comparisons_value:
        if type(entry) is not dict or set(entry) != _COMPARISON_FIELDS:
            raise ValueError("statistics comparison shape is invalid")
        comparisons.append(PairedComparison(**entry))
    seeds_value = document["required_seeds"]
    if type(seeds_value) is not list or any(type(seed) is not int for seed in seeds_value):
        raise ValueError("statistics required seeds must be exact integers")
    return StatisticsContract(
        phase_id=expected_plan.phase_id,
        phase_sha256=expected_plan.phase_sha256,
        metric_version=str(document["metric_version"]),
        required_seeds=tuple(seeds_value),
        comparisons=tuple(comparisons),
        n_bootstrap=int(document["n_bootstrap"]),
        bootstrap_seed=int(document["bootstrap_seed"]),
        canonical_sha256=sha256_bytes(payload),
    )


def build_publication_lock_contract_v1() -> dict[str, object]:
    """Return the repository's single publication-results membership contract."""
    document: dict[str, object] = {
        "schema_version": "publication-lock-contract-v1",
        "contract_id": "gsdiff-publication-results-v1",
        "required_phases": [
            {"phase_id": phase_id, "expected_record_count": count}
            for phase_id, count in _PUBLICATION_PHASE_COUNTS
        ],
        "required_total_records": sum(
            count for _phase_id, count in _PUBLICATION_PHASE_COUNTS
        ),
    }
    return validate_versioned_json(document, "publication-lock-contract-v1")


def load_publication_lock_contract(path: Path) -> PublicationLockContract:
    """Load only the exact generated publication-results contract."""
    document, payload = load_canonical_versioned_json(
        path,
        "publication-lock-contract-v1",
        noun="publication lock contract",
    )
    if set(document) != _PUBLICATION_LOCK_FIELDS:
        raise ValueError("publication lock contract top-level shape is invalid")
    if document["schema_version"] != "publication-lock-contract-v1":
        raise ValueError("publication lock contract schema version is invalid")
    phases = document["required_phases"]
    if type(phases) is not list or any(
        type(entry) is not dict or set(entry) != _PUBLICATION_PHASE_FIELDS
        for entry in phases
    ):
        raise ValueError("publication lock phase shape is invalid")
    expected = build_publication_lock_contract_v1()
    if payload != canonical_json_bytes(expected):
        raise ValueError("publication lock contract is not the repository contract")
    return PublicationLockContract(
        contract_id="gsdiff-publication-results-v1",
        required_phase_counts=MappingProxyType(dict(_PUBLICATION_PHASE_COUNTS)),
        required_total_records=1437,
        canonical_sha256=sha256_bytes(payload),
    )


def load_results_lock(
    path: Path,
    *,
    publication_contract_path: Path,
) -> dict[str, object]:
    """Load a canonical results lock only for the fixed publication scope."""
    publication_contract = load_publication_lock_contract(
        publication_contract_path
    )
    document, _payload = load_canonical_versioned_json(
        path,
        "results-lock-v1",
        noun="results lock",
    )
    if document.get("publication_lock_contract_sha256") != (
        publication_contract.canonical_sha256
    ):
        raise ValueError("results lock binds another publication contract")
    phases = document.get("phases")
    if type(phases) is not list:
        raise ValueError("results lock phases must be an exact array")
    observed: list[tuple[str, int]] = []
    for phase in phases:
        if type(phase) is not dict:
            raise ValueError("results lock phase must be an exact object")
        phase_id = phase.get("phase_id")
        record_count = phase.get("record_count")
        if type(phase_id) is not str or type(record_count) is not int:
            raise ValueError("results lock phase identity is invalid")
        observed.append((phase_id, record_count))
    expected = list(publication_contract.required_phase_counts.items())
    if observed != expected:
        raise ValueError("results lock phases disagree with publication scope")
    if document.get("total_records") != (
        publication_contract.required_total_records
    ):
        raise ValueError("results lock total disagrees with publication scope")
    if sum(count for _phase_id, count in observed) != (
        publication_contract.required_total_records
    ):
        raise ValueError("results lock phase counts do not sum to the total")
    return document


def _logical_keys_from_plan(plan: PhasePlan) -> tuple[LogicalRunKey, ...]:
    if type(plan) is not PhasePlan:
        raise TypeError("phase plan must be an exact PhasePlan")
    _require_string("phase plan ID", plan.phase_id)
    _require_sha256("phase plan SHA-256", plan.phase_sha256)
    if type(plan.expected_runs) is not int or plan.expected_runs <= 0:
        raise ValueError("phase plan expected runs must be a positive exact integer")
    if type(plan.cells) is not tuple or len(plan.cells) != plan.expected_runs:
        raise ValueError("phase plan cells disagree with expected runs")
    if type(plan.expected_datasets) is not int or plan.expected_datasets <= 0:
        raise ValueError(
            "phase plan expected datasets must be a positive exact integer"
        )
    if (
        type(plan.dataset_plans) is not tuple
        or len(plan.dataset_plans) != plan.expected_datasets
    ):
        raise ValueError("phase plan dataset plans disagree with expected datasets")
    for cell in plan.cells:
        if type(cell) is not ExperimentCell:
            raise TypeError("phase plan cell must be an exact ExperimentCell")
        _require_string(
            "phase plan cell scientific contract ID",
            cell.scientific_contract_id,
        )
        _require_sha256(
            "phase plan cell scientific contract SHA-256",
            cell.scientific_contract_sha256,
        )
    for dataset_plan in plan.dataset_plans:
        if type(dataset_plan) is not DatasetPlanKey:
            raise TypeError(
                "phase plan dataset plan must be an exact DatasetPlanKey"
            )
        _require_string(
            "phase plan dataset scientific contract ID",
            dataset_plan.scientific_contract_id,
        )
        _require_sha256(
            "phase plan dataset scientific contract SHA-256",
            dataset_plan.scientific_contract_sha256,
        )
    if len(set(plan.dataset_plans)) != plan.expected_datasets:
        raise ValueError("phase plan contains duplicate dataset plans")
    covered_dataset_plans = {
        DatasetPlanKey(
            scientific_contract_id=cell.scientific_contract_id,
            scientific_contract_sha256=cell.scientific_contract_sha256,
            acquisition_config_id=cell.acquisition_config_id,
            target=cell.target,
            motion=cell.motion,
            seed=cell.seed,
        )
        for cell in plan.cells
    }
    if covered_dataset_plans != set(plan.dataset_plans):
        raise ValueError("phase plan cells do not exactly cover dataset plans")
    keys = tuple(
        sorted(
            LogicalRunKey(
                phase_id=plan.phase_id,
                acquisition_config_id=cell.acquisition_config_id,
                method_config_id=cell.method_config_id,
                method_id=cell.method,
                target_id=cell.target,
                motion_id=cell.motion,
                seed=cell.seed,
            )
            for cell in plan.cells
        )
    )
    if len(keys) != len(set(keys)):
        raise ValueError("phase plan contains duplicate logical keys")
    return keys


def _scientific_contracts_from_plan(
    plan: PhasePlan,
) -> dict[LogicalRunKey, tuple[str, str]]:
    _logical_keys_from_plan(plan)
    contracts = {
        LogicalRunKey(
            phase_id=plan.phase_id,
            acquisition_config_id=cell.acquisition_config_id,
            method_config_id=cell.method_config_id,
            method_id=cell.method,
            target_id=cell.target,
            motion_id=cell.motion,
            seed=cell.seed,
        ): (
            cell.scientific_contract_id,
            cell.scientific_contract_sha256,
        )
        for cell in plan.cells
    }
    if len(contracts) != plan.expected_runs:
        raise ValueError("phase plan scientific contracts are not key-complete")
    return contracts


def _primary_confirmation_rule(
    plan: PhasePlan,
    protocol: Mapping[str, object],
) -> Mapping[str, object]:
    if protocol.get("campaign_id") != "primary-v1":
        raise ValueError("confirmatory statistics require primary-v1")
    rule = protocol.get("confirmation_rule")
    if not isinstance(rule, Mapping):
        raise ValueError("primary confirmation rule is missing")
    required = {
        "rule_id",
        "scientific_contract_id",
        "method",
        "comparator",
        "metric",
        "targets",
        "motions",
        "seeds",
        "minimum_effect_db",
        "require_each_seed_mean_delta_strictly_greater_than_minimum",
        "require_complete_finite_pairs_per_seed",
        "require_method_failure_count_not_greater_than_comparator",
        "inferential_significance_claim",
    }
    if set(rule) != required:
        raise ValueError("primary confirmation rule shape is invalid")
    for field in ("rule_id", "scientific_contract_id", "method", "comparator", "metric"):
        _require_string(f"confirmation rule {field}", rule[field])
    cells = plan.cells
    if {cell.seed for cell in cells} != set(rule["seeds"]):
        raise ValueError("confirmatory plan seeds disagree with confirmation rule")
    if {cell.target for cell in cells} != set(rule["targets"]):
        raise ValueError("confirmatory plan targets disagree with confirmation rule")
    if {cell.motion for cell in cells} != set(rule["motions"]):
        raise ValueError("confirmatory plan motions disagree with confirmation rule")
    if {cell.scientific_contract_id for cell in cells} != {
        rule["scientific_contract_id"]
    }:
        raise ValueError("confirmatory plan scientific contract disagrees with rule")
    required_method_cells = {
        (target, motion, seed, method, "default")
        for target in rule["targets"]
        for motion in rule["motions"]
        for seed in rule["seeds"]
        for method in (rule["method"], rule["comparator"])
    }
    actual_method_cells = {
        (cell.target, cell.motion, cell.seed, cell.method, cell.method_config_id)
        for cell in cells
        if cell.method in {rule["method"], rule["comparator"]}
    }
    if actual_method_cells != required_method_cells:
        raise ValueError("confirmatory plan does not cover the declared comparison")
    return rule


def _require_no_applicable_publication_comparison(
    plan: PhasePlan,
    protocol: Mapping[str, object],
) -> None:
    """Allow an empty comparison list only when no declaration covers the plan."""
    rule = protocol.get("confirmation_rule")
    if rule is None:
        return
    if not isinstance(rule, Mapping):
        raise ValueError("publication comparison declaration is malformed")
    required_fields = {"method", "comparator", "targets", "motions", "seeds"}
    if not required_fields.issubset(rule):
        raise ValueError("publication comparison declaration is incomplete")
    declared_cells = {
        (target, motion, seed, method, "default")
        for target in rule["targets"]
        for motion in rule["motions"]
        for seed in rule["seeds"]
        for method in (rule["method"], rule["comparator"])
    }
    actual_cells = {
        (cell.target, cell.motion, cell.seed, cell.method, cell.method_config_id)
        for cell in plan.cells
    }
    if declared_cells.issubset(actual_cells):
        raise ValueError(
            "phase has an applicable publication comparison declaration"
        )


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


def _logical_key_from_document(value: object) -> LogicalRunKey:
    if type(value) is not dict or set(value) != _LOGICAL_KEY_FIELDS:
        raise ValueError("phase evidence logical key shape is invalid")
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
        raise ValueError("phase evidence logical key is invalid") from error


def _require_string(noun: str, value: object) -> str:
    if type(value) is not str or not value:
        raise TypeError(f"{noun} must be a nonempty exact string")
    return value


def _require_sha256(noun: str, value: object) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{noun} must be a lowercase SHA-256")
    return value


__all__ = [
    "PhaseEvidenceContract",
    "PublicationLockContract",
    "StatisticsContract",
    "build_phase_evidence_contract",
    "build_publication_lock_contract_v1",
    "build_statistics_contract",
    "load_phase_evidence_contract",
    "load_publication_lock_contract",
    "load_results_lock",
    "load_statistics_contract",
]
