from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import re
from types import MappingProxyType

from .identity import canonical_json_bytes, sha256_bytes
from .protocol import ExperimentCell, expand_cells, validate_protocol


_SHA256 = re.compile(r"^[0-9a-f]{64}$", flags=re.ASCII)
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$", flags=re.ASCII)
_ABLATION_CAMPAIGN_ROLE = "ablations-v1"

_PHASE_IDS = (
    "selection-decision-v1",
    "selection-replay-v1",
    "selection-stress-v1",
    "primary-selection-v1",
    "primary-confirmatory-v1",
)
_EXPECTED_COUNTS = {
    "selection-decision-v1": (207, 9),
    "selection-replay-v1": (207, 9),
    "selection-stress-v1": (126, 108),
    "primary-selection-v1": (297, 27),
    "primary-confirmatory-v1": (198, 18),
}
_REQUIRED_CONFIRMATORY_PHASES = (
    ("selection-replay-v1", 207),
    ("selection-stress-v1", 126),
    ("primary-selection-v1", 297),
)
_VERIFIED_PHASE_COMPLETION_FACTORY_TOKEN = object()
_CONFIRMATORY_GATE_FACTORY_TOKEN = object()

_SELECTION_CONFIG_IDS = (
    "ablation-selection-anchor-v1",
    "ablation-representation-recinr-se2-v1",
    "ablation-representation-siren-v1",
    "ablation-representation-grid-v1",
    "ablation-solver-sgd-v1",
    "ablation-solver-hqs-v1",
    "ablation-prior-tv2d-v1",
    "ablation-prior-diffusion-v1",
    "ablation-warmup-0-v1",
    "ablation-warmup-0p1-v1",
    "ablation-warmup-0p4-v1",
    "ablation-temporal-tv-0-v1",
    "ablation-temporal-tv-0p05-v1",
    "ablation-temporal-tv-0p3-v1",
    "ablation-gaussian-count-250-v1",
    "ablation-gaussian-count-500-v1",
    "ablation-gaussian-count-1500-v1",
    "ablation-j1-v1",
    "ablation-j2-v1",
    "ablation-j3-v1",
    "ablation-j4-v1",
    "ablation-j5-v1",
    "ablation-j6-v1",
)
_DIFFUSION_SELECTION_CONFIG_IDS = frozenset(
    {
        "ablation-prior-diffusion-v1",
        "ablation-j1-v1",
        "ablation-j2-v1",
    }
)
_STRESS_ACQUISITION_IDS = (
    "stress-anchor-v1",
    "stress-train-measurements-640-v1",
    "stress-train-measurements-1280-v1",
    "stress-train-measurements-1920-v1",
    "stress-train-measurements-3840-v1",
    "stress-train-measurements-5120-v1",
    "stress-snr-db-15-v1",
    "stress-snr-db-20-v1",
    "stress-snr-db-30-v1",
    "stress-pattern-gaussian-v1",
    "stress-pattern-hadamard-natural-v1",
    "stress-pattern-fourier-v1",
)
_STRESS_MOTION_CONFIG_IDS = (
    "stress-motion-fit-translation-only-v1",
    "stress-motion-fit-rotation-only-v1",
)


@dataclass(frozen=True, order=True)
class DatasetPlanKey:
    scientific_contract_id: str
    scientific_contract_sha256: str
    acquisition_config_id: str
    target: str
    motion: str
    seed: int


@dataclass(frozen=True)
class PhasePlan:
    phase_id: str
    phase_sha256: str
    cells: tuple[ExperimentCell, ...]
    dataset_plans: tuple[DatasetPlanKey, ...]
    expected_runs: int
    expected_datasets: int


@dataclass(frozen=True)
class FrozenSelection:
    method_id: str
    method_config_id: str
    selection_record_sha256: str


@dataclass(frozen=True, init=False)
class VerifiedPhaseCompletion:
    phase_id: str
    phase_sha256: str
    complete_count: int
    publication_experiment_commit: str
    aggregate_sha256: str
    _factory_token: object = field(init=False, repr=False, compare=False)

    def __init__(
        self,
        phase_id: str,
        phase_sha256: str,
        complete_count: int,
        publication_experiment_commit: str,
        aggregate_sha256: str,
        *,
        _factory_token: object | None = None,
    ) -> None:
        if _factory_token is not _VERIFIED_PHASE_COMPLETION_FACTORY_TOKEN:
            raise ValueError("verified phase completion is factory-only")
        object.__setattr__(self, "phase_id", phase_id)
        object.__setattr__(self, "phase_sha256", phase_sha256)
        object.__setattr__(self, "complete_count", complete_count)
        object.__setattr__(
            self,
            "publication_experiment_commit",
            publication_experiment_commit,
        )
        object.__setattr__(self, "aggregate_sha256", aggregate_sha256)
        object.__setattr__(
            self,
            "_factory_token",
            _VERIFIED_PHASE_COMPLETION_FACTORY_TOKEN,
        )


def _verified_phase_completion_from_verified_aggregate_claims(
    *,
    phase_id: str,
    phase_sha256: str,
    complete_count: int,
    publication_experiment_commit: str,
    aggregate_sha256: str,
) -> VerifiedPhaseCompletion:
    """Mint completion claims only for the physical aggregate verifier."""
    if type(phase_id) is not str or phase_id not in _EXPECTED_COUNTS:
        raise ValueError("verified aggregate phase ID is not declared")
    expected_count = _EXPECTED_COUNTS[phase_id][0]
    if type(complete_count) is not int or complete_count != expected_count:
        raise ValueError("verified aggregate count does not match the phase")
    normalized_phase_sha256 = _require_sha256(
        "verified aggregate phase SHA-256",
        phase_sha256,
    )
    normalized_commit = _require_git_commit(
        "verified aggregate publication commit",
        publication_experiment_commit,
    )
    normalized_aggregate_sha256 = _require_sha256(
        "verified canonical aggregate SHA-256",
        aggregate_sha256,
    )
    return VerifiedPhaseCompletion(
        phase_id=phase_id,
        phase_sha256=normalized_phase_sha256,
        complete_count=complete_count,
        publication_experiment_commit=normalized_commit,
        aggregate_sha256=normalized_aggregate_sha256,
        _factory_token=_VERIFIED_PHASE_COMPLETION_FACTORY_TOKEN,
    )


@dataclass(frozen=True, init=False)
class ConfirmatoryGateEvidence:
    prerequisite_sha256: str
    publication_experiment_commit: str
    verified_phase_counts: Mapping[str, int]
    verified_phase_sha256s: Mapping[str, str]
    verified_aggregate_sha256s: Mapping[str, str]
    _factory_token: object = field(init=False, repr=False, compare=False)

    def __init__(
        self,
        prerequisite_sha256: str,
        publication_experiment_commit: str,
        verified_phase_counts: Mapping[str, int],
        verified_phase_sha256s: Mapping[str, str] | None = None,
        verified_aggregate_sha256s: Mapping[str, str] | None = None,
        *,
        _factory_token: object | None = None,
    ) -> None:
        if _factory_token is not _CONFIRMATORY_GATE_FACTORY_TOKEN:
            raise ValueError(
                "confirmatory gate evidence is factory-only; "
                "use verify_confirmatory_prerequisites"
            )
        if verified_phase_sha256s is None:
            raise ValueError("confirmatory gate phase hashes are required")
        if verified_aggregate_sha256s is None:
            raise ValueError("confirmatory gate aggregate hashes are required")
        object.__setattr__(self, "prerequisite_sha256", prerequisite_sha256)
        object.__setattr__(
            self,
            "publication_experiment_commit",
            publication_experiment_commit,
        )
        object.__setattr__(
            self,
            "verified_phase_counts",
            MappingProxyType(dict(verified_phase_counts)),
        )
        object.__setattr__(
            self,
            "verified_phase_sha256s",
            MappingProxyType(dict(verified_phase_sha256s)),
        )
        object.__setattr__(
            self,
            "verified_aggregate_sha256s",
            MappingProxyType(dict(verified_aggregate_sha256s)),
        )
        object.__setattr__(
            self,
            "_factory_token",
            _CONFIRMATORY_GATE_FACTORY_TOKEN,
        )


def _confirmatory_prerequisite(
    *,
    publication_experiment_commit: str,
    phase_counts: Mapping[str, int],
    phase_sha256s: Mapping[str, str],
    aggregate_sha256s: Mapping[str, str],
) -> dict[str, object]:
    phase_records = [
        {
            "phase_id": phase_id,
            "phase_sha256": phase_sha256s[phase_id],
            "aggregate_sha256": aggregate_sha256s[phase_id],
            "complete_count": phase_counts[phase_id],
        }
        for phase_id, _ in _REQUIRED_CONFIRMATORY_PHASES
    ]
    return {
        "schema_version": "confirmatory-prerequisite-v1",
        "publication_experiment_commit": publication_experiment_commit,
        "required_total_runs": sum(
            phase_counts[phase_id] for phase_id, _ in _REQUIRED_CONFIRMATORY_PHASES
        ),
        "required_phases": phase_records,
    }


def verify_confirmatory_prerequisites(
    completions: Sequence[VerifiedPhaseCompletion],
) -> ConfirmatoryGateEvidence:
    """Verify the exact 207 + 126 + 297 prerequisite set."""
    if isinstance(completions, (str, bytes)) or not isinstance(
        completions, Sequence
    ):
        raise TypeError("phase completions must be a sequence")

    by_phase: dict[str, VerifiedPhaseCompletion] = {}
    for completion in completions:
        if type(completion) is not VerifiedPhaseCompletion:
            raise TypeError("phase completion has an invalid type")
        if completion.phase_id in by_phase:
            raise ValueError("phase completion IDs must be unique")
        by_phase[completion.phase_id] = completion

    required_counts = dict(_REQUIRED_CONFIRMATORY_PHASES)
    if set(by_phase) != set(required_counts):
        raise ValueError("confirmatory gate requires the exact required phases")

    commits: set[str] = set()
    phase_sha256s: dict[str, str] = {}
    aggregate_sha256s: dict[str, str] = {}
    for phase_id, expected_count in _REQUIRED_CONFIRMATORY_PHASES:
        completion = by_phase[phase_id]
        if (
            getattr(completion, "_factory_token", None)
            is not _VERIFIED_PHASE_COMPLETION_FACTORY_TOKEN
        ):
            raise ValueError("phase completion is not verifier-produced")
        phase_sha256s[phase_id] = _require_sha256(
            "phase SHA-256", completion.phase_sha256
        )
        aggregate_sha256s[phase_id] = _require_sha256(
            "canonical aggregate SHA-256",
            completion.aggregate_sha256,
        )
        if type(completion.complete_count) is not int:
            raise ValueError("phase complete count must be an exact integer")
        if completion.complete_count != expected_count:
            raise ValueError("phase complete count does not match the contract")
        _require_git_commit(
            "publication experiment commit",
            completion.publication_experiment_commit,
        )
        commits.add(completion.publication_experiment_commit)

    if len(commits) != 1:
        raise ValueError("all phases must use the same publication experiment commit")
    publication_commit = next(iter(commits))
    prerequisite = _confirmatory_prerequisite(
        publication_experiment_commit=publication_commit,
        phase_counts=required_counts,
        phase_sha256s=phase_sha256s,
        aggregate_sha256s=aggregate_sha256s,
    )
    return ConfirmatoryGateEvidence(
        prerequisite_sha256=sha256_bytes(canonical_json_bytes(prerequisite)),
        publication_experiment_commit=publication_commit,
        verified_phase_counts=required_counts,
        verified_phase_sha256s=phase_sha256s,
        verified_aggregate_sha256s=aggregate_sha256s,
        _factory_token=_CONFIRMATORY_GATE_FACTORY_TOKEN,
    )


def materialize_phase(
    phase_id: str,
    *,
    ablation_protocol: Mapping[str, object],
    primary_protocol: Mapping[str, object],
    frozen_selection: FrozenSelection | None = None,
    gate: ConfirmatoryGateEvidence | None = None,
) -> PhasePlan:
    """Materialize one exact protocol-v1 scheduling phase.

    Phase metadata remains outside ``ExperimentCell`` scientific fields.  The
    confirmatory gate is checked before either protocol is inspected.
    """
    if type(phase_id) is not str or phase_id not in _PHASE_IDS:
        raise ValueError("unknown phase ID")
    confirmatory_prerequisite_sha256: str | None = None
    if phase_id == "primary-confirmatory-v1":
        verified_gate = _require_confirmatory_gate(gate)
        confirmatory_prerequisite_sha256 = verified_gate.prerequisite_sha256

    if phase_id in {"selection-decision-v1", "selection-replay-v1"}:
        cells, dataset_plans, source_protocol_sha256 = _selection_membership(
            ablation_protocol
        )
        selection = None
    elif phase_id == "selection-stress-v1":
        selection = _require_frozen_selection(frozen_selection)
        cells, dataset_plans, source_protocol_sha256 = _stress_membership(
            ablation_protocol,
            selection,
        )
    else:
        selection = None
        required_seeds = (
            frozenset({7, 11, 42})
            if phase_id == "primary-selection-v1"
            else frozenset({73, 101})
        )
        cells, dataset_plans, source_protocol_sha256 = _primary_membership(
            primary_protocol,
            required_seeds=required_seeds,
        )

    expected_runs, expected_datasets = _EXPECTED_COUNTS[phase_id]
    if len(cells) != expected_runs or len(set(cells)) != expected_runs:
        raise ValueError("phase run membership count drift")
    if (
        len(dataset_plans) != expected_datasets
        or len(set(dataset_plans)) != expected_datasets
    ):
        raise ValueError("phase dataset membership count drift")

    phase_sha256 = _phase_sha256(
        phase_id=phase_id,
        source_protocol_sha256=source_protocol_sha256,
        cells=cells,
        dataset_plans=dataset_plans,
        frozen_selection=selection,
        confirmatory_prerequisite_sha256=confirmatory_prerequisite_sha256,
    )
    return PhasePlan(
        phase_id=phase_id,
        phase_sha256=phase_sha256,
        cells=cells,
        dataset_plans=dataset_plans,
        expected_runs=expected_runs,
        expected_datasets=expected_datasets,
    )


def _selection_membership(
    protocol: Mapping[str, object],
) -> tuple[tuple[ExperimentCell, ...], tuple[DatasetPlanKey, ...], str]:
    contract_id, contract_sha256, protocol_sha256, anchors, seeds = (
        _ablation_fields(protocol)
    )
    cells = tuple(
        ExperimentCell(
            scientific_contract_id=contract_id,
            scientific_contract_sha256=contract_sha256,
            campaign_id=_ABLATION_CAMPAIGN_ROLE,
            target=target,
            motion=motion,
            seed=seed,
            method=(
                "gsdiff_diffusion"
                if config_id in _DIFFUSION_SELECTION_CONFIG_IDS
                else "gsdiff_tv"
            ),
            acquisition_config_id="base",
            method_config_id=config_id,
        )
        for target, motion in anchors
        for seed in seeds
        for config_id in _SELECTION_CONFIG_IDS
    )
    dataset_plans = tuple(
        DatasetPlanKey(
            scientific_contract_id=contract_id,
            scientific_contract_sha256=contract_sha256,
            acquisition_config_id="base",
            target=target,
            motion=motion,
            seed=seed,
        )
        for target, motion in anchors
        for seed in seeds
    )
    return cells, dataset_plans, protocol_sha256


def _stress_membership(
    protocol: Mapping[str, object],
    selection: FrozenSelection,
) -> tuple[tuple[ExperimentCell, ...], tuple[DatasetPlanKey, ...], str]:
    contract_id, contract_sha256, protocol_sha256, anchors, seeds = (
        _ablation_fields(protocol)
    )
    cells: list[ExperimentCell] = []
    for target, motion in anchors:
        for seed in seeds:
            cells.extend(
                ExperimentCell(
                    scientific_contract_id=contract_id,
                    scientific_contract_sha256=contract_sha256,
                    campaign_id=_ABLATION_CAMPAIGN_ROLE,
                    target=target,
                    motion=motion,
                    seed=seed,
                    method=selection.method_id,
                    acquisition_config_id=acquisition_id,
                    method_config_id=selection.method_config_id,
                )
                for acquisition_id in _STRESS_ACQUISITION_IDS
            )
            cells.extend(
                ExperimentCell(
                    scientific_contract_id=contract_id,
                    scientific_contract_sha256=contract_sha256,
                    campaign_id=_ABLATION_CAMPAIGN_ROLE,
                    target=target,
                    motion=motion,
                    seed=seed,
                    method=selection.method_id,
                    acquisition_config_id="stress-anchor-v1",
                    method_config_id=method_config_id,
                )
                for method_config_id in _STRESS_MOTION_CONFIG_IDS
            )
    dataset_plans = tuple(
        DatasetPlanKey(
            scientific_contract_id=contract_id,
            scientific_contract_sha256=contract_sha256,
            acquisition_config_id=acquisition_id,
            target=target,
            motion=motion,
            seed=seed,
        )
        for target, motion in anchors
        for seed in seeds
        for acquisition_id in _STRESS_ACQUISITION_IDS
    )
    return tuple(cells), dataset_plans, protocol_sha256


def _primary_membership(
    protocol: Mapping[str, object],
    *,
    required_seeds: frozenset[int],
) -> tuple[tuple[ExperimentCell, ...], tuple[DatasetPlanKey, ...], str]:
    validate_protocol(protocol)
    if (
        protocol.get("document_kind") != "campaign"
        or protocol.get("campaign_id") != "primary-v1"
    ):
        raise ValueError("primary phases require the locked primary-v1 campaign")
    protocol_sha256 = _require_sha256(
        "primary protocol SHA-256", protocol.get("protocol_sha256")
    )
    cells = tuple(
        cell for cell in expand_cells(protocol) if cell.seed in required_seeds
    )
    seen: set[DatasetPlanKey] = set()
    dataset_plans: list[DatasetPlanKey] = []
    for cell in cells:
        key = DatasetPlanKey(
            scientific_contract_id=cell.scientific_contract_id,
            scientific_contract_sha256=cell.scientific_contract_sha256,
            acquisition_config_id=cell.acquisition_config_id,
            target=cell.target,
            motion=cell.motion,
            seed=cell.seed,
        )
        if key not in seen:
            seen.add(key)
            dataset_plans.append(key)
    return cells, tuple(dataset_plans), protocol_sha256


def _ablation_fields(
    protocol: Mapping[str, object],
) -> tuple[str, str, str, tuple[tuple[str, str], ...], tuple[int, ...]]:
    validate_protocol(protocol)
    if protocol.get("document_kind") != "ablation":
        raise ValueError("selection phases require the locked ablation protocol")
    contract_id = _require_string(
        "ablation scientific contract ID",
        protocol.get("scientific_contract_id"),
    )
    contract_sha256 = _require_sha256(
        "ablation scientific contract SHA-256",
        protocol.get("scientific_contract_sha256"),
    )
    protocol_sha256 = _require_sha256(
        "ablation protocol SHA-256",
        protocol.get("protocol_sha256"),
    )
    raw_anchors = protocol.get("selection_anchors")
    raw_seeds = protocol.get("selection_seeds")
    if not isinstance(raw_anchors, list) or not isinstance(raw_seeds, list):
        raise ValueError("ablation anchors and seeds are invalid")
    anchors = tuple(
        (
            _require_string("selection target", anchor.get("target")),
            _require_string("selection motion", anchor.get("motion")),
        )
        for anchor in raw_anchors
        if isinstance(anchor, Mapping)
    )
    if len(anchors) != len(raw_anchors):
        raise ValueError("ablation selection anchor is invalid")
    if any(type(seed) is not int for seed in raw_seeds):
        raise ValueError("ablation selection seed is invalid")
    seeds = tuple(raw_seeds)
    return contract_id, contract_sha256, protocol_sha256, anchors, seeds


def _require_frozen_selection(
    selection: FrozenSelection | None,
) -> FrozenSelection:
    if type(selection) is not FrozenSelection:
        raise ValueError("stress phase requires an explicit frozen selection")
    _require_string("frozen selection method ID", selection.method_id)
    _require_string(
        "frozen selection method config ID", selection.method_config_id
    )
    _require_sha256(
        "frozen selection record SHA-256",
        selection.selection_record_sha256,
    )
    if selection.method_config_id not in _SELECTION_CONFIG_IDS:
        raise ValueError("frozen selection is outside the declared candidates")
    expected_method = (
        "gsdiff_diffusion"
        if selection.method_config_id in _DIFFUSION_SELECTION_CONFIG_IDS
        else "gsdiff_tv"
    )
    if selection.method_id != expected_method:
        raise ValueError("frozen selection method and config disagree")
    return selection


def _require_confirmatory_gate(
    gate: ConfirmatoryGateEvidence | None,
) -> ConfirmatoryGateEvidence:
    if (
        type(gate) is not ConfirmatoryGateEvidence
        or getattr(gate, "_factory_token", None)
        is not _CONFIRMATORY_GATE_FACTORY_TOKEN
    ):
        raise ValueError("a valid confirmatory gate is required")
    prerequisite_sha256 = _require_sha256(
        "confirmatory gate SHA-256", gate.prerequisite_sha256
    )
    publication_commit = _require_git_commit(
        "confirmatory gate publication commit",
        gate.publication_experiment_commit,
    )
    if not isinstance(gate.verified_phase_counts, Mapping):
        raise ValueError("confirmatory gate phase counts are invalid")
    if not isinstance(gate.verified_phase_sha256s, Mapping):
        raise ValueError("confirmatory gate phase hashes are invalid")
    if not isinstance(gate.verified_aggregate_sha256s, Mapping):
        raise ValueError("confirmatory gate aggregate hashes are invalid")

    expected_counts = dict(_REQUIRED_CONFIRMATORY_PHASES)
    expected_phase_ids = set(expected_counts)
    if set(gate.verified_phase_counts) != expected_phase_ids:
        raise ValueError("confirmatory gate phase counts are invalid")
    if set(gate.verified_phase_sha256s) != expected_phase_ids:
        raise ValueError("confirmatory gate phase hashes are invalid")
    if set(gate.verified_aggregate_sha256s) != expected_phase_ids:
        raise ValueError("confirmatory gate aggregate hashes are invalid")

    phase_counts: dict[str, int] = {}
    phase_sha256s: dict[str, str] = {}
    aggregate_sha256s: dict[str, str] = {}
    for phase_id, expected_count in _REQUIRED_CONFIRMATORY_PHASES:
        count = gate.verified_phase_counts[phase_id]
        if type(count) is not int or count != expected_count:
            raise ValueError("confirmatory gate phase counts are invalid")
        phase_counts[phase_id] = count
        phase_sha256s[phase_id] = _require_sha256(
            "confirmatory gate phase SHA-256",
            gate.verified_phase_sha256s[phase_id],
        )
        aggregate_sha256s[phase_id] = _require_sha256(
            "confirmatory gate aggregate SHA-256",
            gate.verified_aggregate_sha256s[phase_id],
        )

    prerequisite = _confirmatory_prerequisite(
        publication_experiment_commit=publication_commit,
        phase_counts=phase_counts,
        phase_sha256s=phase_sha256s,
        aggregate_sha256s=aggregate_sha256s,
    )
    expected_sha256 = sha256_bytes(canonical_json_bytes(prerequisite))
    if prerequisite_sha256 != expected_sha256:
        raise ValueError("confirmatory gate SHA-256 does not match its evidence")
    return gate


def _phase_sha256(
    *,
    phase_id: str,
    source_protocol_sha256: str,
    cells: tuple[ExperimentCell, ...],
    dataset_plans: tuple[DatasetPlanKey, ...],
    frozen_selection: FrozenSelection | None,
    confirmatory_prerequisite_sha256: str | None,
) -> str:
    value: dict[str, object] = {
        "schema_version": "experiment-phase-membership-v1",
        "phase_id": phase_id,
        "source_protocol_sha256": source_protocol_sha256,
        "expected_runs": len(cells),
        "expected_datasets": len(dataset_plans),
        "cells": [_cell_record(cell) for cell in cells],
        "dataset_plans": [_dataset_record(key) for key in dataset_plans],
    }
    if frozen_selection is not None:
        value["frozen_selection"] = {
            "method_id": frozen_selection.method_id,
            "method_config_id": frozen_selection.method_config_id,
            "selection_record_sha256": frozen_selection.selection_record_sha256,
        }
    if confirmatory_prerequisite_sha256 is not None:
        value["confirmatory_prerequisite_sha256"] = _require_sha256(
            "confirmatory prerequisite SHA-256",
            confirmatory_prerequisite_sha256,
        )
    return sha256_bytes(canonical_json_bytes(value))


def _cell_record(cell: ExperimentCell) -> dict[str, object]:
    return {
        "scientific_contract_id": cell.scientific_contract_id,
        "scientific_contract_sha256": cell.scientific_contract_sha256,
        "campaign_id": cell.campaign_id,
        "target": cell.target,
        "motion": cell.motion,
        "seed": cell.seed,
        "method": cell.method,
        "acquisition_config_id": cell.acquisition_config_id,
        "method_config_id": cell.method_config_id,
    }


def _dataset_record(key: DatasetPlanKey) -> dict[str, object]:
    return {
        "scientific_contract_id": key.scientific_contract_id,
        "scientific_contract_sha256": key.scientific_contract_sha256,
        "acquisition_config_id": key.acquisition_config_id,
        "target": key.target,
        "motion": key.motion,
        "seed": key.seed,
    }


def _require_string(name: str, value: object) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty exact string")
    return value


def _require_sha256(name: str, value: object) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _require_git_commit(name: str, value: object) -> str:
    if type(value) is not str or _GIT_COMMIT.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase 40-hex commit")
    return value
