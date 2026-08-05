from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from gsdiff.experiments.phases import (
    ConfirmatoryGateEvidence,
    DatasetPlanKey,
    FrozenSelection,
    PhasePlan,
    VerifiedPhaseCompletion,
    materialize_phase,
    _verified_phase_completion_from_verified_aggregate_claims,
    verify_confirmatory_prerequisites,
)
from gsdiff.experiments.protocol import expand_cells, load_protocol


verified_phase_completion_from_canonical_aggregate = (
    _verified_phase_completion_from_verified_aggregate_claims
)


REPO_ROOT = Path(__file__).resolve().parents[2]
ABLATION_PROTOCOL = REPO_ROOT / "configs" / "protocols" / "ablations-v1.yaml"
PRIMARY_PROTOCOL = REPO_ROOT / "configs" / "protocols" / "primary-v1.yaml"

SELECTION_CONFIG_IDS = {
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
}
DIFFUSION_SELECTION_CONFIG_IDS = {
    "ablation-prior-diffusion-v1",
    "ablation-j1-v1",
    "ablation-j2-v1",
}
STRESS_ACQUISITION_IDS = {
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
}
STRESS_MOTION_CONFIG_IDS = {
    "stress-motion-fit-translation-only-v1",
    "stress-motion-fit-rotation-only-v1",
}
PRIMARY_METHODS = {
    "dgi",
    "static_cs",
    "perframe_cs",
    "tv3d",
    "monin",
    "gidc3dtv",
    "recinr",
    "siren",
    "recinr_se2",
    "gsdiff_tv",
    "gsdiff_diffusion",
}
SELECTION_ANCHORS = {
    ("tank", "trans"),
    ("digit5", "rot"),
    ("usaf", "transrot"),
}


@pytest.fixture
def ablation_protocol() -> dict[str, object]:
    return load_protocol(ABLATION_PROTOCOL)


@pytest.fixture
def primary_protocol() -> dict[str, object]:
    return load_protocol(PRIMARY_PROTOCOL)


@pytest.fixture
def frozen_selection() -> FrozenSelection:
    return FrozenSelection(
        method_id="gsdiff_tv",
        method_config_id="ablation-j4-v1",
        selection_record_sha256="a" * 64,
    )


def _valid_completions(
    *, commit: str = "1" * 40
) -> tuple[VerifiedPhaseCompletion, ...]:
    return (
        verified_phase_completion_from_canonical_aggregate(
            phase_id="selection-replay-v1",
            phase_sha256="a" * 64,
            complete_count=207,
            publication_experiment_commit=commit,
            aggregate_sha256="d" * 64,
        ),
        verified_phase_completion_from_canonical_aggregate(
            phase_id="selection-stress-v1",
            phase_sha256="b" * 64,
            complete_count=126,
            publication_experiment_commit=commit,
            aggregate_sha256="e" * 64,
        ),
        verified_phase_completion_from_canonical_aggregate(
            phase_id="primary-selection-v1",
            phase_sha256="c" * 64,
            complete_count=297,
            publication_experiment_commit=commit,
            aggregate_sha256="f" * 64,
        ),
    )


def _gate() -> ConfirmatoryGateEvidence:
    return verify_confirmatory_prerequisites(_valid_completions())


def _corrupt_gate(**changes: object) -> ConfirmatoryGateEvidence:
    """Simulate corrupted evidence without using the factory-only constructor."""
    gate = _gate()
    for field_name, value in changes.items():
        object.__setattr__(gate, field_name, value)
    return gate


def _corrupt_completion(
    completion: VerifiedPhaseCompletion,
    **changes: object,
) -> VerifiedPhaseCompletion:
    for field_name, value in changes.items():
        object.__setattr__(completion, field_name, value)
    return completion


class _ExplodingMapping(Mapping[str, object]):
    """Proves a failed confirmatory gate does not inspect protocol content."""

    def __getitem__(self, key: str) -> object:
        raise AssertionError(f"protocol accessed before gate: {key}")

    def __iter__(self) -> Iterator[str]:
        raise AssertionError("protocol iterated before gate")

    def __len__(self) -> int:
        raise AssertionError("protocol sized before gate")


def test_phase_contract_dataclasses_are_frozen() -> None:
    dataset = DatasetPlanKey("contract", "a" * 64, "base", "tank", "trans", 7)
    selection = FrozenSelection("gsdiff_tv", "config", "b" * 64)
    completion = _valid_completions()[0]
    gate = _gate()
    plan = PhasePlan("phase", "0" * 64, (), (), 0, 0)

    for instance, field in (
        (dataset, "seed"),
        (selection, "method_id"),
        (completion, "complete_count"),
        (gate, "prerequisite_sha256"),
        (plan, "expected_runs"),
    ):
        with pytest.raises(FrozenInstanceError):
            setattr(instance, field, None)


def test_verified_phase_completion_cannot_be_constructed_directly() -> None:
    with pytest.raises(ValueError, match="factory"):
        VerifiedPhaseCompletion(
            "selection-replay-v1",
            "a" * 64,
            207,
            "1" * 40,
            "e" * 64,
        )


def test_canonical_aggregate_factory_creates_exact_completion_evidence() -> None:
    completion = verified_phase_completion_from_canonical_aggregate(
        phase_id="selection-replay-v1",
        phase_sha256="a" * 64,
        complete_count=207,
        publication_experiment_commit="1" * 40,
        aggregate_sha256="d" * 64,
    )

    assert (
        completion.phase_id,
        completion.phase_sha256,
        completion.complete_count,
        completion.publication_experiment_commit,
        completion.aggregate_sha256,
    ) == (
        "selection-replay-v1",
        "a" * 64,
        207,
        "1" * 40,
        "d" * 64,
    )


@pytest.mark.parametrize(
    ("phase_id", "expected_runs", "expected_datasets"),
    [
        ("selection-decision-v1", 207, 9),
        ("selection-replay-v1", 207, 9),
        ("selection-stress-v1", 126, 108),
        ("primary-selection-v1", 297, 27),
        ("primary-confirmatory-v1", 198, 18),
    ],
)
def test_all_phase_counts_are_exact(
    phase_id: str,
    expected_runs: int,
    expected_datasets: int,
    ablation_protocol: dict[str, object],
    primary_protocol: dict[str, object],
    frozen_selection: FrozenSelection,
) -> None:
    plan = materialize_phase(
        phase_id,
        ablation_protocol=ablation_protocol,
        primary_protocol=primary_protocol,
        frozen_selection=(
            frozen_selection if phase_id == "selection-stress-v1" else None
        ),
        gate=_gate() if phase_id == "primary-confirmatory-v1" else None,
    )

    assert plan.expected_runs == expected_runs == len(plan.cells)
    assert plan.expected_datasets == expected_datasets == len(plan.dataset_plans)
    assert len(plan.phase_sha256) == 64
    assert plan.phase_sha256 == plan.phase_sha256.lower()


def test_decision_and_replay_have_the_same_exact_scientific_cells(
    ablation_protocol: dict[str, object],
    primary_protocol: dict[str, object],
) -> None:
    decision = materialize_phase(
        "selection-decision-v1",
        ablation_protocol=ablation_protocol,
        primary_protocol=primary_protocol,
    )
    replay = materialize_phase(
        "selection-replay-v1",
        ablation_protocol=ablation_protocol,
        primary_protocol=primary_protocol,
    )

    assert decision.cells == replay.cells
    assert decision.dataset_plans == replay.dataset_plans
    assert decision.phase_sha256 != replay.phase_sha256
    assert {cell.method_config_id for cell in decision.cells} == SELECTION_CONFIG_IDS
    assert {(cell.target, cell.motion) for cell in decision.cells} == SELECTION_ANCHORS
    assert {cell.seed for cell in decision.cells} == {7, 11, 42}
    assert {cell.acquisition_config_id for cell in decision.cells} == {"base"}
    assert {cell.campaign_id for cell in decision.cells} == {"ablations-v1"}
    assert all(
        cell.method
        == (
            "gsdiff_diffusion"
            if cell.method_config_id in DIFFUSION_SELECTION_CONFIG_IDS
            else "gsdiff_tv"
        )
        for cell in decision.cells
    )


def test_stress_is_an_explicit_union_and_reuses_anchor_acquisition(
    ablation_protocol: dict[str, object],
    primary_protocol: dict[str, object],
    frozen_selection: FrozenSelection,
) -> None:
    plan = materialize_phase(
        "selection-stress-v1",
        ablation_protocol=ablation_protocol,
        primary_protocol=primary_protocol,
        frozen_selection=frozen_selection,
    )

    observed_conditions = {
        (
            cell.method_config_id
            if cell.method_config_id in STRESS_MOTION_CONFIG_IDS
            else cell.acquisition_config_id
        )
        for cell in plan.cells
    }
    assert observed_conditions == STRESS_ACQUISITION_IDS | STRESS_MOTION_CONFIG_IDS
    assert {cell.method for cell in plan.cells} == {frozen_selection.method_id}
    assert {(cell.target, cell.motion) for cell in plan.cells} == SELECTION_ANCHORS
    assert {cell.seed for cell in plan.cells} == {7, 11, 42}
    assert {
        plan_key.acquisition_config_id for plan_key in plan.dataset_plans
    } == STRESS_ACQUISITION_IDS
    assert all(
        cell.acquisition_config_id == "stress-anchor-v1"
        for cell in plan.cells
        if cell.method_config_id in STRESS_MOTION_CONFIG_IDS
    )
    assert all(
        cell.method_config_id == frozen_selection.method_config_id
        for cell in plan.cells
        if cell.acquisition_config_id in STRESS_ACQUISITION_IDS
        and cell.method_config_id not in STRESS_MOTION_CONFIG_IDS
    )


def test_stress_requires_a_frozen_selection(
    ablation_protocol: dict[str, object],
    primary_protocol: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="frozen selection"):
        materialize_phase(
            "selection-stress-v1",
            ablation_protocol=ablation_protocol,
            primary_protocol=primary_protocol,
        )


@pytest.mark.parametrize(
    "selection",
    [
        FrozenSelection("gsdiff_tv", "unselected-config-v1", "a" * 64),
        FrozenSelection(
            "gsdiff_tv", "ablation-prior-diffusion-v1", "a" * 64
        ),
        FrozenSelection("gsdiff_diffusion", "ablation-j4-v1", "a" * 64),
        FrozenSelection("gsdiff_tv", "ablation-j4-v1", "not-a-sha"),
    ],
)
def test_stress_rejects_a_selection_outside_the_declared_23_candidates(
    selection: FrozenSelection,
    ablation_protocol: dict[str, object],
    primary_protocol: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="frozen selection"):
        materialize_phase(
            "selection-stress-v1",
            ablation_protocol=ablation_protocol,
            primary_protocol=primary_protocol,
            frozen_selection=selection,
        )


def test_stress_phase_hash_binds_the_selection_record(
    ablation_protocol: dict[str, object],
    primary_protocol: dict[str, object],
    frozen_selection: FrozenSelection,
) -> None:
    left = materialize_phase(
        "selection-stress-v1",
        ablation_protocol=ablation_protocol,
        primary_protocol=primary_protocol,
        frozen_selection=frozen_selection,
    )
    right = materialize_phase(
        "selection-stress-v1",
        ablation_protocol=ablation_protocol,
        primary_protocol=primary_protocol,
        frozen_selection=FrozenSelection(
            frozen_selection.method_id,
            frozen_selection.method_config_id,
            "b" * 64,
        ),
    )

    assert left.cells == right.cells
    assert left.dataset_plans == right.dataset_plans
    assert left.phase_sha256 != right.phase_sha256


def test_primary_phases_partition_the_current_primary_campaign(
    ablation_protocol: dict[str, object],
    primary_protocol: dict[str, object],
) -> None:
    selection = materialize_phase(
        "primary-selection-v1",
        ablation_protocol=ablation_protocol,
        primary_protocol=primary_protocol,
    )
    confirmatory = materialize_phase(
        "primary-confirmatory-v1",
        ablation_protocol=ablation_protocol,
        primary_protocol=primary_protocol,
        gate=_gate(),
    )
    expanded = set(expand_cells(primary_protocol))

    assert set(selection.cells).isdisjoint(confirmatory.cells)
    assert set(selection.cells) | set(confirmatory.cells) == expanded
    assert {cell.seed for cell in selection.cells} == {7, 11, 42}
    assert {cell.seed for cell in confirmatory.cells} == {73, 101}
    assert {cell.method for cell in selection.cells} == PRIMARY_METHODS
    assert {cell.method for cell in confirmatory.cells} == PRIMARY_METHODS
    assert {cell.campaign_id for cell in selection.cells} == {"primary-v1"}
    assert {cell.campaign_id for cell in confirmatory.cells} == {"primary-v1"}


def test_unknown_phase_is_rejected_before_protocol_access() -> None:
    protocol = _ExplodingMapping()
    with pytest.raises(ValueError, match="unknown phase"):
        materialize_phase(
            "user-filtered-seeds",
            ablation_protocol=protocol,
            primary_protocol=protocol,
        )


@pytest.mark.parametrize(
    "gate_changes",
    [
        None,
        {"verified_phase_counts": {"selection-replay-v1": 207}},
        {"prerequisite_sha256": "not-a-sha"},
    ],
)
def test_invalid_confirmatory_gate_stops_before_protocol_access(
    gate_changes: dict[str, object] | None,
) -> None:
    protocol = _ExplodingMapping()
    gate = None if gate_changes is None else _corrupt_gate(**gate_changes)
    with pytest.raises(ValueError, match="confirmatory gate"):
        materialize_phase(
            "primary-confirmatory-v1",
            ablation_protocol=protocol,
            primary_protocol=protocol,
            gate=gate,
        )


def test_confirmatory_gate_rejects_a_valid_but_unbound_hash_before_protocol_access(
) -> None:
    protocol = _ExplodingMapping()
    forged = _corrupt_gate(prerequisite_sha256="f" * 64)

    with pytest.raises(ValueError, match="confirmatory gate"):
        materialize_phase(
            "primary-confirmatory-v1",
            ablation_protocol=protocol,
            primary_protocol=protocol,
            gate=forged,
        )


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_confirmatory_gate_rejects_nonexact_phase_hash_keys_before_protocol_access(
    mutation: str,
) -> None:
    protocol = _ExplodingMapping()
    phase_hashes = dict(_gate().verified_phase_sha256s)
    if mutation == "missing":
        phase_hashes.pop("selection-stress-v1")
    else:
        phase_hashes["selection-decision-v1"] = "d" * 64
    forged = _corrupt_gate(verified_phase_sha256s=phase_hashes)

    with pytest.raises(ValueError, match="confirmatory gate"):
        materialize_phase(
            "primary-confirmatory-v1",
            ablation_protocol=protocol,
            primary_protocol=protocol,
            gate=forged,
        )


def test_confirmatory_gate_rejects_a_changed_phase_hash_before_protocol_access(
) -> None:
    protocol = _ExplodingMapping()
    phase_hashes = dict(_gate().verified_phase_sha256s)
    phase_hashes["selection-replay-v1"] = "d" * 64
    forged = _corrupt_gate(verified_phase_sha256s=phase_hashes)

    with pytest.raises(ValueError, match="confirmatory gate"):
        materialize_phase(
            "primary-confirmatory-v1",
            ablation_protocol=protocol,
            primary_protocol=protocol,
            gate=forged,
        )


def test_confirmatory_gate_cannot_be_constructed_directly() -> None:
    with pytest.raises(ValueError, match="factory"):
        ConfirmatoryGateEvidence(
            "a" * 64,
            "1" * 40,
            {
                "selection-replay-v1": 207,
                "selection-stress-v1": 126,
                "primary-selection-v1": 297,
            },
            {
                "selection-replay-v1": "a" * 64,
                "selection-stress-v1": "b" * 64,
                "primary-selection-v1": "c" * 64,
            },
        )


def test_confirmatory_prerequisites_require_exact_630_and_one_commit() -> None:
    gate = verify_confirmatory_prerequisites(_valid_completions())

    assert gate.publication_experiment_commit == "1" * 40
    assert dict(gate.verified_phase_counts) == {
        "selection-replay-v1": 207,
        "selection-stress-v1": 126,
        "primary-selection-v1": 297,
    }
    assert dict(gate.verified_phase_sha256s) == {
        "selection-replay-v1": "a" * 64,
        "selection-stress-v1": "b" * 64,
        "primary-selection-v1": "c" * 64,
    }
    assert dict(gate.verified_aggregate_sha256s) == {
        "selection-replay-v1": "d" * 64,
        "selection-stress-v1": "e" * 64,
        "primary-selection-v1": "f" * 64,
    }
    assert sum(gate.verified_phase_counts.values()) == 630
    assert len(gate.prerequisite_sha256) == 64


@pytest.mark.parametrize("changed_index", [0, 1, 2])
def test_prerequisite_hash_binds_each_canonical_aggregate(
    changed_index: int,
) -> None:
    original_completions = list(_valid_completions())
    original = verify_confirmatory_prerequisites(original_completions)
    selected = original_completions[changed_index]
    original_completions[changed_index] = (
        verified_phase_completion_from_canonical_aggregate(
            phase_id=selected.phase_id,
            phase_sha256=selected.phase_sha256,
            complete_count=selected.complete_count,
            publication_experiment_commit=(
                selected.publication_experiment_commit
            ),
            aggregate_sha256="9" * 64,
        )
    )

    changed = verify_confirmatory_prerequisites(original_completions)

    assert changed.prerequisite_sha256 != original.prerequisite_sha256


def test_confirmatory_phase_hash_binds_verified_prerequisite(
    ablation_protocol: dict[str, object],
    primary_protocol: dict[str, object],
) -> None:
    original_completions = list(_valid_completions())
    changed_completion = original_completions[0]
    changed_completions = list(original_completions)
    changed_completions[0] = (
        verified_phase_completion_from_canonical_aggregate(
            phase_id=changed_completion.phase_id,
            phase_sha256=changed_completion.phase_sha256,
            complete_count=changed_completion.complete_count,
            publication_experiment_commit=(
                changed_completion.publication_experiment_commit
            ),
            aggregate_sha256="9" * 64,
        )
    )
    original = materialize_phase(
        "primary-confirmatory-v1",
        ablation_protocol=ablation_protocol,
        primary_protocol=primary_protocol,
        gate=verify_confirmatory_prerequisites(original_completions),
    )
    changed = materialize_phase(
        "primary-confirmatory-v1",
        ablation_protocol=ablation_protocol,
        primary_protocol=primary_protocol,
        gate=verify_confirmatory_prerequisites(changed_completions),
    )

    assert changed.cells == original.cells
    assert changed.dataset_plans == original.dataset_plans
    assert changed.phase_sha256 != original.phase_sha256


def test_prerequisite_hash_changes_when_a_phase_hash_changes() -> None:
    original = verify_confirmatory_prerequisites(_valid_completions())
    changed_completions = list(_valid_completions())
    changed_completions[0] = verified_phase_completion_from_canonical_aggregate(
        phase_id="selection-replay-v1",
        phase_sha256="d" * 64,
        complete_count=207,
        publication_experiment_commit="1" * 40,
        aggregate_sha256="d" * 64,
    )
    changed = verify_confirmatory_prerequisites(changed_completions)

    assert changed.prerequisite_sha256 != original.prerequisite_sha256
    assert changed.verified_phase_sha256s["selection-replay-v1"] == "d" * 64


def test_decision_completion_cannot_substitute_for_replay() -> None:
    completions = (
        verified_phase_completion_from_canonical_aggregate(
            phase_id="selection-decision-v1",
            phase_sha256="a" * 64,
            complete_count=207,
            publication_experiment_commit="1" * 40,
            aggregate_sha256="9" * 64,
        ),
        *_valid_completions()[1:],
    )

    with pytest.raises(ValueError, match="required phases"):
        verify_confirmatory_prerequisites(completions)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "count", "hash"])
def test_malformed_phase_completion_is_rejected(mutation: str) -> None:
    completions = list(_valid_completions())
    if mutation == "missing":
        completions.pop()
    elif mutation == "duplicate":
        completions.append(completions[0])
    elif mutation == "count":
        completions[1] = _corrupt_completion(
            completions[1],
            complete_count=125,
        )
    else:
        completions[1] = _corrupt_completion(
            completions[1],
            phase_sha256="not-a-sha",
        )

    with pytest.raises(ValueError):
        verify_confirmatory_prerequisites(completions)


def test_mixed_publication_commits_are_rejected() -> None:
    completions = list(_valid_completions())
    completions[2] = verified_phase_completion_from_canonical_aggregate(
        phase_id="primary-selection-v1",
        phase_sha256="c" * 64,
        complete_count=297,
        publication_experiment_commit="2" * 40,
        aggregate_sha256="f" * 64,
    )

    with pytest.raises(ValueError, match="same publication"):
        verify_confirmatory_prerequisites(completions)


def test_phase_and_prerequisite_hashes_are_canonical_and_order_invariant(
    ablation_protocol: dict[str, object],
    primary_protocol: dict[str, object],
) -> None:
    first = materialize_phase(
        "selection-decision-v1",
        ablation_protocol=ablation_protocol,
        primary_protocol=primary_protocol,
    )
    second = materialize_phase(
        "selection-decision-v1",
        ablation_protocol=load_protocol(ABLATION_PROTOCOL),
        primary_protocol=load_protocol(PRIMARY_PROTOCOL),
    )
    gate_forward = verify_confirmatory_prerequisites(_valid_completions())
    gate_reverse = verify_confirmatory_prerequisites(
        tuple(reversed(_valid_completions()))
    )

    assert first == second
    assert gate_forward.prerequisite_sha256 == gate_reverse.prerequisite_sha256
