from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
from pathlib import Path

import pytest

import gsdiff.experiments.protocol as protocol


REPO = Path(__file__).resolve().parents[2]
PROTOCOLS = REPO / "configs" / "protocols"
FILES = (
    "pilot-v1.yaml",
    "primary-v1.yaml",
    "supplement-grid-v1.yaml",
    "ood-v1.yaml",
    "failure-v1.yaml",
    "ablations-v1.yaml",
    "methods-v1.yaml",
    "scientific-contracts-v1.yaml",
    "noise-calibration-v1.yaml",
)
METHODS = (
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
)


def _canonical_sha(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _protocol_payload(document: dict[str, object]) -> dict[str, object]:
    payload = copy.deepcopy(document)
    payload.pop("protocol_sha256", None)
    return payload


def _membership_records(document: dict[str, object]) -> list[dict[str, object]]:
    matrix = document["matrix"]
    assert isinstance(matrix, dict)
    records: list[dict[str, object]] = []
    for target, motion, seed, method, acquisition_config_id in itertools.product(
        matrix["targets"],
        matrix["motions"],
        matrix["seeds"],
        matrix["methods"],
        matrix["acquisition_config_ids"],
    ):
        method_config_ids = matrix["method_config_ids"]
        assert isinstance(method_config_ids, dict)
        records.append(
            {
                "acquisition_config_id": acquisition_config_id,
                "method": method,
                "method_config_id": method_config_ids[method],
                "motion": motion,
                "seed": seed,
                "target": target,
            }
        )
    return records


def _refresh_hashes(document: dict[str, object]) -> None:
    if document["document_kind"] == "campaign":
        try:
            records = _membership_records(document)
        except KeyError:
            records = None
        if records is not None:
            document["campaign_sha256"] = _canonical_sha(
                sorted(
                    records,
                    key=lambda record: json.dumps(
                        record,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                    ),
                )
            )
    document["protocol_sha256"] = _canonical_sha(_protocol_payload(document))


def _load(name: str) -> dict[str, object]:
    return protocol.load_protocol(PROTOCOLS / name)


def _run_key(cell: protocol.ExperimentCell) -> tuple[object, ...]:
    return (
        cell.scientific_contract_id,
        cell.scientific_contract_sha256,
        cell.target,
        cell.motion,
        cell.seed,
        cell.method,
        cell.acquisition_config_id,
        cell.method_config_id,
    )


def _acquisition_key(cell: protocol.ExperimentCell) -> tuple[object, ...]:
    return (
        cell.scientific_contract_id,
        cell.scientific_contract_sha256,
        cell.target,
        cell.motion,
        cell.seed,
        cell.acquisition_config_id,
    )


def test_all_nine_versioned_documents_load_and_validate():
    loaded = [_load(name) for name in FILES]

    assert [document["schema_version"] for document in loaded] == [
        "experiment-protocol-v1"
    ] * 9
    assert [document["document_kind"] for document in loaded] == [
        "campaign",
        "campaign",
        "campaign",
        "campaign",
        "campaign",
        "ablation",
        "methods-registry",
        "scientific-contracts-registry",
        "noise-calibration-registry",
    ]
    assert all(
        document["protocol_sha256"] == _canonical_sha(_protocol_payload(document))
        for document in loaded
    )


def test_primary_matrix_expands_in_declared_order_to_495_unique_cells():
    cells = protocol.expand_cells(_load("primary-v1.yaml"))

    assert len(cells) == 3 * 3 * 5 * 11 == 495
    assert len({_run_key(cell) for cell in cells}) == 495
    assert [
        (
            cell.target,
            cell.motion,
            cell.seed,
            cell.method,
            cell.acquisition_config_id,
            cell.method_config_id,
        )
        for cell in cells[:12]
    ] == [
        ("tank", "trans", 7, "dgi", "base", "default"),
        ("tank", "trans", 7, "static_cs", "base", "default"),
        ("tank", "trans", 7, "perframe_cs", "base", "default"),
        ("tank", "trans", 7, "tv3d", "base", "default"),
        ("tank", "trans", 7, "monin", "base", "default"),
        ("tank", "trans", 7, "gidc3dtv", "base", "default"),
        ("tank", "trans", 7, "recinr", "base", "default"),
        ("tank", "trans", 7, "siren", "base", "default"),
        ("tank", "trans", 7, "recinr_se2", "base", "default"),
        ("tank", "trans", 7, "gsdiff_tv", "base", "default"),
        ("tank", "trans", 7, "gsdiff_diffusion", "base", "default"),
        ("tank", "trans", 11, "dgi", "base", "default"),
    ]


def test_primary_and_supplement_overlap_is_exact_at_run_and_acquisition_grain():
    primary = protocol.expand_cells(_load("primary-v1.yaml"))
    supplement = protocol.expand_cells(_load("supplement-grid-v1.yaml"))
    primary_runs = {_run_key(cell) for cell in primary}
    supplement_runs = {_run_key(cell) for cell in supplement}
    primary_acquisitions = {_acquisition_key(cell) for cell in primary}
    supplement_acquisitions = {_acquisition_key(cell) for cell in supplement}

    assert len(supplement_runs) == 4 * 4 * 3 * 11 == 528
    assert len(primary_runs & supplement_runs) == 3 * 3 * 3 * 11 == 297
    assert len(supplement_runs - primary_runs) == 231
    assert len(primary_runs | supplement_runs) == 726
    assert len(primary_acquisitions) == 45
    assert len(supplement_acquisitions) == 48
    assert len(primary_acquisitions & supplement_acquisitions) == 27
    assert len(primary_acquisitions | supplement_acquisitions) == 66
    overlap_key = next(iter(primary_runs & supplement_runs))
    primary_cell = next(cell for cell in primary if _run_key(cell) == overlap_key)
    supplement_cell = next(
        cell for cell in supplement if _run_key(cell) == overlap_key
    )
    assert primary_cell.campaign_id == "primary-v1"
    assert supplement_cell.campaign_id == "supplement-grid-v1"
    assert primary_cell.campaign_id != supplement_cell.campaign_id
    assert _run_key(primary_cell) == _run_key(supplement_cell)


def test_campaign_matrices_and_shared_acquisition_are_exact_literals():
    expected = {
        "primary-v1.yaml": {
            "targets": ["tank", "digit5", "usaf"],
            "motions": ["trans", "rot", "transrot"],
            "seeds": [7, 11, 42, 73, 101],
            "methods": list(METHODS),
            "acquisition_config_ids": ["base"],
        },
        "supplement-grid-v1.yaml": {
            "targets": ["tank", "digit5", "letterR", "usaf"],
            "motions": ["trans", "rot", "transrot", "accel"],
            "seeds": [7, 11, 42],
            "methods": list(METHODS),
            "acquisition_config_ids": ["base"],
        },
        "ood-v1.yaml": {
            "targets": ["cx_camera", "cx_clutter"],
            "motions": ["trans", "rot", "transrot"],
            "seeds": [7, 11, 42],
            "methods": list(METHODS),
            "acquisition_config_ids": ["base"],
        },
        "failure-v1.yaml": {
            "targets": ["cx_coins", "cx_text"],
            "motions": ["transrot"],
            "seeds": [7, 11, 42],
            "methods": [
                "dgi",
                "tv3d",
                "recinr_se2",
                "gsdiff_tv",
                "gsdiff_diffusion",
            ],
            "acquisition_config_ids": [
                "m320",
                "m640",
                "m1280",
                "m2560",
                "m3840",
                "m5120",
            ],
        },
    }
    for name, matrix_literals in expected.items():
        document = _load(name)
        matrix = document["matrix"]
        for field, value in matrix_literals.items():
            assert matrix[field] == value
        assert matrix["method_config_ids"] == {
            method: "default" for method in matrix["methods"]
        }
        if name != "failure-v1.yaml":
            assert document["acquisition_configs"]["base"] == {
                "image_size": [64, 64],
                "num_frames": 20,
                "train_measurements": 2560,
                "holdout_measurements": 250,
                "pattern_family": "bernoulli",
                "pattern_values": [0, 1],
                "pattern_order": "stratified",
                "time_assignment": "uniform",
                "holdout_pattern_family": "uniform-random",
                "snr_db": 25,
                "noise_calibration_id": "detector-absolute-v1",
            }
        assert document["metric_version"] == "metrics-v1"


def test_pilot_is_exactly_structural_and_uses_all_canonical_methods():
    pilot = _load("pilot-v1.yaml")

    assert pilot["scientific_contract_id"] == "gsdiff-pilot-v1"
    assert pilot["matrix"] == {
        "targets": ["tank"],
        "motions": ["transrot"],
        "seeds": [7],
        "methods": list(METHODS),
        "acquisition_config_ids": ["base"],
        "method_config_ids": {method: "default" for method in METHODS},
    }
    assert pilot["acquisition_configs"]["base"] == {
        "image_size": [32, 32],
        "num_frames": 4,
        "train_measurements": 128,
        "holdout_measurements": 16,
        "pattern_family": "bernoulli",
        "pattern_values": [0, 1],
        "pattern_order": "stratified",
        "time_assignment": "uniform",
        "holdout_pattern_family": "uniform-random",
        "snr_db": 25,
        "noise_calibration_id": "detector-absolute-v1",
    }


def test_ood_and_failure_matrices_have_exact_run_and_dataset_counts():
    ood = protocol.expand_cells(_load("ood-v1.yaml"))
    failure = protocol.expand_cells(_load("failure-v1.yaml"))

    assert len(ood) == 2 * 3 * 3 * 11 == 198
    assert len({_acquisition_key(cell) for cell in ood}) == 18
    assert len(failure) == 2 * 1 * 3 * 5 * 6 == 180
    assert len({_acquisition_key(cell) for cell in failure}) == 36
    assert {cell.acquisition_config_id for cell in failure} == {
        "m320",
        "m640",
        "m1280",
        "m2560",
        "m3840",
        "m5120",
    }


def test_campaign_hash_is_hand_derived_from_order_insensitive_membership():
    primary = _load("primary-v1.yaml")
    expected_records = [
        {
            "acquisition_config_id": "base",
            "method": method,
            "method_config_id": "default",
            "motion": motion,
            "seed": seed,
            "target": target,
        }
        for target, motion, seed, method in itertools.product(
            ("tank", "digit5", "usaf"),
            ("trans", "rot", "transrot"),
            (7, 11, 42, 73, 101),
            METHODS,
        )
    ]
    expected_records.sort(
        key=lambda record: json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
    )

    assert primary["campaign_sha256"] == _canonical_sha(expected_records)

    reordered = copy.deepcopy(primary)
    reordered["matrix"]["targets"].reverse()
    reordered["matrix"]["methods"].reverse()
    original_campaign_hash = reordered["campaign_sha256"]
    reordered["protocol_sha256"] = _canonical_sha(_protocol_payload(reordered))
    protocol.validate_protocol(reordered)
    assert reordered["campaign_sha256"] == original_campaign_hash


def test_protocol_hash_is_order_sensitive_while_expansion_preserves_source_order():
    primary = _load("primary-v1.yaml")
    reordered = copy.deepcopy(primary)
    reordered["matrix"]["targets"] = ["usaf", "digit5", "tank"]

    with pytest.raises(ValueError, match="protocol_sha256"):
        protocol.validate_protocol(reordered)

    reordered["protocol_sha256"] = _canonical_sha(_protocol_payload(reordered))
    protocol.validate_protocol(reordered)
    assert reordered["protocol_sha256"] != primary["protocol_sha256"]
    assert protocol.expand_cells(reordered)[0].target == "usaf"


def test_shared_scientific_contract_is_reusable_but_changed_content_is_not():
    names = (
        "primary-v1.yaml",
        "supplement-grid-v1.yaml",
        "ood-v1.yaml",
        "failure-v1.yaml",
    )
    documents = [_load(name) for name in names]

    assert {document["scientific_contract_id"] for document in documents} == {
        "gsdiff-sim-v1"
    }
    assert len(
        {document["scientific_contract_sha256"] for document in documents}
    ) == 1
    assert all(protocol.expand_cells(document) for document in documents)

    registry = _load("scientific-contracts-v1.yaml")
    duplicate = copy.deepcopy(registry["contracts"][0])
    duplicate["content"]["generator"]["version"] = "unsafe-generator-v0"
    duplicate["sha256"] = _canonical_sha(duplicate["content"])
    registry["contracts"].append(duplicate)
    registry["protocol_sha256"] = _canonical_sha(_protocol_payload(registry))
    with pytest.raises(ValueError, match="scientific contract.*reus|duplicate"):
        protocol.validate_protocol(registry)


def test_contract_and_campaign_hashes_match_independent_canonical_json():
    contracts = _load("scientific-contracts-v1.yaml")
    by_id = {entry["id"]: entry for entry in contracts["contracts"]}

    for name in (
        "pilot-v1.yaml",
        "primary-v1.yaml",
        "supplement-grid-v1.yaml",
        "ood-v1.yaml",
        "failure-v1.yaml",
    ):
        document = _load(name)
        entry = by_id[document["scientific_contract_id"]]
        assert entry["sha256"] == _canonical_sha(entry["content"])
        assert document["scientific_contract_sha256"] == entry["sha256"]
        assert document["campaign_sha256"] == _canonical_sha(
            sorted(
                _membership_records(document),
                key=lambda record: json.dumps(
                    record,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ),
            )
        )


def test_corrected_targets_motions_rng_and_noise_are_fully_bound():
    contracts = _load("scientific-contracts-v1.yaml")
    sim = next(entry for entry in contracts["contracts"] if entry["id"] == "gsdiff-sim-v1")
    content = sim["content"]

    assert content["generator"] == {
        "id": "gsdiff-corrected-sim",
        "version": "generator-v1",
        "motion_type": "custom_se2",
        "normalized_time": [0, 1],
    }
    assert content["targets"] == {
        "tank": "assets/tank.png",
        "digit5": "char:5",
        "letterR": "char:R",
        "usaf": "assets/usaf_tar_small.png",
        "cx_camera": "assets/cx_camera.png",
        "cx_clutter": "assets/cx_clutter.png",
        "cx_coins": "assets/cx_coins.png",
        "cx_text": "assets/cx_text.png",
    }
    assert content["motions"] == {
        "trans": {
            "velocity": [8, 8],
            "acceleration": [0, 0],
            "omega": 0,
            "beta": 0,
        },
        "rot": {
            "velocity": [0, 0],
            "acceleration": [0, 0],
            "omega": 0.4,
            "beta": 0,
        },
        "transrot": {
            "velocity": [8, 8],
            "acceleration": [0, 0],
            "omega": 0.3,
            "beta": 0,
        },
        "accel": {
            "velocity": [6, 6],
            "acceleration": [3, 3],
            "omega": 0.2,
            "beta": 0.1,
        },
    }
    assert content["acquisition"] == {
        "pattern_family": "bernoulli",
        "pattern_values": [0, 1],
        "pattern_order": "stratified",
        "time_assignment": "uniform",
        "holdout_pattern_family": "uniform-random",
    }
    assert content["rng"] == {
        "bit_generator": "PCG64",
        "seed_sequence": "SeedSequence(entropy=seed, spawn_key=(stream_id,))",
        "streams": {
            "train-pattern": 0,
            "train-noise": 1,
            "holdout-pattern": 2,
            "holdout-noise": 3,
        },
    }
    assert content["noise"] == {
        "calibration_id": "detector-absolute-v1",
        "mode": "detector-absolute",
        "reference": "corresponding-bernoulli-reference-cell",
        "sigma_formula": "sqrt(var(y_reference,ddof=0))*10**(-snr_db/20)",
        "reuse": ["train", "holdout", "alternate-pattern"],
    }


def test_failure_budgets_are_identity_distinct_and_bind_full_acquisition():
    failure = _load("failure-v1.yaml")
    configs = failure["acquisition_configs"]

    assert list(configs) == ["m320", "m640", "m1280", "m2560", "m3840", "m5120"]
    assert [config["train_measurements"] for config in configs.values()] == [
        320,
        640,
        1280,
        2560,
        3840,
        5120,
    ]
    for config in configs.values():
        assert config["holdout_measurements"] == 250
        assert config["pattern_family"] == "bernoulli"
        assert config["pattern_values"] == [0, 1]
        assert config["pattern_order"] == "stratified"
        assert config["time_assignment"] == "uniform"
        assert config["noise_calibration_id"] == "detector-absolute-v1"


def test_method_registry_locks_canonical_lanes_and_confirmation_pair():
    methods = _load("methods-v1.yaml")
    entries = methods["methods"]

    assert [entry["id"] for entry in entries] == list(METHODS)
    assert methods["primary_method"] == "gsdiff_tv"
    assert methods["secondary_diffusion_method"] == "gsdiff_diffusion"
    assert methods["confirmation_pair"] == {
        "method": "gsdiff_tv",
        "comparator": "recinr_se2",
    }
    assert methods["compatibility_aliases"] == {
        "gsdiff_diff": "gsdiff_diffusion"
    }
    assert next(entry for entry in entries if entry["id"] == "gsdiff_tv")["lane"] == "tv"
    assert next(
        entry for entry in entries if entry["id"] == "gsdiff_diffusion"
    )["lane"] == "diffusion"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda document: document["methods"][0]["command_template"].__setitem__(3, "dgi-copy"),
            "command template",
        ),
        (
            lambda document: document["methods"][0]["required_child_outputs"].append("extra.json"),
            "child outputs",
        ),
        (
            lambda document: next(entry for entry in document["methods"] if entry["id"] == "gsdiff_diffusion")["checkpoints"][0].__setitem__("sha256", "0" * 64),
            "checkpoint",
        ),
        (
            lambda document: next(entry for entry in document["methods"] if entry["id"] == "gsdiff_tv")["profiles"]["publication-v1"]["semantic_config"]["solver"].__setitem__("outer_iterations", 79),
            "semantic config",
        ),
    ],
)
def test_method_registry_nested_contract_mutations_fail_after_rehash(mutate, message):
    invalid = copy.deepcopy(_load("methods-v1.yaml"))
    mutate(invalid)
    _refresh_hashes(invalid)
    with pytest.raises(ValueError, match=message):
        protocol.validate_protocol(invalid)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda document: document["methods"][0]["profiles"]["publication-v1"].__setitem__("method_config_id", "changed"),
        lambda document: document["methods"][0]["profiles"]["controller-cpu-smoke-v1"].__setitem__("method_config_id", "changed"),
        lambda document: document["methods"][0]["profiles"]["ablation-selection-v1"].__setitem__("method_config_id", "changed"),
        lambda document: next(entry for entry in document["methods"] if entry["id"] == "gsdiff_diffusion")["profiles"]["publication-v1"].update({"publication_eligible": True, "selection_eligible": True, "promotion_eligible": True, "execution_ready": True, "execution_blockers": []}),
    ],
)
def test_method_registry_rejects_rehashed_profile_policy_mutations(mutate):
    invalid = copy.deepcopy(_load("methods-v1.yaml"))
    mutate(invalid)
    _refresh_hashes(invalid)
    with pytest.raises(ValueError, match="profile policy"):
        protocol.validate_protocol(invalid)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda document: document["methods"][0]["profiles"]["publication-v1"].__setitem__("publication_eligible", 1),
        lambda document: document["methods"][0]["profiles"]["controller-cpu-smoke-v1"].__setitem__("publication_eligible", 0),
        lambda document: document["methods"][0]["profiles"]["publication-v1"].__setitem__("selection_eligible", 1.0),
        lambda document: document["methods"][0]["profiles"]["controller-cpu-smoke-v1"].__setitem__("selection_eligible", 0.0),
    ],
)
def test_method_registry_rejects_rehashed_profile_policy_boolean_type_spoofs(mutate):
    invalid = copy.deepcopy(_load("methods-v1.yaml"))
    mutate(invalid)
    _refresh_hashes(invalid)
    with pytest.raises(ValueError, match="profile policy"):
        protocol.validate_protocol(invalid)


def test_ablation_axes_shortlist_lanes_and_workflow_counts_are_locked():
    ablations = _load("ablations-v1.yaml")

    assert ablations["axes"] == {
        "representation": ["gaussian", "siren", "grid", "recinr_se2"],
        "solver": ["sgd", "hqs", "admm"],
        "prior": ["tv2d", "tv3d_corrected", "diffusion"],
        "motion_warmup_fraction": [0, 0.1, 0.2, 0.4],
        "temporal_tv_weight": [0, 0.05, 0.1, 0.3],
        "train_measurements": [640, 1280, 1920, 2560, 3840, 5120],
        "snr_db": [15, 20, 25, 30],
        "pattern": ["bernoulli", "gaussian", "hadamard_natural", "fourier"],
        "motion_fit": ["matched", "translation_only", "rotation_only"],
        "gaussian_count": [250, 500, 1000, 1500],
    }
    assert ablations["selection_anchors"] == [
        {"target": "tank", "motion": "trans"},
        {"target": "digit5", "motion": "rot"},
        {"target": "usaf", "motion": "transrot"},
    ]
    assert ablations["selection_seeds"] == [7, 11, 42]
    assert not ({73, 101} & set(ablations["selection_seeds"]))
    assert ablations["semantic_anchor"] == {
        "representation": "gaussian",
        "solver": "admm",
        "prior": "tv3d_corrected",
        "motion_warmup_fraction": 0.2,
        "temporal_tv_weight": 0.1,
        "gaussian_count": 1000,
    }
    assert ablations["joint_shortlist"] == [
        {
            "id": "j1",
            "representation": "recinr_se2",
            "solver": "hqs",
            "prior": "diffusion",
            "motion_warmup_fraction": 0.2,
            "temporal_tv_weight": 0.1,
            "gaussian_count": None,
            "method": "gsdiff_diffusion",
            "method_config_id": "ablation-j1-v1",
        },
        {
            "id": "j2",
            "representation": "grid",
            "solver": "admm",
            "prior": "diffusion",
            "motion_warmup_fraction": 0.2,
            "temporal_tv_weight": 0.1,
            "gaussian_count": None,
            "method": "gsdiff_diffusion",
            "method_config_id": "ablation-j2-v1",
        },
        {
            "id": "j3",
            "representation": "siren",
            "solver": "sgd",
            "prior": "tv3d_corrected",
            "motion_warmup_fraction": 0.1,
            "temporal_tv_weight": 0.05,
            "gaussian_count": None,
            "method": "gsdiff_tv",
            "method_config_id": "ablation-j3-v1",
        },
        {
            "id": "j4",
            "representation": "gaussian",
            "solver": "hqs",
            "prior": "tv2d",
            "motion_warmup_fraction": 0.1,
            "temporal_tv_weight": 0.05,
            "gaussian_count": 1500,
            "method": "gsdiff_tv",
            "method_config_id": "ablation-j4-v1",
        },
        {
            "id": "j5",
            "representation": "recinr_se2",
            "solver": "sgd",
            "prior": "tv2d",
            "motion_warmup_fraction": 0.4,
            "temporal_tv_weight": 0.3,
            "gaussian_count": None,
            "method": "gsdiff_tv",
            "method_config_id": "ablation-j5-v1",
        },
        {
            "id": "j6",
            "representation": "grid",
            "solver": "hqs",
            "prior": "tv3d_corrected",
            "motion_warmup_fraction": 0.4,
            "temporal_tv_weight": 0.05,
            "gaussian_count": None,
            "method": "gsdiff_tv",
            "method_config_id": "ablation-j6-v1",
        },
    ]
    assert ablations["workflow_counts"] == {
        "one_factor_unique_configurations": 17,
        "joint_shortlist_configurations": 6,
        "decision_cells": 207,
        "post_freeze_replay_cells": 207,
        "stress_configurations": 14,
        "stress_cells": 126,
        "publication_commit_cells": 333,
        "total_before_retries": 540,
    }
    assert ablations["resolver"] == {
        "hqs": {"solver": "admm", "mode": "hqs"},
        "tv2d": {"method": "gsdiff_tv", "lane": "tv"},
        "tv3d_corrected": {"method": "gsdiff_tv", "lane": "tv"},
        "diffusion": {"method": "gsdiff_diffusion", "lane": "diffusion"},
        "recinr": {"method": "recinr", "lane": "native-baseline"},
    }
    assert len(set(ablations["method_config_ids"].values())) == len(
        ablations["method_config_ids"]
    )


def test_selection_confirmation_and_pilot_execution_guards_are_locked():
    primary = _load("primary-v1.yaml")
    ablations = _load("ablations-v1.yaml")
    pilot = _load("pilot-v1.yaml")

    assert primary["confirmation_rule"] == {
        "rule_id": "primary-confirmation-v1",
        "scientific_contract_id": "gsdiff-sim-v1",
        "method": "gsdiff_tv",
        "comparator": "recinr_se2",
        "metric": "psnr_global_affine",
        "targets": ["tank", "digit5", "usaf"],
        "motions": ["trans", "rot", "transrot"],
        "seeds": [73, 101],
        "minimum_effect_db": 0.0,
        "require_each_seed_mean_delta_strictly_greater_than_minimum": True,
        "require_complete_finite_pairs_per_seed": 9,
        "require_method_failure_count_not_greater_than_comparator": True,
        "inferential_significance_claim": False,
    }
    assert ablations["selection_rule"] == {
        "objective": "heldout_normalized_l2",
        "objective_formula": "l2(pred-y)/max(l2(y),1e-12)",
        "aggregation": "unweighted_mean_over_3_anchors_x_3_seeds",
        "required_complete_cells": 9,
        "residual_tie_relative_tolerance": 0.005,
        "convergence": {
            "required_history_samples": 21,
            "final_window_samples": 3,
            "minimum_relative_improvement_from_initial": 0.01,
            "maximum_final_window_to_best_ratio": 1.05,
            "require_all_finite": True,
        },
        "compute_cap": {
            "wall_time_seconds_per_run": 1800,
            "peak_vram_bytes": 15032385536,
            "on_exceed": "ineligible-retain-artifacts",
        },
        "tie_break_order": [
            "median_runtime_seconds",
            "peak_vram_bytes",
            "config_id",
        ],
    }
    assert pilot["execution_ready"] is False
    assert pilot["execution_profile"] == "pilot-smoke-v1"
    assert pilot["method_budgets"] is None

    unsafe = copy.deepcopy(pilot)
    unsafe["execution_ready"] = True
    _refresh_hashes(unsafe)
    with pytest.raises(ValueError, match="execution_ready|method_budgets"):
        protocol.validate_protocol(unsafe)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda document: document["matrix"]["methods"].append("dgi"),
            "duplicate method",
        ),
        (
            lambda document: document["matrix"]["targets"].append("unknown"),
            "unknown target",
        ),
        (
            lambda document: document["matrix"]["motions"].append("unknown"),
            "unknown motion",
        ),
        (lambda document: document["matrix"].__setitem__("seeds", []), "seed"),
        (
            lambda document: document["acquisition_configs"]["base"].__setitem__(
                "train_measurements", 0
            ),
            "train_measurements",
        ),
        (
            lambda document: document["acquisition_configs"]["base"].__setitem__(
                "holdout_measurements", 0
            ),
            "holdout_measurements",
        ),
        (
            lambda document: document.__delitem__("metric_version"),
            "metric_version|missing",
        ),
    ],
)
def test_invalid_campaigns_are_rejected_after_hashes_are_refreshed(
    mutation, message: str
):
    invalid = copy.deepcopy(_load("primary-v1.yaml"))
    mutation(invalid)
    _refresh_hashes(invalid)

    with pytest.raises((TypeError, ValueError), match=message):
        protocol.validate_protocol(invalid)


def test_campaign_rejects_declared_membership_hash_mismatch():
    invalid = copy.deepcopy(_load("primary-v1.yaml"))
    invalid["campaign_sha256"] = "0" * 64
    invalid["protocol_sha256"] = _canonical_sha(_protocol_payload(invalid))

    with pytest.raises(ValueError, match="campaign_sha256"):
        protocol.validate_protocol(invalid)


@pytest.mark.parametrize(
    ("path", "replacement", "message"),
    [
        (("matrix", "seeds"), [True], "seed|integer"),
        (("matrix", "seeds"), [7.0], "seed|integer"),
        (("matrix", "seeds"), ["7"], "seed|integer"),
        (("expected_runs",), True, "expected_runs|integer"),
        (("expected_runs",), 495.0, "expected_runs|integer"),
        (("expected_runs",), "495", "expected_runs|integer"),
        (
            ("acquisition_configs", "base", "train_measurements"),
            True,
            "train_measurements|integer",
        ),
        (
            ("acquisition_configs", "base", "holdout_measurements"),
            250.0,
            "holdout_measurements|integer",
        ),
        (("matrix", "acquisition_config_ids"), [""], "nonempty|acquisition"),
        (("matrix", "acquisition_config_ids"), [1], "string|acquisition"),
    ],
)
def test_schema_slots_reject_scalar_coercion(
    path: tuple[str, ...], replacement: object, message: str
):
    invalid = copy.deepcopy(_load("primary-v1.yaml"))
    cursor = invalid
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = replacement
    _refresh_hashes(invalid)

    with pytest.raises((TypeError, ValueError), match=message):
        protocol.validate_protocol(invalid)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda d: d["matrix"].__setitem__("acquisition_config_ids", []),
            "acquisition",
        ),
        (
            lambda d: d["matrix"].__setitem__(
                "acquisition_config_ids", ["missing"]
            ),
            "acquisition",
        ),
        (
            lambda d: d["matrix"].__setitem__(
                "acquisition_config_ids", ["base", "base"]
            ),
            "duplicate|acquisition",
        ),
        (
            lambda d: d["acquisition_configs"].__setitem__(
                "extra", copy.deepcopy(d["acquisition_configs"]["base"])
            ),
            "acquisition_configs",
        ),
        (
            lambda d: d["matrix"]["method_config_ids"].pop("dgi"),
            "method_config_ids",
        ),
        (
            lambda d: d["matrix"]["method_config_ids"].__setitem__(
                "unknown", "default"
            ),
            "method_config_ids",
        ),
        (
            lambda d: d["matrix"]["method_config_ids"].__setitem__(
                "dgi", "wrong-config"
            ),
            "method_config_id",
        ),
    ],
)
def test_acquisition_and_method_config_links_fail_closed(mutate, message: str):
    invalid = copy.deepcopy(_load("primary-v1.yaml"))
    mutate(invalid)
    _refresh_hashes(invalid)

    with pytest.raises((TypeError, ValueError), match=message):
        protocol.validate_protocol(invalid)


def _mutate_contract(
    document: dict[str, object], path: tuple[str, ...], replacement: object
) -> None:
    entry = next(
        item
        for item in document["contracts"]
        if item["id"] == "gsdiff-sim-v1"
    )
    cursor = entry["content"]
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = replacement
    entry["sha256"] = _canonical_sha(entry["content"])
    document["protocol_sha256"] = _canonical_sha(_protocol_payload(document))


@pytest.mark.parametrize(
    ("path", "replacement", "message"),
    [
        (("motions", "trans", "velocity"), [9, 8], "motion"),
        (("motions", "accel", "acceleration"), [4, 3], "motion"),
        (("motions", "rot", "omega"), 0.5, "motion"),
        (("motions", "accel", "beta"), 0.2, "motion"),
        (("acquisition", "pattern_family"), "gaussian", "acquisition"),
        (("acquisition", "pattern_values"), [-1, 1], "acquisition"),
        (("acquisition", "pattern_order"), "random", "acquisition"),
        (("acquisition", "time_assignment"), "interpolation", "acquisition"),
        (
            ("acquisition", "holdout_pattern_family"),
            "bernoulli",
            "acquisition",
        ),
        (("rng", "bit_generator"), "MT19937", "rng|PCG64"),
        (("rng", "seed_sequence"), "seed", "rng|SeedSequence"),
        (("rng", "streams", "train-pattern"), 10, "rng|stream"),
        (("rng", "streams", "train-noise"), 10, "rng|stream"),
        (("rng", "streams", "holdout-pattern"), 10, "rng|stream"),
        (("rng", "streams", "holdout-noise"), 10, "rng|stream"),
        (("noise", "mode"), "snr_db", "noise|detector-absolute"),
        (("noise", "reference"), "per-signal", "noise|reference"),
        (("noise", "sigma_formula"), "legacy", "noise|formula"),
        (("noise", "reuse"), ["train"], "noise|reuse"),
    ],
)
def test_rehashed_contract_rejects_each_locked_physics_family(
    path: tuple[str, ...], replacement: object, message: str
):
    invalid = copy.deepcopy(_load("scientific-contracts-v1.yaml"))
    _mutate_contract(invalid, path, replacement)

    with pytest.raises(ValueError, match=message):
        protocol.validate_protocol(invalid)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("motions", "trans", "velocity"), [8.0, 8.0]),
        (("motions", "trans", "omega"), 0.0),
        (("acquisition", "pattern_values"), [0.0, 1.0]),
    ],
)
def test_rehashed_contract_rejects_numeric_type_equality_spoofs(
    path: tuple[str, ...], replacement: object
):
    invalid = copy.deepcopy(_load("scientific-contracts-v1.yaml"))
    _mutate_contract(invalid, path, replacement)

    with pytest.raises((TypeError, ValueError), match="contract|motion|acquisition"):
        protocol.validate_protocol(invalid)


def test_rehashed_campaign_rejects_float_snr_type_spoof():
    invalid = copy.deepcopy(_load("primary-v1.yaml"))
    invalid["acquisition_configs"]["base"]["snr_db"] = 25.0
    _refresh_hashes(invalid)

    with pytest.raises((TypeError, ValueError), match="snr_db|integer"):
        protocol.validate_protocol(invalid)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("representation", "gaussian"),
        ("solver", "sgd"),
        ("prior", "tv2d"),
        ("motion_warmup_fraction", 0.4),
        ("temporal_tv_weight", 0.3),
        ("gaussian_count", 1500),
        ("method", "gsdiff_tv"),
        ("method_config_id", "ablation-j2-v1"),
    ],
)
def test_rehashed_ablation_rejects_each_shortlist_field_mutation(
    field: str, replacement: object
):
    invalid = copy.deepcopy(_load("ablations-v1.yaml"))
    invalid["joint_shortlist"][0][field] = replacement
    invalid["protocol_sha256"] = _canonical_sha(_protocol_payload(invalid))

    with pytest.raises(ValueError, match="shortlist|method_config|lane"):
        protocol.validate_protocol(invalid)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (
            ("scientific_contract_sha256",),
            "0" * 64,
            "scientific_contract",
        ),
        (
            ("acquisition_configs", "base", "pattern_family"),
            "random",
            "pattern_family|bernoulli",
        ),
        (
            ("acquisition_configs", "base", "noise_calibration_id"),
            "legacy-snr",
            "noise|detector-absolute",
        ),
    ],
)
def test_campaign_contract_bindings_cannot_be_mutated(
    path: tuple[str, ...], value: object, message: str
):
    invalid = copy.deepcopy(_load("primary-v1.yaml"))
    cursor = invalid
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    _refresh_hashes(invalid)

    with pytest.raises(ValueError, match=message):
        protocol.validate_protocol(invalid)


@pytest.mark.parametrize(
    ("document_name", "mutate", "message"),
    [
        (
            "methods-v1.yaml",
            lambda document: document["methods"].append(
                copy.deepcopy(document["methods"][0])
            ),
            "duplicate method",
        ),
        (
            "methods-v1.yaml",
            lambda document: next(
                entry
                for entry in document["methods"]
                if entry["id"] == "gsdiff_diffusion"
            ).__setitem__("lane", "tv"),
            "diffusion|lane",
        ),
        (
            "scientific-contracts-v1.yaml",
            lambda document: next(
                entry
                for entry in document["contracts"]
                if entry["id"] == "gsdiff-sim-v1"
            )["content"]["rng"].__setitem__("bit_generator", "MT19937"),
            "PCG64|rng",
        ),
        (
            "scientific-contracts-v1.yaml",
            lambda document: next(
                entry
                for entry in document["contracts"]
                if entry["id"] == "gsdiff-sim-v1"
            )["content"]["generator"].__setitem__("motion_type", "transrot"),
            "custom_se2|motion_type",
        ),
        (
            "noise-calibration-v1.yaml",
            lambda document: document["calibrations"][0].__setitem__(
                "variance_ddof", 1
            ),
            "ddof|population",
        ),
        (
            "ablations-v1.yaml",
            lambda document: document["selection_seeds"].append(73),
            "confirmatory|selection",
        ),
        (
            "ablations-v1.yaml",
            lambda document: document["resolver"]["diffusion"].__setitem__(
                "method", "gsdiff_tv"
            ),
            "diffusion|gsdiff_diffusion",
        ),
    ],
)
def test_locked_registry_and_ablation_literals_reject_semantic_mutation(
    document_name: str, mutate, message: str
):
    invalid = copy.deepcopy(_load(document_name))
    mutate(invalid)
    if document_name == "scientific-contracts-v1.yaml":
        for entry in invalid["contracts"]:
            entry["sha256"] = _canonical_sha(entry["content"])
    invalid["protocol_sha256"] = _canonical_sha(_protocol_payload(invalid))

    with pytest.raises(ValueError, match=message):
        protocol.validate_protocol(invalid)


@pytest.mark.parametrize(
    ("name", "old", "new", "message"),
    [
        (
            "anchor",
            "schema_version: experiment-protocol-v1",
            "schema_version: &schema experiment-protocol-v1",
            "anchor|alias",
        ),
        (
            "alias",
            "document_kind: campaign",
            "document_kind: &kind campaign\nunsafe_copy: *kind",
            "anchor|alias",
        ),
        (
            "merge",
            "campaign_id: primary-v1",
            "defaults: &defaults {campaign_id: primary-v1}\n<<: *defaults",
            "anchor|alias",
        ),
        (
            "timestamp",
            "scientific_contract_id: gsdiff-sim-v1",
            "scientific_contract_id: 2026-07-27",
            "JSON|scalar|timestamp",
        ),
        (
            "binary",
            "scientific_contract_id: gsdiff-sim-v1",
            "scientific_contract_id: !!binary Z3NkaWZmLXNpbS12MQ==",
            "JSON|scalar|binary",
        ),
        (
            "nan",
            "expected_runs: 495",
            "expected_runs: .nan",
            "finite|JSON|scalar",
        ),
        (
            "infinity",
            "expected_runs: 495",
            "expected_runs: .inf",
            "finite|JSON|scalar",
        ),
    ],
)
def test_public_loader_rejects_unsafe_or_non_json_yaml_before_use(
    tmp_path: Path, name: str, old: str, new: str, message: str
):
    source = (PROTOCOLS / "primary-v1.yaml").read_text(encoding="utf-8")
    assert old in source
    path = tmp_path / f"{name}.yaml"
    path.write_text(source.replace(old, new, 1), encoding="utf-8")

    with pytest.raises((TypeError, ValueError), match=message):
        protocol.load_protocol(path)


def test_public_loader_rejects_unknown_top_level_key(tmp_path: Path):
    source = (PROTOCOLS / "primary-v1.yaml").read_text(encoding="utf-8")
    path = tmp_path / "unknown.yaml"
    path.write_text(source + "\nunknown_top_level: true\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unknown.*unknown_top_level|top-level"):
        protocol.load_protocol(path)


def test_supported_json_scalars_round_trip_from_real_yaml_documents():
    ablations = _load("ablations-v1.yaml")
    pilot = _load("pilot-v1.yaml")

    assert pilot["method_budgets"] is None
    assert pilot["execution_ready"] is False
    assert isinstance(pilot["acquisition_configs"]["base"]["image_size"][0], int)
    assert isinstance(ablations["axes"]["motion_warmup_fraction"][1], float)
    assert math.isfinite(ablations["axes"]["motion_warmup_fraction"][1])
    assert isinstance(ablations["selection_rule"]["objective_formula"], str)
