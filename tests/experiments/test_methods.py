from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from gsdiff.experiments.methods import (
    CANONICAL_METHOD_IDS,
    canonical_method_id,
    derive_algorithm_seed,
    resolve_method_semantics,
)
from gsdiff.experiments.identity import canonical_json_bytes


REGISTRY = Path("configs/protocols/methods-v1.yaml")
METHODS = (
    "dgi", "static_cs", "perframe_cs", "tv3d", "monin", "gidc3dtv",
    "recinr", "siren", "recinr_se2", "gsdiff_tv", "gsdiff_diffusion",
)


def _measurements_metadata() -> dict[str, object]:
    return {"H": 32, "W": 32, "T": 4, "K": 128, "holdout_K": 16}


def resolve_publication(method_id: str, *, base_config: dict[str, object]):
    return resolve_method_semantics(
        method_id, method_config_id="default", base_config=base_config,
        measurements_metadata=_measurements_metadata(), execution_profile="publication-v1",
        registry_path=REGISTRY,
    )


def test_registry_contains_exactly_eleven_canonical_ids() -> None:
    assert CANONICAL_METHOD_IDS == METHODS
    assert len(CANONICAL_METHOD_IDS) == len(set(CANONICAL_METHOD_IDS))


def test_alias_is_input_only() -> None:
    assert canonical_method_id("gsdiff_diff") == "gsdiff_diffusion"
    assert canonical_method_id("gsdiff_diffusion") == "gsdiff_diffusion"
    with pytest.raises(ValueError, match="unknown method"):
        canonical_method_id("gsdiff-admm")


def test_campaign_profile_aliases_reuse_one_method_identity() -> None:
    primary = resolve_method_semantics("dgi", method_config_id="default", base_config={}, measurements_metadata=_measurements_metadata(), execution_profile="primary-full-v1", registry_path=REGISTRY)
    supplement = resolve_method_semantics("dgi", method_config_id="default", base_config={}, measurements_metadata=_measurements_metadata(), execution_profile="supplement-full-v1", registry_path=REGISTRY)
    assert primary.execution_profile == supplement.execution_profile == "publication-v1"
    assert primary.method_config_sha256 == supplement.method_config_sha256


def test_locked_pilot_default_normalizes_to_distinct_smoke_config() -> None:
    pilot = resolve_method_semantics("dgi", method_config_id="default", base_config={}, measurements_metadata=_measurements_metadata(), execution_profile="pilot-smoke-v1", registry_path=REGISTRY)
    publication = resolve_publication("dgi", base_config={})
    assert pilot.requested_method_config_id == "default"
    assert pilot.method_config_id == "smoke-default-v1"
    assert pilot.execution_profile == "controller-cpu-smoke-v1"
    assert pilot.method_config_sha256 != publication.method_config_sha256


def test_declared_ablation_is_hashed_but_execution_blocked() -> None:
    method = resolve_method_semantics("gsdiff_diffusion", method_config_id="ablation-j1-v1", base_config={"representation": "recinr_se2", "solver": "hqs", "prior": "diffusion", "motion_warmup_fraction": 0.2, "temporal_tv_weight": 0.1, "gaussian_count": None}, measurements_metadata=_measurements_metadata(), execution_profile="ablation-selection-v1", registry_path=REGISTRY)
    assert method.execution_ready is False
    assert method.execution_blockers == ("missing-versioned-ablation-native-budgets",)
    assert method.selection_eligible is False


def test_tv_and_diffusion_resolve_to_distinct_semantics() -> None:
    tv = resolve_publication("gsdiff_tv", base_config={"gaussian_count": 1000})
    diffusion = resolve_publication("gsdiff_diff", base_config={"gaussian_count": 1000})
    assert tv.semantic_config["solver"]["prior_type"] == "tv"
    assert diffusion.method_id == "gsdiff_diffusion"
    assert diffusion.semantic_config["solver"]["prior_type"] == "diffusion"
    assert tv.method_config_sha256 != diffusion.method_config_sha256


def test_absolute_path_cannot_enter_semantic_identity() -> None:
    with pytest.raises(ValueError, match="path"):
        resolve_publication("dgi", base_config={"output_dir": r"D:\\leak"})


@pytest.mark.parametrize("method_id", [m for m in METHODS if not m.startswith("gsdiff_")])
def test_gaussian_count_is_rejected_when_inactive(method_id: str) -> None:
    with pytest.raises(ValueError, match="gaussian_count"):
        resolve_publication(method_id, base_config={"gaussian_count": 1000})


def test_null_inactive_gaussian_count_hashes_like_absent() -> None:
    assert resolve_publication("siren", base_config={}).method_config_sha256 == resolve_publication("siren", base_config={"gaussian_count": None}).method_config_sha256


def test_algorithm_seed_matches_locked_domain_derivation() -> None:
    result = derive_algorithm_seed(cell_seed=42, dataset_identity_sha256="1" * 64, method_id="gsdiff_tv", method_config_sha256="2" * 64)
    payload = canonical_json_bytes({"domain": "algorithm-seed-v1", "cell_seed": 42, "dataset_identity_sha256": "1" * 64, "method_id": "gsdiff_tv", "method_config_sha256": "2" * 64})
    digest = hashlib.sha256(payload).hexdigest()
    assert result.derivation_sha256 == digest
    assert result.seed_u32 == int.from_bytes(bytes.fromhex(digest)[:4], "big")


def test_method_hash_is_reconstructed_without_requested_aliases() -> None:
    method = resolve_method_semantics("gsdiff_diff", method_config_id="default", base_config={"gaussian_count": 1000}, measurements_metadata=_measurements_metadata(), execution_profile="primary-full-v1", registry_path=REGISTRY)
    payload = {
        "method_id": "gsdiff_diffusion", "method_config_id": "default",
        "execution_family": method.execution_family, "execution_profile": "publication-v1",
        "command_template": list(method.command_template),
        "semantic_config": {key: value for key, value in method.semantic_config.items()},
        "checkpoint_requirements": [{"logical_id": item.logical_id, "sha256": item.sha256, "provenance_status": item.provenance_status} for item in method.checkpoint_requirements],
        "required_child_outputs": ["reconstruction.npz", "method-info.json"],
        "profile_policy": {"publication_eligible": method.publication_eligible, "selection_eligible": method.selection_eligible, "promotion_eligible": method.promotion_eligible, "convergence_status": method.convergence_status, "execution_ready": method.execution_ready, "execution_blockers": list(method.execution_blockers)},
    }
    assert method.method_config_sha256 == hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def test_pilot_alias_matches_direct_smoke_identity() -> None:
    pilot = resolve_method_semantics("dgi", method_config_id="default", base_config={}, measurements_metadata=_measurements_metadata(), execution_profile="pilot-smoke-v1", registry_path=REGISTRY)
    direct = resolve_method_semantics("dgi", method_config_id="smoke-default-v1", base_config={}, measurements_metadata=_measurements_metadata(), execution_profile="controller-cpu-smoke-v1", registry_path=REGISTRY)
    assert pilot.method_config_sha256 == direct.method_config_sha256


@pytest.mark.parametrize(
    ("method_id", "method_config_id", "base_config"),
    [
        ("gsdiff_diffusion", "ablation-j1-v1", {"representation": "recinr_se2", "solver": "hqs", "prior": "diffusion", "motion_warmup_fraction": 0.2, "temporal_tv_weight": 0.1, "gaussian_count": None}),
        ("gsdiff_diffusion", "ablation-j2-v1", {"representation": "grid", "solver": "admm", "prior": "diffusion", "motion_warmup_fraction": 0.2, "temporal_tv_weight": 0.1, "gaussian_count": None}),
        ("gsdiff_tv", "ablation-j3-v1", {"representation": "siren", "solver": "sgd", "prior": "tv3d_corrected", "motion_warmup_fraction": 0.1, "temporal_tv_weight": 0.05, "gaussian_count": None}),
        ("gsdiff_tv", "ablation-j4-v1", {"representation": "gaussian", "solver": "hqs", "prior": "tv2d", "motion_warmup_fraction": 0.1, "temporal_tv_weight": 0.05, "gaussian_count": 1500}),
        ("gsdiff_tv", "ablation-j5-v1", {"representation": "recinr_se2", "solver": "sgd", "prior": "tv2d", "motion_warmup_fraction": 0.4, "temporal_tv_weight": 0.3, "gaussian_count": None}),
        ("gsdiff_tv", "ablation-j6-v1", {"representation": "grid", "solver": "hqs", "prior": "tv3d_corrected", "motion_warmup_fraction": 0.4, "temporal_tv_weight": 0.05, "gaussian_count": None}),
    ],
)
def test_ablation_joint_configs_are_cross_locked(method_id: str, method_config_id: str, base_config: dict[str, object]) -> None:
    result = resolve_method_semantics(method_id, method_config_id=method_config_id, base_config=base_config, measurements_metadata=_measurements_metadata(), execution_profile="ablation-selection-v1", registry_path=REGISTRY)
    assert dict(result.semantic_config) == base_config


def test_semantics_are_recursively_frozen_and_detached() -> None:
    method = resolve_publication("gsdiff_tv", base_config={"gaussian_count": 1000})
    with pytest.raises(TypeError):
        method.semantic_config["scene"] = {}  # type: ignore[index]
    with pytest.raises(TypeError):
        method.semantic_config["scene"]["gaussian_count"] = 10  # type: ignore[index]
    assert isinstance(method.semantic_config["solver"]["outer_iterations"], int)  # type: ignore[index]


@pytest.mark.parametrize("cell_seed", [True, -1, 1.0])
def test_algorithm_seed_rejects_invalid_cell_seed(cell_seed: object) -> None:
    with pytest.raises(ValueError):
        derive_algorithm_seed(cell_seed=cell_seed, dataset_identity_sha256="1" * 64, method_id="dgi", method_config_sha256="2" * 64)  # type: ignore[arg-type]


@pytest.mark.parametrize("invalid_hash", ["A" * 64, "g" * 64, "1" * 63, "1" * 63 + "\n"])
def test_algorithm_seed_rejects_noncanonical_hashes(invalid_hash: str) -> None:
    with pytest.raises(ValueError):
        derive_algorithm_seed(cell_seed=1, dataset_identity_sha256=invalid_hash, method_id="dgi", method_config_sha256="2" * 64)


@pytest.mark.parametrize("value", ["/tmp/leak", r"\\server\share", "${OUTPUT_DIR}", {"nested": r"D:\\leak"}])
def test_path_hygiene_precedes_override_validation(value: object) -> None:
    with pytest.raises(ValueError, match="path"):
        resolve_publication("dgi", base_config={"unexpected": value})
