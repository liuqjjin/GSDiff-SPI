from __future__ import annotations

import hashlib
import math
from pathlib import Path

import pytest

from gsdiff.experiments import methods as methods_module
from gsdiff.experiments.methods import (
    CANONICAL_METHOD_IDS,
    canonical_method_id,
    derive_algorithm_seed,
    native_iteration_contract_v1,
    resolve_method_semantics,
)
from gsdiff.experiments.identity import canonical_json_bytes


REGISTRY = Path("configs/protocols/methods-v1.yaml")
METHODS = (
    "dgi", "static_cs", "perframe_cs", "tv3d", "monin", "gidc3dtv",
    "recinr", "siren", "recinr_se2", "gsdiff_tv", "gsdiff_diffusion",
)
EXPECTED_DIFFUSION_CONFIG = {
    "denoise_steps": 1,
    "clamp_range": (0.0, 1.0),
    "in_channels": 1,
    "base_channels": 32,
    "channel_mults": (1, 2, 4),
    "emb_dim": 128,
    "sigma_min": 0.002,
    "sigma_max": 0.5,
    "sigma_start": 0.3,
    "sigma_end": 0.05,
    "renoise": False,
    "ddim_spacing": "linear",
}


def _measurements_metadata() -> dict[str, object]:
    return {"H": 32, "W": 32, "T": 4, "K": 128, "holdout_K": 16}


def test_method_resolution_request_snapshots_raw_resolver_inputs() -> None:
    base_config: dict[str, object] = {"gaussian_count": 1000}
    metadata = _measurements_metadata()

    request = methods_module.MethodResolutionRequest(
        requested_method_id="gsdiff_diff",
        requested_method_config_id="default",
        base_config=base_config,
        measurements_metadata=metadata,
        requested_execution_profile="primary-full-v1",
    )
    base_config["gaussian_count"] = 5
    metadata["T"] = 99

    assert request.requested_method_id == "gsdiff_diff"
    assert request.requested_method_config_id == "default"
    assert dict(request.base_config) == {"gaussian_count": 1000}
    assert dict(request.measurements_metadata) == {
        "H": 32,
        "W": 32,
        "T": 4,
        "K": 128,
        "holdout_K": 16,
    }
    assert request.requested_execution_profile == "primary-full-v1"
    with pytest.raises(TypeError):
        request.base_config["gaussian_count"] = 7  # type: ignore[index]
    with pytest.raises(TypeError):
        request.measurements_metadata["T"] = 7  # type: ignore[index]


def test_methods_registry_protocol_anchor_is_locked() -> None:
    assert methods_module.METHODS_REGISTRY_PROTOCOL_SHA256 == (
        "c2dbf832389948b6a43174bbcd37874116a26794b233715067597b67f7a962bf"
    )


def resolve_publication(method_id: str, *, base_config: dict[str, object]):
    return resolve_method_semantics(
        method_id, method_config_id="default", base_config=base_config,
        measurements_metadata=_measurements_metadata(), execution_profile="publication-v1",
        registry_path=REGISTRY,
    )


def test_registry_contains_exactly_eleven_canonical_ids() -> None:
    assert CANONICAL_METHOD_IDS == METHODS
    assert len(CANONICAL_METHOD_IDS) == len(set(CANONICAL_METHOD_IDS))


@pytest.mark.parametrize(
    ("profile", "config_id", "expected_budgets"),
    [
        (
            "publication-v1",
            "default",
            (1, 150, 120, 500, 150, 2500, 1900, 4000, 3000, 80, 80),
        ),
        (
            "controller-cpu-smoke-v1",
            "smoke-default-v1",
            (1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1),
        ),
    ],
)
def test_native_iteration_contract_derives_every_method_budget(
    profile: str,
    config_id: str,
    expected_budgets: tuple[int, ...],
) -> None:
    expected_units = (
        "pass",
        "admm-iteration",
        "admm-iteration",
        "primal-dual-iteration",
        "admm-iteration",
        "adam-step",
        "optimization-step",
        "sgd-step",
        "sgd-step",
        "outer-iteration",
        "outer-iteration",
    )

    for method_id, expected_unit, expected_budget in zip(
        METHODS,
        expected_units,
        expected_budgets,
        strict=True,
    ):
        base_config = (
            {"gaussian_count": 1000}
            if method_id in {"gsdiff_tv", "gsdiff_diffusion"}
            else {}
        )
        method = resolve_method_semantics(
            method_id,
            method_config_id=config_id,
            base_config=base_config,
            measurements_metadata=_measurements_metadata(),
            execution_profile=profile,
            registry_path=REGISTRY,
        )

        assert native_iteration_contract_v1(method) == {
            "unit": expected_unit,
            "budget": expected_budget,
        }


def test_native_iteration_contract_returns_a_fresh_mapping() -> None:
    method = resolve_publication("dgi", base_config={})
    first = native_iteration_contract_v1(method)
    first["budget"] = 99

    assert native_iteration_contract_v1(method) == {"unit": "pass", "budget": 1}


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


@pytest.mark.parametrize(
    ("profile", "config_id"),
    [
        ("publication-v1", "default"),
        ("controller-cpu-smoke-v1", "smoke-default-v1"),
    ],
)
def test_diffusion_profiles_declare_every_scientific_constructor_value(
    profile: str,
    config_id: str,
) -> None:
    method = resolve_method_semantics(
        "gsdiff_diffusion",
        method_config_id=config_id,
        base_config={"gaussian_count": 1000},
        measurements_metadata=_measurements_metadata(),
        execution_profile=profile,
        registry_path=REGISTRY,
    )

    assert dict(method.semantic_config["diffusion"]) == (
        EXPECTED_DIFFUSION_CONFIG
    )


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


def test_all_methods_use_the_locked_execution_family_and_entrypoint() -> None:
    expected_families = {
        **{method_id: "baseline" for method_id in METHODS[:7]},
        **{method_id: "gsdiff" for method_id in METHODS[7:]},
    }
    for method_id, family in expected_families.items():
        base_config = {"gaussian_count": 1000} if method_id.startswith("gsdiff_") else {}
        method = resolve_publication(method_id, base_config=base_config)
        expected_entrypoint = "train.py" if family == "gsdiff" else "scripts/run_baselines.py"
        assert method.execution_family == family
        assert method.command_template[:2] == ("${PYTHON}", expected_entrypoint)


@pytest.mark.parametrize(
    "method_id", ["siren", "recinr_se2", "gsdiff_tv", "gsdiff_diffusion"]
)
@pytest.mark.parametrize(
    ("profile", "config_id"),
    [("publication-v1", "default"), ("controller-cpu-smoke-v1", "smoke-default-v1")],
)
def test_gsdiff_profiles_bind_all_shared_constructor_semantics(method_id: str, profile: str, config_id: str) -> None:
    base_config = {"gaussian_count": 1000} if method_id.startswith("gsdiff_") else {}
    method = resolve_method_semantics(method_id, method_config_id=config_id, base_config=base_config, measurements_metadata=_measurements_metadata(), execution_profile=profile, registry_path=REGISTRY)
    assert dict(method.semantic_config["motion"]) == {
        "enable_rotation": True,
        "polynomial_degree": 1,
        "enable_affine": False,
    }
    solver = method.semantic_config["solver"]
    assert solver["loss_norm"] == "zscore"
    assert solver["lr_motion"] == 0.15
    assert solver["tv_weight"] == 0.005
    if method_id == "siren":
        assert solver["motion_warmup_steps"] == 0


def test_tv_smoke_motion_warmup_integer_matches_zero_fraction() -> None:
    method = resolve_method_semantics("gsdiff_tv", method_config_id="smoke-default-v1", base_config={"gaussian_count": 1000}, measurements_metadata=_measurements_metadata(), execution_profile="controller-cpu-smoke-v1", registry_path=REGISTRY)
    solver = method.semantic_config["solver"]
    assert solver["motion_warmup_fraction"] == 0.0
    assert solver["motion_warmup_outer"] == 0
    assert solver["motion_warmup_outer"] == math.ceil(solver["motion_warmup_fraction"] * solver["outer_iterations"])
