from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Any

import yaml
from yaml.tokens import AliasToken, AnchorToken

from .identity import canonical_json_bytes, sha256_bytes


_SCHEMA_VERSION = "experiment-protocol-v1"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$", flags=re.ASCII)
_METHODS = (
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
_METHOD_EXECUTION_FAMILIES = {
    "dgi": "baseline",
    "static_cs": "baseline",
    "perframe_cs": "baseline",
    "tv3d": "baseline",
    "monin": "baseline",
    "gidc3dtv": "baseline",
    "recinr": "baseline",
    "siren": "gsdiff",
    "recinr_se2": "gsdiff",
    "gsdiff_tv": "gsdiff",
    "gsdiff_diffusion": "gsdiff",
}
_METHOD_PROFILE_NAMES = (
    "publication-v1",
    "controller-cpu-smoke-v1",
    "ablation-selection-v1",
)
_METHOD_PROFILE_SEMANTIC_DIGESTS = {
    "dgi": ("43ca8aa144afd1dc592a2dab12d90e2536b3dbd478024f70605fd88774588834", "3c0929cc135db0fdee8042bd84f018d2854d8a60d866951e31a459f8d0b81868", "96e8e763676f162bf902f18242f530d8e256b347ef89b8cbc016599a8332109f"),
    "static_cs": ("a32acad6eef82e11954d091908deb97e72398030b1423a74eff178489fd9fdb7", "5a462259025ec7a0411d670c99bc03af1c1c355a52707ca902c8b55c889be7b9", "96e8e763676f162bf902f18242f530d8e256b347ef89b8cbc016599a8332109f"),
    "perframe_cs": ("e1891788145d28ab1b20cda5b5490c4f1c7773d97e53b7bdd4b4fb79d515d64a", "5a462259025ec7a0411d670c99bc03af1c1c355a52707ca902c8b55c889be7b9", "96e8e763676f162bf902f18242f530d8e256b347ef89b8cbc016599a8332109f"),
    "tv3d": ("92e600f8984f660f782bc95e907d06043e7f4717ad1a5c535409e226e0b88496", "44766d015599b48c0b14dac06bad9c6371a178931e5c544e5e1ee5613c3a5ae3", "96e8e763676f162bf902f18242f530d8e256b347ef89b8cbc016599a8332109f"),
    "monin": ("96d41e9ee44120c30572c9d6202dd7aa5464a2fc4bc275da0dd324c34848e11e", "99f0ef76bdf5cdd0781c1f752db00881fb5d799e3be1382b1430b92849a1e855", "96e8e763676f162bf902f18242f530d8e256b347ef89b8cbc016599a8332109f"),
    "gidc3dtv": ("d56f9f22fb904c91eb24ad11614ab330778b57596bf01d95d01eb70c3afa24aa", "5780f7de67e97716c400243d923b0f42195839a73aeaa09a5ff3f2dc9a890082", "96e8e763676f162bf902f18242f530d8e256b347ef89b8cbc016599a8332109f"),
    "recinr": ("adb866544a3bf45bc65f6556458ddb3077905a9cbe87d09cd47f82550d7e0218", "7b684716b1a6b9ea1a1f65f2c2d84f3ff3654535f9e4c552af972f55e274dfd8", "96e8e763676f162bf902f18242f530d8e256b347ef89b8cbc016599a8332109f"),
    "siren": ("cc1773d78bcb005cdf6ca049a3b20d641a66dd75b2d50750b44f4963492bac39", "068ae79f110320fa209cb317a27b362392051a1792e38b708ca7463e44eff7a4", "96e8e763676f162bf902f18242f530d8e256b347ef89b8cbc016599a8332109f"),
    "recinr_se2": ("e5ed145b33a028e568f24dfbf1985ff2ff33120e730578e17b04d62a6ede8cf4", "08ecea4d8243771a0e9e077156efc010d6655e463ad4bb6fe17f32677e9e4540", "96e8e763676f162bf902f18242f530d8e256b347ef89b8cbc016599a8332109f"),
    "gsdiff_tv": ("cc7b8b968367b5671a3cb906abee7d88b975ef83b1857120b5f71883c9c0436e", "93f5876cdfe2111e62babed734ef69ac462606b1409a007dac59707ff8440246", "b35e21f70c6b3a2a32235709a2f5a9bf16cbe3fe0917ed178565c89982a8c905"),
    "gsdiff_diffusion": ("77d71af07e653f9e7a5fe69e8fb981942959013196be3d580aa5ec52c279eeed", "5391552fc212c91c66fe11dc336fc9152b34f95f5c02d3eda5a6d27d7bcfb1fd", "fe76b8f92ba4ba6fc735648d6c528c43d1dbf2c17b91b46dd5fe8d0d1a376477"),
}
_METHOD_PROFILE_SEMANTIC_SHA256 = {
    method_id: dict(zip(_METHOD_PROFILE_NAMES, digests, strict=True))
    for method_id, digests in _METHOD_PROFILE_SEMANTIC_DIGESTS.items()
}
_TARGETS = {
    "tank": "assets/tank.png",
    "digit5": "char:5",
    "letterR": "char:R",
    "usaf": "assets/usaf_tar_small.png",
    "cx_camera": "assets/cx_camera.png",
    "cx_clutter": "assets/cx_clutter.png",
    "cx_coins": "assets/cx_coins.png",
    "cx_text": "assets/cx_text.png",
}
_MOTIONS = {
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
_ACQUISITION_PHYSICS = {
    "pattern_family": "bernoulli",
    "pattern_values": [0, 1],
    "pattern_order": "stratified",
    "time_assignment": "uniform",
    "holdout_pattern_family": "uniform-random",
}
_RNG = {
    "bit_generator": "PCG64",
    "seed_sequence": "SeedSequence(entropy=seed, spawn_key=(stream_id,))",
    "streams": {
        "train-pattern": 0,
        "train-noise": 1,
        "holdout-pattern": 2,
        "holdout-noise": 3,
    },
}
_NOISE = {
    "calibration_id": "detector-absolute-v1",
    "mode": "detector-absolute",
    "reference": "corresponding-bernoulli-reference-cell",
    "sigma_formula": "sqrt(var(y_reference,ddof=0))*10**(-snr_db/20)",
    "reuse": ["train", "holdout", "alternate-pattern"],
}
_GENERATOR = {
    "id": "gsdiff-corrected-sim",
    "version": "generator-v1",
    "motion_type": "custom_se2",
    "normalized_time": [0, 1],
}


def _contract_content(purpose: str) -> dict[str, object]:
    return {
        "purpose": purpose,
        "generator": _GENERATOR,
        "targets": _TARGETS,
        "motions": _MOTIONS,
        "acquisition": _ACQUISITION_PHYSICS,
        "rng": _RNG,
        "noise": _NOISE,
    }


_CONTRACT_CONTENTS = {
    "gsdiff-pilot-v1": _contract_content("pilot-smoke"),
    "gsdiff-sim-v1": _contract_content("corrected-simulation"),
    "gsdiff-ablation-v1": _contract_content("controlled-ablation"),
}
_CONTRACT_HASHES = {
    contract_id: sha256_bytes(canonical_json_bytes(content))
    for contract_id, content in _CONTRACT_CONTENTS.items()
}

_CAMPAIGNS = {
    "pilot-v1": {
        "contract": "gsdiff-pilot-v1",
        "targets": {"tank"},
        "motions": {"transrot"},
        "seeds": {7},
        "methods": set(_METHODS),
        "acquisition_ids": {"base"},
        "runs": 11,
        "datasets": 1,
        "profile": "pilot-smoke-v1",
    },
    "primary-v1": {
        "contract": "gsdiff-sim-v1",
        "targets": {"tank", "digit5", "usaf"},
        "motions": {"trans", "rot", "transrot"},
        "seeds": {7, 11, 42, 73, 101},
        "methods": set(_METHODS),
        "acquisition_ids": {"base"},
        "runs": 495,
        "datasets": 45,
        "profile": "primary-full-v1",
    },
    "supplement-grid-v1": {
        "contract": "gsdiff-sim-v1",
        "targets": {"tank", "digit5", "letterR", "usaf"},
        "motions": {"trans", "rot", "transrot", "accel"},
        "seeds": {7, 11, 42},
        "methods": set(_METHODS),
        "acquisition_ids": {"base"},
        "runs": 528,
        "datasets": 48,
        "profile": "supplement-full-v1",
    },
    "ood-v1": {
        "contract": "gsdiff-sim-v1",
        "targets": {"cx_camera", "cx_clutter"},
        "motions": {"trans", "rot", "transrot"},
        "seeds": {7, 11, 42},
        "methods": set(_METHODS),
        "acquisition_ids": {"base"},
        "runs": 198,
        "datasets": 18,
        "profile": "ood-full-v1",
    },
    "failure-v1": {
        "contract": "gsdiff-sim-v1",
        "targets": {"cx_coins", "cx_text"},
        "motions": {"transrot"},
        "seeds": {7, 11, 42},
        "methods": {
            "dgi",
            "tv3d",
            "recinr_se2",
            "gsdiff_tv",
            "gsdiff_diffusion",
        },
        "acquisition_ids": {
            "m320",
            "m640",
            "m1280",
            "m2560",
            "m3840",
            "m5120",
        },
        "runs": 180,
        "datasets": 36,
        "profile": "failure-budget-v1",
    },
}

_CAMPAIGN_KEYS = {
    "schema_version",
    "document_kind",
    "campaign_id",
    "execution_ready",
    "execution_profile",
    "method_budgets",
    "scientific_contract_id",
    "scientific_contract_sha256",
    "metric_version",
    "acquisition_configs",
    "matrix",
    "expected_runs",
    "expected_datasets",
    "campaign_sha256",
    "protocol_sha256",
}
_MATRIX_KEYS = {
    "targets",
    "motions",
    "seeds",
    "methods",
    "acquisition_config_ids",
    "method_config_ids",
}
_ACQUISITION_KEYS = {
    "image_size",
    "num_frames",
    "train_measurements",
    "holdout_measurements",
    "pattern_family",
    "pattern_values",
    "pattern_order",
    "time_assignment",
    "holdout_pattern_family",
    "snr_db",
    "noise_calibration_id",
}
_CONFIRMATION_KEYS = {
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
_EXPECTED_CONFIRMATION = {
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


@dataclass(frozen=True)
class ExperimentCell:
    scientific_contract_id: str
    scientific_contract_sha256: str
    campaign_id: str
    target: str
    motion: str
    seed: int
    method: str
    acquisition_config_id: str = "base"
    method_config_id: str = "default"


class _RestrictedSafeLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _RestrictedSafeLoader, node: yaml.nodes.MappingNode, deep: bool = False
) -> dict[object, object]:
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as error:
            raise ValueError("YAML mapping keys must be JSON strings") from error
        if duplicate:
            raise ValueError(f"duplicate YAML mapping key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_RestrictedSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _require_exact_keys(
    name: str,
    value: object,
    expected: set[str],
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ValueError(
            f"{name} keys do not match: missing={missing}, unknown={unknown}"
        )
    return value


def _require_list(name: str, value: object) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a list")
    return value


def _require_string(name: str, value: object) -> str:
    if type(value) is not str or not value:
        raise TypeError(f"{name} must be a nonempty exact string")
    return value


def _require_integer(name: str, value: object, *, positive: bool = False) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact integer")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _require_sha256(name: str, value: object) -> str:
    if type(value) is not str or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _validate_json_domain(value: object, path: str = "$") -> None:
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} must contain finite JSON numbers")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _validate_json_domain(child, f"{path}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if type(key) is not str:
                raise TypeError(f"{path} mapping keys must be JSON strings")
            _validate_json_domain(child, f"{path}.{key}")
        return
    raise TypeError(
        f"{path} contains non-JSON scalar or collection type "
        f"{type(value).__name__}"
    )


def _without_protocol_hash(protocol: Mapping[str, object]) -> dict[str, object]:
    return {key: value for key, value in protocol.items() if key != "protocol_sha256"}


def _protocol_sha256(protocol: Mapping[str, object]) -> str:
    return sha256_bytes(canonical_json_bytes(_without_protocol_hash(protocol)))


def _canonical_equal(left: object, right: object) -> bool:
    return canonical_json_bytes(left) == canonical_json_bytes(right)


def _membership_records(protocol: Mapping[str, object]) -> list[dict[str, object]]:
    matrix = protocol["matrix"]
    if not isinstance(matrix, Mapping):
        raise TypeError("matrix must be a mapping")
    method_config_ids = matrix["method_config_ids"]
    if not isinstance(method_config_ids, Mapping):
        raise TypeError("matrix.method_config_ids must be a mapping")
    records = []
    for target in matrix["targets"]:  # type: ignore[union-attr]
        for motion in matrix["motions"]:  # type: ignore[union-attr]
            for seed in matrix["seeds"]:  # type: ignore[union-attr]
                for method in matrix["methods"]:  # type: ignore[union-attr]
                    for acquisition_config_id in matrix[
                        "acquisition_config_ids"
                    ]:  # type: ignore[union-attr]
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


def _campaign_sha256(protocol: Mapping[str, object]) -> str:
    records = sorted(_membership_records(protocol), key=canonical_json_bytes)
    return sha256_bytes(canonical_json_bytes(records))


def load_protocol(path: Path) -> dict[str, object]:
    if not isinstance(path, Path):
        raise TypeError("path must be a Path")
    try:
        text = path.read_text(encoding="utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError("protocol YAML must be strict UTF-8") from error
    try:
        tokens = yaml.scan(text, Loader=_RestrictedSafeLoader)
        for token in tokens:
            if isinstance(token, (AnchorToken, AliasToken)):
                raise ValueError(
                    "YAML anchors and aliases are forbidden before construction"
                )
        loaded = yaml.load(text, Loader=_RestrictedSafeLoader)
    except yaml.YAMLError as error:
        raise ValueError("invalid restricted YAML") from error
    if not isinstance(loaded, dict):
        raise TypeError("protocol document must be a YAML mapping")
    _validate_json_domain(loaded)
    validate_protocol(loaded)
    return loaded


def expand_cells(protocol: Mapping[str, object]) -> tuple[ExperimentCell, ...]:
    validate_protocol(protocol)
    if protocol["document_kind"] != "campaign":
        raise ValueError("only campaign documents expand to experiment cells")
    matrix = protocol["matrix"]
    assert isinstance(matrix, Mapping)
    method_config_ids = matrix["method_config_ids"]
    assert isinstance(method_config_ids, Mapping)
    return tuple(
        ExperimentCell(
            scientific_contract_id=protocol["scientific_contract_id"],  # type: ignore[arg-type]
            scientific_contract_sha256=protocol[
                "scientific_contract_sha256"
            ],  # type: ignore[arg-type]
            campaign_id=protocol["campaign_id"],  # type: ignore[arg-type]
            target=target,  # type: ignore[arg-type]
            motion=motion,  # type: ignore[arg-type]
            seed=seed,  # type: ignore[arg-type]
            method=method,  # type: ignore[arg-type]
            acquisition_config_id=acquisition_config_id,  # type: ignore[arg-type]
            method_config_id=method_config_ids[method],  # type: ignore[arg-type]
        )
        for target in matrix["targets"]  # type: ignore[union-attr]
        for motion in matrix["motions"]  # type: ignore[union-attr]
        for seed in matrix["seeds"]  # type: ignore[union-attr]
        for method in matrix["methods"]  # type: ignore[union-attr]
        for acquisition_config_id in matrix[
            "acquisition_config_ids"
        ]  # type: ignore[union-attr]
    )


def validate_protocol(protocol: Mapping[str, object]) -> None:
    if not isinstance(protocol, Mapping):
        raise TypeError("protocol must be a mapping")
    _validate_json_domain(protocol)
    if protocol.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {_SCHEMA_VERSION!r}")
    kind = protocol.get("document_kind")
    if kind == "campaign":
        _validate_campaign(protocol)
    elif kind == "methods-registry":
        _validate_methods_registry(protocol)
    elif kind == "scientific-contracts-registry":
        _validate_contracts_registry(protocol)
    elif kind == "noise-calibration-registry":
        _validate_noise_registry(protocol)
    elif kind == "ablation":
        _validate_ablation(protocol)
    else:
        raise ValueError(f"unknown document_kind: {kind!r}")
    declared = _require_sha256("protocol_sha256", protocol["protocol_sha256"])
    expected = _protocol_sha256(protocol)
    if declared != expected:
        raise ValueError("protocol_sha256 does not hash the normalized document")


def _validate_campaign(protocol: Mapping[str, object]) -> None:
    keys = set(_CAMPAIGN_KEYS)
    campaign_id = protocol.get("campaign_id")
    if campaign_id == "primary-v1":
        keys.add("confirmation_rule")
    document = _require_exact_keys("campaign top-level", protocol, keys)
    campaign_id = _require_string("campaign_id", document["campaign_id"])
    if campaign_id not in _CAMPAIGNS:
        raise ValueError(f"unknown campaign_id: {campaign_id!r}")
    locked = _CAMPAIGNS[campaign_id]

    if type(document["execution_ready"]) is not bool:
        raise TypeError("execution_ready must be an exact boolean")
    if document["execution_profile"] != locked["profile"]:
        raise ValueError("execution_profile does not match the locked campaign")
    budgets = document["method_budgets"]
    if document["execution_ready"]:
        if not isinstance(budgets, Mapping) or set(budgets) != set(
            _require_list("matrix.methods", document["matrix"]["methods"])  # type: ignore[index]
        ):
            raise ValueError(
                "execution_ready campaigns require exact per-method method_budgets"
            )
        for method, budget in budgets.items():
            _require_integer(f"method_budgets[{method!r}]", budget, positive=True)
    elif budgets is not None:
        raise ValueError("non-ready campaign method_budgets must be null")
    if campaign_id == "pilot-v1" and document["execution_ready"] is not False:
        raise ValueError("pilot-v1 execution_ready must remain false in Task 1")

    contract_id = _require_string(
        "scientific_contract_id", document["scientific_contract_id"]
    )
    if contract_id != locked["contract"]:
        raise ValueError("scientific_contract_id does not match campaign")
    contract_sha = _require_sha256(
        "scientific_contract_sha256",
        document["scientific_contract_sha256"],
    )
    if contract_sha != _CONTRACT_HASHES[contract_id]:
        raise ValueError("scientific_contract_sha256 does not match contract content")
    if document["metric_version"] != "metrics-v1":
        raise ValueError("metric_version must be exactly 'metrics-v1'")

    matrix = _require_exact_keys("matrix", document["matrix"], _MATRIX_KEYS)
    targets = _validate_unique_strings("matrix.targets", matrix["targets"])
    motions = _validate_unique_strings("matrix.motions", matrix["motions"])
    seeds = _require_list("matrix.seeds", matrix["seeds"])
    if not seeds:
        raise ValueError("matrix seed list must not be empty")
    for seed in seeds:
        _require_integer("matrix seed", seed)
    methods = _validate_unique_strings("matrix.methods", matrix["methods"])
    acquisition_ids = _validate_unique_strings(
        "matrix.acquisition_config_ids", matrix["acquisition_config_ids"]
    )
    unknown_targets = set(targets) - set(_TARGETS)
    unknown_motions = set(motions) - set(_MOTIONS)
    unknown_methods = set(methods) - set(_METHODS)
    if unknown_targets:
        raise ValueError(f"unknown target IDs: {sorted(unknown_targets)}")
    if unknown_motions:
        raise ValueError(f"unknown motion IDs: {sorted(unknown_motions)}")
    if unknown_methods:
        raise ValueError(f"unknown method IDs: {sorted(unknown_methods)}")
    method_configs = matrix["method_config_ids"]
    if not isinstance(method_configs, Mapping):
        raise TypeError("matrix.method_config_ids must be a mapping")
    if set(method_configs) != set(methods):
        raise ValueError("matrix.method_config_ids must cover every method exactly")
    if any(value != "default" for value in method_configs.values()):
        raise ValueError("campaign method_config_id must be exactly 'default'")

    if set(targets) != locked["targets"]:
        raise ValueError("campaign target membership does not match the locked matrix")
    if set(motions) != locked["motions"]:
        raise ValueError("campaign motion membership does not match the locked matrix")
    if set(seeds) != locked["seeds"] or len(seeds) != len(set(seeds)):
        raise ValueError("campaign seed membership does not match the locked matrix")
    if set(methods) != locked["methods"]:
        raise ValueError("campaign method membership does not match the locked matrix")
    if set(acquisition_ids) != locked["acquisition_ids"]:
        raise ValueError(
            "campaign acquisition-config membership does not match the locked matrix"
        )

    configs = document["acquisition_configs"]
    if not isinstance(configs, Mapping):
        raise TypeError("acquisition_configs must be a mapping")
    if set(configs) != set(acquisition_ids):
        raise ValueError("acquisition_configs must cover matrix IDs exactly")
    for config_id, config in configs.items():
        _validate_acquisition_config(campaign_id, str(config_id), config)

    expected_runs = _require_integer(
        "expected_runs", document["expected_runs"], positive=True
    )
    expected_datasets = _require_integer(
        "expected_datasets", document["expected_datasets"], positive=True
    )
    if expected_runs != locked["runs"] or expected_datasets != locked["datasets"]:
        raise ValueError("declared campaign counts do not match locked counts")
    records = _membership_records(document)
    if len(records) != expected_runs or len({canonical_json_bytes(r) for r in records}) != len(
        records
    ):
        raise ValueError("expanded campaign membership is not exact and unique")
    acquisition_count = (
        len(targets) * len(motions) * len(seeds) * len(acquisition_ids)
    )
    if acquisition_count != expected_datasets:
        raise ValueError("expanded acquisition count does not match expected_datasets")
    declared_campaign_sha = _require_sha256(
        "campaign_sha256", document["campaign_sha256"]
    )
    if declared_campaign_sha != _campaign_sha256(document):
        raise ValueError(
            "campaign_sha256 does not hash the sorted logical membership set"
        )
    if campaign_id == "primary-v1":
        confirmation = _require_exact_keys(
            "confirmation_rule",
            document["confirmation_rule"],
            _CONFIRMATION_KEYS,
        )
        if not _canonical_equal(confirmation, _EXPECTED_CONFIRMATION):
            raise ValueError("confirmation rule does not match the locked declaration")


def _validate_unique_strings(name: str, value: object) -> list[str]:
    values = _require_list(name, value)
    normalized = [_require_string(f"{name} item", item) for item in values]
    if len(normalized) != len(set(normalized)):
        noun = "method" if "method" in name else name
        raise ValueError(f"duplicate {noun} ID in {name}")
    return normalized


def _validate_acquisition_config(
    campaign_id: str, config_id: str, value: object
) -> None:
    config = _require_exact_keys(
        f"acquisition_configs[{config_id!r}]",
        value,
        _ACQUISITION_KEYS,
    )
    image_size = _require_list("image_size", config["image_size"])
    if len(image_size) != 2:
        raise ValueError("image_size must have exactly two dimensions")
    for dimension in image_size:
        _require_integer("image_size dimension", dimension, positive=True)
    _require_integer("num_frames", config["num_frames"], positive=True)
    train_measurements = _require_integer(
        "train_measurements", config["train_measurements"], positive=True
    )
    _require_integer(
        "holdout_measurements",
        config["holdout_measurements"],
        positive=True,
    )
    if not _canonical_equal({
        key: config[key]
        for key in (
            "pattern_family",
            "pattern_values",
            "pattern_order",
            "time_assignment",
            "holdout_pattern_family",
        )
    }, _ACQUISITION_PHYSICS):
        raise ValueError(
            "acquisition must use locked Bernoulli pattern_family and corrected timing"
        )
    if _require_integer("snr_db", config["snr_db"]) != 25:
        raise ValueError("snr_db must be exactly 25")
    if config["noise_calibration_id"] != "detector-absolute-v1":
        raise ValueError("noise must use detector-absolute-v1 calibration")
    if campaign_id == "pilot-v1":
        expected = ([32, 32], 4, 128, 16)
    else:
        expected = (
            [64, 64],
            20,
            int(config_id[1:]) if campaign_id == "failure-v1" else 2560,
            250,
        )
    observed = (
        image_size,
        config["num_frames"],
        train_measurements,
        config["holdout_measurements"],
    )
    if observed != expected:
        raise ValueError(
            f"acquisition config {config_id!r} does not bind the locked full K/config"
        )


def _expected_method_profile_policy(
    method_id: str, profile_name: str
) -> dict[str, object]:
    if profile_name == "publication-v1":
        policy = {
            "method_config_id": "default",
            "publication_eligible": True,
            "selection_eligible": True,
            "promotion_eligible": True,
            "convergence_status": (
                "not-applicable" if method_id == "dgi" else "convergence-required"
            ),
            "execution_ready": True,
            "execution_blockers": [],
        }
        if method_id == "gsdiff_diffusion":
            policy.update(
                {
                    "publication_eligible": False,
                    "selection_eligible": False,
                    "promotion_eligible": False,
                    "execution_ready": False,
                    "execution_blockers": [
                        "missing-reproducible-checkpoint-locator",
                        "missing-checkpoint-training-provenance",
                    ],
                }
            )
        return policy
    if profile_name == "controller-cpu-smoke-v1":
        return {
            "method_config_id": "smoke-default-v1",
            "publication_eligible": False,
            "selection_eligible": False,
            "promotion_eligible": False,
            "convergence_status": "smoke-only/not-convergence-assessed",
            "execution_ready": True,
            "execution_blockers": [],
        }
    if profile_name == "ablation-selection-v1":
        return {
            "method_config_id": "ablation-joint-shortlist-v1",
            "publication_eligible": False,
            "selection_eligible": False,
            "promotion_eligible": False,
            "convergence_status": "not-applicable",
            "execution_ready": False,
            "execution_blockers": ["missing-versioned-ablation-native-budgets"],
        }
    raise ValueError(f"unknown method profile: {profile_name!r}")


def _validate_methods_registry(protocol: Mapping[str, object]) -> None:
    keys = {
        "schema_version",
        "document_kind",
        "campaign_execution_profile_aliases",
        "pilot_method_config_aliases",
        "methods",
        "primary_method",
        "secondary_diffusion_method",
        "confirmation_pair",
        "compatibility_aliases",
        "protocol_sha256",
    }
    document = _require_exact_keys("methods-registry top-level", protocol, keys)
    expected_lanes = {
        "dgi": "analytic",
        "static_cs": "static",
        "perframe_cs": "per-frame",
        "tv3d": "tv",
        "monin": "motion-estimation",
        "gidc3dtv": "deep-image-prior",
        "recinr": "native-baseline",
        "siren": "implicit-neural",
        "recinr_se2": "se2-inr",
        "gsdiff_tv": "tv",
        "gsdiff_diffusion": "diffusion",
    }
    if document["campaign_execution_profile_aliases"] != {
        "primary-full-v1": "publication-v1",
        "supplement-full-v1": "publication-v1",
        "ood-full-v1": "publication-v1",
        "failure-budget-v1": "publication-v1",
        "pilot-smoke-v1": "controller-cpu-smoke-v1",
        "ablation-selection-v1": "ablation-selection-v1",
    }:
        raise ValueError("campaign execution profile aliases are not locked")
    if document["pilot_method_config_aliases"] != {
        "pilot-smoke-v1": {"default": "smoke-default-v1"}
    }:
        raise ValueError("pilot method config aliases are not locked")
    entries = _require_list("methods", document["methods"])
    ids = []
    for entry in entries:
        method = _require_exact_keys(
            "method entry", entry,
            {"id", "lane", "execution_family", "command_template",
             "required_child_outputs", "checkpoints", "profiles"},
        )
        ids.append(_require_string("method.id", method["id"]))
        if method["lane"] != expected_lanes.get(method["id"]):
            raise ValueError(f"method {method['id']!r} has an invalid diffusion/TV lane")
        family = _METHOD_EXECUTION_FAMILIES[method["id"]]
        if method["execution_family"] != family:
            raise ValueError("method execution family is not locked")
        entrypoint = "train.py" if family == "gsdiff" else "scripts/run_baselines.py"
        expected_command = [
            "${PYTHON}", entrypoint, "--method", method["id"], "--dataset",
            "${MEASUREMENTS_PATH}", "--dataset-identity-sha256",
            "${DATASET_IDENTITY_SHA256}", "--method-config",
            "${METHOD_CONFIG_PATH}", "--algorithm-seed", "${ALGORITHM_SEED}",
            "--device", "${DEVICE}", "--output-dir", "${OUTPUT_DIR}",
        ]
        if method["id"] == "gsdiff_diffusion":
            expected_command += ["--checkpoint", "gsdiff-diffusion-prior-v1=${CHECKPOINT:gsdiff-diffusion-prior-v1}"]
        if method["command_template"] != expected_command:
            raise ValueError("method command template is not locked")
        if method["required_child_outputs"] != ["reconstruction.npz", "method-info.json"]:
            raise ValueError("method required child outputs are not locked")
        expected_checkpoints: list[object] = []
        if method["id"] == "gsdiff_diffusion":
            expected_checkpoints = [{
                "logical_id": "gsdiff-diffusion-prior-v1",
                "sha256": "667948800911acb9f9a7271e20af5692b0f007007d0fc32a15ac169eba32c5dd",
                "provenance_status": "blocked-missing-training-provenance",
            }]
        if method["checkpoints"] != expected_checkpoints:
            raise ValueError("method checkpoint declaration is not locked")
        profiles = _require_exact_keys(
            "method profiles", method["profiles"],
            set(_METHOD_PROFILE_NAMES),
        )
        for profile_name, profile in profiles.items():
            _require_exact_keys(
                f"method profile {profile_name}", profile,
                {"method_config_id", "publication_eligible", "selection_eligible",
                 "promotion_eligible", "convergence_status", "execution_ready",
                 "execution_blockers", "semantic_config"},
            )
            if not isinstance(profile["semantic_config"], Mapping):
                raise ValueError("method semantic config must be a mapping")
            for field in (
                "publication_eligible",
                "selection_eligible",
                "promotion_eligible",
                "execution_ready",
            ):
                if type(profile[field]) is not bool:
                    raise ValueError("method profile policy booleans must be exact booleans")
            profile_policy = {
                key: value
                for key, value in profile.items()
                if key != "semantic_config"
            }
            if profile_policy != _expected_method_profile_policy(
                method["id"], profile_name
            ):
                raise ValueError("method profile policy does not match locked values")
            expected_semantic_sha = _METHOD_PROFILE_SEMANTIC_SHA256[method["id"]][profile_name]
            if sha256_bytes(canonical_json_bytes(profile["semantic_config"])) != expected_semantic_sha:
                raise ValueError("method semantic config does not match locked values")
        publication = profiles["publication-v1"]
        if not isinstance(publication, Mapping):
            raise ValueError("publication profile must be a mapping")
        if method["id"] == "dgi":
            if publication["convergence_status"] != "not-applicable":
                raise ValueError("DGI publication profile is analytic")
        elif publication["convergence_status"] != "convergence-required":
            raise ValueError("optimizing publication profile requires convergence")
        if not isinstance(publication["semantic_config"].get("compute_cap"), Mapping):
            raise ValueError("publication semantic config must bind compute cap")
        if method["id"] == "gsdiff_tv":
            for profile_name in ("publication-v1", "controller-cpu-smoke-v1"):
                semantic_config = profiles[profile_name]["semantic_config"]
                assert isinstance(semantic_config, Mapping)
                solver = semantic_config["solver"]
                assert isinstance(solver, Mapping)
                expected_warmup = math.ceil(
                    solver["motion_warmup_fraction"]
                    * solver["outer_iterations"]
                )
                if solver["motion_warmup_outer"] != expected_warmup:
                    raise ValueError(
                        "GSDiff TV motion warmup integer does not match ceil-derived value"
                    )
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate method ID in methods registry")
    if ids != list(_METHODS):
        raise ValueError("methods registry must preserve canonical declared order")
    if document["primary_method"] != "gsdiff_tv":
        raise ValueError("primary method must remain gsdiff_tv")
    if document["secondary_diffusion_method"] != "gsdiff_diffusion":
        raise ValueError("diffusion secondary method must remain gsdiff_diffusion")
    if document["confirmation_pair"] != {
        "method": "gsdiff_tv",
        "comparator": "recinr_se2",
    }:
        raise ValueError("confirmation pair must be gsdiff_tv versus recinr_se2")
    if document["compatibility_aliases"] != {
        "gsdiff_diff": "gsdiff_diffusion"
    }:
        raise ValueError("gsdiff_diff may only be a compatibility alias")


def _validate_contracts_registry(protocol: Mapping[str, object]) -> None:
    keys = {
        "schema_version",
        "document_kind",
        "contracts",
        "protocol_sha256",
    }
    document = _require_exact_keys(
        "scientific-contracts-registry top-level", protocol, keys
    )
    entries = _require_list("contracts", document["contracts"])
    observed: dict[str, bytes] = {}
    for entry in entries:
        contract = _require_exact_keys(
            "scientific contract entry", entry, {"id", "sha256", "content"}
        )
        contract_id = _require_string("scientific contract id", contract["id"])
        content = contract["content"]
        _require_exact_keys(
            "scientific contract content",
            content,
            {
                "purpose",
                "generator",
                "targets",
                "motions",
                "acquisition",
                "rng",
                "noise",
            },
        )
        encoded = canonical_json_bytes(content)
        declared_sha = _require_sha256(
            "scientific contract sha256", contract["sha256"]
        )
        if declared_sha != sha256_bytes(encoded):
            raise ValueError("scientific contract sha256 does not hash its content")
        if contract_id in observed:
            if observed[contract_id] != encoded:
                raise ValueError(
                    "scientific contract ID reuse has changed contract content"
                )
            raise ValueError("duplicate scientific contract ID")
        observed[contract_id] = encoded
        if contract_id not in _CONTRACT_CONTENTS:
            raise ValueError(f"unknown scientific contract ID: {contract_id!r}")
        if not _canonical_equal(content, _CONTRACT_CONTENTS[contract_id]):
            _validate_contract_literals(content)
            raise ValueError(
                f"scientific contract {contract_id!r} content is not locked"
            )
    if list(observed) != list(_CONTRACT_CONTENTS):
        raise ValueError("scientific contracts registry is incomplete or reordered")


def _validate_contract_literals(content: object) -> None:
    if not isinstance(content, Mapping):
        raise TypeError("scientific contract content must be a mapping")
    generator = content.get("generator")
    if isinstance(generator, Mapping):
        if generator.get("motion_type") != "custom_se2":
            raise ValueError("generator motion_type must resolve aliases to custom_se2")
        if generator.get("version") != "generator-v1":
            raise ValueError("generator version must bind corrected generator-v1")
    if not _canonical_equal(content.get("rng"), _RNG):
        raise ValueError("rng must use PCG64 and locked SeedSequence streams")
    if not _canonical_equal(content.get("noise"), _NOISE):
        raise ValueError("noise must use detector-absolute population calibration")
    if not _canonical_equal(content.get("targets"), _TARGETS):
        raise ValueError("target mappings do not match the locked assets")
    if not _canonical_equal(content.get("motions"), _MOTIONS):
        raise ValueError("motion aliases do not match locked custom_se2 parameters")
    if not _canonical_equal(content.get("acquisition"), _ACQUISITION_PHYSICS):
        raise ValueError("acquisition physics does not match corrected literals")


def _validate_noise_registry(protocol: Mapping[str, object]) -> None:
    keys = {
        "schema_version",
        "document_kind",
        "calibrations",
        "protocol_sha256",
    }
    document = _require_exact_keys(
        "noise-calibration-registry top-level", protocol, keys
    )
    calibrations = _require_list("calibrations", document["calibrations"])
    if len(calibrations) != 1:
        raise ValueError("noise calibration registry must contain one locked entry")
    calibration = _require_exact_keys(
        "noise calibration",
        calibrations[0],
        {
            "id",
            "mode",
            "reference",
            "variance_ddof",
            "sigma_formula",
            "reuse",
        },
    )
    expected = {
        "id": "detector-absolute-v1",
        "mode": "detector-absolute",
        "reference": "corresponding-bernoulli-reference-cell",
        "variance_ddof": 0,
        "sigma_formula": "sqrt(var(y_reference,ddof=0))*10**(-snr_db/20)",
        "reuse": ["train", "holdout", "alternate-pattern"],
    }
    if not _canonical_equal(calibration, expected):
        if calibration.get("variance_ddof") != 0:
            raise ValueError("noise variance ddof must be population ddof=0")
        raise ValueError("noise calibration does not match detector-absolute-v1")


def _validate_ablation(protocol: Mapping[str, object]) -> None:
    keys = {
        "schema_version",
        "document_kind",
        "scientific_contract_id",
        "scientific_contract_sha256",
        "metric_version",
        "execution_ready",
        "execution_profile",
        "method_budgets",
        "selection_anchors",
        "selection_seeds",
        "forbidden_confirmation_seeds",
        "axes",
        "semantic_anchor",
        "joint_shortlist",
        "method_config_ids",
        "resolver",
        "selection_rule",
        "workflow_counts",
        "protocol_sha256",
    }
    document = _require_exact_keys("ablation top-level", protocol, keys)
    if document["scientific_contract_id"] != "gsdiff-ablation-v1":
        raise ValueError("ablation scientific contract ID is not locked")
    if (
        document["scientific_contract_sha256"]
        != _CONTRACT_HASHES["gsdiff-ablation-v1"]
    ):
        raise ValueError("ablation scientific contract hash is not locked")
    if document["metric_version"] != "metrics-v1":
        raise ValueError("ablation metric_version must be metrics-v1")
    if document["execution_ready"] is not False or document["method_budgets"] is not None:
        raise ValueError("ablation is not execution_ready before method budgets")
    if document["execution_profile"] != "ablation-selection-v1":
        raise ValueError("ablation execution_profile is not locked")
    expected_anchors = [
        {"target": "tank", "motion": "trans"},
        {"target": "digit5", "motion": "rot"},
        {"target": "usaf", "motion": "transrot"},
    ]
    if not _canonical_equal(document["selection_anchors"], expected_anchors):
        raise ValueError("ablation selection anchors are not locked")
    selection_seeds = _require_list("selection_seeds", document["selection_seeds"])
    for seed in selection_seeds:
        _require_integer("selection seed", seed)
    if selection_seeds != [7, 11, 42]:
        if {73, 101} & set(document["selection_seeds"]):  # type: ignore[arg-type]
            raise ValueError("confirmatory seeds are forbidden during selection")
        raise ValueError("ablation selection seeds are not locked")
    forbidden_seeds = _require_list(
        "forbidden_confirmation_seeds",
        document["forbidden_confirmation_seeds"],
    )
    for seed in forbidden_seeds:
        _require_integer("forbidden confirmation seed", seed)
    if forbidden_seeds != [73, 101]:
        raise ValueError("ablation confirmatory seed guard is not locked")
    expected_axes = {
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
    if not _canonical_equal(document["axes"], expected_axes):
        raise ValueError("ablation axes are not locked")
    expected_anchor = {
        "representation": "gaussian",
        "solver": "admm",
        "prior": "tv3d_corrected",
        "motion_warmup_fraction": 0.2,
        "temporal_tv_weight": 0.1,
        "gaussian_count": 1000,
    }
    if not _canonical_equal(document["semantic_anchor"], expected_anchor):
        raise ValueError("ablation semantic anchor is not locked")
    shortlist = _require_list("joint_shortlist", document["joint_shortlist"])
    expected_shortlist = [
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
    if not _canonical_equal(shortlist, expected_shortlist):
        raise ValueError("ablation joint shortlist is not the exact six configs")
    method_config_ids = document["method_config_ids"]
    if not isinstance(method_config_ids, Mapping):
        raise TypeError("ablation method_config_ids must be a mapping")
    if set(method_config_ids) != {"j1", "j2", "j3", "j4", "j5", "j6"}:
        raise ValueError("ablation method_config_ids are incomplete")
    values = list(method_config_ids.values())
    if len(values) != len(set(values)):
        raise ValueError("ablation method_config_ids must be identity-distinct")
    if any(
        method_config_ids[entry["id"]] != entry["method_config_id"]
        for entry in shortlist
    ):
        raise ValueError("shortlist method_config_id linkage is inconsistent")
    expected_resolver = {
        "hqs": {"solver": "admm", "mode": "hqs"},
        "tv2d": {"method": "gsdiff_tv", "lane": "tv"},
        "tv3d_corrected": {"method": "gsdiff_tv", "lane": "tv"},
        "diffusion": {"method": "gsdiff_diffusion", "lane": "diffusion"},
        "recinr": {"method": "recinr", "lane": "native-baseline"},
    }
    if not _canonical_equal(document["resolver"], expected_resolver):
        diffusion = document["resolver"].get("diffusion", {})  # type: ignore[union-attr]
        if diffusion.get("method") != "gsdiff_diffusion":
            raise ValueError("diffusion prior must resolve to gsdiff_diffusion lane")
        raise ValueError("ablation resolver does not match locked lane mappings")
    _validate_selection_rule(document["selection_rule"])
    expected_counts = {
        "one_factor_unique_configurations": 17,
        "joint_shortlist_configurations": 6,
        "decision_cells": 207,
        "post_freeze_replay_cells": 207,
        "stress_configurations": 14,
        "stress_cells": 126,
        "publication_commit_cells": 333,
        "total_before_retries": 540,
    }
    if not _canonical_equal(document["workflow_counts"], expected_counts):
        raise ValueError("ablation workflow counts are not locked")


def _validate_selection_rule(value: object) -> None:
    rule = _require_exact_keys(
        "selection_rule",
        value,
        {
            "objective",
            "objective_formula",
            "aggregation",
            "required_complete_cells",
            "residual_tie_relative_tolerance",
            "convergence",
            "compute_cap",
            "tie_break_order",
        },
    )
    expected = {
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
    if not _canonical_equal(rule, expected):
        raise ValueError("selection rule does not match the locked declaration")
