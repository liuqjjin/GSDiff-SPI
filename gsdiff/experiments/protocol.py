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


def _validate_methods_registry(protocol: Mapping[str, object]) -> None:
    keys = {
        "schema_version",
        "document_kind",
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
    entries = _require_list("methods", document["methods"])
    ids = []
    for entry in entries:
        method = _require_exact_keys("method entry", entry, {"id", "lane"})
        ids.append(_require_string("method.id", method["id"]))
        if method["lane"] != expected_lanes.get(method["id"]):
            raise ValueError(f"method {method['id']!r} has an invalid diffusion/TV lane")
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
