"""Fail-closed resolution for the versioned experiment method registry."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from types import MappingProxyType

from .identity import canonical_json_bytes, sha256_bytes
from .protocol import load_protocol


CANONICAL_METHOD_IDS = (
    "dgi", "static_cs", "perframe_cs", "tv3d", "monin", "gidc3dtv",
    "recinr", "siren", "recinr_se2", "gsdiff_tv", "gsdiff_diffusion",
)
_METHOD_ALIASES = {"gsdiff_diff": "gsdiff_diffusion"}
_SHA256 = re.compile(r"[0-9a-f]{64}", flags=re.ASCII)
_WINDOWS_ABSOLUTE = re.compile(r"(?:[A-Za-z]:[\\/]|\\\\)")
_STAGING_UUID = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
    flags=re.IGNORECASE,
)
METHODS_REGISTRY_PROTOCOL_SHA256 = (
    "ef3d613267360538f4ac0e6301f6a854b8d0487eecde5ed98cf51ee6a8a0275c"
)


@dataclass(frozen=True)
class CheckpointRequirement:
    logical_id: str
    sha256: str
    provenance_status: str


@dataclass(frozen=True)
class ResolvedMethod:
    method_id: str
    requested_method_config_id: str
    method_config_id: str
    execution_family: str
    command_template: tuple[str, ...]
    semantic_config: Mapping[str, object]
    method_config_sha256: str
    required_child_outputs: tuple[str, ...]
    checkpoint_requirements: tuple[CheckpointRequirement, ...]
    execution_profile: str
    publication_eligible: bool
    selection_eligible: bool
    promotion_eligible: bool
    convergence_status: str
    execution_ready: bool
    execution_blockers: tuple[str, ...]


@dataclass(frozen=True)
class MethodResolutionRequest:
    requested_method_id: str
    requested_method_config_id: str
    base_config: Mapping[str, object]
    measurements_metadata: Mapping[str, object]
    requested_execution_profile: str

    def __post_init__(self) -> None:
        for name in (
            "requested_method_id",
            "requested_method_config_id",
            "requested_execution_profile",
        ):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise ValueError(f"{name} must be a nonempty exact string")
        object.__setattr__(
            self,
            "base_config",
            _snapshot_json_mapping(self.base_config, name="base_config"),
        )
        object.__setattr__(
            self,
            "measurements_metadata",
            _snapshot_json_mapping(
                self.measurements_metadata,
                name="measurements_metadata",
            ),
        )


@dataclass(frozen=True)
class AlgorithmSeed:
    derivation_sha256: str
    seed_u32: int


def canonical_method_id(method_id: str) -> str:
    if type(method_id) is not str:
        raise TypeError("method_id must be a string")
    canonical = _METHOD_ALIASES.get(method_id, method_id)
    if canonical not in CANONICAL_METHOD_IDS:
        raise ValueError(f"unknown method: {method_id!r}")
    return canonical


def resolve_method_semantics(
    method_id: str,
    *,
    method_config_id: str,
    base_config: Mapping[str, object],
    measurements_metadata: Mapping[str, object],
    execution_profile: str,
    registry_path: Path = Path("configs/protocols/methods-v1.yaml"),
) -> ResolvedMethod:
    """Resolve only declared, path-free method semantics from the registry."""
    canonical_id = canonical_method_id(method_id)
    if type(method_config_id) is not str or not method_config_id:
        raise ValueError("method_config_id must be a nonempty string")
    if not isinstance(base_config, Mapping) or not isinstance(measurements_metadata, Mapping):
        raise TypeError("base_config and measurements_metadata must be mappings")
    if type(execution_profile) is not str:
        raise TypeError("execution_profile must be a string")
    _reject_unsafe_semantic_value(base_config)

    registry = load_protocol(registry_path)
    if registry.get("protocol_sha256") != METHODS_REGISTRY_PROTOCOL_SHA256:
        raise ValueError("methods registry protocol hash is not the locked version")
    aliases = registry["campaign_execution_profile_aliases"]
    assert isinstance(aliases, Mapping)
    requested_profile = execution_profile
    normalized_profile = aliases.get(requested_profile, requested_profile)
    if type(normalized_profile) is not str:
        raise ValueError("unknown execution profile")
    entries = registry["methods"]
    assert isinstance(entries, list)
    entry = next((item for item in entries if item["id"] == canonical_id), None)
    if entry is None:
        raise ValueError(f"unknown method: {method_id!r}")
    assert isinstance(entry, Mapping)
    profiles = entry["profiles"]
    assert isinstance(profiles, Mapping)
    profile = profiles.get(normalized_profile)
    if not isinstance(profile, Mapping):
        raise ValueError(f"unknown execution profile: {execution_profile!r}")

    normalized_config_id = method_config_id
    pilot_aliases = registry["pilot_method_config_aliases"]
    assert isinstance(pilot_aliases, Mapping)
    if requested_profile in pilot_aliases:
        profile_aliases = pilot_aliases[requested_profile]
        assert isinstance(profile_aliases, Mapping)
        normalized_config_id = profile_aliases.get(method_config_id, method_config_id)
    if type(normalized_config_id) is not str:
        raise ValueError("method config alias is invalid")

    declared_config_id = profile["method_config_id"]
    if normalized_profile == "ablation-selection-v1":
        semantics = _resolve_ablation_semantics(canonical_id, method_config_id, base_config, profile)
        normalized_config_id = method_config_id
    else:
        if normalized_config_id != declared_config_id:
            raise ValueError("method config ID does not match the selected profile")
        semantics = profile["semantic_config"]
        if not isinstance(semantics, Mapping):
            raise ValueError("semantic config must be a mapping")
        _validate_base_config(canonical_id, base_config, semantics)

    frozen_semantics = _freeze_json(semantics)
    if not isinstance(frozen_semantics, Mapping):
        raise ValueError("semantic config must be a mapping")
    checkpoints = _checkpoint_requirements(entry["checkpoints"])
    command_template = tuple(entry["command_template"])
    outputs = tuple(entry["required_child_outputs"])
    policy = {
        name: profile[name]
        for name in (
            "publication_eligible", "selection_eligible", "promotion_eligible",
            "convergence_status", "execution_ready", "execution_blockers",
        )
    }
    checkpoint_payload = [
        {"logical_id": item.logical_id, "sha256": item.sha256, "provenance_status": item.provenance_status}
        for item in checkpoints
    ]
    payload = {
        "method_id": canonical_id,
        "method_config_id": normalized_config_id,
        "execution_family": entry["execution_family"],
        "execution_profile": normalized_profile,
        "command_template": list(command_template),
        "semantic_config": thaw_json(frozen_semantics),
        "checkpoint_requirements": checkpoint_payload,
        "required_child_outputs": ["reconstruction.npz", "method-info.json"],
        "profile_policy": {
            "publication_eligible": policy["publication_eligible"],
            "selection_eligible": policy["selection_eligible"],
            "promotion_eligible": policy["promotion_eligible"],
            "convergence_status": policy["convergence_status"],
            "execution_ready": policy["execution_ready"],
            "execution_blockers": list(policy["execution_blockers"]),
        },
    }
    return ResolvedMethod(
        method_id=canonical_id,
        requested_method_config_id=method_config_id,
        method_config_id=normalized_config_id,
        execution_family=entry["execution_family"],
        command_template=command_template,
        semantic_config=frozen_semantics,
        method_config_sha256=sha256_bytes(canonical_json_bytes(payload)),
        required_child_outputs=outputs,
        checkpoint_requirements=checkpoints,
        execution_profile=normalized_profile,
        publication_eligible=profile["publication_eligible"],
        selection_eligible=profile["selection_eligible"],
        promotion_eligible=profile["promotion_eligible"],
        convergence_status=profile["convergence_status"],
        execution_ready=profile["execution_ready"],
        execution_blockers=tuple(profile["execution_blockers"]),
    )


def derive_algorithm_seed(
    *, cell_seed: int, dataset_identity_sha256: str, method_id: str,
    method_config_sha256: str,
) -> AlgorithmSeed:
    if type(cell_seed) is not int or cell_seed < 0:
        raise ValueError("cell_seed must be a nonnegative integer")
    _require_sha256("dataset_identity_sha256", dataset_identity_sha256)
    _require_sha256("method_config_sha256", method_config_sha256)
    payload = {
        "domain": "algorithm-seed-v1", "cell_seed": cell_seed,
        "dataset_identity_sha256": dataset_identity_sha256,
        "method_id": canonical_method_id(method_id),
        "method_config_sha256": method_config_sha256,
    }
    digest = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return AlgorithmSeed(digest, int.from_bytes(bytes.fromhex(digest)[:4], "big"))


def thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: thaw_json(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(child) for child in value]
    return value


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_json(child) for key, child in value.items()})
    if isinstance(value, list) or isinstance(value, tuple):
        return tuple(_freeze_json(child) for child in value)
    return value


def _snapshot_json_mapping(
    value: object,
    *,
    name: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    try:
        snapshot = json.loads(
            canonical_json_bytes(dict(value)).decode("utf-8", errors="strict")
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain exact JSON values") from error
    frozen = _freeze_json(snapshot)
    if not isinstance(frozen, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return frozen


def _validate_base_config(method_id: str, base_config: Mapping[str, object], semantics: Mapping[str, object]) -> None:
    unknown = set(base_config) - {"gaussian_count"}
    if unknown:
        raise ValueError(f"undeclared base_config override: {sorted(unknown)!r}")
    scene = semantics.get("scene")
    gaussian_count = scene.get("gaussian_count") if isinstance(scene, Mapping) else None
    supplied = base_config.get("gaussian_count")
    if gaussian_count is None:
        if supplied is not None:
            raise ValueError("gaussian_count is inactive for this method")
        return
    if type(supplied) is not int or type(supplied) is bool or supplied != gaussian_count:
        raise ValueError("gaussian_count must equal the locked Gaussian count")
    if method_id == "recinr" and ({"scene", "solver"} & set(base_config)):
        raise ValueError("native recinr does not accept generic scene or solver")


def _resolve_ablation_semantics(method_id: str, method_config_id: str, base_config: Mapping[str, object], profile: Mapping[str, object]) -> Mapping[str, object]:
    semantic_config = profile["semantic_config"]
    assert isinstance(semantic_config, Mapping)
    declared = semantic_config["declared_joint_configs"]
    assert isinstance(declared, Mapping)
    candidate = declared.get(method_config_id)
    if not isinstance(candidate, Mapping):
        raise ValueError("unknown declared ablation method config")
    if canonical_json_bytes(dict(base_config)) != canonical_json_bytes(candidate):
        raise ValueError("ablation base_config does not exactly match its declared config")
    return candidate


def _checkpoint_requirements(value: object) -> tuple[CheckpointRequirement, ...]:
    if not isinstance(value, list):
        raise ValueError("checkpoints must be a list")
    return tuple(CheckpointRequirement(item["logical_id"], item["sha256"], item["provenance_status"]) for item in value)


def _require_sha256(name: str, value: object) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be 64 lowercase hexadecimal characters")
    return value


def _reject_unsafe_semantic_value(value: object) -> None:
    if isinstance(value, Mapping):
        for child in value.values():
            _reject_unsafe_semantic_value(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _reject_unsafe_semantic_value(child)
    elif type(value) is str:
        if value.startswith("/") or _WINDOWS_ABSOLUTE.search(value) or _STAGING_UUID.search(value) or "${" in value:
            raise ValueError("path or token cannot enter semantic identity")
