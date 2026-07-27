"""Canonical identity and generic array validation for SPI artifacts."""

import hashlib
import json
from typing import Mapping

import numpy as np


_HEX_DIGITS = frozenset("0123456789abcdef")


class ArtifactValidationError(ValueError):
    """An artifact does not satisfy its declared schema or identity."""


def json_native(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_native(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_native(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ArtifactValidationError(
        f"value of type {type(value).__name__} is not JSON-native"
    )


def canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            json_native(value),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ArtifactValidationError("metadata is not canonical JSON") from exc


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def validate_sha256(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX_DIGITS for character in value)
    ):
        raise ArtifactValidationError(
            f"{field} must be 64 lowercase hexadecimal characters"
        )
    return value


def readonly_array(value: object, field: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise ArtifactValidationError(f"{field} cannot use object dtype")
    array = np.ascontiguousarray(array).copy()
    array.flags.writeable = False
    return array


def optional_readonly_array(
    value: object | None, field: str
) -> np.ndarray | None:
    return None if value is None else readonly_array(value, field)


def array_descriptor(
    array: np.ndarray | None,
) -> Mapping[str, object] | None:
    if array is None:
        return None
    if array.dtype.hasobject:
        raise ArtifactValidationError("object arrays cannot be serialized")
    content = np.ascontiguousarray(array).tobytes(order="C")
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": sha256_bytes(content),
    }


def validate_array_descriptor(
    name: str, array: np.ndarray, descriptor: object
) -> None:
    if not isinstance(descriptor, Mapping):
        raise ArtifactValidationError(f"missing array descriptor for {name}")
    expected = array_descriptor(array)
    if canonical_json_bytes(descriptor) != canonical_json_bytes(expected):
        raise ArtifactValidationError(
            f"schema/content-hash mismatch for array {name}"
        )


def validate_exact_keys(
    value: Mapping[str, object], expected: set[str], label: str
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ArtifactValidationError(
            f"{label} keys mismatch; missing={missing}, extra={extra}"
        )


def validate_generation_config(
    config: Mapping[str, object],
) -> Mapping[str, object]:
    native = json_native(config)
    if not isinstance(native, dict):
        raise ArtifactValidationError("resolved_generation_config must be a mapping")
    required = {
        "H",
        "W",
        "T",
        "K",
        "seed",
        "target",
        "pattern",
        "time_assignment",
        "noise",
        "motion",
        "holdout",
    }
    missing = required - set(native)
    if missing:
        raise ArtifactValidationError(
            f"resolved generation config missing {sorted(missing)}"
        )
    nested_requirements = {
        "pattern": {"family", "order"},
        "time_assignment": {"mode"},
        "noise": {"convention", "parameters"},
        "motion": {"model", "parameters"},
        "holdout": {"count", "pattern_family", "seed_offset"},
    }
    for name, keys in nested_requirements.items():
        section = native[name]
        if not isinstance(section, dict) or not keys.issubset(section):
            raise ArtifactValidationError(
                f"resolved generation config has incomplete {name} section"
            )
    motion_parameters = native["motion"]["parameters"]
    if not isinstance(motion_parameters, dict) or not {
        "velocity",
        "acceleration",
        "omega",
        "beta",
    }.issubset(motion_parameters):
        raise ArtifactValidationError(
            "motion parameters must include velocity, acceleration, omega, and beta"
        )
    if not isinstance(native["noise"]["parameters"], dict):
        raise ArtifactValidationError("noise parameters must be a mapping")
    canonical_json_bytes(native)
    return native


def acquisition_identity_spec(
    *,
    arrays: Mapping[str, np.ndarray | None],
    H: int,
    W: int,
    T: int,
    K: int,
    resolved_generation_config: Mapping[str, object],
    generator_code_version: str,
    target_asset_sha256: str,
    seed: int,
    pattern_family: str,
    pattern_order: str,
    time_assignment_mode: str,
    noise_convention: str,
    noise_parameters: Mapping[str, object],
    motion_model: str,
    motion_parameters: Mapping[str, object],
    schema: str,
) -> Mapping[str, object]:
    return {
        "schema": schema,
        "resolved_generation_config": json_native(resolved_generation_config),
        "generator_code_version": generator_code_version,
        "target_asset_sha256": target_asset_sha256,
        "seed": int(seed),
        "pattern_family": pattern_family,
        "pattern_order": pattern_order,
        "time_assignment_mode": time_assignment_mode,
        "noise": {
            "convention": noise_convention,
            "parameters": json_native(noise_parameters),
        },
        "motion": {
            "model": motion_model,
            "parameters": json_native(motion_parameters),
        },
        "dimensions": {"H": int(H), "W": int(W), "T": int(T), "K": int(K)},
        "arrays": {
            name: array_descriptor(array) for name, array in arrays.items()
        },
    }
