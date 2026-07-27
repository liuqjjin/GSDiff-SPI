"""Canonical identity and generic array validation for SPI artifacts."""

import hashlib
import json
import math
import re
from types import MappingProxyType
from typing import Mapping

import numpy as np


_HEX_DIGITS = frozenset("0123456789abcdef")
_TARGET_DESCRIPTOR_PATTERN = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z", re.ASCII
)
_BUILTIN_CHAR_TARGET_PATTERN = re.compile(r"char:[A-Za-z0-9]\Z", re.ASCII)
_TARGET_KINDS = frozenset({"builtin", "asset"})
_TARGET_DESCRIPTOR_RESERVED_TOKENS = (
    "truth",
    "evaluation",
    "evaluator",
    "canonical",
    "trajectory",
    "metric",
    "display",
    "normalized",
)
_TARGET_DESCRIPTOR_RESERVED_SEGMENTS = frozenset({"gt"})


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
    contiguous = np.ascontiguousarray(array)
    immutable_bytes = contiguous.tobytes(order="C")
    return np.frombuffer(immutable_bytes, dtype=contiguous.dtype).reshape(
        contiguous.shape
    )


def optional_readonly_array(
    value: object | None, field: str
) -> np.ndarray | None:
    return None if value is None else readonly_array(value, field)


def deep_freeze_json(value: object) -> object:
    native = json_native(value)

    def freeze(item: object) -> object:
        if isinstance(item, Mapping):
            return MappingProxyType(
                {str(key): freeze(child) for key, child in item.items()}
            )
        if isinstance(item, list):
            return tuple(freeze(child) for child in item)
        return item

    return freeze(native)


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


def validate_exact_int(
    value: object,
    field: str,
    *,
    minimum: int | None = None,
) -> int:
    if type(value) is not int:
        raise ArtifactValidationError(f"{field} must be an exact integer")
    if minimum is not None and value < minimum:
        raise ArtifactValidationError(f"{field} must be at least {minimum}")
    return value


def validate_finite_number(
    value: object,
    field: str,
    *,
    minimum: float | None = None,
) -> int | float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ArtifactValidationError(f"{field} must be a finite number")
    if minimum is not None and value < minimum:
        raise ArtifactValidationError(f"{field} must be at least {minimum}")
    return value


def validate_real_finite_array(
    value: object,
    field: str,
    *,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ArtifactValidationError(
            f"{field} must be a real numeric finite array"
        ) from exc
    if (
        np.issubdtype(array.dtype, np.bool_)
        or not np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.complexfloating)
    ):
        raise ArtifactValidationError(
            f"{field} must be a real numeric finite array"
        )
    if shape is not None and array.shape != shape:
        raise ArtifactValidationError(f"{field} must have shape {shape}")
    if not np.all(np.isfinite(array)):
        raise ArtifactValidationError(
            f"{field} must be a real numeric finite array"
        )
    return array


def validate_index_array(
    value: object,
    field: str,
    *,
    shape: tuple[int, ...],
    upper_bound: int,
) -> np.ndarray:
    array = np.asarray(value)
    if (
        np.issubdtype(array.dtype, np.bool_)
        or not np.issubdtype(array.dtype, np.integer)
    ):
        raise ArtifactValidationError(f"{field} must use an integer dtype")
    if array.shape != shape:
        raise ArtifactValidationError(f"{field} must have shape {shape}")
    dtype_range = np.iinfo(array.dtype)
    if dtype_range.min > 0 or dtype_range.max < upper_bound - 1:
        raise ArtifactValidationError(
            f"{field} dtype must represent [0, {upper_bound})"
        )
    if np.any(array < 0) or np.any(array >= upper_bound):
        raise ArtifactValidationError(
            f"{field} values must be in [0, {upper_bound})"
        )
    return array


def validate_time_grid(
    value: object,
    field: str,
    *,
    length: int,
) -> np.ndarray:
    array = validate_real_finite_array(value, field, shape=(length,))
    if length > 1 and not np.all(array[1:] > array[:-1]):
        raise ArtifactValidationError(f"{field} must be strictly increasing")
    return array


def _validate_nonempty_string(value: object, field: str) -> str:
    if type(value) is not str or not value.strip():
        raise ArtifactValidationError(f"{field} must be a non-empty string")
    return value


def _validate_target_descriptor(kind: str, value: object) -> str:
    descriptor = _validate_nonempty_string(value, "target.descriptor")
    if (
        kind == "builtin"
        and _BUILTIN_CHAR_TARGET_PATTERN.fullmatch(descriptor) is not None
    ):
        return descriptor
    descriptor_lower = descriptor.lower()
    if (
        _TARGET_DESCRIPTOR_PATTERN.fullmatch(descriptor) is None
        or ".." in descriptor
        or any(
            token in descriptor_lower
            for token in _TARGET_DESCRIPTOR_RESERVED_TOKENS
        )
        or any(
            segment in _TARGET_DESCRIPTOR_RESERVED_SEGMENTS
            for segment in re.split(r"[._-]+", descriptor_lower)
        )
    ):
        raise ArtifactValidationError(
            "target.descriptor must be an opaque logical target ID"
        )
    return descriptor


def _validate_mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ArtifactValidationError(f"{field} must be a mapping")
    if any(type(key) is not str for key in value):
        raise ArtifactValidationError(f"{field} keys must be strings")
    return value


def _validate_vector(
    value: object, field: str, *, length: int
) -> list[int | float]:
    if type(value) not in (list, tuple) or len(value) != length:
        raise ArtifactValidationError(
            f"{field} must be a numeric sequence of length {length}"
        )
    return [
        validate_finite_number(item, f"{field}[{index}]")
        for index, item in enumerate(value)
    ]


def validate_generation_config(
    config: Mapping[str, object],
) -> Mapping[str, object]:
    config = _validate_mapping(config, "resolved_generation_config")
    required = {
        "schema",
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
    validate_exact_keys(config, required, "resolved generation config")
    if config["schema"] != "measurements-v1":
        raise ArtifactValidationError("resolved generation config schema mismatch")

    dimensions = {
        name: validate_exact_int(
            config[name], f"resolved_generation_config.{name}", minimum=1
        )
        for name in ("H", "W", "T", "K")
    }
    seed = validate_exact_int(
        config["seed"], "resolved_generation_config.seed"
    )

    target = _validate_mapping(
        config["target"], "resolved_generation_config.target"
    )
    validate_exact_keys(target, {"kind", "descriptor"}, "target")
    target_kind = _validate_nonempty_string(target["kind"], "target.kind")
    if target_kind not in _TARGET_KINDS:
        raise ArtifactValidationError("target.kind is unsupported")
    target_native = {
        "kind": target_kind,
        "descriptor": _validate_target_descriptor(
            target_kind, target["descriptor"]
        ),
    }

    pattern = _validate_mapping(
        config["pattern"], "resolved_generation_config.pattern"
    )
    validate_exact_keys(pattern, {"family", "order"}, "pattern")
    pattern_family = _validate_nonempty_string(
        pattern["family"], "pattern.family"
    )
    pattern_order = _validate_nonempty_string(pattern["order"], "pattern.order")
    if pattern_order not in {"sequential", "stratified", "random"}:
        raise ArtifactValidationError("pattern.order is unsupported")

    time_assignment = _validate_mapping(
        config["time_assignment"],
        "resolved_generation_config.time_assignment",
    )
    validate_exact_keys(time_assignment, {"mode"}, "time_assignment")
    time_mode = _validate_nonempty_string(
        time_assignment["mode"], "time_assignment.mode"
    )
    if time_mode not in {"uniform", "interpolation"}:
        raise ArtifactValidationError("time_assignment.mode is unsupported")

    noise = _validate_mapping(
        config["noise"], "resolved_generation_config.noise"
    )
    validate_exact_keys(noise, {"convention", "parameters"}, "noise")
    noise_convention = _validate_nonempty_string(
        noise["convention"], "noise.convention"
    )
    noise_parameters = _validate_mapping(
        noise["parameters"], "noise.parameters"
    )
    validate_exact_keys(
        noise_parameters, {"snr_db", "sigma_abs"}, "noise.parameters"
    )
    snr_db = validate_finite_number(
        noise_parameters["snr_db"], "noise.parameters.snr_db"
    )
    sigma_abs = noise_parameters["sigma_abs"]
    if sigma_abs is not None:
        sigma_abs = validate_finite_number(
            sigma_abs, "noise.parameters.sigma_abs", minimum=0.0
        )

    motion = _validate_mapping(
        config["motion"], "resolved_generation_config.motion"
    )
    validate_exact_keys(motion, {"model", "parameters"}, "motion")
    motion_model = _validate_nonempty_string(motion["model"], "motion.model")
    motion_parameters = _validate_mapping(
        motion["parameters"], "motion.parameters"
    )
    validate_exact_keys(
        motion_parameters,
        {
            "velocity",
            "acceleration",
            "omega",
            "beta",
            "speed_factor",
            "motion_mode",
        },
        "motion.parameters",
    )
    velocity = _validate_vector(
        motion_parameters["velocity"], "motion.parameters.velocity", length=2
    )
    acceleration = _validate_vector(
        motion_parameters["acceleration"],
        "motion.parameters.acceleration",
        length=2,
    )
    omega = validate_finite_number(
        motion_parameters["omega"], "motion.parameters.omega"
    )
    beta = validate_finite_number(
        motion_parameters["beta"], "motion.parameters.beta"
    )
    speed_factor = validate_finite_number(
        motion_parameters["speed_factor"],
        "motion.parameters.speed_factor",
        minimum=0.0,
    )
    motion_mode = validate_exact_int(
        motion_parameters["motion_mode"], "motion.parameters.motion_mode"
    )
    if motion_mode not in {1, 2}:
        raise ArtifactValidationError("motion.parameters.motion_mode must be 1 or 2")

    holdout = _validate_mapping(
        config["holdout"], "resolved_generation_config.holdout"
    )
    validate_exact_keys(
        holdout,
        {"present", "count", "pattern_family", "seed_offset"},
        "holdout",
    )
    if type(holdout["present"]) is not bool:
        raise ArtifactValidationError("holdout.present must be boolean")
    holdout_count = validate_exact_int(
        holdout["count"], "holdout.count", minimum=0
    )
    if holdout["present"] != (holdout_count > 0):
        raise ArtifactValidationError(
            "holdout.present must equal whether holdout.count is positive"
        )
    holdout_family = _validate_nonempty_string(
        holdout["pattern_family"], "holdout.pattern_family"
    )
    holdout_seed_offset = validate_exact_int(
        holdout["seed_offset"], "holdout.seed_offset"
    )

    native = {
        "schema": "measurements-v1",
        **dimensions,
        "seed": seed,
        "target": target_native,
        "pattern": {"family": pattern_family, "order": pattern_order},
        "time_assignment": {"mode": time_mode},
        "noise": {
            "convention": noise_convention,
            "parameters": {"snr_db": snr_db, "sigma_abs": sigma_abs},
        },
        "motion": {
            "model": motion_model,
            "parameters": {
                "velocity": velocity,
                "acceleration": acceleration,
                "omega": omega,
                "beta": beta,
                "speed_factor": speed_factor,
                "motion_mode": motion_mode,
            },
        },
        "holdout": {
            "present": holdout["present"],
            "count": holdout_count,
            "pattern_family": holdout_family,
            "seed_offset": holdout_seed_offset,
        },
    }
    canonical_json_bytes(native)
    return native


_ACQUISITION_IDENTITY_ARRAYS = {
    "patterns",
    "measurements",
    "frame_indices",
    "time_grid",
    "holdout_patterns",
    "holdout_measurements",
    "holdout_frame_indices",
}


def _validate_identity_array_descriptor(
    descriptor: object,
    field: str,
) -> Mapping[str, object]:
    descriptor = _validate_mapping(descriptor, field)
    validate_exact_keys(descriptor, {"dtype", "shape", "sha256"}, field)
    dtype_text = _validate_nonempty_string(descriptor["dtype"], f"{field}.dtype")
    try:
        dtype = np.dtype(dtype_text)
    except TypeError as exc:
        raise ArtifactValidationError(f"{field}.dtype is invalid") from exc
    if dtype.hasobject:
        raise ArtifactValidationError(f"{field}.dtype cannot be object")
    shape = descriptor["shape"]
    if type(shape) not in (list, tuple):
        raise ArtifactValidationError(f"{field}.shape must be a sequence")
    shape_native = [
        validate_exact_int(value, f"{field}.shape[{index}]", minimum=0)
        for index, value in enumerate(shape)
    ]
    return {
        "dtype": dtype_text,
        "shape": shape_native,
        "sha256": validate_sha256(descriptor["sha256"], f"{field}.sha256"),
    }


def validate_acquisition_identity_spec(
    spec: object,
) -> Mapping[str, object]:
    spec = _validate_mapping(spec, "dataset_identity_spec")
    validate_exact_keys(
        spec,
        {
            "schema",
            "resolved_generation_config",
            "generator_code_version",
            "target_asset_sha256",
            "seed",
            "pattern_family",
            "pattern_order",
            "time_assignment_mode",
            "noise",
            "motion",
            "dimensions",
            "arrays",
        },
        "dataset identity spec",
    )
    if spec["schema"] != "measurements-v1":
        raise ArtifactValidationError("dataset identity spec schema mismatch")
    config = validate_generation_config(spec["resolved_generation_config"])
    _validate_nonempty_string(
        spec["generator_code_version"], "generator_code_version"
    )
    validate_sha256(spec["target_asset_sha256"], "target asset hash")

    dimensions = _validate_mapping(spec["dimensions"], "identity dimensions")
    validate_exact_keys(dimensions, {"H", "W", "T", "K"}, "identity dimensions")
    normalized_dimensions = {
        name: validate_exact_int(
            dimensions[name], f"identity dimensions.{name}", minimum=1
        )
        for name in ("H", "W", "T", "K")
    }
    if normalized_dimensions != {
        name: config[name] for name in ("H", "W", "T", "K")
    }:
        raise ArtifactValidationError(
            "identity dimensions disagree with resolved generation config"
        )

    expected_scalars = {
        "seed": config["seed"],
        "pattern_family": config["pattern"]["family"],
        "pattern_order": config["pattern"]["order"],
        "time_assignment_mode": config["time_assignment"]["mode"],
    }
    validate_exact_int(spec["seed"], "identity seed")
    for name in (
        "pattern_family",
        "pattern_order",
        "time_assignment_mode",
    ):
        _validate_nonempty_string(spec[name], f"identity {name}")
    if any(spec[name] != value for name, value in expected_scalars.items()):
        raise ArtifactValidationError(
            "identity fields disagree with resolved generation config"
        )

    noise = _validate_mapping(spec["noise"], "identity noise")
    validate_exact_keys(noise, {"convention", "parameters"}, "identity noise")
    if canonical_json_bytes(noise) != canonical_json_bytes(config["noise"]):
        raise ArtifactValidationError(
            "identity noise disagrees with resolved generation config"
        )
    motion = _validate_mapping(spec["motion"], "identity motion")
    validate_exact_keys(motion, {"model", "parameters"}, "identity motion")
    if canonical_json_bytes(motion) != canonical_json_bytes(config["motion"]):
        raise ArtifactValidationError(
            "identity motion disagrees with resolved generation config"
        )

    arrays = _validate_mapping(spec["arrays"], "identity arrays")
    validate_exact_keys(
        arrays, _ACQUISITION_IDENTITY_ARRAYS, "identity arrays"
    )
    normalized_descriptors = {}
    for name in _ACQUISITION_IDENTITY_ARRAYS:
        descriptor = arrays[name]
        if descriptor is None:
            normalized_descriptors[name] = None
        else:
            normalized_descriptors[name] = _validate_identity_array_descriptor(
                descriptor, f"identity arrays.{name}"
            )
    expected_shapes = {
        "patterns": [config["K"], config["H"], config["W"]],
        "measurements": [config["K"]],
        "frame_indices": [config["K"]],
        "time_grid": [config["T"]],
    }
    for name, expected_shape in expected_shapes.items():
        descriptor = normalized_descriptors[name]
        if descriptor is None or descriptor["shape"] != expected_shape:
            raise ArtifactValidationError(
                f"identity arrays.{name} shape disagrees with config"
            )
    holdout_names = (
        "holdout_patterns",
        "holdout_measurements",
        "holdout_frame_indices",
    )
    holdout_present = config["holdout"]["present"]
    if any(
        (normalized_descriptors[name] is not None) != holdout_present
        for name in holdout_names
    ):
        raise ArtifactValidationError(
            "identity holdout descriptors disagree with config"
        )
    if holdout_present:
        count = config["holdout"]["count"]
        holdout_shapes = {
            "holdout_patterns": [count, config["H"], config["W"]],
            "holdout_measurements": [count],
            "holdout_frame_indices": [count],
        }
        if any(
            normalized_descriptors[name]["shape"] != shape
            for name, shape in holdout_shapes.items()
        ):
            raise ArtifactValidationError(
                "identity holdout shapes disagree with config"
            )
    return config


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
