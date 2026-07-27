"""Corrected deterministic acquisition generation and semantic identities."""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import re
import stat
from types import MappingProxyType
from typing import Mapping

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import rotate as nd_rotate
from scipy.ndimage import shift as nd_shift

from ._artifact_identity import (
    ArtifactValidationError,
    array_descriptor,
    deep_freeze_json,
    readonly_array,
    validate_exact_json_native,
    validate_path_free_opaque_id,
    validate_sha256,
)
from ._artifact_models import EvaluationTruth, SPIAcquisitionData
from .patterns import generate_patterns


_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z", re.ASCII)
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
_MOTION_KEYS = {
    "id",
    "velocity",
    "acceleration",
    "omega",
    "beta",
}
_CALIBRATION_KEYS = {
    "id",
    "mode",
    "reference",
    "variance_ddof",
    "sigma_formula",
    "reuse",
}
_CORRECTED_CONFIG_KEYS = {
    "schema_version",
    "dimensions",
    "target",
    "motion",
    "acquisition",
    "rng",
    "motion_renderer",
    "serializer",
}
_CALIBRATION_RECORD_KEYS = {
    "schema_version",
    "calibration",
    "scientific_contract",
    "target_id",
    "motion_id",
    "seed",
    "reference_cell_sha256",
    "reference_measurements",
    "requested_snr_db",
    "ddof",
    "reference_variance",
    "sigma_absolute",
    "realized_snr_db",
    "generator",
    "generator_config_sha256",
    "runtime",
}
_PATTERN_FAMILIES = frozenset(
    {
        "bernoulli",
        "gaussian",
        "random",
        "hadamard",
        "hadamard_cc",
        "hadamard_walsh",
        "hadamard_natural",
        "fourier",
        "s_matrix",
        "s_matrix_m",
    }
)
_REPARSE_POINT = 0x400


def _strict_native(value: object) -> object:
    validate_exact_json_native(value)
    if type(value) in (dict, MappingProxyType):
        return {
            key: _strict_native(child)  # type: ignore[union-attr]
            for key, child in value.items()  # type: ignore[union-attr]
        }
    if type(value) in (list, tuple):
        return [_strict_native(child) for child in value]  # type: ignore[union-attr]
    return value


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        _strict_native(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _require_mapping(
    value: object,
    field: str,
    keys: set[str],
) -> Mapping[str, object]:
    if type(value) not in (dict, MappingProxyType):
        raise TypeError(f"{field} must be an exact dict")
    if any(type(key) is not str for key in value):
        raise TypeError(f"{field} keys must be exact strings")
    actual = set(value)
    if actual != keys:
        raise ArtifactValidationError(
            f"{field} keys mismatch; "
            f"missing={sorted(keys - actual)}, extra={sorted(actual - keys)}"
        )
    validate_exact_json_native(value, field)
    return value


def _require_string(value: object, field: str) -> str:
    if type(value) is not str or not value:
        raise TypeError(f"{field} must be a nonempty exact string")
    return value


def _require_int(
    value: object,
    field: str,
    *,
    minimum: int | None = None,
) -> int:
    if type(value) is not int:
        raise TypeError(f"{field} must be an exact integer")
    if minimum is not None and value < minimum:
        raise ArtifactValidationError(f"{field} must be at least {minimum}")
    return value


def _require_number(
    value: object,
    field: str,
    *,
    minimum: float | None = None,
) -> int | float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise TypeError(f"{field} must be a finite native number")
    if minimum is not None and value < minimum:
        raise ArtifactValidationError(f"{field} must be at least {minimum}")
    return value


def _require_sequence(
    value: object,
    field: str,
    *,
    length: int | None = None,
) -> list[object]:
    if type(value) not in (list, tuple):
        raise TypeError(f"{field} must be an exact list or tuple")
    if length is not None and len(value) != length:
        raise ArtifactValidationError(
            f"{field} must contain exactly {length} values"
        )
    validate_exact_json_native(value, field)
    return list(value)


def _require_named_sha(
    value: object,
    field: str,
) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ArtifactValidationError(f"{field} must be a lowercase SHA-256")
    return value


def _is_reparse_or_link(path_stat: os.stat_result) -> bool:
    return stat.S_ISLNK(path_stat.st_mode) or bool(
        getattr(path_stat, "st_file_attributes", 0) & _REPARSE_POINT
    )


def _validate_existing_ancestors(path: Path) -> None:
    current = path.absolute()
    while True:
        if os.path.lexists(current):
            current_stat = os.lstat(current)
            if _is_reparse_or_link(current_stat):
                raise ArtifactValidationError(
                    f"linked or reparse path rejected: {current}"
                )
        parent = current.parent
        if parent == current:
            return
        current = parent


def _read_regular_snapshot(path: Path) -> bytes:
    path = Path(path)
    _validate_existing_ancestors(path)
    before = os.lstat(path)
    if _is_reparse_or_link(before) or not stat.S_ISREG(before.st_mode):
        raise ArtifactValidationError("target asset must be a regular file")
    flags = os.O_RDONLY
    flags |= getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ArtifactValidationError("cannot safely open target asset") from error
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ArtifactValidationError("target asset is not a regular file")
        if (
            getattr(before, "st_dev", None),
            getattr(before, "st_ino", None),
        ) != (
            getattr(opened, "st_dev", None),
            getattr(opened, "st_ino", None),
        ):
            raise ArtifactValidationError("target asset changed during open")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after_fd = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = os.lstat(path)
    observed = (
        getattr(opened, "st_dev", None),
        getattr(opened, "st_ino", None),
        opened.st_size,
        getattr(opened, "st_mtime_ns", None),
    )
    if observed != (
        getattr(after_fd, "st_dev", None),
        getattr(after_fd, "st_ino", None),
        after_fd.st_size,
        getattr(after_fd, "st_mtime_ns", None),
    ) or observed != (
        getattr(after, "st_dev", None),
        getattr(after, "st_ino", None),
        after.st_size,
        getattr(after, "st_mtime_ns", None),
    ):
        raise ArtifactValidationError("target asset changed during read")
    payload = b"".join(chunks)
    if len(payload) != opened.st_size:
        raise ArtifactValidationError("target asset size changed during read")
    return payload


@dataclass(frozen=True)
class TargetSnapshot:
    """One safely-read target snapshot used for both hashing and decoding."""

    target_id: str
    descriptor: str
    assets_sha256: Mapping[str, object]
    canonical_image: np.ndarray
    renderer: Mapping[str, object] | None

    def __post_init__(self) -> None:
        validate_path_free_opaque_id(self.target_id, "target_id")
        _require_string(self.descriptor, "target descriptor")
        if type(self.assets_sha256) not in (dict, MappingProxyType):
            raise TypeError("assets_sha256 must be an exact dict")
        if not self.assets_sha256:
            raise ArtifactValidationError("assets_sha256 cannot be empty")
        for name, digest in self.assets_sha256.items():
            _require_string(name, "asset name")
            _require_named_sha(digest, f"asset hash {name}")
        validate_exact_json_native(self.assets_sha256, "assets_sha256")
        if self.renderer is not None:
            if type(self.renderer) not in (dict, MappingProxyType):
                raise TypeError("renderer must be an exact dict or None")
            validate_exact_json_native(self.renderer, "renderer")
        image = np.asarray(self.canonical_image)
        if (
            image.ndim != 2
            or not np.issubdtype(image.dtype, np.floating)
            or not np.isfinite(image).all()
            or np.any(image < 0)
            or np.any(image > 1)
        ):
            raise ArtifactValidationError(
                "canonical_image must be a finite floating [H,W] image in [0,1]"
            )
        object.__setattr__(
            self,
            "assets_sha256",
            deep_freeze_json(
                {
                    name: self.assets_sha256[name]
                    for name in sorted(self.assets_sha256)
                }
            ),
        )
        object.__setattr__(
            self,
            "renderer",
            None if self.renderer is None else deep_freeze_json(self.renderer),
        )
        object.__setattr__(
            self,
            "canonical_image",
            readonly_array(image.astype(np.float32), "canonical_image"),
        )


@dataclass(frozen=True, kw_only=True)
class CorrectedDataset:
    dataset_identity_sha256: str
    dataset_identity_spec: InitVar[Mapping[str, object]]
    resolved_generator_config: InitVar[Mapping[str, object]]
    noise_calibration_record: InitVar[Mapping[str, object]]
    noise_calibration_sha256: str
    acquisition: SPIAcquisitionData
    truth: EvaluationTruth
    _dataset_identity_spec: Mapping[str, object] = field(
        init=False, repr=False
    )
    _resolved_generator_config: Mapping[str, object] = field(
        init=False, repr=False
    )
    _noise_calibration_record: Mapping[str, object] = field(
        init=False, repr=False
    )

    def __post_init__(
        self,
        dataset_identity_spec: Mapping[str, object],
        resolved_generator_config: Mapping[str, object],
        noise_calibration_record: Mapping[str, object],
    ) -> None:
        validate_sha256(self.dataset_identity_sha256, "dataset identity")
        validate_sha256(
            self.noise_calibration_sha256, "noise calibration"
        )
        snapshots = {
            "_dataset_identity_spec": dataset_identity_spec,
            "_resolved_generator_config": resolved_generator_config,
            "_noise_calibration_record": noise_calibration_record,
        }
        for private_name, value in snapshots.items():
            public_name = private_name.removeprefix("_")
            validate_exact_json_native(value, public_name)
            if type(value) not in (dict, MappingProxyType):
                raise TypeError(f"{public_name} must be an exact dict")
            object.__setattr__(
                self, private_name, deep_freeze_json(value)
            )
        if (
            _sha256_json(self._dataset_identity_spec)
            != self.dataset_identity_sha256
            or _sha256_json(self._noise_calibration_record)
            != self.noise_calibration_sha256
            or self._dataset_identity_spec["generator_config_sha256"]
            != _sha256_json(self._resolved_generator_config)
        ):
            raise ArtifactValidationError(
                "corrected dataset semantic hashes are inconsistent"
            )
        if (
            type(self.acquisition) is not SPIAcquisitionData
            or type(self.truth) is not EvaluationTruth
            or self.acquisition.dataset_identity_sha256
            != self.dataset_identity_sha256
            or self.truth.dataset_identity_sha256
            != self.dataset_identity_sha256
        ):
            raise ArtifactValidationError(
                "corrected dataset artifacts disagree with identity"
            )

    @staticmethod
    def _native_copy(value: Mapping[str, object]) -> dict[str, object]:
        native = _strict_native(value)
        if type(native) is not dict:
            raise RuntimeError("frozen semantic snapshot is not an object")
        return native

    @property
    def dataset_identity_spec(self) -> dict[str, object]:
        return self._native_copy(self._dataset_identity_spec)

    @property
    def resolved_generator_config(self) -> dict[str, object]:
        return self._native_copy(self._resolved_generator_config)

    @property
    def noise_calibration_record(self) -> dict[str, object]:
        return self._native_copy(self._noise_calibration_record)


def _decode_image(payload: bytes, H: int, W: int) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(payload)) as image:
            grayscale = image.convert("L")
            resized = grayscale.resize(
                (W, H), Image.Resampling.LANCZOS
            )
            return (
                np.asarray(resized, dtype=np.float32) / np.float32(255.0)
            )
    except (OSError, ValueError) as error:
        raise ArtifactValidationError("target asset is not a valid image") from error


def _bundled_dejavu_font_path() -> Path:
    specification = importlib.util.find_spec("matplotlib")
    if specification is None or specification.origin is None:
        raise ArtifactValidationError("matplotlib bundled font is unavailable")
    path = (
        Path(specification.origin).parent
        / "mpl-data"
        / "fonts"
        / "ttf"
        / "DejaVuSans.ttf"
    )
    if not path.is_file():
        raise ArtifactValidationError("bundled DejaVu Sans font is unavailable")
    return path


def _render_glyph(
    descriptor: str,
    font_bytes: bytes,
    H: int,
    W: int,
    renderer: Mapping[str, object],
) -> np.ndarray:
    supersample = int(renderer["supersample"])
    fill_fraction = float(renderer["fill_fraction"])
    image = Image.new("L", (W * supersample, H * supersample), 0)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(
        io.BytesIO(font_bytes),
        max(1, int(H * supersample * fill_fraction)),
    )
    glyph = descriptor.split(":", 1)[1]
    bounds = draw.textbbox((0, 0), glyph, font=font)
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    draw.text(
        (
            (W * supersample - width) / 2 - bounds[0],
            (H * supersample - height) / 2 - bounds[1],
        ),
        glyph,
        fill=255,
        font=font,
    )
    resized = image.resize((W, H), Image.Resampling.LANCZOS)
    return np.asarray(resized, dtype=np.float32) / np.float32(255.0)


def resolve_target_snapshot(
    *,
    repo_root: Path,
    target_id: str,
    descriptor: str,
    H: int,
    W: int,
) -> TargetSnapshot:
    if not isinstance(repo_root, Path):
        raise TypeError("repo_root must be a Path")
    validate_path_free_opaque_id(target_id, "target_id")
    _require_string(descriptor, "target descriptor")
    _require_int(H, "H", minimum=1)
    _require_int(W, "W", minimum=1)
    if descriptor.startswith("char:"):
        if (
            len(descriptor) != 6
            or not descriptor[-1].isascii()
            or not descriptor[-1].isalnum()
        ):
            raise ArtifactValidationError("invalid built-in glyph descriptor")
        renderer = {
            "font_family": "DejaVu Sans",
            "fill_fraction": 0.8,
            "resample": "lanczos",
            "supersample": 4,
        }
        font_bytes = _read_regular_snapshot(_bundled_dejavu_font_path())
        assets = {
            "descriptor": hashlib.sha256(
                descriptor.encode("utf-8")
            ).hexdigest(),
            "font": hashlib.sha256(font_bytes).hexdigest(),
            "renderer": _sha256_json(renderer),
        }
        return TargetSnapshot(
            target_id=target_id,
            descriptor=descriptor,
            assets_sha256=assets,
            canonical_image=_render_glyph(
                descriptor, font_bytes, H, W, renderer
            ),
            renderer=renderer,
        )

    normalized = descriptor.replace("\\", "/")
    parts = normalized.split("/")
    if (
        normalized != descriptor
        or normalized.startswith("/")
        or any(part in ("", ".", "..") for part in parts)
        or ":" in descriptor
    ):
        raise ArtifactValidationError(
            "target descriptor must be a safe repository-relative path"
        )
    _validate_existing_ancestors(repo_root)
    root = repo_root.resolve(strict=True)
    _validate_existing_ancestors(root)
    asset = root.joinpath(*parts)
    try:
        common = os.path.commonpath((str(root), str(asset.absolute())))
    except ValueError as error:
        raise ArtifactValidationError("target asset escapes repository") from error
    if common != str(root):
        raise ArtifactValidationError("target asset escapes repository")
    raw = _read_regular_snapshot(asset)
    renderer = {
        "color_mode": "grayscale",
        "resample": "lanczos",
    }
    return TargetSnapshot(
        target_id=target_id,
        descriptor=descriptor,
        assets_sha256={descriptor: hashlib.sha256(raw).hexdigest()},
        canonical_image=_decode_image(raw, H, W),
        renderer=renderer,
    )


def acquisition_rng(seed: int, stream_id: int) -> np.random.Generator:
    _require_int(seed, "seed")
    _require_int(stream_id, "stream_id", minimum=0)
    if stream_id not in {0, 1, 2, 3}:
        raise ArtifactValidationError("stream_id must be one of 0, 1, 2, 3")
    sequence = np.random.SeedSequence(
        entropy=seed,
        spawn_key=(stream_id,),
    )
    return np.random.Generator(np.random.PCG64(sequence))


def _pattern_values(family: str) -> list[object]:
    if family == "gaussian":
        return ["real"]
    if family.startswith("hadamard"):
        return [-1, 1]
    return [0, 1]


def _validate_generation_inputs(
    *,
    scientific_contract: Mapping[str, object],
    target_snapshot: TargetSnapshot,
    motion: Mapping[str, object],
    seed: int,
    acquisition_config: Mapping[str, object],
    noise_calibration_entry: Mapping[str, object],
    generator: Mapping[str, object],
    runtime: Mapping[str, object],
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    contract = _require_mapping(
        scientific_contract,
        "scientific_contract",
        {"id", "sha256"},
    )
    contract_native = {
        "id": validate_path_free_opaque_id(
            contract["id"], "scientific_contract.id"
        ),
        "sha256": _require_named_sha(
            contract["sha256"], "scientific_contract.sha256"
        ),
    }
    if type(target_snapshot) is not TargetSnapshot:
        raise TypeError("target_snapshot must be an exact TargetSnapshot")
    motion_map = _require_mapping(motion, "motion", _MOTION_KEYS)
    velocity = [
        _require_number(item, f"motion.velocity[{index}]")
        for index, item in enumerate(
            _require_sequence(
                motion_map["velocity"], "motion.velocity", length=2
            )
        )
    ]
    acceleration = [
        _require_number(item, f"motion.acceleration[{index}]")
        for index, item in enumerate(
            _require_sequence(
                motion_map["acceleration"],
                "motion.acceleration",
                length=2,
            )
        )
    ]
    motion_native = {
        "id": validate_path_free_opaque_id(motion_map["id"], "motion.id"),
        "velocity": velocity,
        "acceleration": acceleration,
        "omega": _require_number(motion_map["omega"], "motion.omega"),
        "beta": _require_number(motion_map["beta"], "motion.beta"),
    }
    _require_int(seed, "seed")
    config = _require_mapping(
        acquisition_config,
        "acquisition_config",
        _ACQUISITION_KEYS,
    )
    image_size = [
        _require_int(item, f"image_size[{index}]", minimum=1)
        for index, item in enumerate(
            _require_sequence(config["image_size"], "image_size", length=2)
        )
    ]
    H, W = image_size
    if target_snapshot.canonical_image.shape != (H, W):
        raise ArtifactValidationError(
            "target snapshot dimensions disagree with acquisition config"
        )
    family = _require_string(
        config["pattern_family"], "pattern_family"
    )
    if family not in _PATTERN_FAMILIES:
        raise ArtifactValidationError("unsupported pattern_family")
    pattern_values = _require_sequence(
        config["pattern_values"], "pattern_values"
    )
    if pattern_values != _pattern_values(family):
        raise ArtifactValidationError(
            "pattern_values disagree with pattern_family"
        )
    acquisition_native = {
        "image_size": image_size,
        "num_frames": _require_int(
            config["num_frames"], "num_frames", minimum=1
        ),
        "train_measurements": _require_int(
            config["train_measurements"],
            "train_measurements",
            minimum=1,
        ),
        "holdout_measurements": _require_int(
            config["holdout_measurements"],
            "holdout_measurements",
            minimum=0,
        ),
        "pattern_family": family,
        "pattern_values": pattern_values,
        "pattern_order": _require_string(
            config["pattern_order"], "pattern_order"
        ),
        "time_assignment": _require_string(
            config["time_assignment"], "time_assignment"
        ),
        "holdout_pattern_family": _require_string(
            config["holdout_pattern_family"],
            "holdout_pattern_family",
        ),
        "snr_db": _require_number(config["snr_db"], "snr_db"),
        "noise_calibration_id": validate_path_free_opaque_id(
            config["noise_calibration_id"], "noise_calibration_id"
        ),
    }
    if acquisition_native["pattern_order"] not in {
        "sequential",
        "stratified",
        "random",
    }:
        raise ArtifactValidationError("unsupported pattern_order")
    if acquisition_native["time_assignment"] != "uniform":
        raise ArtifactValidationError(
            "corrected generator requires uniform time assignment"
        )
    if acquisition_native["holdout_pattern_family"] != "uniform-random":
        raise ArtifactValidationError(
            "corrected generator requires uniform-random holdout patterns"
        )
    calibration = _require_mapping(
        noise_calibration_entry,
        "noise_calibration_entry",
        _CALIBRATION_KEYS,
    )
    calibration_native = _strict_native(calibration)
    assert isinstance(calibration_native, dict)
    if (
        _require_string(calibration_native["id"], "calibration.id")
        != acquisition_native["noise_calibration_id"]
        or calibration_native["mode"] != "detector-absolute"
        or calibration_native["reference"]
        != "corresponding-bernoulli-reference-cell"
        or type(calibration_native["variance_ddof"]) is not int
        or calibration_native["variance_ddof"] != 0
    ):
        raise ArtifactValidationError(
            "noise calibration entry does not match detector-absolute-v1"
        )
    _require_string(
        calibration_native["sigma_formula"], "calibration.sigma_formula"
    )
    reuse = _require_sequence(calibration_native["reuse"], "calibration.reuse")
    if set(reuse) != {"train", "holdout", "alternate-pattern"}:
        raise ArtifactValidationError("calibration.reuse is incomplete")
    generator_map = _require_mapping(
        generator, "generator", {"id", "version", "git_commit"}
    )
    generator_native = {
        "id": validate_path_free_opaque_id(
            generator_map["id"], "generator.id"
        ),
        "version": validate_path_free_opaque_id(
            generator_map["version"], "generator.version"
        ),
        "git_commit": _require_string(
            generator_map["git_commit"], "generator.git_commit"
        ),
    }
    if _COMMIT_RE.fullmatch(generator_native["git_commit"]) is None:
        raise ArtifactValidationError(
            "generator.git_commit must be a full lowercase clean commit"
        )
    runtime_map = _require_mapping(
        runtime,
        "runtime",
        {"dependencies_sha256", "environment_lock_sha256"},
    )
    runtime_native = {
        "dependencies_sha256": _require_named_sha(
            runtime_map["dependencies_sha256"],
            "runtime.dependencies_sha256",
        ),
        "environment_lock_sha256": _require_named_sha(
            runtime_map["environment_lock_sha256"],
            "runtime.environment_lock_sha256",
        ),
    }
    return (
        contract_native,
        motion_native,
        acquisition_native,
        calibration_native,
        generator_native,
        runtime_native,
    )


def _resolved_generator_config(
    target: TargetSnapshot,
    motion: Mapping[str, object],
    acquisition: Mapping[str, object],
) -> dict[str, object]:
    H, W = acquisition["image_size"]  # type: ignore[misc]
    return {
        "schema_version": "corrected-generator-config-v1",
        "dimensions": {
            "H": H,
            "W": W,
            "T": acquisition["num_frames"],
            "K": acquisition["train_measurements"],
            "holdout_K": acquisition["holdout_measurements"],
        },
        "target": {
            "id": target.target_id,
            "descriptor": target.descriptor,
            "assets_sha256": target.assets_sha256,
            "renderer": target.renderer,
        },
        "motion": motion,
        "acquisition": {
            key: acquisition[key]
            for key in (
                "pattern_family",
                "pattern_values",
                "pattern_order",
                "time_assignment",
                "holdout_pattern_family",
                "snr_db",
                "noise_calibration_id",
            )
        },
        "rng": {
            "bit_generator": "PCG64",
            "seed_sequence": (
                "SeedSequence(entropy=seed, spawn_key=(stream_id,))"
            ),
            "streams": {
                "train-pattern": 0,
                "train-noise": 1,
                "holdout-pattern": 2,
                "holdout-noise": 3,
            },
        },
        "motion_renderer": {
            "acceleration_factor": 1.0,
            "angular_acceleration_factor": 1.0,
            "interpolation_order": 1,
            "mode": "constant",
            "normalized_time": [0, 1],
        },
        "serializer": {
            "measurements_schema": "measurements-blind-v1",
            "truth_schema": "evaluation-truth-v2",
        },
    }


def resolve_corrected_dataset_request(
    *,
    scientific_contract: Mapping[str, object],
    target_snapshot: TargetSnapshot,
    motion: Mapping[str, object],
    seed: int,
    acquisition_config: Mapping[str, object],
    noise_calibration_entry: Mapping[str, object],
    generator: Mapping[str, object],
    runtime: Mapping[str, object],
) -> dict[str, object]:
    """Resolve immutable scientific content without generating measurements."""

    (
        contract,
        motion_native,
        acquisition_native,
        calibration,
        generator_native,
        runtime_native,
    ) = _validate_generation_inputs(
        scientific_contract=scientific_contract,
        target_snapshot=target_snapshot,
        motion=motion,
        seed=seed,
        acquisition_config=acquisition_config,
        noise_calibration_entry=noise_calibration_entry,
        generator=generator,
        runtime=runtime,
    )
    config = _resolved_generator_config(
        target_snapshot,
        motion_native,
        acquisition_native,
    )
    request = {
        "schema_version": "corrected-dataset-request-v1",
        "scientific_contract": contract,
        "target": {
            "id": target_snapshot.target_id,
            "descriptor": target_snapshot.descriptor,
            "assets_sha256": target_snapshot.assets_sha256,
            "renderer": target_snapshot.renderer,
        },
        "motion": motion_native,
        "seed": seed,
        "acquisition_config": acquisition_native,
        "noise_calibration": {
            "id": calibration["id"],
            "registry_entry_sha256": _sha256_json(calibration),
            "entry": calibration,
        },
        "generator": generator_native,
        "runtime": runtime_native,
        "resolved_generator_config": config,
    }
    native = _strict_native(request)
    if type(native) is not dict:
        raise RuntimeError("resolved corrected request is not an object")
    return native


def _ranked_patterns(
    H: int,
    W: int,
    K: int,
    family: str,
    rng: np.random.Generator,
) -> np.ndarray:
    if family == "bernoulli":
        return rng.integers(
            0, 2, size=(K, H, W), dtype=np.int8
        ).astype(np.float32)
    if family == "random":
        return rng.random((K, H, W)).astype(np.float32)
    if family == "gaussian":
        return rng.standard_normal((K, H, W)).astype(np.float32)
    return generate_patterns(H, W, K, family, seed=0).astype(np.float32)


def _order_patterns(
    patterns: np.ndarray,
    order: str,
    T: int,
    rng: np.random.Generator,
) -> np.ndarray:
    K = patterns.shape[0]
    if order == "sequential":
        indices = np.arange(K)
    elif order == "random":
        indices = rng.permutation(K)
    else:
        patterns_per_frame = max(1, int(np.ceil(K / T)))
        frame = np.clip(
            np.arange(K) // patterns_per_frame, 0, T - 1
        )
        rank = frame + (np.arange(K) - frame * patterns_per_frame) * T
        indices = np.argsort(
            np.argsort(rank, kind="stable"), kind="stable"
        )
    return np.ascontiguousarray(patterns[indices], dtype=np.float32)


def _generate_frames(
    canonical: np.ndarray,
    T: int,
    motion: Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time_grid = np.linspace(0.0, 1.0, T).astype(np.float32)
    velocity = np.asarray(motion["velocity"], dtype=np.float64)
    acceleration = np.asarray(motion["acceleration"], dtype=np.float64)
    omega = float(motion["omega"])
    beta = float(motion["beta"])
    translation = (
        time_grid.astype(np.float64)[:, None] * velocity
        + time_grid.astype(np.float64)[:, None] ** 2 * acceleration
    )
    rotation = (
        time_grid.astype(np.float64) * omega
        + time_grid.astype(np.float64) ** 2 * beta
    )
    frames = np.empty(
        (T, canonical.shape[0], canonical.shape[1]), dtype=np.float32
    )
    source = np.asarray(canonical, dtype=np.float64)
    for index in range(T):
        rotated = nd_rotate(
            source,
            np.degrees(rotation[index]),
            reshape=False,
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
        moved = nd_shift(
            rotated,
            translation[index],
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
        frames[index] = np.clip(moved, 0.0, 1.0).astype(np.float32)
    return (
        frames,
        translation.astype(np.float32),
        rotation.astype(np.float32),
    )


def _frame_indices(K: int, T: int) -> np.ndarray:
    per_frame = max(1, int(np.ceil(K / T)))
    return np.clip(
        np.arange(K) // per_frame, 0, T - 1
    ).astype(np.int64)


def _measure(
    patterns: np.ndarray,
    frames: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    return np.einsum(
        "khw,khw->k",
        patterns.astype(np.float64),
        frames[indices].astype(np.float64),
        optimize=False,
    )


def _noise(
    size: int,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if sigma == 0.0:
        return np.zeros(size, dtype=np.float64)
    return rng.normal(0.0, sigma, size=size).astype(np.float64)


def _realized_snr(
    signal: np.ndarray,
    noise: np.ndarray,
) -> float | None:
    signal_variance = float(np.var(signal, ddof=0))
    noise_variance = float(np.var(noise, ddof=0))
    if signal_variance == 0.0 or noise_variance == 0.0:
        return None
    return float(10.0 * np.log10(signal_variance / noise_variance))


def _build_dataset_identity_spec(
    *,
    scientific_contract: Mapping[str, object],
    target: TargetSnapshot,
    motion_id: str,
    seed: int,
    generator_config_sha256: str,
    calibration_id: str,
    noise_calibration_sha256: str,
    generator: Mapping[str, object],
    runtime: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": "dataset-identity-v1",
        "scientific_contract": scientific_contract,
        "target": {
            "id": target.target_id,
            "assets_sha256": target.assets_sha256,
        },
        "motion": {"id": motion_id},
        "seed": seed,
        "generator_config_sha256": generator_config_sha256,
        "noise_calibration": {
            "id": calibration_id,
            "sha256": noise_calibration_sha256,
        },
        "generator": generator,
        "runtime": runtime,
    }


def validate_dataset_identity_spec(
    spec: object,
) -> Mapping[str, object]:
    mapping = _require_mapping(
        spec,
        "dataset_identity_spec",
        {
            "schema_version",
            "scientific_contract",
            "target",
            "motion",
            "seed",
            "generator_config_sha256",
            "noise_calibration",
            "generator",
            "runtime",
        },
    )
    if mapping["schema_version"] != "dataset-identity-v1":
        raise ArtifactValidationError("dataset identity schema mismatch")
    _require_mapping(
        mapping["scientific_contract"],
        "scientific_contract",
        {"id", "sha256"},
    )
    target = _require_mapping(
        mapping["target"], "target", {"id", "assets_sha256"}
    )
    validate_path_free_opaque_id(target["id"], "target.id")
    if type(target["assets_sha256"]) not in (dict, MappingProxyType):
        raise TypeError("target.assets_sha256 must be an exact dict")
    for name, digest in target["assets_sha256"].items():  # type: ignore[union-attr]
        _require_string(name, "asset name")
        _require_named_sha(digest, f"asset hash {name}")
    motion = _require_mapping(mapping["motion"], "motion", {"id"})
    validate_path_free_opaque_id(motion["id"], "motion.id")
    _require_int(mapping["seed"], "seed")
    _require_named_sha(
        mapping["generator_config_sha256"], "generator_config_sha256"
    )
    calibration = _require_mapping(
        mapping["noise_calibration"],
        "noise_calibration",
        {"id", "sha256"},
    )
    validate_path_free_opaque_id(
        calibration["id"], "noise_calibration.id"
    )
    _require_named_sha(
        calibration["sha256"], "noise_calibration.sha256"
    )
    generator = _require_mapping(
        mapping["generator"],
        "generator",
        {"id", "version", "git_commit"},
    )
    validate_path_free_opaque_id(generator["id"], "generator.id")
    validate_path_free_opaque_id(generator["version"], "generator.version")
    if (
        type(generator["git_commit"]) is not str
        or _COMMIT_RE.fullmatch(generator["git_commit"]) is None
    ):
        raise ArtifactValidationError("generator.git_commit is invalid")
    runtime = _require_mapping(
        mapping["runtime"],
        "runtime",
        {"dependencies_sha256", "environment_lock_sha256"},
    )
    _require_named_sha(
        runtime["dependencies_sha256"], "runtime.dependencies_sha256"
    )
    _require_named_sha(
        runtime["environment_lock_sha256"],
        "runtime.environment_lock_sha256",
    )
    return mapping


def _validate_corrected_config(
    config: object,
    identity: Mapping[str, object],
) -> tuple[
    Mapping[str, object],
    Mapping[str, object],
    Mapping[str, object],
]:
    mapping = _require_mapping(
        config,
        "resolved_generator_config",
        _CORRECTED_CONFIG_KEYS,
    )
    if mapping["schema_version"] != "corrected-generator-config-v1":
        raise ArtifactValidationError("generator config schema mismatch")
    dimensions = _require_mapping(
        mapping["dimensions"],
        "generator config dimensions",
        {"H", "W", "T", "K", "holdout_K"},
    )
    for name in ("H", "W", "T", "K"):
        _require_int(
            dimensions[name],
            f"generator config dimensions.{name}",
            minimum=1,
        )
    _require_int(
        dimensions["holdout_K"],
        "generator config dimensions.holdout_K",
        minimum=0,
    )

    target = _require_mapping(
        mapping["target"],
        "generator config target",
        {"id", "descriptor", "assets_sha256", "renderer"},
    )
    validate_path_free_opaque_id(target["id"], "generator config target.id")
    _require_string(target["descriptor"], "generator config target.descriptor")
    if type(target["assets_sha256"]) not in (dict, MappingProxyType):
        raise TypeError("generator config target assets must be an exact dict")
    if not target["assets_sha256"]:
        raise ArtifactValidationError("generator config target assets are empty")
    for name, digest in target["assets_sha256"].items():  # type: ignore[union-attr]
        _require_string(name, "generator config asset name")
        _require_named_sha(digest, f"generator config asset hash {name}")
    renderer = target["renderer"]
    if renderer is not None and type(renderer) not in (
        dict,
        MappingProxyType,
    ):
        raise TypeError("generator config renderer must be an exact dict or None")
    if _canonical_json_bytes(
        {"id": target["id"], "assets_sha256": target["assets_sha256"]}
    ) != _canonical_json_bytes(identity["target"]):
        raise ArtifactValidationError(
            "generator config target disagrees with dataset identity"
        )

    motion = _require_mapping(
        mapping["motion"], "generator config motion", _MOTION_KEYS
    )
    validate_path_free_opaque_id(motion["id"], "generator config motion.id")
    for name in ("velocity", "acceleration"):
        values = _require_sequence(
            motion[name], f"generator config motion.{name}", length=2
        )
        for index, value in enumerate(values):
            _require_number(
                value, f"generator config motion.{name}[{index}]"
            )
    for name in ("omega", "beta"):
        _require_number(
            motion[name], f"generator config motion.{name}"
        )
    if motion["id"] != identity["motion"]["id"]:  # type: ignore[index]
        raise ArtifactValidationError(
            "generator config motion disagrees with dataset identity"
        )

    acquisition = _require_mapping(
        mapping["acquisition"],
        "generator config acquisition",
        {
            "pattern_family",
            "pattern_values",
            "pattern_order",
            "time_assignment",
            "holdout_pattern_family",
            "snr_db",
            "noise_calibration_id",
        },
    )
    family = _require_string(
        acquisition["pattern_family"],
        "generator config acquisition.pattern_family",
    )
    if family not in _PATTERN_FAMILIES:
        raise ArtifactValidationError("generator config pattern is unsupported")
    if _require_sequence(
        acquisition["pattern_values"],
        "generator config acquisition.pattern_values",
    ) != _pattern_values(family):
        raise ArtifactValidationError(
            "generator config pattern values disagree with family"
        )
    if acquisition["pattern_order"] not in {
        "sequential",
        "stratified",
        "random",
    }:
        raise ArtifactValidationError("generator config order is unsupported")
    if acquisition["time_assignment"] != "uniform":
        raise ArtifactValidationError(
            "generator config time assignment is unsupported"
        )
    if acquisition["holdout_pattern_family"] != "uniform-random":
        raise ArtifactValidationError(
            "generator config holdout pattern is unsupported"
        )
    _require_number(
        acquisition["snr_db"], "generator config acquisition.snr_db"
    )
    validate_path_free_opaque_id(
        acquisition["noise_calibration_id"],
        "generator config acquisition.noise_calibration_id",
    )

    expected_rng = {
        "bit_generator": "PCG64",
        "seed_sequence": (
            "SeedSequence(entropy=seed, spawn_key=(stream_id,))"
        ),
        "streams": {
            "train-pattern": 0,
            "train-noise": 1,
            "holdout-pattern": 2,
            "holdout-noise": 3,
        },
    }
    expected_renderer = {
        "acceleration_factor": 1.0,
        "angular_acceleration_factor": 1.0,
        "interpolation_order": 1,
        "mode": "constant",
        "normalized_time": [0, 1],
    }
    expected_serializer = {
        "measurements_schema": "measurements-blind-v1",
        "truth_schema": "evaluation-truth-v2",
    }
    for field, expected in (
        ("rng", expected_rng),
        ("motion_renderer", expected_renderer),
        ("serializer", expected_serializer),
    ):
        if _canonical_json_bytes(mapping[field]) != _canonical_json_bytes(
            expected
        ):
            raise ArtifactValidationError(
                f"generator config {field} contract mismatch"
            )
    if _sha256_json(mapping) != identity["generator_config_sha256"]:
        raise ArtifactValidationError(
            "generator config hash disagrees with dataset identity"
        )
    return dimensions, motion, acquisition


def validate_corrected_truth(data: EvaluationTruth) -> None:
    """Validate evaluator-only bindings for ``dataset-identity-v1``."""

    if type(data) is not EvaluationTruth:
        raise TypeError("corrected truth must be an exact EvaluationTruth")
    identity = validate_dataset_identity_spec(data.dataset_identity_spec)
    if _sha256_json(identity) != data.dataset_identity_sha256:
        raise ArtifactValidationError("dataset identity mismatch")
    metadata = _require_mapping(
        data.evaluator_metadata,
        "corrected evaluator metadata",
        {"resolved_generator_config", "noise_calibration_record"},
    )
    config = metadata["resolved_generator_config"]
    dimensions, motion, acquisition = _validate_corrected_config(
        config, identity
    )

    record = _require_mapping(
        metadata["noise_calibration_record"],
        "noise calibration record",
        _CALIBRATION_RECORD_KEYS,
    )
    if record["schema_version"] != "noise-calibration-record-v1":
        raise ArtifactValidationError("noise calibration schema mismatch")
    calibration = _require_mapping(
        record["calibration"],
        "noise calibration descriptor",
        {"id", "registry_entry_sha256"},
    )
    validate_path_free_opaque_id(
        calibration["id"], "noise calibration record id"
    )
    _require_named_sha(
        calibration["registry_entry_sha256"],
        "noise calibration registry entry hash",
    )
    if calibration["id"] != acquisition["noise_calibration_id"]:
        raise ArtifactValidationError(
            "noise calibration record disagrees with generator config"
        )
    if _canonical_json_bytes(
        record["scientific_contract"]
    ) != _canonical_json_bytes(identity["scientific_contract"]):
        raise ArtifactValidationError(
            "noise calibration contract disagrees with dataset identity"
        )
    if (
        record["target_id"] != identity["target"]["id"]  # type: ignore[index]
        or record["motion_id"] != identity["motion"]["id"]  # type: ignore[index]
        or record["seed"] != identity["seed"]
    ):
        raise ArtifactValidationError(
            "noise calibration cell disagrees with dataset identity"
        )
    _require_named_sha(
        record["reference_cell_sha256"], "reference cell hash"
    )
    reference_measurements = _require_mapping(
        record["reference_measurements"],
        "reference measurements descriptor",
        {"dtype", "shape", "sha256"},
    )
    _require_string(
        reference_measurements["dtype"],
        "reference measurements dtype",
    )
    reference_shape = _require_sequence(
        reference_measurements["shape"],
        "reference measurements shape",
        length=1,
    )
    if (
        _require_int(
            reference_shape[0],
            "reference measurements shape[0]",
            minimum=1,
        )
        != dimensions["K"]
    ):
        raise ArtifactValidationError(
            "reference measurements count disagrees with generator config"
        )
    _require_named_sha(
        reference_measurements["sha256"],
        "reference measurements hash",
    )
    requested_snr = _require_number(
        record["requested_snr_db"], "requested SNR"
    )
    if requested_snr != acquisition["snr_db"]:
        raise ArtifactValidationError(
            "noise calibration SNR disagrees with generator config"
        )
    if record["ddof"] != 0 or type(record["ddof"]) is not int:
        raise ArtifactValidationError("noise calibration ddof must be zero")
    variance = _require_number(
        record["reference_variance"],
        "noise calibration reference variance",
        minimum=0.0,
    )
    sigma = _require_number(
        record["sigma_absolute"],
        "noise calibration absolute sigma",
        minimum=0.0,
    )
    expected_sigma = math.sqrt(float(variance)) * 10.0 ** (
        -float(requested_snr) / 20.0
    )
    if sigma != expected_sigma:
        raise ArtifactValidationError(
            "noise calibration sigma disagrees with locked formula"
        )
    realized = _require_mapping(
        record["realized_snr_db"],
        "realized SNR",
        {"train", "holdout"},
    )
    for name in ("train", "holdout"):
        if realized[name] is not None:
            _require_number(realized[name], f"realized SNR {name}")
    if (
        _canonical_json_bytes(record["generator"])
        != _canonical_json_bytes(identity["generator"])
        or _canonical_json_bytes(record["runtime"])
        != _canonical_json_bytes(identity["runtime"])
        or record["generator_config_sha256"]
        != identity["generator_config_sha256"]
    ):
        raise ArtifactValidationError(
            "noise calibration provenance disagrees with dataset identity"
        )
    if _sha256_json(record) != identity["noise_calibration"]["sha256"]:  # type: ignore[index]
        raise ArtifactValidationError(
            "noise calibration hash disagrees with dataset identity"
        )
    if calibration["id"] != identity["noise_calibration"]["id"]:  # type: ignore[index]
        raise ArtifactValidationError(
            "noise calibration id disagrees with dataset identity"
        )

    for name in ("H", "W", "T"):
        _require_int(getattr(data, name), f"truth {name}", minimum=1)
        if getattr(data, name) != dimensions[name]:
            raise ArtifactValidationError(
                "truth dimensions disagree with generator config"
            )
    expected_shapes = {
        "canonical_image": (data.H, data.W),
        "gt_frames": (data.T, data.H, data.W),
        "translation_trajectory": (data.T, 2),
        "rotation_trajectory": (data.T,),
        "gt_velocity": (2,),
        "gt_acceleration": (2,),
    }
    for name, shape in expected_shapes.items():
        array = np.asarray(getattr(data, name))
        if (
            array.shape != shape
            or array.dtype.hasobject
            or not np.issubdtype(array.dtype, np.number)
            or np.iscomplexobj(array)
            or not np.isfinite(array).all()
        ):
            raise ArtifactValidationError(
                f"{name} must be a finite real array with shape {shape}"
            )
    if np.any(data.canonical_image < 0) or np.any(data.canonical_image > 1):
        raise ArtifactValidationError("canonical image must be in [0,1]")
    if type(data.motion_model) is not str or data.motion_model != motion["id"]:
        raise ArtifactValidationError(
            "truth motion model disagrees with generator config"
        )
    if (
        type(data.gt_omega) not in (int, float)
        or not math.isfinite(data.gt_omega)
        or type(data.gt_beta) not in (int, float)
        or not math.isfinite(data.gt_beta)
        or data.gt_omega != motion["omega"]
        or data.gt_beta != motion["beta"]
    ):
        raise ArtifactValidationError(
            "truth angular motion disagrees with generator config"
        )
    if not np.array_equal(
        data.gt_velocity,
        np.asarray(motion["velocity"], dtype=data.gt_velocity.dtype),
    ) or not np.array_equal(
        data.gt_acceleration,
        np.asarray(
            motion["acceleration"], dtype=data.gt_acceleration.dtype
        ),
    ):
        raise ArtifactValidationError(
            "truth linear motion disagrees with generator config"
        )
    expected_frames, expected_translation, expected_rotation = (
        _generate_frames(data.canonical_image, data.T, motion)
    )
    if (
        not np.array_equal(data.gt_frames, expected_frames)
        or not np.array_equal(
            data.translation_trajectory, expected_translation
        )
        or not np.array_equal(
            data.rotation_trajectory, expected_rotation
        )
    ):
        raise ArtifactValidationError(
            "truth rendering or trajectories disagree with generator config"
        )


def generate_corrected_dataset(
    *,
    scientific_contract: Mapping[str, object],
    target_snapshot: TargetSnapshot,
    motion: Mapping[str, object],
    seed: int,
    acquisition_config: Mapping[str, object],
    noise_calibration_entry: Mapping[str, object],
    generator: Mapping[str, object],
    runtime: Mapping[str, object],
) -> CorrectedDataset:
    request = resolve_corrected_dataset_request(
        scientific_contract=scientific_contract,
        target_snapshot=target_snapshot,
        motion=motion,
        seed=seed,
        acquisition_config=acquisition_config,
        noise_calibration_entry=noise_calibration_entry,
        generator=generator,
        runtime=runtime,
    )
    contract = request["scientific_contract"]
    motion_native = request["motion"]
    acquisition_native = request["acquisition_config"]
    calibration_descriptor = request["noise_calibration"]
    generator_native = request["generator"]
    runtime_native = request["runtime"]
    config = request["resolved_generator_config"]
    assert isinstance(contract, dict)
    assert isinstance(motion_native, dict)
    assert isinstance(acquisition_native, dict)
    assert isinstance(calibration_descriptor, dict)
    assert isinstance(generator_native, dict)
    assert isinstance(runtime_native, dict)
    assert isinstance(config, dict)
    calibration = calibration_descriptor["entry"]
    assert isinstance(calibration, dict)
    config_sha256 = _sha256_json(config)
    dimensions = config["dimensions"]
    assert isinstance(dimensions, dict)
    H = int(dimensions["H"])
    W = int(dimensions["W"])
    T = int(dimensions["T"])
    K = int(dimensions["K"])
    holdout_K = int(dimensions["holdout_K"])
    frames, translation, rotation = _generate_frames(
        target_snapshot.canonical_image, T, motion_native
    )
    time_grid = np.linspace(0.0, 1.0, T).astype(np.float32)
    train_pattern_rng = acquisition_rng(seed, 0)
    patterns = _order_patterns(
        _ranked_patterns(
            H,
            W,
            K,
            str(acquisition_native["pattern_family"]),
            train_pattern_rng,
        ),
        str(acquisition_native["pattern_order"]),
        T,
        train_pattern_rng,
    )
    frame_indices = _frame_indices(K, T)
    train_noiseless = _measure(patterns, frames, frame_indices)

    reference_rng = acquisition_rng(seed, 0)
    reference_patterns = _order_patterns(
        _ranked_patterns(H, W, K, "bernoulli", reference_rng),
        str(acquisition_native["pattern_order"]),
        T,
        reference_rng,
    )
    reference_measurements = _measure(
        reference_patterns, frames, frame_indices
    )
    reference_variance = float(
        np.var(reference_measurements, ddof=0)
    )
    requested_snr = float(acquisition_native["snr_db"])
    sigma = float(
        math.sqrt(reference_variance)
        * 10.0 ** (-requested_snr / 20.0)
    )
    train_noise = _noise(K, sigma, acquisition_rng(seed, 1))
    measurements = (train_noiseless + train_noise).astype(np.float32)

    holdout_patterns = None
    holdout_measurements = None
    holdout_frame_indices = None
    holdout_noiseless = np.empty(0, dtype=np.float64)
    holdout_noise = np.empty(0, dtype=np.float64)
    if holdout_K:
        holdout_patterns = acquisition_rng(seed, 2).random(
            (holdout_K, H, W)
        ).astype(np.float32)
        holdout_frame_indices = np.clip(
            (np.arange(holdout_K) * T) // holdout_K,
            0,
            T - 1,
        ).astype(np.int64)
        holdout_noiseless = _measure(
            holdout_patterns, frames, holdout_frame_indices
        )
        holdout_noise = _noise(
            holdout_K, sigma, acquisition_rng(seed, 3)
        )
        holdout_measurements = (
            holdout_noiseless + holdout_noise
        ).astype(np.float32)

    reference_config = _strict_native(config)
    assert isinstance(reference_config, dict)
    reference_acquisition = reference_config["acquisition"]
    assert isinstance(reference_acquisition, dict)
    reference_acquisition["pattern_family"] = "bernoulli"
    reference_acquisition["pattern_values"] = [0, 1]
    reference_cell = {
        "schema_version": "noise-reference-cell-v1",
        "scientific_contract": contract,
        "target": {
            "id": target_snapshot.target_id,
            "assets_sha256": target_snapshot.assets_sha256,
        },
        "motion": {"id": motion_native["id"]},
        "seed": seed,
        "generator_config_sha256": _sha256_json(reference_config),
        "calibration": {
            "id": calibration["id"],
            "registry_entry_sha256": _sha256_json(calibration),
        },
        "generator": generator_native,
        "runtime": runtime_native,
    }
    calibration_record = {
        "schema_version": "noise-calibration-record-v1",
        "calibration": {
            "id": calibration["id"],
            "registry_entry_sha256": _sha256_json(calibration),
        },
        "scientific_contract": contract,
        "target_id": target_snapshot.target_id,
        "motion_id": motion_native["id"],
        "seed": seed,
        "reference_cell_sha256": _sha256_json(reference_cell),
        "reference_measurements": array_descriptor(
            reference_measurements
        ),
        "requested_snr_db": acquisition_native["snr_db"],
        "ddof": 0,
        "reference_variance": reference_variance,
        "sigma_absolute": sigma,
        "realized_snr_db": {
            "train": _realized_snr(train_noiseless, train_noise),
            "holdout": (
                _realized_snr(holdout_noiseless, holdout_noise)
                if holdout_K
                else None
            ),
        },
        "generator": generator_native,
        "generator_config_sha256": config_sha256,
        "runtime": runtime_native,
    }
    calibration_sha256 = _sha256_json(calibration_record)
    identity_spec = _build_dataset_identity_spec(
        scientific_contract=contract,
        target=target_snapshot,
        motion_id=str(motion_native["id"]),
        seed=seed,
        generator_config_sha256=config_sha256,
        calibration_id=str(calibration["id"]),
        noise_calibration_sha256=calibration_sha256,
        generator=generator_native,
        runtime=runtime_native,
    )
    validate_dataset_identity_spec(identity_spec)
    dataset_identity_sha256 = _sha256_json(identity_spec)
    arrays: dict[str, np.ndarray | None] = {
        "patterns": patterns,
        "measurements": measurements,
        "frame_indices": frame_indices,
        "time_grid": time_grid,
        "holdout_patterns": holdout_patterns,
        "holdout_measurements": holdout_measurements,
        "holdout_frame_indices": holdout_frame_indices,
    }
    descriptors = {
        name: array_descriptor(value)
        for name, value in arrays.items()
        if value is not None
    }
    acquisition = SPIAcquisitionData(
        dataset_identity_sha256=dataset_identity_sha256,
        patterns=patterns,
        measurements=measurements,
        frame_indices=frame_indices,
        time_grid=time_grid,
        holdout_patterns=holdout_patterns,
        holdout_measurements=holdout_measurements,
        holdout_frame_indices=holdout_frame_indices,
        H=H,
        W=W,
        T=T,
        K=K,
        holdout_K=holdout_K,
        acquisition={
            "pattern_family": acquisition_native["pattern_family"],
            "pattern_values": acquisition_native["pattern_values"],
            "pattern_order": acquisition_native["pattern_order"],
            "time_assignment": acquisition_native["time_assignment"],
            "holdout_pattern_family": acquisition_native[
                "holdout_pattern_family"
            ],
            "noise_convention": "detector-absolute",
            "noise_sigma_absolute": sigma,
        },
        array_descriptors=descriptors,
    )
    truth = EvaluationTruth(
        dataset_identity_sha256=dataset_identity_sha256,
        dataset_identity_spec=identity_spec,
        canonical_image=target_snapshot.canonical_image,
        gt_frames=frames,
        translation_trajectory=translation,
        rotation_trajectory=rotation,
        gt_velocity=np.asarray(
            motion_native["velocity"], dtype=np.float32
        ),
        gt_acceleration=np.asarray(
            motion_native["acceleration"], dtype=np.float32
        ),
        gt_omega=float(motion_native["omega"]),
        gt_beta=float(motion_native["beta"]),
        motion_model=str(motion_native["id"]),
        H=H,
        W=W,
        T=T,
        evaluator_metadata={
            "resolved_generator_config": config,
            "noise_calibration_record": calibration_record,
        },
    )
    from ._artifact_dataset import _validate_acquisition_identity
    from ._artifact_truth import _validate_truth

    _validate_acquisition_identity(acquisition)
    _validate_truth(truth)
    return CorrectedDataset(
        dataset_identity_sha256=dataset_identity_sha256,
        dataset_identity_spec=identity_spec,
        resolved_generator_config=config,
        noise_calibration_record=calibration_record,
        noise_calibration_sha256=calibration_sha256,
        acquisition=acquisition,
        truth=truth,
    )
