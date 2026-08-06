"""Fail-closed control plane for the target-disjoint diffusion prior v2.

The large dataset and 200-epoch run are intentionally external artifacts.  This
module fixes their identities, validates every hand-off, and keeps the three
command-line entry points free of scientific overrides.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Callable, Mapping, Sequence

from jsonschema import Draft202012Validator
import numpy as np
from scipy.ndimage import rotate as nd_rotate, shift as nd_shift
import torch
from torch.utils.data import DataLoader, TensorDataset
import yaml

from gsdiff.data.simulation import load_image, make_test_image
from gsdiff.experiments.identity import (
    canonical_json_bytes as _compact_json_bytes,
    git_state,
    sha256_file,
)
from gsdiff.experiments.protocol import load_protocol
from gsdiff.prior.unet3d import UNet3D


REPO_ROOT = Path(__file__).resolve().parents[2]
AUTHORITATIVE_PYTHON = Path(r"D:\conda\envs\gsdiff-spi\python.exe")
CONTRACT_PATH = REPO_ROOT / "configs" / "training" / "diffusion-prior-v2.json"
CONTRACT_SCHEMA_PATH = REPO_ROOT / "schemas" / "diffusion-prior-contract-v2.schema.json"
PROVENANCE_SCHEMA_PATH = REPO_ROOT / "schemas" / "diffusion-prior-provenance-v2.schema.json"
SCIENTIFIC_CONTRACTS_PATH = REPO_ROOT / "configs" / "protocols" / "scientific-contracts-v1.yaml"
ENVIRONMENT_LOCK_PATH = REPO_ROOT / "docs" / "reproducibility" / "environment-lock.json"
REQUIREMENTS_LOCK_PATH = REPO_ROOT / "requirements-lock.txt"
PROVENANCE_PATH = REPO_ROOT / "docs" / "reproducibility" / "diffusion-prior-v2-provenance.json"

SOURCES = (
    "procedural:7",
    "procedural:L",
    "procedural:T",
    "procedural:circle",
)
_SOURCE_NAMES = dict(zip(SOURCES, ("7", "L", "T", "circle"), strict=True))
CONTROL_FILES = (
    "configs/training/diffusion-prior-v2.json",
    "schemas/diffusion-prior-contract-v2.schema.json",
    "schemas/diffusion-prior-provenance-v2.schema.json",
    "gsdiff/prior/training_v2.py",
    "scripts/generate_diffusion_prior_v2_dataset.py",
    "scripts/train_diffusion_prior_v2.py",
    "scripts/verify_diffusion_prior_v2.py",
)

REPRODUCIBILITY_POLICY = {
    "id": "seeded-best-effort-v1",
    "seed": 42,
    "cudnn_deterministic": True,
    "cudnn_benchmark": False,
    "cuda_matmul_tf32": False,
    "cudnn_tf32": False,
    "amp": False,
    "workers": 0,
    "cublas_workspace_config": None,
    "strict_cuda_determinism": False,
    "strict_cuda_determinism_unavailable_reason": (
        "locked 3D trilinear-upsample backward path"
    ),
    "claim": (
        "fixed seeds/order/environment and exact data/checkpoint hashes; "
        "bitwise retraining is not claimed"
    ),
}


class ContractError(ValueError):
    pass


class LeakageError(ContractError):
    pass


class ArtifactError(RuntimeError):
    pass


class AnchorError(RuntimeError):
    pass


class EnvironmentError(RuntimeError):
    pass


class TrainingError(RuntimeError):
    pass


def canonical_json_bytes(value: object) -> bytes:
    """Canonical compact UTF-8 JSON with one repository-standard newline."""
    return _compact_json_bytes(value) + b"\n"


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_unique_json_bytes(raw: bytes, *, noun: str) -> object:
    try:
        return json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ContractError(f"invalid JSON constant: {token}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"{noun} is not valid unique-key JSON") from exc


def _load_schema(path: Path) -> dict[str, object]:
    value = _load_unique_json_bytes(path.read_bytes(), noun=path.name)
    if type(value) is not dict:
        raise ContractError(f"{path.name} must be a JSON object")
    Draft202012Validator.check_schema(value)
    return value


def load_contract(path: Path | str = CONTRACT_PATH) -> dict[str, object]:
    destination = Path(path)
    try:
        raw = destination.read_bytes()
    except OSError as exc:
        raise ContractError(f"contract cannot be read: {destination}") from exc
    value = _load_unique_json_bytes(raw, noun="diffusion prior v2 contract")
    if type(value) is not dict:
        raise ContractError("diffusion prior v2 contract must be an object")
    if raw != canonical_json_bytes(value):
        raise ContractError("diffusion prior v2 contract is not canonical JSON")
    errors = sorted(
        Draft202012Validator(_load_schema(CONTRACT_SCHEMA_PATH)).iter_errors(value),
        key=lambda error: list(error.absolute_path),
    )
    if errors:
        raise ContractError(
            "invalid diffusion prior v2 contract: "
            + "; ".join(error.message for error in errors)
        )
    dataset = value["dataset"]
    training = value["training"]
    assert type(dataset) is dict and type(training) is dict
    if dataset["num_videos"] != sum(dataset["source_counts"].values()):
        raise ContractError("dataset source counts do not sum to num_videos")
    expected_batches = dataset["num_videos"] // training["batch_size"]
    if training["batches_per_epoch"] != expected_batches:
        raise ContractError("batches_per_epoch is not derived from dataset and batch size")
    if training["optimizer_steps"] != training["epochs"] * expected_batches:
        raise ContractError("optimizer_steps is not epochs times batches_per_epoch")
    return value


def contract_sha256(contract: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_json_bytes(dict(contract))).hexdigest()


def artifact_root(
    contract: Mapping[str, object], *, repo_root: Path = REPO_ROOT
) -> Path:
    return repo_root / "artifacts" / "diffusion-prior-v2" / contract_sha256(contract)


def sample_video_parameters(index: int, rng: np.random.RandomState) -> dict[str, object]:
    if type(index) is not int or index < 0:
        raise ContractError("video index must be a nonnegative integer")
    return {
        "source": SOURCES[index % len(SOURCES)],
        "vy": rng.uniform(-12.0, 12.0),
        "vx": rng.uniform(-12.0, 12.0),
        "omega": rng.uniform(-0.5, 0.5),
        "child_seed": int(rng.randint(0, 2**31)),
    }


def make_source_image(
    descriptor: str,
    height: int,
    width: int,
    *,
    image_factory: Callable[[str, int, int], np.ndarray] = make_test_image,
) -> np.ndarray:
    if descriptor not in _SOURCE_NAMES:
        raise ContractError(f"unknown procedural training source: {descriptor!r}")
    image = np.asarray(image_factory(_SOURCE_NAMES[descriptor], height, width))
    if image.shape != (height, width):
        raise ContractError("procedural image factory returned the wrong shape")
    image = np.ascontiguousarray(image, dtype=np.float32)
    if not np.isfinite(image).all() or image.min() < 0.0 or image.max() > 1.0:
        raise ContractError("procedural image is nonfinite or outside [0,1]")
    return image


def render_se2_video(
    image: np.ndarray,
    *,
    frames: int,
    vy: float,
    vx: float,
    omega: float,
) -> np.ndarray:
    canonical = np.asarray(image, dtype=np.float32)
    if canonical.ndim != 2 or type(frames) is not int or frames < 1:
        raise ContractError("SE(2) input shape or frame count is invalid")
    video = np.empty((frames, *canonical.shape), dtype=np.float32)
    for index, time_value in enumerate(np.linspace(0.0, 1.0, frames)):
        rotated = nd_rotate(
            canonical,
            np.degrees(float(omega) * float(time_value)),
            reshape=False,
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
        shifted = nd_shift(
            rotated,
            shift=(float(vy) * float(time_value), float(vx) * float(time_value)),
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
        video[index] = np.clip(shifted, 0.0, 1.0).astype(np.float32, copy=False)
    return video


def _canonical_image_hash(image: np.ndarray) -> str:
    value = np.ascontiguousarray(image, dtype=np.float32)
    if value.shape != (64, 64) or not np.isfinite(value).all():
        raise LeakageError("canonical leakage image is invalid")
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def _default_evaluation_image(descriptor: str, height: int, width: int) -> np.ndarray:
    if descriptor.startswith("char:"):
        return make_test_image(descriptor, height, width)
    return load_image(str(REPO_ROOT / descriptor), (height, width))


def audit_target_disjointness(
    registry_path: Path | str,
    *,
    training_sources: Sequence[str] = SOURCES,
    image_loader: Callable[[str, int, int], np.ndarray] = _default_evaluation_image,
    source_factory: Callable[[str, int, int], np.ndarray] | None = None,
) -> dict[str, object]:
    path = Path(registry_path)
    try:
        registry = load_protocol(path)
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise LeakageError("scientific contracts registry is invalid") from exc
    contracts = registry.get("contracts")
    if type(contracts) is not list:
        raise LeakageError("scientific contracts registry has no contracts")
    descriptors: list[str] = []
    for entry in contracts:
        if type(entry) is not dict or type(entry.get("content")) is not dict:
            raise LeakageError("scientific contract entry is malformed")
        targets = entry["content"].get("targets")
        if type(targets) is not dict:
            raise LeakageError("scientific contract targets are malformed")
        for descriptor in targets.values():
            if type(descriptor) is not str:
                raise LeakageError("evaluation descriptor is not text")
            if descriptor not in descriptors:
                descriptors.append(descriptor)
    descriptor_intersection = sorted(set(descriptors).intersection(training_sources))
    if descriptor_intersection:
        raise LeakageError(f"descriptor leakage collision: {descriptor_intersection}")

    evaluation_hashes = {
        descriptor: _canonical_image_hash(image_loader(descriptor, 64, 64))
        for descriptor in descriptors
    }
    training_hashes: dict[str, str] = {}
    for descriptor in training_sources:
        if source_factory is None:
            image = make_source_image(descriptor, 64, 64)
        else:
            image = source_factory(descriptor, 64, 64)
        training_hashes[descriptor] = _canonical_image_hash(image)
    pixel_intersection = sorted(set(evaluation_hashes.values()).intersection(training_hashes.values()))
    if pixel_intersection:
        raise LeakageError(f"canonical pixel leakage collision: {pixel_intersection}")
    return {
        "descriptor_intersection": [],
        "canonical_image_sha256_intersection": [],
        "evaluation_descriptor_count": len(descriptors),
        "training_descriptor_count": len(training_sources),
        "evaluation_canonical_image_sha256": evaluation_hashes,
        "training_canonical_image_sha256": training_hashes,
        "protocol_sha256": registry["protocol_sha256"],
        "registry_file_sha256": sha256_file(path),
        "claim": "exact descriptor/pixel disjointness only; not semantic or distributional independence",
    }


def streaming_tensor_sha256(tensor: torch.Tensor, *, chunk_elements: int = 1 << 20) -> str:
    if tensor.device.type != "cpu" or not tensor.is_contiguous():
        tensor = tensor.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(b"tensor-content-v1\0")
    digest.update(str(tensor.dtype).encode("ascii") + b"\0")
    digest.update(_compact_json_bytes(list(tensor.shape)) + b"\0")
    flat = tensor.view(-1)
    for offset in range(0, flat.numel(), chunk_elements):
        digest.update(flat[offset : offset + chunk_elements].numpy().tobytes(order="C"))
    return digest.hexdigest()


def _dataset_expectations(contract: Mapping[str, object]) -> tuple[int, int, int, int]:
    dataset = contract.get("dataset")
    if type(dataset) is not dict:
        raise ArtifactError("contract dataset section is invalid")
    shape = dataset.get("shape")
    if type(shape) is not list or len(shape) != 4 or any(type(item) is not int for item in shape):
        raise ArtifactError("contract dataset shape is invalid")
    return tuple(shape)  # type: ignore[return-value]


def validate_dataset_payload(
    payload: object, contract: Mapping[str, object]
) -> dict[str, object]:
    if type(payload) is not dict or set(payload) != {"videos"}:
        raise ArtifactError("dataset payload must contain only videos")
    videos = payload["videos"]
    if type(videos) is not torch.Tensor:
        raise ArtifactError("dataset videos are not a tensor")
    if videos.device.type != "cpu" or tuple(videos.shape) != _dataset_expectations(contract):
        raise ArtifactError("dataset tensor device or shape mismatch")
    if videos.dtype is not torch.float32:
        raise ArtifactError("dataset tensor dtype mismatch")
    if not torch.isfinite(videos).all().item():
        raise ArtifactError("dataset tensor contains nonfinite values")
    minimum = float(videos.min().item())
    maximum = float(videos.max().item())
    if minimum < 0.0 or maximum > 1.0:
        raise ArtifactError("dataset tensor range is outside [0,1]")
    return {
        "shape": list(videos.shape),
        "dtype": "float32",
        "device": "cpu",
        "finite": True,
        "minimum": minimum,
        "maximum": maximum,
        "content_sha256": streaming_tensor_sha256(videos),
    }


def make_dataset_manifest(
    dataset_path: Path,
    *,
    contract: Mapping[str, object],
    contract_sha256: str,
    source_anchor: Mapping[str, object],
    environment: Mapping[str, object],
    leakage_audit: Mapping[str, object],
) -> dict[str, object]:
    try:
        payload = torch.load(dataset_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ArtifactError("dataset file cannot be loaded safely") from exc
    evidence = validate_dataset_payload(payload, contract)
    dataset = contract["dataset"]
    assert type(dataset) is dict
    return {
        "schema": "diffusion-prior-dataset-manifest-v2",
        "logical_id": "gsdiff-diffusion-prior-v2",
        "contract_sha256": contract_sha256,
        "dataset": {
            **evidence,
            "file_sha256": sha256_file(dataset_path),
            "size_bytes": dataset_path.stat().st_size,
            "source_counts": deepcopy(dataset["source_counts"]),
        },
        "source_anchor": deepcopy(dict(source_anchor)),
        "environment": deepcopy(dict(environment)),
        "leakage_audit": deepcopy(dict(leakage_audit)),
    }


def _load_canonical_json(path: Path, *, noun: str) -> dict[str, object]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ArtifactError(f"{noun} cannot be read") from exc
    try:
        value = _load_unique_json_bytes(raw, noun=noun)
    except ContractError as exc:
        raise ArtifactError(str(exc)) from exc
    if type(value) is not dict or raw != canonical_json_bytes(value):
        raise ArtifactError(f"{noun} is not canonical JSON")
    return value


def _require_sha256(value: object, *, noun: str) -> str:
    if type(value) is not str or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise ArtifactError(f"{noun} is not a lowercase SHA-256")
    return value


def _validate_source_anchor_shape(anchor: object) -> dict[str, object]:
    if type(anchor) is not dict or set(anchor) != {"commit", "control_files"}:
        raise ArtifactError("source anchor shape is invalid")
    commit = anchor["commit"]
    files = anchor["control_files"]
    if type(commit) is not str or len(commit) != 40 or any(ch not in "0123456789abcdef" for ch in commit):
        raise ArtifactError("source anchor commit is invalid")
    if type(files) is not dict or set(files) != set(CONTROL_FILES):
        raise ArtifactError("source anchor control-file inventory is invalid")
    for path, digest in files.items():
        if type(path) is not str:
            raise ArtifactError("source anchor control-file path is invalid")
        _require_sha256(digest, noun="control-file hash")
    return anchor


def _validate_environment_shape(environment: object) -> dict[str, object]:
    if type(environment) is not dict or set(environment) != {
        "dependencies_sha256", "fingerprint_sha256", "policy"
    }:
        raise ArtifactError("environment evidence shape is invalid")
    _require_sha256(environment["dependencies_sha256"], noun="dependencies hash")
    _require_sha256(environment["fingerprint_sha256"], noun="environment fingerprint")
    if environment["policy"] != REPRODUCIBILITY_POLICY:
        raise ArtifactError("reproducibility policy mismatch")
    return environment


def validate_dataset_pair(
    dataset_path: Path,
    manifest_path: Path,
    *,
    contract: Mapping[str, object],
    source_anchor: Mapping[str, object] | None = None,
    environment: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if not dataset_path.is_file() or not manifest_path.is_file():
        raise ArtifactError("dataset pair is incomplete")
    manifest = _load_canonical_json(manifest_path, noun="dataset manifest")
    if set(manifest) != {
        "schema", "logical_id", "contract_sha256", "dataset", "source_anchor",
        "environment", "leakage_audit"
    }:
        raise ArtifactError("dataset manifest fields are invalid")
    if manifest["schema"] != "diffusion-prior-dataset-manifest-v2" or manifest["logical_id"] != "gsdiff-diffusion-prior-v2":
        raise ArtifactError("dataset manifest identity is invalid")
    if manifest["contract_sha256"] != contract_sha256(contract):
        raise ArtifactError("dataset manifest contract hash mismatch")
    recorded_anchor = _validate_source_anchor_shape(manifest["source_anchor"])
    recorded_environment = _validate_environment_shape(manifest["environment"])
    if source_anchor is not None:
        require_matching_anchor(recorded_anchor, source_anchor, require_same_commit=True)
    if environment is not None:
        require_matching_environment(recorded_environment, environment)
    leakage = manifest["leakage_audit"]
    if type(leakage) is not dict or leakage.get("descriptor_intersection") != [] or leakage.get("canonical_image_sha256_intersection") != []:
        raise ArtifactError("dataset leakage audit is absent or nonempty")
    try:
        payload = torch.load(dataset_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ArtifactError("dataset file cannot be loaded safely") from exc
    evidence = validate_dataset_payload(payload, contract)
    recorded = manifest["dataset"]
    if type(recorded) is not dict or set(recorded) != {
        "shape", "dtype", "device", "finite", "minimum", "maximum",
        "content_sha256", "file_sha256", "size_bytes", "source_counts"
    }:
        raise ArtifactError("dataset evidence fields are invalid")
    expected = {
        **evidence,
        "file_sha256": sha256_file(dataset_path),
        "size_bytes": dataset_path.stat().st_size,
        "source_counts": deepcopy(contract["dataset"]["source_counts"]),
    }
    if recorded != expected:
        if recorded.get("file_sha256") != expected["file_sha256"]:
            raise ArtifactError("dataset file hash mismatch")
        raise ArtifactError("dataset evidence mismatch")
    return manifest


def _atomic_torch_save(
    value: object,
    destination: Path,
    *,
    pre_promote_check: Callable[[Path], None] | None = None,
) -> None:
    temporary = destination.with_name(destination.name + ".tmp")
    if os.path.lexists(temporary):
        raise ArtifactError(f"stale exact sibling temporary file exists: {temporary.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary.open("xb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        if pre_promote_check is not None:
            pre_promote_check(temporary)
        os.replace(temporary, destination)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def _atomic_json_write(value: Mapping[str, object], destination: Path) -> None:
    temporary = destination.with_name(destination.name + ".tmp")
    if os.path.lexists(temporary):
        raise ArtifactError(f"stale exact sibling temporary file exists: {temporary.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary.open("xb") as stream:
            stream.write(canonical_json_bytes(dict(value)))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def generate_dataset_artifact(
    *,
    contract: Mapping[str, object],
    artifact_root: Path,
    source_anchor: Mapping[str, object],
    environment: Mapping[str, object],
    leakage_audit: Mapping[str, object],
    video_factory: Callable[[str, int, np.random.RandomState], torch.Tensor] | None = None,
    pre_promote_check: Callable[[], None] | None = None,
) -> dict[str, object]:
    dataset_path = artifact_root / "dataset.pt"
    manifest_path = artifact_root / "dataset-manifest.json"
    dataset_exists = dataset_path.exists()
    manifest_exists = manifest_path.exists()
    if dataset_exists != manifest_exists:
        raise ArtifactError("one-sided dataset output refuses reuse")
    if dataset_exists:
        manifest = validate_dataset_pair(
            dataset_path,
            manifest_path,
            contract=contract,
            source_anchor=source_anchor,
            environment=environment,
        )
        if manifest["leakage_audit"] != dict(leakage_audit):
            raise ArtifactError("existing dataset leakage audit mismatch")
        return manifest

    shape = _dataset_expectations(contract)
    videos = torch.empty(shape, dtype=torch.float32, device="cpu")
    rng = np.random.RandomState(0)
    counts = {source: 0 for source in SOURCES}
    for index in range(shape[0]):
        parameters = sample_video_parameters(index, rng)
        source = str(parameters["source"])
        counts[source] += 1
        if video_factory is None:
            image = make_source_image(source, shape[2], shape[3])
            rendered = torch.from_numpy(
                render_se2_video(
                    image,
                    frames=shape[1],
                    vy=float(parameters["vy"]),
                    vx=float(parameters["vx"]),
                    omega=float(parameters["omega"]),
                )
            )
        else:
            rendered = video_factory(source, index, rng)
        if type(rendered) is not torch.Tensor or tuple(rendered.shape) != shape[1:] or rendered.dtype is not torch.float32:
            raise ArtifactError("video factory returned an invalid tensor")
        videos[index].copy_(rendered.to(device="cpu"))
    if counts != contract["dataset"]["source_counts"]:
        raise ArtifactError("generated source counts disagree with contract")
    validate_dataset_payload({"videos": videos}, contract)
    artifact_root.mkdir(parents=True, exist_ok=True)
    def validate_before_promotion(temporary: Path) -> None:
        try:
            staged = torch.load(temporary, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise ArtifactError("staged dataset cannot be loaded safely") from exc
        validate_dataset_payload(staged, contract)
        if pre_promote_check is not None:
            pre_promote_check()

    _atomic_torch_save(
        {"videos": videos},
        dataset_path,
        pre_promote_check=validate_before_promotion,
    )
    manifest = make_dataset_manifest(
        dataset_path,
        contract=contract,
        contract_sha256=contract_sha256(contract),
        source_anchor=source_anchor,
        environment=environment,
        leakage_audit=leakage_audit,
    )
    _atomic_json_write(manifest, manifest_path)
    return validate_dataset_pair(
        dataset_path,
        manifest_path,
        contract=contract,
        source_anchor=source_anchor,
        environment=environment,
    )


def require_clean_git_state(state: Mapping[str, object]) -> dict[str, object]:
    if state.get("dirty") is not False and state.get("clean") is not True:
        raise AnchorError("Git source tree is dirty")
    commit = state.get("commit")
    if type(commit) is not str or len(commit) != 40:
        raise AnchorError("Git source commit is invalid")
    return dict(state)


def require_matching_anchor(
    recorded: Mapping[str, object],
    actual: Mapping[str, object],
    *,
    require_same_commit: bool,
) -> None:
    left = _validate_source_anchor_shape(dict(recorded))
    right = _validate_source_anchor_shape(dict(actual))
    if require_same_commit and left["commit"] != right["commit"]:
        raise AnchorError("source commit mismatch")
    if left["control_files"] != right["control_files"]:
        raise AnchorError("control-file hashes mismatch")


def require_matching_environment(
    recorded: Mapping[str, object], actual: Mapping[str, object]
) -> None:
    try:
        left = _validate_environment_shape(dict(recorded))
        right = _validate_environment_shape(dict(actual))
    except ArtifactError as exc:
        raise EnvironmentError(str(exc)) from exc
    if left != right:
        raise EnvironmentError("strict environment evidence mismatch")


def _git_output(*args: str) -> bytes:
    completed = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise AnchorError(f"Git command failed: git {' '.join(args)}")
    return completed.stdout


def _control_hashes_at_commit(commit: str) -> dict[str, str]:
    result: dict[str, str] = {}
    object_type = _git_output("cat-file", "-t", commit).strip()
    if object_type != b"commit":
        raise AnchorError("recorded training source is not a Git commit")
    for relative in CONTROL_FILES:
        payload = _git_output("show", f"{commit}:{relative}")
        result[relative] = hashlib.sha256(payload).hexdigest()
    return result


def current_source_anchor(*, require_clean: bool = True) -> dict[str, object]:
    state = git_state(REPO_ROOT, tuple(Path(path) for path in CONTROL_FILES))
    if require_clean:
        require_clean_git_state(state)
    commit = state["commit"]
    assert type(commit) is str
    hashes = {
        relative: sha256_file(REPO_ROOT / relative)
        for relative in CONTROL_FILES
    }
    committed = _control_hashes_at_commit(commit)
    if hashes != committed:
        raise AnchorError("live control files differ from HEAD blobs")
    return {"commit": commit, "control_files": hashes}


def verify_historical_source_anchor(anchor: Mapping[str, object]) -> None:
    recorded = _validate_source_anchor_shape(dict(anchor))
    if _control_hashes_at_commit(str(recorded["commit"])) != recorded["control_files"]:
        raise AnchorError("recorded control-file hashes disagree with historical Git blobs")


def collect_environment_evidence() -> dict[str, object]:
    if Path(torch.__file__).resolve().is_relative_to(Path(r"D:\conda\envs\spi")):
        raise EnvironmentError("legacy environment is forbidden")
    try:
        from scripts.reproducibility.verify_environment_lock import verify_environment_lock

        summary = verify_environment_lock(
            ENVIRONMENT_LOCK_PATH,
            strict=True,
            requirements_lock=REQUIREMENTS_LOCK_PATH,
        )
    except Exception as exc:
        raise EnvironmentError("strict environment lock verification failed") from exc
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") is not None:
        raise EnvironmentError("CUBLAS_WORKSPACE_CONFIG must remain null")
    if torch.are_deterministic_algorithms_enabled():
        raise EnvironmentError("strict deterministic algorithms must remain disabled")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return {
        "dependencies_sha256": summary["dependencies_sha256"],
        "fingerprint_sha256": summary["fingerprint_sha256"],
        "policy": deepcopy(REPRODUCIBILITY_POLICY),
    }


def require_resources(contract: Mapping[str, object], root: Path) -> None:
    resources = contract.get("resources")
    if type(resources) is not dict:
        raise ArtifactError("contract resource declaration is invalid")
    probe = root
    while not probe.exists():
        probe = probe.parent
    if shutil.disk_usage(probe).free < resources["minimum_free_disk_bytes"]:
        raise ArtifactError("declared free-disk requirement is not met")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise TrainingError("exact cuda:0 runtime is unavailable")
    properties = torch.cuda.get_device_properties(0)
    if properties.total_memory < resources["minimum_cuda_memory_bytes"]:
        raise TrainingError("declared CUDA-memory requirement is not met")


def _configure_seeded_best_effort(seed: int) -> None:
    if type(seed) is not int:
        raise TrainingError("training seed must be an integer")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _make_model() -> UNet3D:
    return UNet3D(
        in_channels=1,
        base_channels=32,
        channel_mults=[1, 2, 4],
        emb_dim=128,
    )


def _forward(model: torch.nn.Module, noisy: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    try:
        return model(noisy, sigma)
    except TypeError:
        return model(noisy)


def _make_optimizer(model: torch.nn.Module, options: Mapping[str, object]) -> torch.optim.AdamW:
    if options.get("name") != "AdamW":
        raise TrainingError("optimizer must be AdamW")
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(options["lr"]),
        weight_decay=float(options["weight_decay"]),
        betas=tuple(options["betas"]),
        eps=float(options["eps"]),
        amsgrad=bool(options["amsgrad"]),
        maximize=bool(options["maximize"]),
        foreach=bool(options["foreach"]),
        fused=bool(options["fused"]),
        capturable=bool(options["capturable"]),
        differentiable=bool(options["differentiable"]),
    )


def _initial_ema(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def _update_ema(
    ema: dict[str, torch.Tensor], model: torch.nn.Module, decay: float
) -> None:
    with torch.no_grad():
        for key, value in model.state_dict().items():
            if value.is_floating_point():
                ema[key].mul_(decay).add_(value.detach(), alpha=1.0 - decay)
            else:
                ema[key].copy_(value)


def _noise_batch(
    clean: torch.Tensor, *, sigma_min: float, sigma_max: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = clean.shape[0]
    uniform = torch.rand(batch, device=clean.device)
    sigma = torch.exp(
        (1.0 - uniform) * math.log(sigma_min) + uniform * math.log(sigma_max)
    )
    epsilon = torch.randn_like(clean)
    return clean + sigma[:, None, None, None, None] * epsilon, sigma, epsilon


def _validate_state_dict(
    state: object,
    expected: Mapping[str, torch.Tensor],
    *,
    noun: str,
) -> dict[str, torch.Tensor]:
    if type(state) is not dict or set(state) != set(expected):
        raise ArtifactError(f"{noun} checkpoint keys mismatch")
    checked: dict[str, torch.Tensor] = {}
    for key, reference in expected.items():
        value = state[key]
        if type(value) is not torch.Tensor:
            raise ArtifactError(f"{noun} checkpoint value is not a tensor")
        if value.shape != reference.shape:
            raise ArtifactError(f"{noun} checkpoint shape mismatch for {key}")
        if value.dtype != reference.dtype:
            raise ArtifactError(f"{noun} checkpoint dtype mismatch for {key}")
        if value.is_floating_point() and not torch.isfinite(value).all().item():
            raise ArtifactError(f"{noun} checkpoint contains nonfinite tensor")
        checked[key] = value
    return checked


def verify_checkpoint_pair(
    raw_path: Path,
    ema_path: Path,
    *,
    model_factory: Callable[[], torch.nn.Module] = _make_model,
) -> dict[str, object]:
    if not raw_path.is_file() or not ema_path.is_file():
        raise ArtifactError("final checkpoint pair is incomplete")
    expected = model_factory().state_dict()
    try:
        raw = torch.load(raw_path, map_location="cpu", weights_only=True)
        ema = torch.load(ema_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ArtifactError("checkpoint cannot be loaded safely") from exc
    raw_checked = _validate_state_dict(raw, expected, noun="raw")
    ema_checked = _validate_state_dict(ema, expected, noun="EMA")
    if all(torch.equal(raw_checked[key], ema_checked[key]) for key in expected):
        raise ArtifactError("raw and EMA checkpoint tensors are identical")
    raw_hash = sha256_file(raw_path)
    ema_hash = sha256_file(ema_path)
    if raw_hash == ema_hash:
        raise ArtifactError("raw and EMA checkpoint file hashes are identical")
    return {
        "raw_file_sha256": raw_hash,
        "ema_file_sha256": ema_hash,
        "promotable": "ema-final.pt",
    }


def _training_candidate(
    *,
    contract: Mapping[str, object],
    dataset_manifest_path: Path,
    source_anchor: Mapping[str, object],
    environment: Mapping[str, object],
    epoch_losses: list[float],
    optimizer_steps: int,
    checkpoint_summary: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema": "diffusion-prior-training-candidate-v2",
        "logical_id": "gsdiff-diffusion-prior-v2",
        "contract_sha256": contract_sha256(contract),
        "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
        "source_anchor": deepcopy(dict(source_anchor)),
        "environment": deepcopy(dict(environment)),
        "epochs_completed": len(epoch_losses),
        "optimizer_steps": optimizer_steps,
        "epoch_mean_losses": epoch_losses,
        "checkpoints": deepcopy(dict(checkpoint_summary)),
    }


def validate_training_candidate(
    path: Path,
    *,
    contract: Mapping[str, object],
    dataset_manifest_path: Path,
    raw_path: Path,
    ema_path: Path,
    model_factory: Callable[[], torch.nn.Module] = _make_model,
) -> dict[str, object]:
    candidate = _load_canonical_json(path, noun="training candidate")
    expected_fields = {
        "schema", "logical_id", "contract_sha256", "dataset_manifest_sha256",
        "source_anchor", "environment", "epochs_completed", "optimizer_steps",
        "epoch_mean_losses", "checkpoints"
    }
    if set(candidate) != expected_fields:
        raise ArtifactError("training candidate fields are invalid")
    if candidate["schema"] != "diffusion-prior-training-candidate-v2" or candidate["logical_id"] != "gsdiff-diffusion-prior-v2":
        raise ArtifactError("training candidate identity is invalid")
    if candidate["contract_sha256"] != contract_sha256(contract) or candidate["dataset_manifest_sha256"] != sha256_file(dataset_manifest_path):
        raise ArtifactError("training candidate cross-hash mismatch")
    _validate_source_anchor_shape(candidate["source_anchor"])
    _validate_environment_shape(candidate["environment"])
    training = contract["training"]
    assert type(training) is dict
    losses = candidate["epoch_mean_losses"]
    if (
        type(candidate["epochs_completed"]) is not int
        or candidate["epochs_completed"] != training["epochs"]
        or type(candidate["optimizer_steps"]) is not int
        or candidate["optimizer_steps"] != training["optimizer_steps"]
        or type(losses) is not list
        or len(losses) != training["epochs"]
        or any(type(loss) not in (int, float) or isinstance(loss, bool) or not math.isfinite(loss) for loss in losses)
    ):
        raise ArtifactError("training candidate counts or losses are invalid")
    checkpoints = verify_checkpoint_pair(raw_path, ema_path, model_factory=model_factory)
    if candidate["checkpoints"] != checkpoints:
        raise ArtifactError("training candidate checkpoint hashes mismatch")
    return candidate


def run_training(
    *,
    contract: Mapping[str, object],
    dataset_path: Path,
    manifest_path: Path,
    candidate_path: Path,
    raw_path: Path,
    ema_path: Path,
    source_anchor: Mapping[str, object],
    environment: Mapping[str, object],
    device: torch.device,
    model_factory: Callable[[], torch.nn.Module] = _make_model,
    loss_hook: Callable[[torch.Tensor, int], torch.Tensor] | None = None,
    gradient_hook: Callable[[torch.nn.Parameter, int], None] | None = None,
    pre_candidate_check: Callable[[], None] | None = None,
) -> dict[str, object]:
    if candidate_path.exists() or raw_path.exists() or ema_path.exists():
        raise TrainingError("training output already exists; resume and selection are forbidden")
    validate_dataset_pair(
        dataset_path,
        manifest_path,
        contract=contract,
        source_anchor=source_anchor,
        environment=environment,
    )
    training = contract["training"]
    noise = contract["noise"]
    assert type(training) is dict and type(noise) is dict
    if str(device) not in {"cpu", "cuda:0"}:
        raise TrainingError("device is not the exact test fixture or production cuda:0")
    if str(device) == "cuda:0" and (not torch.cuda.is_available() or torch.cuda.current_device() != 0):
        raise TrainingError("silent CPU or nonzero-CUDA fallback is forbidden")
    _configure_seeded_best_effort(int(contract["rng"]["training_seed"]))
    payload = torch.load(dataset_path, map_location="cpu", weights_only=True)
    videos = payload["videos"]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(training["dataloader_generator_seed"]))
    loader = DataLoader(
        TensorDataset(videos),
        batch_size=int(training["batch_size"]),
        shuffle=True,
        generator=generator,
        drop_last=True,
        num_workers=0,
    )
    model = model_factory().to(device)
    optimizer = _make_optimizer(model, training["optimizer"])
    ema = _initial_ema(model)
    step = 0
    epoch_means: list[float] = []
    try:
        for _epoch in range(int(training["epochs"])):
            total = 0.0
            batches = 0
            for (clean_cpu,) in loader:
                clean = clean_cpu.unsqueeze(1).to(device)
                noisy, sigma, epsilon = _noise_batch(
                    clean,
                    sigma_min=float(noise["sigma_min"]),
                    sigma_max=float(noise["sigma_max"]),
                )
                optimizer.zero_grad(set_to_none=True)
                prediction = _forward(model, noisy, sigma)
                loss = torch.nn.functional.mse_loss(prediction, epsilon)
                if loss_hook is not None:
                    loss = loss_hook(loss, step)
                if loss.numel() != 1 or not torch.isfinite(loss).item():
                    raise TrainingError("nonfinite loss refused")
                loss.backward()
                if gradient_hook is not None:
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            gradient_hook(parameter, step)
                            break
                for parameter in model.parameters():
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all().item():
                        raise TrainingError("nonfinite gradient refused")
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip_norm"]))
                if not torch.isfinite(norm).item():
                    raise TrainingError("nonfinite gradient norm refused")
                optimizer.step()
                step += 1
                _update_ema(ema, model, float(training["ema_decay"]))
                total += float(loss.detach().item())
                batches += 1
            if batches != training["batches_per_epoch"]:
                raise TrainingError("epoch batch count mismatch")
            mean = total / batches
            if not math.isfinite(mean):
                raise TrainingError("nonfinite epoch mean refused")
            epoch_means.append(mean)
        if step != training["optimizer_steps"]:
            raise TrainingError("optimizer step count mismatch")
        raw_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        ema_state = {key: value.detach().cpu() for key, value in ema.items()}
        _validate_state_dict(raw_state, model_factory().state_dict(), noun="raw")
        _validate_state_dict(ema_state, model_factory().state_dict(), noun="EMA")
        _atomic_torch_save(raw_state, raw_path)
        _atomic_torch_save(ema_state, ema_path)
        checkpoint_summary = verify_checkpoint_pair(
            raw_path, ema_path, model_factory=model_factory
        )
        candidate = _training_candidate(
            contract=contract,
            dataset_manifest_path=manifest_path,
            source_anchor=source_anchor,
            environment=environment,
            epoch_losses=epoch_means,
            optimizer_steps=step,
            checkpoint_summary=checkpoint_summary,
        )
        if pre_candidate_check is not None:
            pre_candidate_check()
        _atomic_json_write(candidate, candidate_path)
        return validate_training_candidate(
            candidate_path,
            contract=contract,
            dataset_manifest_path=manifest_path,
            raw_path=raw_path,
            ema_path=ema_path,
            model_factory=model_factory,
        )
    except BaseException:
        if candidate_path.exists():
            candidate_path.unlink()
        raise


def run_preflight(
    *, save_directory: Path, device: torch.device
) -> dict[str, object]:
    if str(device) != "cuda:0" or not torch.cuda.is_available():
        raise TrainingError("preflight requires real cuda:0 without fallback")
    _configure_seeded_best_effort(42)
    before = set(save_directory.iterdir()) if save_directory.exists() else set()
    save_directory.mkdir(parents=True, exist_ok=True)
    model = _make_model().to(device)
    optimizer = _make_optimizer(
        model,
        {
            "name": "AdamW", "lr": 1e-4, "weight_decay": 1e-5,
            "betas": [0.9, 0.999], "eps": 1e-8, "amsgrad": False,
            "maximize": False, "foreach": False, "fused": False,
            "capturable": False, "differentiable": False,
        },
    )
    clean = torch.rand((8, 1, 20, 64, 64), device=device)
    noisy, sigma, epsilon = _noise_batch(clean, sigma_min=0.002, sigma_max=0.5)
    prediction = model(noisy, sigma)
    loss = torch.nn.functional.mse_loss(prediction, epsilon)
    if not torch.isfinite(loss).item():
        raise TrainingError("preflight produced nonfinite loss")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if not torch.isfinite(norm).item():
        raise TrainingError("preflight produced nonfinite gradient norm")
    ema = _initial_ema(model)
    optimizer.step()
    _update_ema(ema, model, 0.999)
    with tempfile.TemporaryDirectory(dir=save_directory, prefix=".preflight-") as temporary:
        checkpoint = Path(temporary) / "ema.pt"
        torch.save({key: value.detach().cpu() for key, value in ema.items()}, checkpoint)
        loaded = torch.load(checkpoint, map_location=device, weights_only=True)
        inference_model = _make_model().to(device)
        inference_model.load_state_dict(loaded, strict=True)
        inference_model.eval()
        with torch.no_grad():
            output = inference_model(clean[:1], sigma[:1])
        if output.shape != clean[:1].shape or not torch.isfinite(output).all().item():
            raise TrainingError("preflight EMA inference failed")
    after = set(save_directory.iterdir())
    if after != before:
        raise TrainingError("preflight left durable artifacts")
    return {
        "batch_shape": [8, 1, 20, 64, 64],
        "device": "cuda:0",
        "durable_writes": 0,
        "optimizer_steps": 1,
    }


def make_provenance(
    *,
    contract_sha256: str,
    dataset_manifest_sha256: str,
    candidate_sha256: str,
    source_anchor: Mapping[str, object],
    environment: Mapping[str, object],
    checkpoints: Mapping[str, object],
    verification_commit: str,
) -> dict[str, object]:
    return {
        "schema": "diffusion-prior-provenance-v2",
        "logical_id": "gsdiff-diffusion-prior-v2",
        "contract_sha256": contract_sha256,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "training_candidate_sha256": candidate_sha256,
        "source_anchor": deepcopy(dict(source_anchor)),
        "environment": deepcopy(dict(environment)),
        "checkpoints": deepcopy(dict(checkpoints)),
        "verification_commit": verification_commit,
    }


def validate_provenance(
    path: Path,
    *,
    current_commit: str | None = None,
) -> dict[str, object]:
    value = _load_canonical_json(path, noun="diffusion prior provenance")
    schema = _load_schema(PROVENANCE_SCHEMA_PATH)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(value),
        key=lambda error: list(error.absolute_path),
    )
    if errors:
        raise ArtifactError(
            "invalid diffusion prior provenance: "
            + "; ".join(error.message for error in errors)
        )
    _validate_source_anchor_shape(value["source_anchor"])
    _validate_environment_shape(value["environment"])
    if value["verification_commit"] != value["source_anchor"]["commit"]:
        raise ArtifactError("provenance verification source commit mismatch")
    if current_commit is not None and (
        type(current_commit) is not str
        or len(current_commit) != 40
        or any(ch not in "0123456789abcdef" for ch in current_commit)
    ):
        raise ArtifactError("current commit is invalid")
    if value["checkpoints"]["raw_file_sha256"] == value["checkpoints"]["ema_file_sha256"]:
        raise ArtifactError("provenance raw and EMA hashes must differ")
    return value


def _verify_ema_inference(ema_path: Path) -> None:
    if not torch.cuda.is_available():
        raise TrainingError("independent verification requires real cuda:0")
    model = _make_model().to("cuda:0")
    state = torch.load(ema_path, map_location="cuda:0", weights_only=True)
    _validate_state_dict(state, model.state_dict(), noun="EMA")
    model.load_state_dict(state, strict=True)
    model.eval()
    with torch.no_grad():
        batch = torch.zeros((1, 1, 20, 64, 64), device="cuda:0")
        sigma = torch.full((1,), 0.1, device="cuda:0")
        output = model(batch, sigma)
    if output.shape != batch.shape or not torch.isfinite(output).all().item():
        raise TrainingError("independent CUDA EMA inference failed")


def verify_and_publish(
    *,
    contract: Mapping[str, object],
    root: Path,
    provenance_path: Path = PROVENANCE_PATH,
) -> dict[str, object]:
    dataset_path = root / "dataset.pt"
    manifest_path = root / "dataset-manifest.json"
    raw_path = root / "raw-final.pt"
    ema_path = root / "ema-final.pt"
    candidate_path = root / "training-candidate.json"
    candidate = validate_training_candidate(
        candidate_path,
        contract=contract,
        dataset_manifest_path=manifest_path,
        raw_path=raw_path,
        ema_path=ema_path,
    )
    source_anchor = candidate["source_anchor"]
    environment = candidate["environment"]
    assert type(source_anchor) is dict and type(environment) is dict
    verify_historical_source_anchor(source_anchor)
    manifest = validate_dataset_pair(
        dataset_path,
        manifest_path,
        contract=contract,
        source_anchor=source_anchor,
        environment=environment,
    )
    current_leakage = audit_target_disjointness(SCIENTIFIC_CONTRACTS_PATH)
    if manifest["leakage_audit"] != current_leakage:
        raise LeakageError("recorded leakage audit disagrees with the real registry")
    current_environment = collect_environment_evidence()
    require_matching_environment(environment, current_environment)
    checkpoints = verify_checkpoint_pair(raw_path, ema_path)
    _verify_ema_inference(ema_path)
    current_commit = _git_output("rev-parse", "HEAD").decode("ascii").strip()
    if provenance_path.exists():
        existing = validate_provenance(provenance_path, current_commit=current_commit)
        verification_commit = str(existing["verification_commit"])
        if _git_output("cat-file", "-t", verification_commit).strip() != b"commit":
            raise ArtifactError("recorded verification source is not a Git commit")
        recomputed = make_provenance(
            contract_sha256=contract_sha256(contract),
            dataset_manifest_sha256=sha256_file(manifest_path),
            candidate_sha256=sha256_file(candidate_path),
            source_anchor=source_anchor,
            environment=environment,
            checkpoints=checkpoints,
            verification_commit=verification_commit,
        )
        if existing != recomputed:
            raise ArtifactError("existing tracked provenance disagrees with recomputation")
        return existing
    provenance = make_provenance(
        contract_sha256=contract_sha256(contract),
        dataset_manifest_sha256=sha256_file(manifest_path),
        candidate_sha256=sha256_file(candidate_path),
        source_anchor=source_anchor,
        environment=environment,
        checkpoints=checkpoints,
        verification_commit=current_commit,
    )
    _atomic_json_write(provenance, provenance_path)
    return validate_provenance(provenance_path, current_commit=current_commit)


def dataset_cli() -> int:
    contract = load_contract()
    root = artifact_root(contract)
    require_resources(contract, root)
    anchor_before = current_source_anchor(require_clean=True)
    environment_before = collect_environment_evidence()
    leakage = audit_target_disjointness(SCIENTIFIC_CONTRACTS_PATH)

    def recheck_before_promotion() -> None:
        require_matching_anchor(
            anchor_before, current_source_anchor(require_clean=True), require_same_commit=True
        )
        require_matching_environment(environment_before, collect_environment_evidence())

    manifest = generate_dataset_artifact(
        contract=contract,
        artifact_root=root,
        source_anchor=anchor_before,
        environment=environment_before,
        leakage_audit=leakage,
        pre_promote_check=recheck_before_promotion,
    )
    anchor_after = current_source_anchor(require_clean=True)
    environment_after = collect_environment_evidence()
    require_matching_anchor(anchor_before, anchor_after, require_same_commit=True)
    require_matching_environment(environment_before, environment_after)
    print(
        "diffusion_prior_v2_dataset=valid "
        f"contract_sha256={contract_sha256(contract)} "
        f"file_sha256={manifest['dataset']['file_sha256']}"
    )
    return 0


def training_cli(*, preflight_only: bool) -> int:
    contract = load_contract()
    root = artifact_root(contract)
    require_resources(contract, root)
    anchor_before = current_source_anchor(require_clean=True)
    environment_before = collect_environment_evidence()
    validate_dataset_pair(
        root / "dataset.pt",
        root / "dataset-manifest.json",
        contract=contract,
        source_anchor=anchor_before,
        environment=environment_before,
    )
    if preflight_only:
        result = run_preflight(save_directory=root, device=torch.device("cuda:0"))
        print(
            "diffusion_prior_v2_preflight=passed "
            f"device={result['device']} optimizer_steps=1"
        )
    else:
        def recheck_before_candidate() -> None:
            require_matching_anchor(
                anchor_before,
                current_source_anchor(require_clean=True),
                require_same_commit=True,
            )
            require_matching_environment(
                environment_before, collect_environment_evidence()
            )

        result = run_training(
            contract=contract,
            dataset_path=root / "dataset.pt",
            manifest_path=root / "dataset-manifest.json",
            candidate_path=root / "training-candidate.json",
            raw_path=root / "raw-final.pt",
            ema_path=root / "ema-final.pt",
            source_anchor=anchor_before,
            environment=environment_before,
            device=torch.device("cuda:0"),
            pre_candidate_check=recheck_before_candidate,
        )
        print(
            "diffusion_prior_v2_training=candidate "
            f"epochs={result['epochs_completed']} optimizer_steps={result['optimizer_steps']}"
        )
    anchor_after = current_source_anchor(require_clean=True)
    environment_after = collect_environment_evidence()
    require_matching_anchor(anchor_before, anchor_after, require_same_commit=True)
    require_matching_environment(environment_before, environment_after)
    return 0


def verification_cli() -> int:
    contract = load_contract()
    provenance = verify_and_publish(contract=contract, root=artifact_root(contract))
    print(
        "diffusion_prior_v2_verification=passed "
        f"contract_sha256={provenance['contract_sha256']}"
    )
    return 0
