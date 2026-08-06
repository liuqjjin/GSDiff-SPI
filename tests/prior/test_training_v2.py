from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest
import torch

from gsdiff.prior import training_v2 as v2


REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = REPO_ROOT / "configs" / "training" / "diffusion-prior-v2.json"


def _tiny_contract() -> dict[str, object]:
    contract = deepcopy(v2.load_contract(CONTRACT_PATH))
    contract["dataset"]["num_videos"] = 4
    contract["dataset"]["shape"][0] = 4
    contract["dataset"]["source_counts"] = {source: 1 for source in v2.SOURCES}
    contract["training"]["batch_size"] = 2
    contract["training"]["epochs"] = 2
    contract["training"]["batches_per_epoch"] = 2
    contract["training"]["optimizer_steps"] = 4
    return contract


def _anchor() -> dict[str, object]:
    return {
        "commit": "1" * 40,
        "control_files": {path: "2" * 64 for path in v2.CONTROL_FILES},
    }


def _environment() -> dict[str, object]:
    return {
        "dependencies_sha256": "3" * 64,
        "fingerprint_sha256": "4" * 64,
        "policy": deepcopy(v2.REPRODUCIBILITY_POLICY),
    }


def _dataset_payload(contract: dict[str, object]) -> dict[str, object]:
    videos = torch.arange(4 * 20 * 64 * 64, dtype=torch.float32)
    videos = (videos.remainder(997) / 996.0).reshape(4, 20, 64, 64)
    return {"videos": videos}


def _write_dataset_pair(root: Path, contract: dict[str, object]) -> tuple[Path, Path]:
    root.mkdir(parents=True)
    dataset_path = root / "dataset.pt"
    torch.save(_dataset_payload(contract), dataset_path)
    manifest = v2.make_dataset_manifest(
        dataset_path,
        contract=contract,
        contract_sha256=v2.contract_sha256(contract),
        source_anchor=_anchor(),
        environment=_environment(),
        leakage_audit={
            "descriptor_intersection": [],
            "canonical_image_sha256_intersection": [],
            "evaluation_descriptor_count": 8,
            "training_descriptor_count": 4,
            "protocol_sha256": "5" * 64,
            "registry_file_sha256": "6" * 64,
        },
    )
    manifest_path = root / "dataset-manifest.json"
    manifest_path.write_bytes(v2.canonical_json_bytes(manifest))
    return dataset_path, manifest_path


def test_contract_is_canonical_fixed_and_step_counts_are_bound() -> None:
    raw = CONTRACT_PATH.read_bytes()
    contract = v2.load_contract(CONTRACT_PATH)

    assert raw == v2.canonical_json_bytes(contract)
    assert contract["logical_id"] == "gsdiff-diffusion-prior-v2"
    assert contract["dataset"] == {
        "device": "cpu",
        "dtype": "float32",
        "num_videos": 5000,
        "shape": [5000, 20, 64, 64],
        "source_counts": {source: 1250 for source in v2.SOURCES},
        "sources": list(v2.SOURCES),
    }
    assert contract["training"]["batches_per_epoch"] == 625
    assert contract["training"]["optimizer_steps"] == 125000
    assert contract["training"]["device"] == "cuda:0"
    assert contract["training"]["optimizer"] == {
        "amsgrad": False,
        "betas": [0.9, 0.999],
        "capturable": False,
        "differentiable": False,
        "eps": 1e-8,
        "foreach": False,
        "fused": False,
        "lr": 1e-4,
        "maximize": False,
        "name": "AdamW",
        "weight_decay": 1e-5,
    }


@pytest.mark.parametrize("mutation", ["extra", "missing", "bool-as-int", "tampered"])
def test_contract_rejects_noncanonical_or_invalid_content(
    tmp_path: Path, mutation: str
) -> None:
    contract = deepcopy(v2.load_contract(CONTRACT_PATH))
    if mutation == "extra":
        contract["unexpected"] = True
    elif mutation == "missing":
        del contract["training"]["epochs"]
    elif mutation == "bool-as-int":
        contract["training"]["epochs"] = True
    else:
        contract["training"]["optimizer_steps"] = 124999
    path = tmp_path / "contract.json"
    path.write_bytes(v2.canonical_json_bytes(contract))

    with pytest.raises(v2.ContractError):
        v2.load_contract(path)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema":"x","schema":"y"}', encoding="utf-8")
    with pytest.raises(v2.ContractError, match="duplicate"):
        v2.load_contract(duplicate)


def test_sources_are_closed_round_robin_and_sampling_matches_mt19937() -> None:
    rng = np.random.RandomState(0)
    observed = [v2.sample_video_parameters(i, rng) for i in range(8)]
    reference = np.random.RandomState(0)
    expected = []
    for i in range(8):
        expected.append(
            {
                "source": v2.SOURCES[i % 4],
                "vy": reference.uniform(-12.0, 12.0),
                "vx": reference.uniform(-12.0, 12.0),
                "omega": reference.uniform(-0.5, 0.5),
                "child_seed": int(reference.randint(0, 2**31)),
            }
        )

    assert observed == expected
    assert {source: sum(row["source"] == source for row in observed) for source in v2.SOURCES} == {
        source: 2 for source in v2.SOURCES
    }


def test_source_mapping_calls_only_closed_procedural_factory() -> None:
    calls: list[tuple[str, int, int]] = []

    def factory(name: str, height: int, width: int) -> np.ndarray:
        calls.append((name, height, width))
        return np.ones((height, width), dtype=np.float32)

    for descriptor in v2.SOURCES:
        result = v2.make_source_image(descriptor, 64, 64, image_factory=factory)
        assert result.shape == (64, 64)
    assert calls == [("7", 64, 64), ("L", 64, 64), ("T", 64, 64), ("circle", 64, 64)]
    with pytest.raises(v2.ContractError):
        v2.make_source_image("char:7", 64, 64, image_factory=factory)


def test_custom_se2_is_rotation_then_shift_float32_and_bounded() -> None:
    image = np.zeros((64, 64), dtype=np.float32)
    image[16:24, 20:28] = 1.0
    video = v2.render_se2_video(image, frames=20, vy=8.0, vx=-5.0, omega=0.3)

    assert video.shape == (20, 64, 64)
    assert video.dtype == np.float32
    assert np.isfinite(video).all()
    assert 0.0 <= float(video.min()) <= float(video.max()) <= 1.0
    assert np.array_equal(video[0], image)


def test_real_registry_leakage_audit_and_injected_collisions() -> None:
    audit = v2.audit_target_disjointness(
        REPO_ROOT / "configs" / "protocols" / "scientific-contracts-v1.yaml"
    )
    assert audit["evaluation_descriptor_count"] == 8
    assert audit["training_descriptor_count"] == 4
    assert audit["descriptor_intersection"] == []
    assert audit["canonical_image_sha256_intersection"] == []

    with pytest.raises(v2.LeakageError, match="descriptor"):
        v2.audit_target_disjointness(
            REPO_ROOT / "configs" / "protocols" / "scientific-contracts-v1.yaml",
            training_sources=("assets/tank.png",),
        )

    def colliding_factory(_descriptor: str, _height: int, _width: int) -> np.ndarray:
        return np.zeros((64, 64), dtype=np.float32)

    with pytest.raises(v2.LeakageError, match="pixel"):
        v2.audit_target_disjointness(
            REPO_ROOT / "configs" / "protocols" / "scientific-contracts-v1.yaml",
            training_sources=("procedural:7",),
            image_loader=lambda *_args: np.zeros((64, 64), dtype=np.float32),
            source_factory=colliding_factory,
        )


@pytest.mark.parametrize("corruption", ["shape", "dtype", "range", "finite"])
def test_dataset_validation_rejects_tensor_corruption(corruption: str) -> None:
    contract = _tiny_contract()
    payload = _dataset_payload(contract)
    videos = payload["videos"]
    if corruption == "shape":
        payload["videos"] = videos[:3]
    elif corruption == "dtype":
        payload["videos"] = videos.double()
    elif corruption == "range":
        videos[0, 0, 0, 0] = 2.0
    else:
        videos[0, 0, 0, 0] = torch.nan

    with pytest.raises(v2.ArtifactError):
        v2.validate_dataset_payload(payload, contract)


def test_dataset_manifest_binds_file_and_streaming_tensor_hashes(tmp_path: Path) -> None:
    contract = _tiny_contract()
    dataset_path, manifest_path = _write_dataset_pair(tmp_path / "artifact", contract)
    manifest = v2.validate_dataset_pair(
        dataset_path,
        manifest_path,
        contract=contract,
        source_anchor=_anchor(),
        environment=_environment(),
    )
    assert manifest["dataset"]["file_sha256"] == hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    assert manifest["dataset"]["content_sha256"] == v2.streaming_tensor_sha256(
        torch.load(dataset_path, map_location="cpu", weights_only=True)["videos"]
    )

    dataset_path.write_bytes(dataset_path.read_bytes() + b"tamper")
    with pytest.raises(v2.ArtifactError, match="file"):
        v2.validate_dataset_pair(dataset_path, manifest_path, contract=contract)


def test_dataset_generation_is_complete_last_and_refuses_one_sided_reuse(tmp_path: Path) -> None:
    contract = _tiny_contract()
    artifact_root = tmp_path / "artifact"
    publication_checks: list[str] = []

    def video_factory(_source: str, index: int, _rng: np.random.RandomState) -> torch.Tensor:
        return torch.full((20, 64, 64), index / 4.0, dtype=torch.float32)

    def pre_promote_check() -> None:
        assert not (artifact_root / "dataset.pt").exists()
        assert (artifact_root / "dataset.pt.tmp").is_file()
        assert not (artifact_root / "dataset-manifest.json").exists()
        publication_checks.append("checked")

    v2.generate_dataset_artifact(
        contract=contract,
        artifact_root=artifact_root,
        source_anchor=_anchor(),
        environment=_environment(),
        leakage_audit={
            "descriptor_intersection": [],
            "canonical_image_sha256_intersection": [],
            "evaluation_descriptor_count": 8,
            "training_descriptor_count": 4,
            "protocol_sha256": "5" * 64,
            "registry_file_sha256": "6" * 64,
        },
        video_factory=video_factory,
        pre_promote_check=pre_promote_check,
    )
    assert publication_checks == ["checked"]
    assert (artifact_root / "dataset.pt").is_file()
    assert (artifact_root / "dataset-manifest.json").is_file()
    before = {path.name: path.read_bytes() for path in artifact_root.iterdir()}
    v2.generate_dataset_artifact(
        contract=contract,
        artifact_root=artifact_root,
        source_anchor=_anchor(),
        environment=_environment(),
        leakage_audit=json.loads((artifact_root / "dataset-manifest.json").read_text())["leakage_audit"],
        video_factory=video_factory,
    )
    assert {path.name: path.read_bytes() for path in artifact_root.iterdir()} == before

    with pytest.raises(v2.ArtifactError, match="leakage"):
        v2.generate_dataset_artifact(
            contract=contract,
            artifact_root=artifact_root,
            source_anchor=_anchor(),
            environment=_environment(),
            leakage_audit={
                "descriptor_intersection": ["procedural:7"],
                "canonical_image_sha256_intersection": [],
            },
            video_factory=video_factory,
        )

    (artifact_root / "dataset-manifest.json").unlink()
    with pytest.raises(v2.ArtifactError, match="one-sided"):
        v2.generate_dataset_artifact(
            contract=contract,
            artifact_root=artifact_root,
            source_anchor=_anchor(),
            environment=_environment(),
            leakage_audit={},
            video_factory=video_factory,
        )


def test_anchor_and_environment_mismatch_fail_closed() -> None:
    recorded = _anchor()
    different = deepcopy(recorded)
    different["commit"] = "9" * 40
    with pytest.raises(v2.AnchorError):
        v2.require_matching_anchor(recorded, different, require_same_commit=True)
    with pytest.raises(v2.AnchorError, match="dirty"):
        v2.require_clean_git_state({**recorded, "clean": False})

    expected = _environment()
    actual = deepcopy(expected)
    actual["fingerprint_sha256"] = "8" * 64
    with pytest.raises(v2.EnvironmentError):
        v2.require_matching_environment(expected, actual)


def test_tiny_training_counts_ema_order_and_failure_writes_no_candidate(tmp_path: Path) -> None:
    contract = _tiny_contract()
    dataset_path, manifest_path = _write_dataset_pair(tmp_path / "dataset", contract)
    candidate_path = tmp_path / "candidate.json"

    publication_checks: list[str] = []

    def pre_candidate_check() -> None:
        assert (tmp_path / "raw-final.pt").is_file()
        assert (tmp_path / "ema-final.pt").is_file()
        assert not candidate_path.exists()
        publication_checks.append("checked")

    result = v2.run_training(
        contract=contract,
        dataset_path=dataset_path,
        manifest_path=manifest_path,
        candidate_path=candidate_path,
        raw_path=tmp_path / "raw-final.pt",
        ema_path=tmp_path / "ema-final.pt",
        source_anchor=_anchor(),
        environment=_environment(),
        device=torch.device("cpu"),
        model_factory=lambda: torch.nn.Conv3d(1, 1, 1),
        pre_candidate_check=pre_candidate_check,
    )
    assert publication_checks == ["checked"]
    assert result["epochs_completed"] == 2
    assert result["optimizer_steps"] == 4
    assert len(result["epoch_mean_losses"]) == 2
    assert candidate_path.is_file()
    raw = torch.load(tmp_path / "raw-final.pt", map_location="cpu", weights_only=True)
    ema = torch.load(tmp_path / "ema-final.pt", map_location="cpu", weights_only=True)
    assert raw.keys() == ema.keys()
    assert any(not torch.equal(raw[key], ema[key]) for key in raw)

    candidate_path.unlink()
    with pytest.raises(v2.TrainingError, match="nonfinite loss"):
        v2.run_training(
            contract=contract,
            dataset_path=dataset_path,
            manifest_path=manifest_path,
            candidate_path=candidate_path,
            raw_path=tmp_path / "bad-raw.pt",
            ema_path=tmp_path / "bad-ema.pt",
            source_anchor=_anchor(),
            environment=_environment(),
            device=torch.device("cpu"),
            model_factory=lambda: torch.nn.Conv3d(1, 1, 1),
            loss_hook=lambda _loss, _step: torch.tensor(float("nan")),
        )
    assert not candidate_path.exists()


def test_nonfinite_gradient_is_refused_before_publication(tmp_path: Path) -> None:
    contract = _tiny_contract()
    dataset_path, manifest_path = _write_dataset_pair(tmp_path / "dataset", contract)
    with pytest.raises(v2.TrainingError, match="gradient"):
        v2.run_training(
            contract=contract,
            dataset_path=dataset_path,
            manifest_path=manifest_path,
            candidate_path=tmp_path / "candidate.json",
            raw_path=tmp_path / "raw.pt",
            ema_path=tmp_path / "ema.pt",
            source_anchor=_anchor(),
            environment=_environment(),
            device=torch.device("cpu"),
            model_factory=lambda: torch.nn.Conv3d(1, 1, 1),
            gradient_hook=lambda parameter, _step: parameter.grad.fill_(float("nan")),
        )
    assert not (tmp_path / "candidate.json").exists()


def test_checkpoint_and_provenance_verification_distinguish_raw_and_promoted(tmp_path: Path) -> None:
    model = torch.nn.Conv3d(1, 1, 1)
    raw = {key: value.detach().clone() for key, value in model.state_dict().items()}
    ema = {key: value.detach().clone() + 0.25 for key, value in raw.items()}
    raw_path = tmp_path / "raw.pt"
    ema_path = tmp_path / "ema.pt"
    torch.save(raw, raw_path)
    torch.save(ema, ema_path)
    summary = v2.verify_checkpoint_pair(raw_path, ema_path, model_factory=lambda: torch.nn.Conv3d(1, 1, 1))
    assert summary["promotable"] == "ema-final.pt"
    assert summary["raw_file_sha256"] != summary["ema_file_sha256"]

    broken = deepcopy(ema)
    broken["weight"] = broken["weight"].double()
    torch.save(broken, ema_path)
    with pytest.raises(v2.ArtifactError, match="dtype"):
        v2.verify_checkpoint_pair(raw_path, ema_path, model_factory=lambda: torch.nn.Conv3d(1, 1, 1))


def test_provenance_historical_training_commit_allows_different_current_commit(tmp_path: Path) -> None:
    provenance = v2.make_provenance(
        contract_sha256="a" * 64,
        dataset_manifest_sha256="b" * 64,
        candidate_sha256="c" * 64,
        source_anchor=_anchor(),
        environment=_environment(),
        checkpoints={
            "raw_file_sha256": "d" * 64,
            "ema_file_sha256": "e" * 64,
            "promotable": "ema-final.pt",
        },
        verification_commit="1" * 40,
    )
    path = tmp_path / "provenance.json"
    path.write_bytes(v2.canonical_json_bytes(provenance))
    loaded = v2.validate_provenance(path, current_commit="9" * 40)
    assert loaded["source_anchor"]["commit"] == "1" * 40
    assert loaded["verification_commit"] == "1" * 40

    mismatched = deepcopy(provenance)
    mismatched["verification_commit"] = "f" * 40
    path.write_bytes(v2.canonical_json_bytes(mismatched))
    with pytest.raises(v2.ArtifactError, match="source commit"):
        v2.validate_provenance(path, current_commit="9" * 40)

    path.write_bytes(v2.canonical_json_bytes(provenance) + b"\n")
    with pytest.raises(v2.ArtifactError, match="canonical"):
        v2.validate_provenance(path, current_commit="f" * 40)


@pytest.mark.parametrize(
    "script,allowed",
    [
        ("generate_diffusion_prior_v2_dataset.py", set()),
        ("train_diffusion_prior_v2.py", {"--preflight-only"}),
        ("verify_diffusion_prior_v2.py", set()),
    ],
)
def test_cli_exposes_no_scientific_bypass(script: str, allowed: set[str]) -> None:
    completed = subprocess.run(
        [str(v2.AUTHORITATIVE_PYTHON), str(REPO_ROOT / "scripts" / script), "--help"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    options = {
        token.rstrip(",")
        for line in completed.stdout.splitlines()
        for token in line.split()
        if token.startswith("--")
    }
    assert options == {"--help", *allowed}
    for forbidden in ("device", "output", "force", "source", "shape"):
        assert forbidden not in completed.stdout.lower()
