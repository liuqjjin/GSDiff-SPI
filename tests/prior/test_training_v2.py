from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Callable

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


def _leakage_audit() -> dict[str, object]:
    return {
        "descriptor_intersection": [],
        "canonical_image_sha256_intersection": [],
        "evaluation_descriptor_count": 8,
        "training_descriptor_count": 4,
        "evaluation_canonical_image_sha256": {
            f"evaluation:{index}": f"{index + 1:064x}" for index in range(8)
        },
        "training_canonical_image_sha256": {
            source: f"{index + 20:064x}" for index, source in enumerate(v2.SOURCES)
        },
        "protocol_sha256": "5" * 64,
        "registry_file_sha256": "6" * 64,
        "claim": (
            "exact descriptor/pixel disjointness only; not semantic or "
            "distributional independence"
        ),
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
        leakage_audit=_leakage_audit(),
    )
    manifest_path = root / "dataset-manifest.json"
    manifest_path.write_bytes(v2.canonical_json_bytes(manifest))
    return dataset_path, manifest_path


def _write_candidate_artifacts(
    root: Path, contract: dict[str, object]
) -> tuple[Path, Callable[[], torch.nn.Module]]:
    dataset_path, manifest_path = _write_dataset_pair(root, contract)

    def model_factory() -> torch.nn.Module:
        return torch.nn.Conv3d(1, 1, 1)

    reference = model_factory().state_dict()
    raw = {key: value.detach().clone() for key, value in reference.items()}
    ema = {key: value.detach().clone() + 0.25 for key, value in reference.items()}
    raw_path = root / "raw-final.pt"
    ema_path = root / "ema-final.pt"
    torch.save(raw, raw_path)
    torch.save(ema, ema_path)
    checkpoints = v2.verify_checkpoint_pair(
        raw_path, ema_path, model_factory=model_factory
    )
    candidate = v2._training_candidate(
        contract=contract,
        dataset_manifest_path=manifest_path,
        source_anchor=_anchor(),
        environment=_environment(),
        epoch_losses=[0.5, 0.25],
        optimizer_steps=4,
        checkpoint_summary=checkpoints,
    )
    candidate_path = root / "training-candidate.json"
    candidate_path.write_bytes(v2.canonical_json_bytes(candidate))
    return candidate_path, model_factory


def _isolate_verifier(
    monkeypatch: pytest.MonkeyPatch, current_commit: list[str]
) -> list[str]:
    historical: list[str] = []

    def verify_historical(anchor: dict[str, object]) -> None:
        historical.append(str(anchor["commit"]))

    def git_output(*args: str) -> bytes:
        if args == ("rev-parse", "HEAD"):
            return (current_commit[0] + "\n").encode("ascii")
        if args[:2] == ("cat-file", "-t"):
            return b"commit\n"
        raise AssertionError(f"unexpected Git read: {args}")

    monkeypatch.setattr(v2, "verify_historical_source_anchor", verify_historical)
    monkeypatch.setattr(v2, "audit_target_disjointness", lambda _path: _leakage_audit())
    monkeypatch.setattr(v2, "collect_environment_evidence", _environment)
    monkeypatch.setattr(v2, "_verify_ema_inference", lambda _path: None)
    monkeypatch.setattr(v2, "_git_output", git_output)
    return historical


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


@pytest.mark.parametrize("descriptor", v2.SOURCES)
@pytest.mark.parametrize(
    "vy,vx,omega",
    [(7.25, -4.5, 0.37), (-11.0, 9.125, -0.41)],
)
def test_v2_motion_is_elementwise_identical_to_existing_custom_se2_simulator(
    monkeypatch: pytest.MonkeyPatch,
    descriptor: str,
    vy: float,
    vx: float,
    omega: float,
) -> None:
    from gsdiff.data import simulation

    source_name = descriptor.partition(":")[2]
    monkeypatch.setattr(simulation.os.path, "exists", lambda _path: False)
    reference = simulation.generate_spi_data(
        H=64,
        W=64,
        T=20,
        K=4,
        pattern_type="random",
        motion_type="custom_se2",
        snr_db=0,
        seed=17,
        shape=source_name,
        motion_mode=2,
        gt_velocity=[vy, vx],
        gt_omega=omega,
    ).gt_frames
    actual = v2.render_se2_video(
        v2.make_source_image(descriptor, 64, 64),
        frames=20,
        vy=vy,
        vx=vx,
        omega=omega,
    )

    assert actual.dtype == np.float32
    assert np.array_equal(actual, reference)


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
        leakage_audit=_leakage_audit(),
        source_anchor=_anchor(),
        environment=_environment(),
    )
    assert manifest["dataset"]["file_sha256"] == hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    assert manifest["dataset"]["content_sha256"] == v2.streaming_tensor_sha256(
        torch.load(dataset_path, map_location="cpu", weights_only=True)["videos"]
    )

    dataset_path.write_bytes(dataset_path.read_bytes() + b"tamper")
    with pytest.raises(v2.ArtifactError, match="file"):
        v2.validate_dataset_pair(
            dataset_path,
            manifest_path,
            contract=contract,
            leakage_audit=_leakage_audit(),
        )


@pytest.mark.parametrize(
    "field",
    [
        "descriptor_intersection",
        "canonical_image_sha256_intersection",
        "evaluation_descriptor_count",
        "training_descriptor_count",
        "evaluation_canonical_image_sha256",
        "training_canonical_image_sha256",
        "protocol_sha256",
        "registry_file_sha256",
        "claim",
    ],
)
def test_dataset_manifest_requires_exact_fresh_leakage_evidence(
    tmp_path: Path, field: str
) -> None:
    contract = _tiny_contract()
    expected = _leakage_audit()
    dataset_path, manifest_path = _write_dataset_pair(tmp_path / "artifact", contract)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if field.endswith("intersection"):
        manifest["leakage_audit"][field] = ["collision"]
    elif field.endswith("count"):
        manifest["leakage_audit"][field] += 1
    elif field.endswith("image_sha256"):
        first = next(iter(manifest["leakage_audit"][field]))
        manifest["leakage_audit"][field][first] = "9" * 64
    elif field.endswith("sha256"):
        manifest["leakage_audit"][field] = "9" * 64
    else:
        manifest["leakage_audit"][field] = "expanded semantic claim"
    manifest_path.write_bytes(v2.canonical_json_bytes(manifest))

    with pytest.raises(v2.ArtifactError, match="leakage"):
        v2.validate_dataset_pair(
            dataset_path,
            manifest_path,
            contract=contract,
            source_anchor=_anchor(),
            environment=_environment(),
            leakage_audit=expected,
        )


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
        leakage_audit=_leakage_audit(),
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


def test_environment_capture_rejects_non_authoritative_interpreter_before_lock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = tmp_path / "python.exe"
    fake.write_bytes(b"not the authoritative interpreter")
    monkeypatch.setattr(sys, "executable", str(fake))

    with pytest.raises(v2.EnvironmentError, match="authoritative Python"):
        v2.collect_environment_evidence()


@pytest.mark.parametrize(
    "entrypoint",
    [
        v2.dataset_cli,
        lambda: v2.training_cli(preflight_only=True),
        v2.verification_cli,
    ],
)
def test_cli_control_paths_reject_non_authoritative_interpreter_first(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    entrypoint: Callable[[], int],
) -> None:
    fake = tmp_path / "python.exe"
    fake.write_bytes(b"not the authoritative interpreter")
    monkeypatch.setattr(sys, "executable", str(fake))

    with pytest.raises(v2.EnvironmentError, match="authoritative Python"):
        entrypoint()


def test_historical_git_reads_ignore_inherited_git_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    expected = subprocess.run(
        ["git", "--no-replace-objects", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "hostile-git-dir"))
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "hostile-config"))

    assert v2._git_output("rev-parse", "HEAD") == expected


def test_atomic_json_publication_never_overwrites_existing_destination(tmp_path: Path) -> None:
    destination = tmp_path / "provenance.json"
    destination.write_bytes(b"existing-winner")

    with pytest.raises(v2.ArtifactError, match="already exists"):
        v2._atomic_json_write({"schema": "candidate"}, destination)
    assert destination.read_bytes() == b"existing-winner"


def test_atomic_torch_publication_loses_race_without_clobbering_winner(tmp_path: Path) -> None:
    destination = tmp_path / "raw-final.pt"

    def publish_race_winner(_temporary: Path) -> None:
        destination.write_bytes(b"race-winner")

    with pytest.raises(v2.ArtifactError, match="already exists"):
        v2._atomic_torch_save(
            {"weight": torch.ones(1)},
            destination,
            pre_promote_check=publish_race_winner,
        )
    assert destination.read_bytes() == b"race-winner"


def test_atomic_publication_rejects_dangling_destination_link(tmp_path: Path) -> None:
    destination = tmp_path / "ema-final.pt"
    missing = tmp_path / "missing-target.pt"
    try:
        os.symlink(missing, destination)
    except OSError as exc:
        pytest.skip(f"file symlinks unavailable: {exc}")

    with pytest.raises(v2.ArtifactError, match="already exists"):
        v2._atomic_torch_save({"weight": torch.ones(1)}, destination)
    assert os.path.lexists(destination)
    assert not missing.exists()


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
        leakage_audit=_leakage_audit(),
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
            leakage_audit=_leakage_audit(),
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
            leakage_audit=_leakage_audit(),
            device=torch.device("cpu"),
            model_factory=lambda: torch.nn.Conv3d(1, 1, 1),
            gradient_hook=lambda parameter, _step: parameter.grad.fill_(float("nan")),
        )
    assert not (tmp_path / "candidate.json").exists()


def test_training_updates_ema_after_each_optimizer_step_with_exact_formula(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    contract = _tiny_contract()
    dataset_path, manifest_path = _write_dataset_pair(tmp_path / "dataset", contract)

    def model_factory() -> torch.nn.Module:
        model = torch.nn.Conv3d(1, 1, 1)
        with torch.no_grad():
            model.weight.fill_(0.25)
            model.bias.fill_(0.1)
        return model

    real_update = v2._update_ema
    observed_steps: list[int] = []

    def checked_update(
        ema: dict[str, torch.Tensor], model: torch.nn.Module, decay: float
    ) -> None:
        before = {key: value.clone() for key, value in ema.items()}
        current = {key: value.detach().clone() for key, value in model.state_dict().items()}
        assert any(
            not torch.equal(before[key], current[key])
            for key in before
            if current[key].is_floating_point()
        ), "EMA update ran before optimizer.step"
        real_update(ema, model, decay)
        for key, value in ema.items():
            expected = before[key] * decay + current[key] * (1.0 - decay)
            assert torch.equal(value, expected)
        observed_steps.append(1)

    monkeypatch.setattr(v2, "_update_ema", checked_update)
    v2.run_training(
        contract=contract,
        dataset_path=dataset_path,
        manifest_path=manifest_path,
        candidate_path=tmp_path / "candidate.json",
        raw_path=tmp_path / "raw.pt",
        ema_path=tmp_path / "ema.pt",
        source_anchor=_anchor(),
        environment=_environment(),
        leakage_audit=_leakage_audit(),
        device=torch.device("cpu"),
        model_factory=model_factory,
    )
    assert len(observed_steps) == 4


def test_training_candidate_publication_loses_race_without_deleting_winner(
    tmp_path: Path,
) -> None:
    contract = _tiny_contract()
    dataset_path, manifest_path = _write_dataset_pair(tmp_path / "dataset", contract)
    candidate_path = tmp_path / "candidate.json"

    def publish_race_winner() -> None:
        candidate_path.write_bytes(b"race-winner")

    with pytest.raises(v2.ArtifactError, match="already exists"):
        v2.run_training(
            contract=contract,
            dataset_path=dataset_path,
            manifest_path=manifest_path,
            candidate_path=candidate_path,
            raw_path=tmp_path / "raw.pt",
            ema_path=tmp_path / "ema.pt",
            source_anchor=_anchor(),
            environment=_environment(),
            leakage_audit=_leakage_audit(),
            device=torch.device("cpu"),
            model_factory=lambda: torch.nn.Conv3d(1, 1, 1),
            pre_candidate_check=publish_race_winner,
        )
    assert candidate_path.read_bytes() == b"race-winner"


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


@pytest.mark.parametrize("corruption", ["missing", "extra", "shape", "dtype", "finite"])
def test_checkpoint_verification_rejects_every_state_dict_corruption(
    tmp_path: Path, corruption: str
) -> None:
    factory = lambda: torch.nn.Conv3d(1, 1, 1)
    reference = factory().state_dict()
    raw = {key: value.detach().clone() for key, value in reference.items()}
    ema = {key: value.detach().clone() + 0.25 for key, value in reference.items()}
    if corruption == "missing":
        del ema["bias"]
    elif corruption == "extra":
        ema["unexpected"] = torch.zeros(1)
    elif corruption == "shape":
        ema["weight"] = torch.zeros((2, *ema["weight"].shape[1:]))
    elif corruption == "dtype":
        ema["weight"] = ema["weight"].double()
    else:
        ema["weight"].fill_(float("nan"))
    raw_path = tmp_path / "raw.pt"
    ema_path = tmp_path / "ema.pt"
    torch.save(raw, raw_path)
    torch.save(ema, ema_path)

    with pytest.raises(v2.ArtifactError):
        v2.verify_checkpoint_pair(raw_path, ema_path, model_factory=factory)


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


def test_first_provenance_publication_at_C_then_descendant_P_is_zero_write(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    contract = _tiny_contract()
    _candidate_path, model_factory = _write_candidate_artifacts(tmp_path / "artifact", contract)
    provenance_path = tmp_path / "tracked-provenance.json"
    current = ["1" * 40]
    historical = _isolate_verifier(monkeypatch, current)

    first = v2.verify_and_publish(
        contract=contract,
        root=tmp_path / "artifact",
        provenance_path=provenance_path,
        model_factory=model_factory,
    )
    assert first["verification_commit"] == "1" * 40
    before_bytes = provenance_path.read_bytes()
    before_stat = provenance_path.stat()
    current[0] = "9" * 40
    monkeypatch.setattr(
        v2,
        "_atomic_json_write",
        lambda *_args, **_kwargs: pytest.fail("existing provenance path performed a write"),
    )

    later = v2.verify_and_publish(
        contract=contract,
        root=tmp_path / "artifact",
        provenance_path=provenance_path,
        model_factory=model_factory,
    )
    assert later == first
    assert provenance_path.read_bytes() == before_bytes
    assert provenance_path.stat().st_mtime_ns == before_stat.st_mtime_ns
    assert historical == ["1" * 40, "1" * 40]


def test_first_provenance_publication_from_descendant_P_records_historical_C(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    contract = _tiny_contract()
    _candidate_path, model_factory = _write_candidate_artifacts(tmp_path / "artifact", contract)
    provenance_path = tmp_path / "tracked-provenance.json"
    current = ["9" * 40]
    _isolate_verifier(monkeypatch, current)

    published = v2.verify_and_publish(
        contract=contract,
        root=tmp_path / "artifact",
        provenance_path=provenance_path,
        model_factory=model_factory,
    )
    assert published["source_anchor"]["commit"] == "1" * 40
    assert published["verification_commit"] == "1" * 40
    assert v2.validate_provenance(provenance_path, current_commit="9" * 40) == published


def test_invalid_provenance_is_rejected_before_any_tracked_write(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    contract = _tiny_contract()
    _candidate_path, model_factory = _write_candidate_artifacts(tmp_path / "artifact", contract)
    provenance_path = tmp_path / "tracked-provenance.json"
    _isolate_verifier(monkeypatch, ["1" * 40])
    real_make = v2.make_provenance

    def invalid_make(**kwargs: object) -> dict[str, object]:
        value = real_make(**kwargs)
        value["verification_commit"] = "9" * 40
        return value

    monkeypatch.setattr(v2, "make_provenance", invalid_make)
    with pytest.raises(v2.ArtifactError, match="source commit"):
        v2.verify_and_publish(
            contract=contract,
            root=tmp_path / "artifact",
            provenance_path=provenance_path,
            model_factory=model_factory,
        )
    assert not os.path.lexists(provenance_path)


@pytest.mark.parametrize(
    "field",
    [
        "dataset_manifest_sha256",
        "training_candidate_sha256",
        "raw_file_sha256",
        "ema_file_sha256",
    ],
)
def test_existing_provenance_rejects_every_artifact_cross_hash_tamper_without_write(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, field: str
) -> None:
    contract = _tiny_contract()
    _candidate_path, model_factory = _write_candidate_artifacts(tmp_path / "artifact", contract)
    provenance_path = tmp_path / "tracked-provenance.json"
    _isolate_verifier(monkeypatch, ["1" * 40])
    v2.verify_and_publish(
        contract=contract,
        root=tmp_path / "artifact",
        provenance_path=provenance_path,
        model_factory=model_factory,
    )
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if field in {"raw_file_sha256", "ema_file_sha256"}:
        provenance["checkpoints"][field] = "a" * 64
    else:
        provenance[field] = "a" * 64
    provenance_path.write_bytes(v2.canonical_json_bytes(provenance))
    before = provenance_path.read_bytes()
    monkeypatch.setattr(
        v2,
        "_atomic_json_write",
        lambda *_args, **_kwargs: pytest.fail("tampered provenance path performed a write"),
    )

    with pytest.raises(v2.ArtifactError):
        v2.verify_and_publish(
            contract=contract,
            root=tmp_path / "artifact",
            provenance_path=provenance_path,
            model_factory=model_factory,
        )
    assert provenance_path.read_bytes() == before


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
