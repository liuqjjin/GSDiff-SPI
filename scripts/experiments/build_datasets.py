"""Build or verify immutable datasets selected by an experiment protocol."""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    repository_import_root = str(REPO_ROOT)
    trusted_import_key = os.path.normcase(repository_import_root)
    retained_import_paths: list[str] = []
    for candidate in sys.path:
        try:
            candidate_key = os.path.normcase(
                str(Path(candidate or os.curdir).resolve())
            )
        except OSError:
            retained_import_paths.append(candidate)
            continue
        if candidate_key != trusted_import_key:
            retained_import_paths.append(candidate)
    sys.path[:] = [repository_import_root, *retained_import_paths]

from gsdiff.data.artifacts import (
    ArtifactValidationError,
    DatasetDirectoryDiscovery,
    TargetSnapshot,
    VerifiedDatasetDirectory,
    discover_dataset_directories,
    generate_corrected_dataset,
    publish_dataset,
    resolve_corrected_dataset_request,
    resolve_target_snapshot,
    verify_canonical_dataset_directory_discovery,
    verify_dataset_directory,
    verify_dataset_directory_discovery,
)
from gsdiff.experiments.identity import (
    canonical_json_bytes,
    git_state,
    verify_environment_requirements,
)
from gsdiff.experiments.protocol import expand_cells, load_protocol


_SOURCE_ROOTS = (
    Path("gsdiff"),
    Path("scripts"),
    Path("configs"),
    Path("schemas"),
    Path("assets"),
    Path("train.py"),
    Path("requirements-lock.txt"),
    Path("docs/reproducibility/environment-lock.json"),
)


@dataclass(frozen=True)
class DatasetRequest:
    request_sha256: str
    semantic_content: Mapping[str, object]
    target_snapshot: TargetSnapshot = field(repr=False)
    scientific_contract: Mapping[str, object] = field(repr=False)
    motion: Mapping[str, object] = field(repr=False)
    seed: int
    acquisition_config: Mapping[str, object] = field(repr=False)
    noise_calibration_entry: Mapping[str, object] = field(repr=False)
    generator: Mapping[str, object] = field(repr=False)
    runtime: Mapping[str, object] = field(repr=False)

    def generation_arguments(self) -> dict[str, object]:
        return {
            "scientific_contract": self.scientific_contract,
            "target_snapshot": self.target_snapshot,
            "motion": self.motion,
            "seed": self.seed,
            "acquisition_config": self.acquisition_config,
            "noise_calibration_entry": self.noise_calibration_entry,
            "generator": self.generator,
            "runtime": self.runtime,
        }


@dataclass(frozen=True)
class CampaignDatasetPlan:
    campaign: Mapping[str, object] = field(repr=False)
    expanded_cells: int
    expected_datasets: int
    requests: tuple[DatasetRequest, ...]


@dataclass(frozen=True)
class _CatalogCandidate:
    path: Path
    identity: str
    manifest_sha256: str
    projection: bytes


@dataclass(frozen=True)
class _DatasetCatalog:
    discovery: DatasetDirectoryDiscovery
    candidates: tuple[_CatalogCandidate, ...]
    corrupt_identities: tuple[str, ...]


def _matching_entry(
    entries: object,
    *,
    entry_id: str,
    noun: str,
) -> Mapping[str, object]:
    if not isinstance(entries, list):
        raise TypeError(f"{noun} registry entries must be a list")
    matches = [
        entry
        for entry in entries
        if isinstance(entry, Mapping) and entry.get("id") == entry_id
    ]
    if len(matches) != 1:
        raise ValueError(f"{noun} registry must contain one matching entry")
    return matches[0]


def plan_campaign_datasets(
    *,
    repo_root: Path,
    protocol_path: Path,
    runtime: Mapping[str, object],
    generator_commit: str,
) -> CampaignDatasetPlan:
    campaign = load_protocol(protocol_path)
    contracts = load_protocol(
        repo_root / "configs" / "protocols" / "scientific-contracts-v1.yaml"
    )
    calibrations = load_protocol(
        repo_root / "configs" / "protocols" / "noise-calibration-v1.yaml"
    )
    contract_id = campaign["scientific_contract_id"]
    if type(contract_id) is not str:
        raise TypeError("campaign scientific contract ID must be a string")
    contract_entry = _matching_entry(
        contracts["contracts"],
        entry_id=contract_id,
        noun="scientific contract",
    )
    if contract_entry["sha256"] != campaign["scientific_contract_sha256"]:
        raise ValueError(
            "campaign scientific contract hash does not match registry"
        )
    contract_content = contract_entry["content"]
    if not isinstance(contract_content, Mapping):
        raise TypeError("scientific contract content must be a mapping")
    targets = contract_content["targets"]
    motions = contract_content["motions"]
    generator_contract = contract_content["generator"]
    acquisition_configs = campaign["acquisition_configs"]
    if not all(
        isinstance(value, Mapping)
        for value in (
            targets,
            motions,
            generator_contract,
            acquisition_configs,
        )
    ):
        raise TypeError("campaign scientific content must be mappings")
    cells = expand_cells(campaign)
    scientific_contract = {
        "id": contract_entry["id"],
        "sha256": contract_entry["sha256"],
    }
    generator = {
        "id": generator_contract["id"],
        "version": generator_contract["version"],
        "git_commit": generator_commit,
    }

    unresolved: dict[bytes, tuple[object, ...]] = {}
    for cell in cells:
        acquisition_config = acquisition_configs[cell.acquisition_config_id]
        if not isinstance(acquisition_config, Mapping):
            raise TypeError("acquisition config must be a mapping")
        calibration_id = acquisition_config["noise_calibration_id"]
        if type(calibration_id) is not str:
            raise TypeError("noise calibration ID must be a string")
        calibration_entry = _matching_entry(
            calibrations["calibrations"],
            entry_id=calibration_id,
            noun="noise calibration",
        )
        motion_values = motions[cell.motion]
        if not isinstance(motion_values, Mapping):
            raise TypeError("motion content must be a mapping")
        motion = {"id": cell.motion, **motion_values}
        descriptor = targets[cell.target]
        if type(descriptor) is not str:
            raise TypeError("target descriptor must be a string")
        raw_content = {
            "scientific_contract": scientific_contract,
            "target": {
                "id": cell.target,
                "descriptor": descriptor,
            },
            "motion": motion,
            "seed": cell.seed,
            "acquisition_config": acquisition_config,
            "noise_calibration_entry": calibration_entry,
            "generator": generator,
            "runtime": runtime,
        }
        unresolved.setdefault(
            canonical_json_bytes(raw_content),
            (
                cell.target,
                descriptor,
                motion,
                cell.seed,
                acquisition_config,
                calibration_entry,
            ),
        )

    target_cache: dict[tuple[str, str, int, int], TargetSnapshot] = {}
    resolved_by_content: dict[bytes, DatasetRequest] = {}
    for raw_key in sorted(unresolved):
        (
            target_id,
            descriptor,
            motion,
            seed,
            acquisition_config,
            calibration_entry,
        ) = unresolved[raw_key]
        assert type(target_id) is str
        assert type(descriptor) is str
        assert isinstance(motion, Mapping)
        assert type(seed) is int
        assert isinstance(acquisition_config, Mapping)
        assert isinstance(calibration_entry, Mapping)
        image_size = acquisition_config["image_size"]
        if (
            not isinstance(image_size, list)
            or len(image_size) != 2
            or any(type(value) is not int for value in image_size)
        ):
            raise TypeError("acquisition image_size must be two integers")
        H, W = image_size
        cache_key = (target_id, descriptor, H, W)
        target_snapshot = target_cache.get(cache_key)
        if target_snapshot is None:
            target_snapshot = resolve_target_snapshot(
                repo_root=repo_root,
                target_id=target_id,
                descriptor=descriptor,
                H=H,
                W=W,
            )
            target_cache[cache_key] = target_snapshot
        arguments = {
            "scientific_contract": scientific_contract,
            "target_snapshot": target_snapshot,
            "motion": motion,
            "seed": seed,
            "acquisition_config": acquisition_config,
            "noise_calibration_entry": calibration_entry,
            "generator": generator,
            "runtime": runtime,
        }
        semantic_content = resolve_corrected_dataset_request(**arguments)
        encoded = canonical_json_bytes(semantic_content)
        resolved_by_content.setdefault(
            encoded,
            DatasetRequest(
                request_sha256=hashlib.sha256(encoded).hexdigest(),
                semantic_content=semantic_content,
                target_snapshot=target_snapshot,
                scientific_contract=scientific_contract,
                motion=motion,
                seed=seed,
                acquisition_config=acquisition_config,
                noise_calibration_entry=calibration_entry,
                generator=generator,
                runtime=runtime,
            ),
        )

    expected_datasets = campaign["expected_datasets"]
    if type(expected_datasets) is not int:
        raise TypeError("expected_datasets must be an exact integer")
    requests = tuple(
        sorted(
            resolved_by_content.values(),
            key=lambda request: request.request_sha256,
        )
    )
    if len(requests) != expected_datasets:
        raise ValueError(
            "content-level dataset count does not match expected_datasets"
        )
    return CampaignDatasetPlan(
        campaign=campaign,
        expanded_cells=len(cells),
        expected_datasets=expected_datasets,
        requests=requests,
    )


class _UsageError(ValueError):
    pass


class _DirtyWorktreeError(RuntimeError):
    pass


class _ProvenanceChangedError(RuntimeError):
    pass


class _AmbiguousCurrentDatasetError(RuntimeError):
    pass


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise _UsageError(message)


def _parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(prog="build_datasets.py")
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    return parser


def _from_repo(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _emit_report(report: Mapping[str, object]) -> None:
    sys.stdout.write(canonical_json_bytes(report).decode("utf-8") + "\n")


def _environment() -> dict[str, str]:
    return verify_environment_requirements(
        REPO_ROOT / "requirements-lock.txt",
        REPO_ROOT / "docs" / "reproducibility" / "environment-lock.json",
    )


def _git_state() -> dict[str, object]:
    return git_state(
        REPO_ROOT,
        tuple(REPO_ROOT / path for path in _SOURCE_ROOTS),
    )


def _diagnostic_inventory(
    paths: Sequence[Path],
) -> list[dict[str, object]]:
    counts: dict[str, int] = {}
    for path in paths:
        identity = path.name[1:65]
        counts[identity] = counts.get(identity, 0) + 1
    return [
        {
            "dataset_identity_sha256": identity,
            "count": counts[identity],
        }
        for identity in sorted(counts)
    ]


def _dry_run(
    *,
    protocol_path: Path,
    artifact_root: Path,
) -> int:
    runtime = _environment()
    state = _git_state()
    commit = state["commit"]
    if type(commit) is not str:
        raise TypeError("Git commit must be a string")
    plan = plan_campaign_datasets(
        repo_root=REPO_ROOT,
        protocol_path=protocol_path,
        runtime=runtime,
        generator_commit=commit,
    )
    discovery = discover_dataset_directories(artifact_root)
    existing_by_identity = {
        path.name: path for path in discovery.canonical_directories
    }
    records: list[dict[str, object]] = []
    identities: dict[str, str] = {}
    for request in plan.requests:
        generated = generate_corrected_dataset(
            **request.generation_arguments()
        )
        identity = generated.dataset_identity_sha256
        previous_request = identities.setdefault(
            identity, request.request_sha256
        )
        if previous_request != request.request_sha256:
            raise ValueError(
                "distinct scientific requests produced one dataset identity"
            )
        final_dir = existing_by_identity.get(identity)
        status = "would-create"
        if final_dir is not None:
            verified = verify_dataset_directory(
                final_dir,
                expected_dataset_identity_sha256=identity,
                expected_generated=generated,
            )
            if not verified.expected_generated_verified:
                raise RuntimeError(
                    "dry-run existing dataset lacked generated verification"
                )
            status = "exists-valid"
        records.append(
            {
                "dataset_identity_sha256": identity,
                "request_sha256": request.request_sha256,
                "status": status,
            }
        )
        del generated
    verify_dataset_directory_discovery(discovery)
    if len(identities) != plan.expected_datasets:
        raise ValueError(
            "generated identity count does not match expected_datasets"
        )
    campaign_id = plan.campaign["campaign_id"]
    protocol_sha256 = plan.campaign["protocol_sha256"]
    if type(campaign_id) is not str or type(protocol_sha256) is not str:
        raise TypeError("campaign identity fields must be strings")
    _recheck_exact_state(state)
    _emit_report(
        {
            "schema_version": "dataset-build-report-v1",
            "mode": "dry-run",
            "status": "complete",
            "campaign_id": campaign_id,
            "protocol_sha256": protocol_sha256,
            "expanded_cells": plan.expanded_cells,
            "expected_datasets": plan.expected_datasets,
            "observed_datasets": len(records),
            "publishable": not bool(state["dirty"]),
            "manifest_externally_anchored": False,
            "datasets": records,
            "stale_staging_count": len(
                discovery.stale_staging_directories
            ),
            "stale_staging": _diagnostic_inventory(
                discovery.stale_staging_directories
            ),
            "rejected_count": len(discovery.rejected_directories),
            "rejected": _diagnostic_inventory(
                discovery.rejected_directories
            ),
            "unmatched_datasets": sorted(
                set(existing_by_identity) - set(identities)
            ),
            "errors": [],
        }
    )
    return 0


def _semantic_projection_from_request(
    request: DatasetRequest,
) -> bytes:
    semantic = request.semantic_content
    calibration = semantic["noise_calibration"]
    if not isinstance(calibration, Mapping):
        raise TypeError("request noise calibration must be a mapping")
    projection = {
        "schema_version": "corrected-dataset-semantic-match-v1",
        "scientific_contract": semantic["scientific_contract"],
        "target": semantic["target"],
        "motion": semantic["motion"],
        "seed": semantic["seed"],
        "acquisition_config": semantic["acquisition_config"],
        "noise_calibration": {
            "id": calibration["id"],
            "registry_entry_sha256": calibration[
                "registry_entry_sha256"
            ],
        },
        "generator": semantic["generator"],
        "runtime": semantic["runtime"],
        "resolved_generator_config": semantic[
            "resolved_generator_config"
        ],
    }
    return canonical_json_bytes(projection)


def _semantic_projection_from_manifest(
    manifest: Mapping[str, object],
) -> bytes:
    identity = manifest["dataset_identity_spec"]
    config = manifest["resolved_generator_config"]
    calibration_record = manifest["noise_calibration_record"]
    if not all(
        isinstance(value, Mapping)
        for value in (identity, config, calibration_record)
    ):
        raise TypeError("verified manifest semantic fields must be mappings")
    dimensions = config["dimensions"]
    acquisition = config["acquisition"]
    calibration = calibration_record["calibration"]
    if not all(
        isinstance(value, Mapping)
        for value in (dimensions, acquisition, calibration)
    ):
        raise TypeError("verified manifest config fields must be mappings")
    projection = {
        "schema_version": "corrected-dataset-semantic-match-v1",
        "scientific_contract": identity["scientific_contract"],
        "target": config["target"],
        "motion": config["motion"],
        "seed": identity["seed"],
        "acquisition_config": {
            "image_size": [dimensions["H"], dimensions["W"]],
            "num_frames": dimensions["T"],
            "train_measurements": dimensions["K"],
            "holdout_measurements": dimensions["holdout_K"],
            **dict(acquisition),
        },
        "noise_calibration": {
            "id": calibration["id"],
            "registry_entry_sha256": calibration[
                "registry_entry_sha256"
            ],
        },
        "generator": identity["generator"],
        "runtime": identity["runtime"],
        "resolved_generator_config": config,
    }
    return canonical_json_bytes(projection)


def _catalog_candidate_from_verified(
    *,
    path: Path,
    identity: str,
    verified: VerifiedDatasetDirectory,
) -> _CatalogCandidate:
    if verified.manifest_externally_anchored:
        raise RuntimeError(
            "catalog verification unexpectedly claimed an external anchor"
        )
    if verified.expected_generated_verified:
        raise RuntimeError(
            "catalog verification unexpectedly claimed generated verification"
        )
    manifest = verified.manifest
    projection = _semantic_projection_from_manifest(manifest)
    candidate = _CatalogCandidate(
        path=path,
        identity=identity,
        manifest_sha256=verified.dataset_manifest_sha256,
        projection=projection,
    )
    del manifest
    return candidate


def _scan_dataset_catalog(artifact_root: Path) -> _DatasetCatalog:
    discovery = discover_dataset_directories(artifact_root)
    candidates: list[_CatalogCandidate] = []
    corrupt_identities: list[str] = []
    for path in discovery.canonical_directories:
        identity = path.name
        try:
            verified = verify_dataset_directory(
                path,
                expected_dataset_identity_sha256=identity,
            )
        except (ArtifactValidationError, TypeError):
            corrupt_identities.append(identity)
            continue
        candidate = _catalog_candidate_from_verified(
            path=path,
            identity=identity,
            verified=verified,
        )
        del verified
        candidates.append(candidate)
    return _DatasetCatalog(
        discovery=discovery,
        candidates=tuple(candidates),
        corrupt_identities=tuple(sorted(corrupt_identities)),
    )


def _reverify_catalog_candidate(
    candidate: _CatalogCandidate,
) -> None:
    verified = verify_dataset_directory(
        candidate.path,
        expected_dataset_identity_sha256=candidate.identity,
    )
    if verified.manifest_externally_anchored:
        raise RuntimeError(
            "catalog recheck unexpectedly claimed an external anchor"
        )
    if verified.expected_generated_verified:
        raise RuntimeError(
            "catalog recheck unexpectedly claimed generated verification"
        )
    if verified.dataset_manifest_sha256 != candidate.manifest_sha256:
        del verified
        raise ArtifactValidationError(
            "catalog candidate manifest changed after initial verification"
        )
    manifest = verified.manifest
    observed_projection = _semantic_projection_from_manifest(manifest)
    del manifest
    del verified
    if observed_projection != candidate.projection:
        raise ArtifactValidationError(
            "catalog candidate projection changed after initial verification"
        )


def _catalog_identities(catalog: _DatasetCatalog) -> set[str]:
    identities = {
        candidate.identity for candidate in catalog.candidates
    }
    identities.update(catalog.corrupt_identities)
    return identities


def _refresh_corrupt_catalog_candidates(
    catalog: _DatasetCatalog,
) -> tuple[_DatasetCatalog, bool]:
    candidates = list(catalog.candidates)
    remaining_corrupt: list[str] = []
    changed = False
    for identity in catalog.corrupt_identities:
        path = catalog.discovery.datasets_dir / identity
        try:
            verified = verify_dataset_directory(
                path,
                expected_dataset_identity_sha256=identity,
            )
        except (ArtifactValidationError, TypeError):
            remaining_corrupt.append(identity)
            continue
        candidate = _catalog_candidate_from_verified(
            path=path,
            identity=identity,
            verified=verified,
        )
        del verified
        candidates.append(candidate)
        changed = True
    return (
        _DatasetCatalog(
            discovery=catalog.discovery,
            candidates=tuple(
                sorted(candidates, key=lambda candidate: candidate.identity)
            ),
            corrupt_identities=tuple(remaining_corrupt),
        ),
        changed,
    )


def _rescan_changed_catalog(
    *,
    artifact_root: Path,
    original_discovery: DatasetDirectoryDiscovery,
) -> tuple[_DatasetCatalog, DatasetDirectoryDiscovery]:
    if original_discovery.datasets_dir_exists:
        verify_canonical_dataset_directory_discovery(
            original_discovery
        )
    catalog = _scan_dataset_catalog(artifact_root)
    if not original_discovery.datasets_dir_exists:
        original_discovery = catalog.discovery
    return catalog, original_discovery


def _stabilize_dataset_catalog(
    *,
    artifact_root: Path,
    catalog: _DatasetCatalog,
) -> _DatasetCatalog:
    original_discovery = catalog.discovery
    for _ in range(6):
        try:
            additions = verify_canonical_dataset_directory_discovery(
                catalog.discovery
            )
        except ArtifactValidationError:
            catalog, original_discovery = _rescan_changed_catalog(
                artifact_root=artifact_root,
                original_discovery=original_discovery,
            )
            continue
        if additions:
            catalog, original_discovery = _rescan_changed_catalog(
                artifact_root=artifact_root,
                original_discovery=original_discovery,
            )
            continue

        catalog, corrupt_changed = _refresh_corrupt_catalog_candidates(
            catalog
        )
        if corrupt_changed:
            continue

        fresh_discovery = discover_dataset_directories(artifact_root)
        fresh_identities = {
            path.name for path in fresh_discovery.canonical_directories
        }
        if fresh_identities != _catalog_identities(catalog):
            catalog, original_discovery = _rescan_changed_catalog(
                artifact_root=artifact_root,
                original_discovery=original_discovery,
            )
            continue
        try:
            catalog_additions = (
                verify_canonical_dataset_directory_discovery(
                    catalog.discovery
                )
            )
            fresh_additions = (
                verify_canonical_dataset_directory_discovery(
                    fresh_discovery
                )
            )
        except ArtifactValidationError:
            catalog, original_discovery = _rescan_changed_catalog(
                artifact_root=artifact_root,
                original_discovery=original_discovery,
            )
            continue
        if catalog_additions or fresh_additions:
            catalog, original_discovery = _rescan_changed_catalog(
                artifact_root=artifact_root,
                original_discovery=original_discovery,
            )
            continue

        catalog, corrupt_changed = _refresh_corrupt_catalog_candidates(
            catalog
        )
        if corrupt_changed:
            continue
        original_identities = {
            path.name
            for path in original_discovery.canonical_directories
        }
        expected_additions = (
            _catalog_identities(catalog) - original_identities
        )
        try:
            observed_additions = {
                path.name
                for path in verify_canonical_dataset_directory_discovery(
                    original_discovery
                )
            }
            final_fresh_additions = (
                verify_canonical_dataset_directory_discovery(
                    fresh_discovery
                )
            )
        except ArtifactValidationError:
            catalog, original_discovery = _rescan_changed_catalog(
                artifact_root=artifact_root,
                original_discovery=original_discovery,
            )
            continue
        if (
            observed_additions != expected_additions
            or final_fresh_additions
        ):
            catalog, original_discovery = _rescan_changed_catalog(
                artifact_root=artifact_root,
                original_discovery=original_discovery,
            )
            continue
        return _DatasetCatalog(
            discovery=fresh_discovery,
            candidates=catalog.candidates,
            corrupt_identities=catalog.corrupt_identities,
        )
    raise ArtifactValidationError(
        "canonical dataset catalog did not stabilize"
    )


def _catalog_matches(
    catalog: _DatasetCatalog,
    projection: bytes,
) -> tuple[_CatalogCandidate, ...]:
    return tuple(
        candidate
        for candidate in catalog.candidates
        if candidate.projection == projection
    )


def _campaign_report_fields(
    plan: CampaignDatasetPlan,
) -> tuple[str, str]:
    campaign_id = plan.campaign["campaign_id"]
    protocol_sha256 = plan.campaign["protocol_sha256"]
    if type(campaign_id) is not str or type(protocol_sha256) is not str:
        raise TypeError("campaign identity fields must be strings")
    return campaign_id, protocol_sha256


def _verify_only(
    *,
    protocol_path: Path,
    artifact_root: Path,
) -> int:
    runtime = _environment()
    state = _git_state()
    commit = state["commit"]
    if type(commit) is not str:
        raise TypeError("Git commit must be a string")
    plan = plan_campaign_datasets(
        repo_root=REPO_ROOT,
        protocol_path=protocol_path,
        runtime=runtime,
        generator_commit=commit,
    )
    expected_by_projection: dict[bytes, DatasetRequest] = {}
    for request in plan.requests:
        key = _semantic_projection_from_request(request)
        if key in expected_by_projection:
            raise ValueError("planned dataset semantic projection collision")
        expected_by_projection[key] = request

    discovery = discover_dataset_directories(artifact_root)
    matches: dict[str, list[str]] = {
        request.request_sha256: [] for request in plan.requests
    }
    unmatched: list[str] = []
    errors: list[dict[str, object]] = []
    for dataset_dir in discovery.canonical_directories:
        identity = dataset_dir.name
        try:
            verified = verify_dataset_directory(
                dataset_dir,
                expected_dataset_identity_sha256=identity,
            )
        except ArtifactValidationError:
            errors.append(
                {
                    "code": "corrupt-dataset",
                    "dataset_identity_sha256": identity,
                }
            )
            continue
        if verified.manifest_externally_anchored:
            raise RuntimeError(
                "verify-only unexpectedly claimed an external anchor"
            )
        if verified.expected_generated_verified:
            raise RuntimeError(
                "verify-only unexpectedly claimed generated verification"
            )
        projection = _semantic_projection_from_manifest(
            verified.manifest
        )
        request = expected_by_projection.get(projection)
        if request is None:
            unmatched.append(identity)
        else:
            matches[request.request_sha256].append(identity)
        del verified

    verify_dataset_directory_discovery(discovery)
    records: list[dict[str, object]] = []
    for request in plan.requests:
        identities = sorted(matches[request.request_sha256])
        if not identities:
            errors.append(
                {
                    "code": "missing-current-dataset",
                    "request_sha256": request.request_sha256,
                }
            )
        elif len(identities) > 1:
            errors.append(
                {
                    "code": "ambiguous-current-dataset",
                    "request_sha256": request.request_sha256,
                    "dataset_identity_sha256": identities,
                }
            )
        else:
            records.append(
                {
                    "dataset_identity_sha256": identities[0],
                    "request_sha256": request.request_sha256,
                    "status": "verified",
                }
            )

    campaign_id, protocol_sha256 = _campaign_report_fields(plan)
    succeeded = not errors
    _recheck_exact_state(state)
    _emit_report(
        {
            "schema_version": "dataset-build-report-v1",
            "mode": "verify-only",
            "status": "complete" if succeeded else "failed",
            "campaign_id": campaign_id,
            "protocol_sha256": protocol_sha256,
            "expanded_cells": plan.expanded_cells,
            "expected_datasets": plan.expected_datasets,
            "observed_datasets": len(records),
            "publishable": not bool(state["dirty"]),
            "manifest_externally_anchored": False,
            "datasets": records,
            "stale_staging_count": len(
                discovery.stale_staging_directories
            ),
            "stale_staging": _diagnostic_inventory(
                discovery.stale_staging_directories
            ),
            "rejected_count": len(discovery.rejected_directories),
            "rejected": _diagnostic_inventory(
                discovery.rejected_directories
            ),
            "unmatched_datasets": sorted(unmatched),
            "errors": errors,
        }
    )
    if succeeded:
        return 0
    sys.stderr.write("dataset verification failed\n")
    return 1


def _require_initial_clean_state(state: Mapping[str, object]) -> None:
    if type(state.get("dirty")) is not bool:
        raise TypeError("Git dirty state must be an exact boolean")
    if state["dirty"]:
        raise _DirtyWorktreeError("dataset build requires a clean worktree")


def _recheck_exact_state(expected: Mapping[str, object]) -> None:
    observed = _git_state()
    if canonical_json_bytes(observed) != canonical_json_bytes(expected):
        raise _ProvenanceChangedError(
            "dataset build provenance changed during execution"
        )


def _recheck_clean_state(expected: Mapping[str, object]) -> None:
    _require_initial_clean_state(expected)
    _recheck_exact_state(expected)


def _build(
    *,
    protocol_path: Path,
    artifact_root: Path,
) -> int:
    runtime = _environment()
    initial_state = _git_state()
    _require_initial_clean_state(initial_state)
    commit = initial_state["commit"]
    if type(commit) is not str:
        raise TypeError("Git commit must be a string")
    plan = plan_campaign_datasets(
        repo_root=REPO_ROOT,
        protocol_path=protocol_path,
        runtime=runtime,
        generator_commit=commit,
    )
    expected_by_projection: dict[bytes, DatasetRequest] = {}
    for request in plan.requests:
        projection = _semantic_projection_from_request(request)
        if projection in expected_by_projection:
            raise ValueError("planned dataset semantic projection collision")
        expected_by_projection[projection] = request

    initial_catalog = _scan_dataset_catalog(artifact_root)
    selected: dict[str, _CatalogCandidate] = {}
    missing: set[str] = set()
    for projection, request in expected_by_projection.items():
        matches = _catalog_matches(initial_catalog, projection)
        if len(matches) > 1:
            raise _AmbiguousCurrentDatasetError(
                "multiple current datasets match one scientific request"
            )
        if matches:
            selected[request.request_sha256] = matches[0]
        else:
            missing.add(request.request_sha256)

    for candidate in selected.values():
        _reverify_catalog_candidate(candidate)

    initial_catalog = _stabilize_dataset_catalog(
        artifact_root=artifact_root,
        catalog=initial_catalog,
    )
    selected = {}
    missing = set()
    for projection, request in expected_by_projection.items():
        matches = _catalog_matches(initial_catalog, projection)
        if len(matches) > 1:
            raise _AmbiguousCurrentDatasetError(
                "multiple current datasets match one scientific request"
            )
        if matches:
            selected[request.request_sha256] = matches[0]
        else:
            missing.add(request.request_sha256)

    outcomes: dict[str, tuple[str, str]] = {
        request_sha256: (candidate.identity, "reused")
        for request_sha256, candidate in selected.items()
    }
    identities: dict[str, str] = {
        candidate.identity: request_sha256
        for request_sha256, candidate in selected.items()
    }
    for request in plan.requests:
        if request.request_sha256 not in missing:
            continue
        _recheck_clean_state(initial_state)
        generated = generate_corrected_dataset(
            **request.generation_arguments()
        )
        _recheck_clean_state(initial_state)
        identity = generated.dataset_identity_sha256
        previous_request = identities.setdefault(
            identity,
            request.request_sha256,
        )
        if previous_request != request.request_sha256:
            raise ValueError(
                "distinct scientific requests produced one dataset identity"
            )
        _recheck_clean_state(initial_state)
        publication = publish_dataset(artifact_root, generated)
        _recheck_clean_state(initial_state)
        status = publication.status
        if status not in {"created", "reused"}:
            raise RuntimeError("unknown dataset publication status")
        del publication
        outcomes[request.request_sha256] = (
            identity,
            status,
        )
        del generated

    if (
        len(outcomes) != plan.expected_datasets
        or len(identities) != plan.expected_datasets
    ):
        raise ValueError(
            "resolved identity count does not match expected_datasets"
        )
    _recheck_clean_state(initial_state)
    final_catalog = _scan_dataset_catalog(artifact_root)
    _recheck_clean_state(initial_state)
    final_catalog = _stabilize_dataset_catalog(
        artifact_root=artifact_root,
        catalog=final_catalog,
    )
    _recheck_clean_state(initial_state)

    final_selected: dict[str, _CatalogCandidate] = {}
    for projection, request in expected_by_projection.items():
        matches = _catalog_matches(final_catalog, projection)
        if len(matches) > 1:
            raise _AmbiguousCurrentDatasetError(
                "multiple current datasets match one scientific request"
            )
        if not matches:
            raise ArtifactValidationError(
                "current dataset missing from final catalog"
            )
        final_selected[request.request_sha256] = matches[0]
    for request_sha256, candidate in final_selected.items():
        if outcomes[request_sha256][0] != candidate.identity:
            raise ArtifactValidationError(
                "resolved dataset changed before final catalog"
            )
    _recheck_clean_state(initial_state)

    records = [
        {
            "dataset_identity_sha256": outcomes[request.request_sha256][0],
            "request_sha256": request.request_sha256,
            "status": outcomes[request.request_sha256][1],
        }
        for request in plan.requests
    ]
    requested_projections = set(expected_by_projection)
    corrupt_identities = sorted(set(final_catalog.corrupt_identities))
    unmatched_identities = sorted(
        candidate.identity
        for candidate in final_catalog.candidates
        if candidate.projection not in requested_projections
    )
    if len(final_selected) != plan.expected_datasets:
        raise ArtifactValidationError(
            "final catalog count does not match expected_datasets"
        )
    campaign_id, protocol_sha256 = _campaign_report_fields(plan)
    _emit_report(
        {
            "schema_version": "dataset-build-report-v1",
            "mode": "build",
            "status": "complete",
            "campaign_id": campaign_id,
            "protocol_sha256": protocol_sha256,
            "expanded_cells": plan.expanded_cells,
            "expected_datasets": plan.expected_datasets,
            "observed_datasets": len(records),
            "publishable": True,
            "manifest_externally_anchored": False,
            "datasets": records,
            "stale_staging_count": len(
                final_catalog.discovery.stale_staging_directories
            ),
            "stale_staging": _diagnostic_inventory(
                final_catalog.discovery.stale_staging_directories
            ),
            "rejected_count": len(
                final_catalog.discovery.rejected_directories
            ),
            "rejected": _diagnostic_inventory(
                final_catalog.discovery.rejected_directories
            ),
            "unmatched_datasets": unmatched_identities,
            "corrupt_count": len(corrupt_identities),
            "corrupt_datasets": [
                {"dataset_identity_sha256": identity}
                for identity in corrupt_identities
            ],
            "errors": [],
        }
    )
    return 0


def _operational_error_code(error: Exception) -> str:
    if isinstance(error, _DirtyWorktreeError):
        return "dirty-worktree"
    if isinstance(error, _ProvenanceChangedError):
        return "provenance-changed"
    if isinstance(error, _AmbiguousCurrentDatasetError):
        return "ambiguous-current-dataset"
    if isinstance(error, ArtifactValidationError):
        return "artifact-validation-error"
    if isinstance(
        error,
        (FileNotFoundError, IsADirectoryError, NotADirectoryError),
    ):
        return "input-error"
    if isinstance(error, (TypeError, ValueError)):
        return "validation-error"
    return "runtime-error"


def _emit_operational_failure(*, mode: str, code: str) -> int:
    _emit_report(
        {
            "schema_version": "dataset-build-report-v1",
            "mode": mode,
            "status": "failed",
            "manifest_externally_anchored": False,
            "errors": [{"code": code}],
        }
    )
    sys.stderr.write(f"dataset build failed: {code}\n")
    return 1


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = _parser().parse_args(argv)
    except _UsageError as error:
        sys.stderr.write(f"build_datasets.py: error: {error}\n")
        return 2
    if arguments.dry_run and arguments.verify_only:
        sys.stderr.write(
            "--dry-run and --verify-only are mutually exclusive\n"
        )
        return 2
    mode = (
        "dry-run"
        if arguments.dry_run
        else "verify-only" if arguments.verify_only else "build"
    )
    try:
        if arguments.dry_run:
            return _dry_run(
                protocol_path=_from_repo(arguments.protocol),
                artifact_root=_from_repo(arguments.artifact_root),
            )
        if arguments.verify_only:
            return _verify_only(
                protocol_path=_from_repo(arguments.protocol),
                artifact_root=_from_repo(arguments.artifact_root),
            )
        return _build(
            protocol_path=_from_repo(arguments.protocol),
            artifact_root=_from_repo(arguments.artifact_root),
        )
    except Exception as error:
        return _emit_operational_failure(
            mode=mode,
            code=_operational_error_code(error),
        )


if __name__ == "__main__":
    raise SystemExit(main())
