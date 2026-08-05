"""Execute one versioned campaign through immutable run identities."""

from __future__ import annotations

import sys

if __name__ == "__main__" and not sys.flags.isolated:
    sys.stderr.write(
        "campaign execution refused: isolated-python-required\n"
    )
    raise SystemExit(2)

sys.dont_write_bytecode = True

import argparse
from collections.abc import Mapping
import hashlib
import os
from pathlib import Path
import re
from types import MappingProxyType

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    trusted = str(REPO_ROOT)
    trusted_key = os.path.normcase(trusted)
    retained = []
    for candidate in sys.path:
        try:
            key = os.path.normcase(str(Path(candidate or os.curdir).resolve()))
        except OSError:
            retained.append(candidate)
            continue
        if key != trusted_key:
            retained.append(candidate)
    sys.path[:] = [trusted, *retained]

from gsdiff.data.artifacts import (
    discover_dataset_directories,
    verify_canonical_dataset_directory_discovery,
    verify_dataset_directory,
)
from gsdiff.data._artifact_dataset import blind_acquisition_spec
from gsdiff.experiments.identity import (
    _authoritative_runtime_projection,
    _authoritative_python_executable_evidence,
    build_run_identity,
    canonical_json_bytes,
    collect_runtime_metadata,
    git_state,
    resolved_config_sha256,
    verify_environment_requirements,
)
from gsdiff.experiments.methods import (
    MethodResolutionRequest,
    derive_algorithm_seed,
    native_iteration_contract_v1,
    resolve_method_semantics,
)
from gsdiff.experiments.execution import _materialization_identity_documents
from gsdiff.experiments.child_outputs import build_method_info_contract_v1
from gsdiff.experiments.dataset_binding import build_dataset_input_contract
from gsdiff.experiments.protocol import expand_cells, load_protocol
from gsdiff.experiments.runner import (
    RunOutcome,
    RunExecutionPlan,
    RunRequest,
    _identity_asset_mapping,
    _identity_bound_config,
    _preflight_disk_space,
    _preflight_runtime,
    _validate_authoritative_request,
    run_request,
)
from gsdiff.experiments.source_snapshot import (
    materialize_source_snapshot as _materialize_source_snapshot,
    selected_source_evidence as _selected_source_evidence,
    verify_source_snapshot as _verify_source_snapshot,
)
from scripts.experiments.build_datasets import (
    _semantic_projection_from_manifest,
    _semantic_projection_from_request,
    plan_campaign_datasets,
)


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

_RUNTIME_DEVICE = re.compile(r"(?:cpu|cuda:(?:0|[1-9][0-9]*))\Z", re.ASCII)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_PHASE_ID = re.compile(
    r"[a-z0-9]+(?:-[a-z0-9]+)*-v[1-9][0-9]*\Z",
    re.ASCII,
)
_UNPARTITIONED_CAMPAIGN_IDS = frozenset(
    {"pilot-v1", "supplement-grid-v1", "ood-v1", "failure-v1"}
)
_PRIMARY_PHASE_IDS = frozenset(
    {"primary-selection-v1", "primary-confirmatory-v1"}
)
_ABLATION_PHASE_IDS = frozenset(
    {"selection-decision-v1", "selection-replay-v1", "selection-stress-v1"}
)
_DIRECT_EXECUTION_PHASE_IDS = frozenset({"pilot-v1", "ood-v1", "failure-v1"})


def _canonical_runtime_device(value: str) -> str:
    if _RUNTIME_DEVICE.fullmatch(value) is None:
        raise argparse.ArgumentTypeError(
            "device must be canonical cpu or cuda:N"
        )
    return value


def _canonical_phase_id(value: str) -> str:
    if _PHASE_ID.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("phase must be canonical ...-vN")
    return value


def _phase_identity_base_config(
    *,
    phase_id: str,
    acquisition_config_id: str,
    method_config_sha256: str,
) -> dict[str, str]:
    _canonical_phase_id(phase_id)
    if type(acquisition_config_id) is not str or not acquisition_config_id:
        raise ValueError("acquisition config ID must be a nonempty string")
    if (
        type(method_config_sha256) is not str
        or _SHA256.fullmatch(method_config_sha256) is None
    ):
        raise ValueError("method config digest must be a lowercase SHA-256")
    return {
        "phase_id": phase_id,
        "acquisition_config_id": acquisition_config_id,
        "method_config_sha256": method_config_sha256,
    }


def _preflight_requested_device(requested_runtime_device: str) -> int | None:
    device = _canonical_runtime_device(requested_runtime_device)
    if device == "cpu":
        return None
    cuda = __import__("torch").cuda
    if cuda.is_available() is not True:
        raise ValueError("requested CUDA runtime is unavailable")
    device_count = cuda.device_count()
    if type(device_count) is not int or device_count < 0:
        raise ValueError("CUDA device count is invalid")
    index = int(device.partition(":")[2])
    if index >= device_count:
        raise ValueError("requested CUDA device index is out of range")
    return index


def _runtime_manifest(
    runtime: dict[str, object],
    requested_runtime_device: str,
) -> dict[str, str]:
    device = _canonical_runtime_device(requested_runtime_device)
    gpu_name = ""
    index = _preflight_requested_device(device)
    if index is not None:
        gpu_name = str(__import__("torch").cuda.get_device_name(index))
    return {
        "python": str(runtime["python_version"]),
        "pytorch": str(runtime["torch_version"]),
        "cuda": (
            "" if runtime["cuda_version"] is None else str(runtime["cuda_version"])
        ),
        "gpu": gpu_name,
        "os": str(runtime["os"]),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run_campaign.py")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--phase", type=_canonical_phase_id, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--device", type=_canonical_runtime_device, required=True)
    parser.add_argument("--checkpoint", action="append", default=[])
    parser.add_argument("--minimum-free-bytes", type=int, default=1_073_741_824)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        arguments = _parser().parse_args(argv)
        campaign = load_protocol(arguments.protocol)
        _require_phase_protocol_match(arguments.phase, campaign)
        if campaign.get("execution_ready") is not True:
            print(
                "campaign execution refused: execution-not-ready",
                file=sys.stderr,
            )
            return 1
        _require_versioned_budget_contract(campaign)
        _require_execution_device_contract(
            arguments.phase,
            campaign,
            arguments.device,
        )
        return _run_ready_campaign(arguments, campaign)
    except (OSError, TypeError, ValueError) as error:
        print(
            f"campaign execution refused: {type(error).__name__}",
            file=sys.stderr,
        )
        return 1


def _require_phase_protocol_match(
    phase_id: str,
    protocol: dict[str, object],
) -> None:
    document_kind = protocol.get("document_kind")
    if document_kind == "campaign":
        campaign_id = protocol.get("campaign_id")
        if campaign_id in _UNPARTITIONED_CAMPAIGN_IDS:
            allowed_phases = frozenset({campaign_id})
        elif campaign_id == "primary-v1":
            allowed_phases = _PRIMARY_PHASE_IDS
        else:
            raise ValueError("campaign ID is not phase-enabled")
    elif document_kind == "ablation":
        allowed_phases = _ABLATION_PHASE_IDS
    else:
        raise ValueError("protocol document kind is not phase-enabled")
    if phase_id not in allowed_phases:
        raise ValueError("phase does not match the protocol")


def _require_materialized_phase_execution(phase_id: str) -> None:
    if phase_id not in _DIRECT_EXECUTION_PHASE_IDS:
        raise ValueError(
            "phase execution refused: exact phase materialization is not implemented"
        )


def _require_versioned_budget_contract(campaign: dict[str, object]) -> None:
    if campaign.get("document_kind") != "campaign":
        raise ValueError(
            "native-budget resolution for this protocol kind is not implemented"
        )
    matrix = campaign.get("matrix")
    acquisition_configs = campaign.get("acquisition_configs")
    budgets = campaign.get("method_budgets")
    if not isinstance(matrix, Mapping):
        raise ValueError("campaign matrix must be a mapping")
    if not isinstance(acquisition_configs, Mapping) or not acquisition_configs:
        raise ValueError("campaign acquisition configs must be a nonempty mapping")
    methods = matrix.get("methods")
    method_config_ids = matrix.get("method_config_ids")
    if not isinstance(methods, list) or not all(type(item) is str for item in methods):
        raise TypeError("campaign methods must be an exact string list")
    if len(methods) != len(set(methods)):
        raise ValueError("campaign methods must be unique")
    if not isinstance(method_config_ids, Mapping) or set(method_config_ids) != set(methods):
        raise ValueError("campaign method config IDs must match methods")
    if not isinstance(budgets, Mapping) or set(budgets) != set(methods):
        raise ValueError("campaign method budgets must match methods exactly")

    registry_path = REPO_ROOT / "configs" / "protocols" / "methods-v1.yaml"
    registry = load_protocol(registry_path)
    execution_profile = campaign.get("execution_profile")
    if type(execution_profile) is not str:
        raise TypeError("campaign execution profile must be a string")
    for method_id in methods:
        declared_budget = budgets[method_id]
        if type(declared_budget) is not int or declared_budget <= 0:
            raise TypeError("campaign method budget must be an exact positive integer")
        resolved_budgets: set[int] = set()
        for acquisition in acquisition_configs.values():
            metadata = _budget_measurements_metadata(acquisition)
            base_config = _declared_base_config(
                registry,
                method_id=method_id,
                execution_profile=execution_profile,
            )
            resolved = resolve_method_semantics(
                method_id,
                method_config_id=method_config_ids[method_id],
                base_config=base_config,
                measurements_metadata=metadata,
                execution_profile=execution_profile,
                registry_path=registry_path,
            )
            native = native_iteration_contract_v1(resolved)
            resolved_budgets.add(native["budget"])
        if resolved_budgets != {declared_budget}:
            raise ValueError(
                f"method budget disagrees with resolved native semantics: {method_id}"
            )


def _budget_measurements_metadata(acquisition: object) -> dict[str, int]:
    if not isinstance(acquisition, Mapping):
        raise TypeError("acquisition config must be a mapping")
    image_size = acquisition.get("image_size")
    if (
        not isinstance(image_size, list)
        or len(image_size) != 2
        or any(type(value) is not int or value <= 0 for value in image_size)
    ):
        raise ValueError("acquisition image size must contain two positive integers")
    fields = {
        "T": acquisition.get("num_frames"),
        "K": acquisition.get("train_measurements"),
        "holdout_K": acquisition.get("holdout_measurements"),
    }
    if any(type(value) is not int or value <= 0 for value in fields.values()):
        raise ValueError("acquisition dimensions must be positive integers")
    return {
        "H": image_size[0],
        "W": image_size[1],
        "T": fields["T"],
        "K": fields["K"],
        "holdout_K": fields["holdout_K"],
    }


def _require_execution_device_contract(
    phase_id: str,
    campaign: dict[str, object],
    requested_runtime_device: str,
) -> None:
    """Keep the controller smoke profile on its declared CPU-only lane."""
    if phase_id != "pilot-v1":
        return
    if campaign.get("execution_profile") != "pilot-smoke-v1":
        raise ValueError("pilot phase requires the locked smoke profile")
    if requested_runtime_device != "cpu":
        raise ValueError("pilot smoke execution is CPU-only")


def _require_campaign_method_policy(
    method,
    *,
    phase_id: str,
    campaign: Mapping[str, object],
    requested_runtime_device: str,
) -> None:
    if method.promotion_eligible:
        return
    exact_cpu_pilot_smoke = (
        phase_id == "pilot-v1"
        and campaign.get("campaign_id") == "pilot-v1"
        and campaign.get("execution_profile") == "pilot-smoke-v1"
        and requested_runtime_device == "cpu"
        and method.requested_method_config_id == "default"
        and method.method_config_id == "smoke-default-v1"
        and method.execution_profile == "controller-cpu-smoke-v1"
        and method.publication_eligible is False
        and method.selection_eligible is False
        and method.promotion_eligible is False
        and method.convergence_status == "smoke-only/not-convergence-assessed"
    )
    if not exact_cpu_pilot_smoke:
        raise ValueError("campaign contains a non-promotable method")


def _run_ready_campaign(
    arguments: argparse.Namespace,
    campaign: dict[str, object],
) -> int:
    _require_materialized_phase_execution(arguments.phase)
    if arguments.minimum_free_bytes < 0:
        raise ValueError("minimum free bytes must be nonnegative")
    _preflight_requested_device(arguments.device)
    code = git_state(REPO_ROOT, _SOURCE_ROOTS)
    if code["dirty"] is not False:
        raise ValueError("ready campaigns require a clean source tree")
    source_snapshot = _materialize_source_snapshot(
        REPO_ROOT,
        arguments.artifact_root,
        code["commit"],
        _SOURCE_ROOTS,
    )
    snapshot_campaign = load_protocol(
        _snapshot_protocol_path(source_snapshot.root, arguments.protocol)
    )
    if canonical_json_bytes(snapshot_campaign) != canonical_json_bytes(campaign):
        raise ValueError("campaign changed relative to claimed source snapshot")
    runtime_manifest, runtime_hashes = _authoritative_runtime_projection(
        source_snapshot.root / "requirements-lock.txt",
        source_snapshot.root / "docs/reproducibility/environment-lock.json",
        arguments.device,
    )
    python_executable, python_executable_sha256, _python_signature = (
        _authoritative_python_executable_evidence()
    )
    expected_source_inventory, expected_source_snapshot_sha256 = (
        _selected_source_evidence(source_snapshot)
    )
    checkpoints = _checkpoint_locators(arguments.checkpoint)
    dataset_plan = plan_campaign_datasets(
        repo_root=source_snapshot.root,
        protocol_path=_snapshot_protocol_path(
            source_snapshot.root,
            arguments.protocol,
        ),
        runtime=runtime_hashes,
        generator_commit=code["commit"],
    )
    datasets = _dataset_catalog(
        arguments.artifact_root,
        expected_requests=dataset_plan.requests,
    )
    registry_path = (
        source_snapshot.root / "configs/protocols/methods-v1.yaml"
    )
    registry = load_protocol(registry_path)
    cells = expand_cells(campaign)
    requests: list[RunRequest] = []
    used_checkpoints: set[str] = set()
    for cell in cells:
        dataset_request = _planned_dataset_request_for_cell(
            cell,
            campaign,
            dataset_plan.requests,
        )
        verified = datasets.get(dataset_request.request_sha256)
        if verified is None:
            raise ValueError("campaign dataset is missing")
        acquisition = verified.acquisition
        measurement_metadata = {
            "H": acquisition.H,
            "W": acquisition.W,
            "T": acquisition.T,
            "K": acquisition.K,
            "holdout_K": acquisition.holdout_K,
        }
        base_config = _declared_base_config(
            registry,
            method_id=cell.method,
            execution_profile=campaign["execution_profile"],
        )
        resolution_request = MethodResolutionRequest(
            requested_method_id=cell.method,
            requested_method_config_id=cell.method_config_id,
            base_config=base_config,
            measurements_metadata=measurement_metadata,
            requested_execution_profile=campaign["execution_profile"],
        )
        method = resolve_method_semantics(
            cell.method,
            method_config_id=cell.method_config_id,
            base_config=base_config,
            measurements_metadata=measurement_metadata,
            execution_profile=campaign["execution_profile"],
            registry_path=registry_path,
        )
        if not method.execution_ready or method.execution_blockers:
            raise ValueError("campaign contains an unresolved method")
        _require_campaign_method_policy(
            method,
            phase_id=arguments.phase,
            campaign=campaign,
            requested_runtime_device=arguments.device,
        )
        checkpoint_store: dict[str, Path] = {}
        checkpoint_hashes: dict[str, str] = {}
        for requirement in method.checkpoint_requirements:
            path = checkpoints.get(requirement.logical_id)
            if path is None:
                raise ValueError("required checkpoint locator is missing")
            checkpoint_store[requirement.logical_id] = path
            checkpoint_hashes[requirement.logical_id] = requirement.sha256
            used_checkpoints.add(requirement.logical_id)
        dataset_target = verified.manifest["resolved_generator_config"]["target"]
        assets = _identity_asset_mapping(dataset_target)
        algorithm_seed = derive_algorithm_seed(
            cell_seed=cell.seed,
            dataset_identity_sha256=verified.dataset_identity_sha256,
            method_id=method.method_id,
            method_config_sha256=method.method_config_sha256,
        )
        _materialized_config, materialization_logical = (
            _materialization_identity_documents(
                method=method,
                dataset_identity_sha256=verified.dataset_identity_sha256,
                measurements_file_sha256=(
                    verified.payload_evidence["measurements.npz"].sha256
                ),
                expected_acquisition_spec=blind_acquisition_spec(acquisition),
                algorithm_seed=algorithm_seed,
                source_inventory=[
                    dict(item) for item in expected_source_inventory
                ],
                requested_runtime_device=arguments.device,
            )
        )
        materialization_logical_sha256 = hashlib.sha256(
            canonical_json_bytes(materialization_logical)
        ).hexdigest()
        config_resolved = _identity_bound_config(
            _phase_identity_base_config(
                phase_id=arguments.phase,
                acquisition_config_id=cell.acquisition_config_id,
                method_config_sha256=method.method_config_sha256,
            ),
            requested_runtime_device=arguments.device,
            source_snapshot_sha256=source_snapshot.snapshot_sha256,
            source_projection_sha256=expected_source_snapshot_sha256,
            compute_cap=method.semantic_config["compute_cap"],
            materialization_logical_sha256=materialization_logical_sha256,
            method_info_contract=build_method_info_contract_v1(
                method,
                blind_acquisition_spec(acquisition),
            ),
            dataset_input_contract=build_dataset_input_contract(verified),
            runtime_contract=runtime_manifest,
            python_executable_sha256=python_executable_sha256,
        )
        identity = build_run_identity(
            execution_class="blind_method_child",
            scientific_contract_id=cell.scientific_contract_id,
            scientific_contract_sha256=cell.scientific_contract_sha256,
            method_id=method.method_id,
            target_id=cell.target,
            motion_id=cell.motion,
            seed=cell.seed,
            config_sha256=resolved_config_sha256(config_resolved),
            dataset_identity_sha256=verified.dataset_identity_sha256,
            assets_sha256=assets,
            checkpoints_sha256=checkpoint_hashes,
            code_commit=code["commit"],
            dirty_worktree=False,
            source_tree_hash=None,
            dependencies_sha256=runtime_hashes["dependencies_sha256"],
            environment_lock_sha256=runtime_hashes["environment_lock_sha256"],
            metric_version=campaign["metric_version"],
        )
        plan = RunExecutionPlan(
            identity=identity,
            resolution_request=resolution_request,
            registry_path=registry_path,
            checkpoint_store=MappingProxyType(checkpoint_store),
            python_executable=python_executable,
            source_root=source_snapshot.root,
            requested_runtime_device=arguments.device,
            config_resolved=config_resolved,
            assets_sha256=MappingProxyType(assets),
            code_commit=code["commit"],
            dirty_worktree=False,
            source_tree_hash=None,
            dependencies_sha256=runtime_hashes["dependencies_sha256"],
            environment_lock_sha256=runtime_hashes["environment_lock_sha256"],
            metric_version=campaign["metric_version"],
            runtime_metadata=runtime_manifest,
            expected_dataset_manifest_sha256=verified.dataset_manifest_sha256,
            minimum_free_bytes=arguments.minimum_free_bytes,
            source_snapshot=source_snapshot,
            expected_source_inventory=expected_source_inventory,
            expected_source_snapshot_sha256=(
                expected_source_snapshot_sha256
            ),
        )
        requests.append(
            RunRequest(cell, verified.dataset_dir, method, identity, plan)
        )
    if set(checkpoints) != used_checkpoints:
        raise ValueError("unused checkpoint locator is not allowed")
    _campaign_preflight(requests, arguments.artifact_root)
    outcomes = [
        run_request(request, arguments.artifact_root) for request in requests
    ]
    return _emit_campaign_outcome(campaign["campaign_id"], outcomes)


def _emit_campaign_outcome(
    campaign_id: str,
    outcomes: list[RunOutcome],
) -> int:
    failed = sum(outcome.status == "failed" for outcome in outcomes)
    report = {
        "schema": "campaign-run-report-v1",
        "campaign_id": campaign_id,
        "requested": len(outcomes),
        "complete": sum(outcome.status == "complete" for outcome in outcomes),
        "cached": sum(outcome.status == "cached" for outcome in outcomes),
        "failed": failed,
        "status": "failed" if failed else "complete",
    }
    print(canonical_json_bytes(report).decode("utf-8"))
    return 1 if failed else 0


def _campaign_preflight(
    requests: list[RunRequest],
    artifact_root: Path,
) -> None:
    """Validate every requested run before allowing the first child."""
    logical_cells: set[tuple[object, ...]] = set()
    identities: set[str] = set()
    for request in requests:
        plan = request.execution_plan
        cell = request.cell
        logical_cell = (
            cell.scientific_contract_id,
            cell.scientific_contract_sha256,
            cell.target,
            cell.motion,
            cell.seed,
            cell.method,
            cell.method_config_id,
            cell.acquisition_config_id,
        )
        if logical_cell in logical_cells:
            raise ValueError("campaign contains a duplicate logical cell")
        logical_cells.add(logical_cell)
        identity_sha256 = request.identity.identity_sha256
        if identity_sha256 in identities:
            raise ValueError("campaign contains a duplicate run identity")
        identities.add(identity_sha256)
        _preflight_runtime(plan)
        _validate_authoritative_request(request, plan)

    thresholds = {
        request.execution_plan.minimum_free_bytes for request in requests
    }
    if len(thresholds) != 1:
        raise ValueError("campaign plans disagree on disk safety threshold")
    if thresholds:
        _preflight_disk_space(artifact_root, thresholds.pop())


def _checkpoint_locators(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for assignment in values:
        logical_id, separator, raw_path = assignment.partition("=")
        if not separator or not logical_id or logical_id in result:
            raise ValueError("checkpoint locators must be unique logical-id=path pairs")
        path = Path(raw_path).absolute()
        if not path.is_file():
            raise ValueError("checkpoint locator is not a file")
        result[logical_id] = path
    return result


def _snapshot_protocol_path(snapshot_root: Path, protocol: Path) -> Path:
    candidate = protocol.absolute()
    try:
        relative = candidate.relative_to(REPO_ROOT)
    except ValueError as error:
        raise ValueError("campaign protocol must be tracked inside the repository") from error
    path = snapshot_root.joinpath(*relative.parts)
    if not path.is_file():
        raise ValueError("campaign protocol is absent from claimed source snapshot")
    return path


def _dataset_catalog(artifact_root: Path, *, expected_requests=None):
    discovery = discover_dataset_directories(artifact_root)
    _validate_dataset_catalog_discovery(discovery)
    verified_by_projection: dict[bytes, list[object]] = {}
    verified_by_identity = {}
    for path in discovery.canonical_directories:
        verified = verify_dataset_directory(
            path,
            expected_dataset_identity_sha256=path.name,
        )
        if verified.dataset_identity_sha256 in verified_by_identity:
            raise ValueError("dataset catalog has duplicate identities")
        verified_by_identity[verified.dataset_identity_sha256] = verified
        projection = _semantic_projection_from_manifest(verified.manifest)
        verified_by_projection.setdefault(projection, []).append(verified)
    _postverify_dataset_catalog_discovery(discovery)
    if expected_requests is None:
        return verified_by_identity
    result = {}
    expected_projections: set[bytes] = set()
    for request in expected_requests:
        projection = _semantic_projection_from_request(request)
        if projection in expected_projections:
            raise ValueError("planned dataset semantic projection collision")
        expected_projections.add(projection)
        matches = verified_by_projection.get(projection, [])
        if len(matches) != 1:
            raise ValueError(
                "campaign dataset is missing or ambiguously matched"
            )
        verified = matches[0]
        verified = verify_dataset_directory(
            verified.dataset_dir,
            expected_dataset_identity_sha256=(
                verified.dataset_identity_sha256
            ),
            expected_dataset_manifest_sha256=(
                verified.dataset_manifest_sha256
            ),
        )
        result[request.request_sha256] = verified
    _postverify_dataset_catalog_discovery(discovery)
    return result


def _validate_dataset_catalog_discovery(discovery) -> None:
    if not discovery.datasets_dir_exists:
        raise ValueError("dataset artifact root is missing")
    if discovery.stale_staging_directories or discovery.rejected_directories:
        raise ValueError(
            "dataset catalog contains staging or rejected directories"
        )


def _postverify_dataset_catalog_discovery(discovery) -> None:
    additions = verify_canonical_dataset_directory_discovery(discovery)
    if additions:
        raise ValueError("dataset catalog changed during verification")
    refreshed = discover_dataset_directories(discovery.artifact_root)
    _validate_dataset_catalog_discovery(refreshed)
    if refreshed.canonical_directories != discovery.canonical_directories:
        raise ValueError("dataset catalog changed during verification")
    if verify_canonical_dataset_directory_discovery(refreshed):
        raise ValueError("dataset catalog changed during verification")


def _planned_dataset_request_for_cell(cell, campaign, requests):
    acquisition_configs = campaign["acquisition_configs"]
    expected_acquisition = acquisition_configs[cell.acquisition_config_id]
    matches = []
    for request in requests:
        semantic = request.semantic_content
        if (
            semantic["scientific_contract"]["id"]
            == cell.scientific_contract_id
            and semantic["scientific_contract"]["sha256"]
            == cell.scientific_contract_sha256
            and semantic["target"]["id"] == cell.target
            and semantic["motion"]["id"] == cell.motion
            and semantic["seed"] == cell.seed
            and semantic["acquisition_config"] == expected_acquisition
        ):
            matches.append(request)
    if len(matches) != 1:
        raise ValueError(
            "campaign cell does not resolve one exact planned dataset request"
        )
    return matches[0]


def _declared_base_config(
    registry: dict[str, object],
    *,
    method_id: str,
    execution_profile: str,
) -> dict[str, object]:
    aliases = registry["campaign_execution_profile_aliases"]
    profile_id = aliases.get(execution_profile, execution_profile)
    entry = next(item for item in registry["methods"] if item["id"] == method_id)
    profile = entry["profiles"][profile_id]
    semantics = profile["semantic_config"]
    scene = semantics.get("scene") if isinstance(semantics, dict) else None
    gaussian_count = scene.get("gaussian_count") if isinstance(scene, dict) else None
    return {} if gaussian_count is None else {"gaussian_count": gaussian_count}


if __name__ == "__main__":
    raise SystemExit(main())
