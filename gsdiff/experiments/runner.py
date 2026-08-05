"""Atomic content-addressed execution for strict experiment requests."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
import ctypes
from dataclasses import dataclass
from datetime import datetime, timezone
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import re
import secrets
import shutil
import stat
import subprocess
import threading
import time
from typing import Literal
import uuid

import numpy as np
import torch

from gsdiff.data.artifacts import (
    blind_acquisition_spec,
    discover_dataset_directories,
    verify_canonical_dataset_directory_discovery,
    verify_dataset_directory,
)
from gsdiff.data._artifact_persistence import (
    _delete_windows_owned_path,
    _promote_exact_directory_no_clobber,
)
from gsdiff.evaluation.metrics import evaluate_video_global_affine

from .audit import validate_audit_log
from .dataset_binding import (
    build_dataset_input_contract,
    dataset_measurement_record,
    validate_dataset_protocol_binding,
)
from .child_outputs import (
    build_method_info_contract_v1,
    load_reconstruction_v2,
    validate_method_child_outputs_v2,
)
from .execution import (
    _directory_identity,
    _materialization_identity_documents,
    _read_stable_regular_bytes,
    _reject_linked_ancestors,
    _resolved_regular_file,
    _verify_directory_identity,
    materialize_method_execution,
)
from .identity import (
    RunIdentity,
    _authoritative_runtime_projection,
    _authoritative_python_executable_evidence,
    build_run_identity,
    canonical_json_bytes,
    resolved_config_sha256,
    sha256_file,
)
from .manifest import (
    _RUNNER_ARTIFACT_CONTRACT,
    build_manifest,
    load_complete_manifest,
)
from ._owned_tree import cleanup_pinned_owned_tree as _cleanup_pinned_owned_tree
from .methods import (
    MethodResolutionRequest,
    ResolvedMethod,
    derive_algorithm_seed,
    resolve_method_semantics,
)
from .protocol import ExperimentCell, expand_cells, load_protocol
from .source_snapshot import (
    SourceSnapshot,
    _load_canonical_manifest as _load_source_snapshot_manifest,
    selected_source_evidence,
    verify_source_snapshot,
    verify_source_snapshot_against_git,
)


@dataclass(frozen=True)
class RunExecutionPlan:
    identity: RunIdentity
    resolution_request: MethodResolutionRequest
    registry_path: Path
    checkpoint_store: Mapping[str, Path]
    python_executable: Path
    source_root: Path
    requested_runtime_device: str
    config_resolved: Mapping[str, object]
    assets_sha256: Mapping[str, str]
    code_commit: str
    dirty_worktree: bool
    source_tree_hash: str | None
    dependencies_sha256: str
    environment_lock_sha256: str
    metric_version: str
    runtime_metadata: Mapping[str, object]
    expected_dataset_manifest_sha256: str
    minimum_free_bytes: int
    source_snapshot: SourceSnapshot | None = None
    expected_source_inventory: tuple[Mapping[str, str], ...] = ()
    expected_source_snapshot_sha256: str | None = None


@dataclass(frozen=True)
class RunRequest:
    cell: ExperimentCell
    dataset_dir: Path
    method: ResolvedMethod
    identity: RunIdentity
    execution_plan: RunExecutionPlan


@dataclass(frozen=True)
class RunOutcome:
    status: Literal["complete", "cached", "failed"]
    run_dir: Path | None
    diagnostic_dir: Path | None
    return_code: int

_SHA256 = re.compile(r"^[0-9a-f]{64}$", re.ASCII)
_RUNTIME_DEVICE = re.compile(
    r"(?:cpu|cuda:(?:0|[1-9][0-9]*))\Z",
    re.ASCII,
)
_MOVEFILE_WRITE_THROUGH = 0x00000008
_WINDOWS_ATOMIC_RENAME = os.name == "nt"
_VRAM_SAMPLING_INTERVAL_MS = 250
_TRUSTED_REPO_ROOT = Path(__file__).resolve().parents[2]
_TRUSTED_SOURCE_ROOTS = (
    Path("gsdiff"),
    Path("scripts"),
    Path("configs"),
    Path("schemas"),
    Path("assets"),
    Path("train.py"),
    Path("requirements-lock.txt"),
    Path("docs/reproducibility/environment-lock.json"),
)
_POWERSHELL_SAMPLER_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
$targetPid = [Int32]::Parse($args[0], [Globalization.CultureInfo]::InvariantCulture)
$intervalMs = [Int32]::Parse($args[1], [Globalization.CultureInfo]::InvariantCulture)
$prefix = 'pid_' + $targetPid.ToString([Globalization.CultureInfo]::InvariantCulture) + '_'
[Console]::Out.WriteLine('READY')
[Console]::Out.Flush()
while ($true) {
    $samples = (Get-Counter '\GPU Process Memory(*)\Dedicated Usage' -ErrorAction SilentlyContinue).CounterSamples
    foreach ($sample in $samples) {
        if ($sample.InstanceName.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
            $status = [Convert]::ToInt64($sample.Status)
            if ($status -ne 0 -and $status -ne 1) {
                continue
            }
            $raw = [Double]$sample.CookedValue
            if ([Double]::IsNaN($raw) -or [Double]::IsInfinity($raw) -or $raw -lt 0) {
                throw 'invalid dedicated GPU process memory sample'
            }
            $value = [Convert]::ToInt64($raw)
            if ($raw -ne [Double]$value) {
                throw 'non-integral dedicated GPU process memory sample'
            }
            [Console]::Out.WriteLine(("R`t{0}`t{1}`t{2}" -f $sample.InstanceName, $status, $value))
        }
    }
    [Console]::Out.WriteLine('E')
    [Console]::Out.Flush()
    Start-Sleep -Milliseconds $intervalMs
}
"""


if os.name == "nt":
    from ctypes import wintypes

    class _JobBasicLimitInformation(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _IoCounters(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class _JobExtendedLimitInformation(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _JobBasicLimitInformation),
            ("IoInfo", _IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    class _ThreadEntry32(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ThreadID", wintypes.DWORD),
            ("th32OwnerProcessID", wintypes.DWORD),
            ("tpBasePri", wintypes.LONG),
            ("tpDeltaPri", wintypes.LONG),
            ("dwFlags", wintypes.DWORD),
        ]


class _WindowsKillJob:
    """Non-inheritable Windows job whose final close kills its process tree."""

    def __init__(self, *, active_process_limit: int | None = None) -> None:
        if os.name != "nt":
            raise OSError("Windows Job Objects are unavailable")
        if active_process_limit is not None and (
            type(active_process_limit) is not int or active_process_limit <= 0
        ):
            raise ValueError("active process limit must be a positive integer")
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        create = kernel32.CreateJobObjectW
        create.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        create.restype = wintypes.HANDLE
        handle = create(None, None)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        self._kernel32 = kernel32
        self._handle = handle
        try:
            info = _JobExtendedLimitInformation()
            info.BasicLimitInformation.LimitFlags = 0x00002000
            if active_process_limit is not None:
                info.BasicLimitInformation.ActiveProcessLimit = (
                    active_process_limit
                )
                info.BasicLimitInformation.LimitFlags |= 0x00000008
            set_info = kernel32.SetInformationJobObject
            set_info.argtypes = [
                wintypes.HANDLE,
                ctypes.c_int,
                ctypes.c_void_p,
                wintypes.DWORD,
            ]
            set_info.restype = wintypes.BOOL
            if not set_info(
                handle,
                9,
                ctypes.byref(info),
                ctypes.sizeof(info),
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            set_inherit = kernel32.SetHandleInformation
            set_inherit.argtypes = [
                wintypes.HANDLE,
                wintypes.DWORD,
                wintypes.DWORD,
            ]
            set_inherit.restype = wintypes.BOOL
            if not set_inherit(handle, 0x00000001, 0):
                raise ctypes.WinError(ctypes.get_last_error())
        except BaseException:
            self.close()
            raise

    def assign(self, process_id: int) -> None:
        if self._handle is None:
            raise RuntimeError("job handle is closed")
        open_process = self._kernel32.OpenProcess
        open_process.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        open_process.restype = wintypes.HANDLE
        process_handle = open_process(0x0100 | 0x0001, False, process_id)
        if not process_handle:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            assign = self._kernel32.AssignProcessToJobObject
            assign.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
            assign.restype = wintypes.BOOL
            if not assign(self._handle, process_handle):
                raise ctypes.WinError(ctypes.get_last_error())
        finally:
            self._kernel32.CloseHandle(process_handle)

    def close(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle is not None:
            self._handle = None
            self._kernel32.CloseHandle(handle)

    def __enter__(self):
        return self

    def __exit__(self, *_exc_info) -> None:
        self.close()


def _spawn_suspended_in_job(
    argv: list[str] | tuple[str, ...],
    *,
    cwd: Path,
    env: dict[str, str],
    stdout,
    stderr,
    job: _WindowsKillJob,
    text: bool = False,
) -> subprocess.Popen:
    process = _create_suspended_in_job(
        argv,
        cwd=cwd,
        env=env,
        stdout=stdout,
        stderr=stderr,
        job=job,
        text=text,
    )
    try:
        _resume_suspended_process(process.pid)
    except BaseException:
        job.close()
        process.wait(timeout=5)
        raise
    return process


def _create_suspended_in_job(
    argv: list[str] | tuple[str, ...],
    *,
    cwd: Path,
    env: dict[str, str],
    stdout,
    stderr,
    job: _WindowsKillJob,
    text: bool = False,
) -> subprocess.Popen:
    if os.name != "nt":
        raise OSError("suspended Job Object launch is Windows-only")
    process = subprocess.Popen(
        argv,
        cwd=cwd,
        env=env,
        stdout=stdout,
        stderr=stderr,
        text=text,
        encoding="utf-8" if text else None,
        creationflags=0x00000004,
    )
    try:
        job.assign(process.pid)
    except BaseException:
        process.kill()
        process.wait(timeout=5)
        job.close()
        raise
    return process


def _resume_suspended_process(process_id: int) -> None:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    snapshot_fn = kernel32.CreateToolhelp32Snapshot
    snapshot_fn.argtypes = [wintypes.DWORD, wintypes.DWORD]
    snapshot_fn.restype = wintypes.HANDLE
    snapshot = snapshot_fn(0x00000004, 0)
    if snapshot == wintypes.HANDLE(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    resumed = 0
    entry = _ThreadEntry32()
    entry.dwSize = ctypes.sizeof(entry)
    first = kernel32.Thread32First
    following = kernel32.Thread32Next
    first.argtypes = following.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(_ThreadEntry32),
    ]
    first.restype = following.restype = wintypes.BOOL
    try:
        present = bool(first(snapshot, ctypes.byref(entry)))
        while present:
            if entry.th32OwnerProcessID == process_id:
                open_thread = kernel32.OpenThread
                open_thread.argtypes = [
                    wintypes.DWORD,
                    wintypes.BOOL,
                    wintypes.DWORD,
                ]
                open_thread.restype = wintypes.HANDLE
                thread_handle = open_thread(
                    0x0002,
                    False,
                    entry.th32ThreadID,
                )
                if not thread_handle:
                    raise ctypes.WinError(ctypes.get_last_error())
                try:
                    resume = kernel32.ResumeThread
                    resume.argtypes = [wintypes.HANDLE]
                    resume.restype = wintypes.DWORD
                    if resume(thread_handle) == 0xFFFFFFFF:
                        raise ctypes.WinError(ctypes.get_last_error())
                    resumed += 1
                finally:
                    kernel32.CloseHandle(thread_handle)
            present = bool(following(snapshot, ctypes.byref(entry)))
    finally:
        kernel32.CloseHandle(snapshot)
    if resumed != 1:
        raise RuntimeError("suspended child did not expose exactly one primary thread")


def _identity_bound_config(
    base_config: Mapping[str, object],
    *,
    requested_runtime_device: str,
    source_snapshot_sha256: str,
    source_projection_sha256: str,
    compute_cap: Mapping[str, object],
    materialization_logical_sha256: str,
    method_info_contract: Mapping[str, object],
    dataset_input_contract: Mapping[str, object],
    runtime_contract: Mapping[str, object],
    python_executable_sha256: str,
) -> dict[str, object]:
    if type(base_config) is not dict:
        raise TypeError("identity base config must be an exact dict")
    if "runner_execution" in base_config:
        raise ValueError("identity base config uses reserved runner_execution key")
    _canonical_runtime_device(requested_runtime_device)
    cap_document = _validated_compute_cap_document(compute_cap)
    for name, value in (
        ("source snapshot", source_snapshot_sha256),
        ("source projection", source_projection_sha256),
        ("materialization logical", materialization_logical_sha256),
    ):
        if type(value) is not str or _SHA256.fullmatch(value) is None:
            raise ValueError(f"{name} digest must be canonical SHA-256")
    canonical_json_bytes(dict(base_config))
    canonical_json_bytes(dict(method_info_contract))
    if type(dataset_input_contract) is not dict or set(dataset_input_contract) != {
        "dataset_manifest_sha256",
        "measurements_file_sha256",
        "evaluation_truth_file_sha256",
        "measurement",
    }:
        raise ValueError("dataset input contract has invalid fields")
    for name, value in dataset_input_contract.items():
        if name == "measurement":
            canonical_json_bytes(value)
            continue
        if type(value) is not str or _SHA256.fullmatch(value) is None:
            raise ValueError(f"dataset input contract {name} is invalid")
    if type(runtime_contract) is not dict or set(runtime_contract) != {
        "python",
        "pytorch",
        "cuda",
        "gpu",
        "os",
    } or any(type(value) is not str for value in runtime_contract.values()):
        raise ValueError("runtime identity contract is invalid")
    if (
        type(python_executable_sha256) is not str
        or _SHA256.fullmatch(python_executable_sha256) is None
    ):
        raise ValueError("Python executable identity digest is invalid")
    return {
        **dict(base_config),
        "runner_execution": {
            "schema": "runner-execution-identity-v1",
            "requested_runtime_device": requested_runtime_device,
            "source_snapshot_sha256": source_snapshot_sha256,
            "source_projection_sha256": source_projection_sha256,
            "compute_cap": cap_document,
            "materialization_logical_sha256": materialization_logical_sha256,
            "method_info_contract": dict(method_info_contract),
            "dataset_input_contract": dict(dataset_input_contract),
            "runtime_contract": dict(runtime_contract),
            "python_executable_sha256": python_executable_sha256,
        },
    }


def _canonical_runtime_device(value: object) -> tuple[str, int | None]:
    if type(value) is not str or _RUNTIME_DEVICE.fullmatch(value) is None:
        raise ValueError(
            "requested runtime device must be canonical cpu or cuda:N"
        )
    if value == "cpu":
        return value, None
    return value, int(value.partition(":")[2])


def _identity_asset_mapping(target: object) -> dict[str, str]:
    if type(target) is not dict:
        raise ValueError("dataset target must be an exact object")
    target_id = target.get("id")
    descriptor = target.get("descriptor")
    assets = target.get("assets_sha256")
    if (
        type(target_id) is not str
        or re.fullmatch(r"[a-z0-9][a-z0-9_-]*", target_id, re.ASCII) is None
        or type(descriptor) is not str
        or type(assets) is not dict
    ):
        raise ValueError("dataset target identity fields are invalid")
    for logical_id, digest in assets.items():
        if (
            type(logical_id) is not str
            or not logical_id
            or type(digest) is not str
            or _SHA256.fullmatch(digest) is None
        ):
            raise ValueError("dataset target assets are invalid")
    if descriptor.startswith("char:"):
        if set(assets) != {"descriptor", "font", "renderer"}:
            raise ValueError("glyph target assets are incomplete")
        return dict(assets)
    if len(assets) != 1:
        raise ValueError("file target must bind exactly one physical asset")
    return {target_id: next(iter(assets.values()))}


def _validated_compute_cap_document(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != {
        "wall_time_seconds",
        "peak_vram_bytes",
        "on_exceed",
    }:
        raise ValueError("compute cap must contain exactly the frozen fields")
    wall_time = value["wall_time_seconds"]
    peak_vram = value["peak_vram_bytes"]
    on_exceed = value["on_exceed"]
    if type(wall_time) is not int or wall_time <= 0:
        raise TypeError("compute cap wall time must be a positive integer")
    if type(peak_vram) is not int or peak_vram <= 0:
        raise TypeError("compute cap peak VRAM must be a positive integer")
    if on_exceed != "ineligible-retain-artifacts":
        raise ValueError("compute cap on-exceed policy is unsupported")
    return {
        "wall_time_seconds": wall_time,
        "peak_vram_bytes": peak_vram,
        "on_exceed": on_exceed,
    }


def reusable_run(
    artifact_root: Path,
    expected_identity: RunIdentity,
) -> Path | None:
    if not isinstance(artifact_root, Path):
        raise TypeError("artifact_root must be a Path")
    if type(expected_identity) is not RunIdentity:
        raise TypeError("expected_identity must be a RunIdentity")
    run_dir = artifact_root.absolute() / "runs" / expected_identity.identity_sha256
    if not run_dir.exists():
        return None
    manifest = load_complete_manifest(
        run_dir / "manifest.json",
        artifact_root=artifact_root,
        expected_identity_sha256=expected_identity.identity_sha256,
    )
    if manifest is None:
        raise ValueError("existing run identity directory is not complete")
    return run_dir


def run_request(request: RunRequest, artifact_root: Path) -> RunOutcome:
    if type(request) is not RunRequest:
        raise TypeError("request must be an exact RunRequest")
    if type(request.identity) is not RunIdentity:
        raise TypeError("request identity must be an exact RunIdentity")
    plan = request.execution_plan
    if type(plan) is not RunExecutionPlan:
        raise TypeError("request execution_plan must be an exact RunExecutionPlan")
    if plan.identity != request.identity:
        raise ValueError("execution plan identity does not match request identity")
    _validate_request_dataset_root(request, artifact_root)
    verified, canonical_method, compute_cap = _validate_authoritative_request(
        request,
        plan,
    )
    cached = reusable_run(artifact_root, request.identity)
    if cached is not None:
        return RunOutcome("cached", cached, None, 0)
    _preflight_execution(plan, artifact_root)
    runs_dir = artifact_root.absolute() / "runs"
    claims_dir = artifact_root.absolute() / ".claims"
    failed_dir = artifact_root.absolute() / "failed"
    _ensure_real_directory(artifact_root.absolute())
    _ensure_real_directory(runs_dir)
    _ensure_real_directory(claims_dir)
    _ensure_real_directory(failed_dir)
    with _claim_identity(claims_dir, request.identity) as claim:
        cached = reusable_run(artifact_root, request.identity)
        if cached is not None:
            return RunOutcome("cached", cached, None, 0)
        _preflight_execution(plan, artifact_root)
        _recover_stale_stages(
            runs_dir=runs_dir,
            failed_dir=failed_dir,
            claims_dir=claims_dir,
            identity=request.identity,
            current_claim=claim,
        )
        stage = runs_dir / (
            f"{request.identity.identity_sha256}.tmp-{uuid.uuid4()}"
        )
        stage.mkdir()
        stage_identity = _directory_identity(stage, noun="run staging directory")
        owner_path = stage / ".owner.json"
        _write_file_durable(
            owner_path,
            canonical_json_bytes(
                {
                    "schema": "run-stage-owner-v1",
                    "owner_token": claim["owner_token"],
                    "fence": claim["fence"],
                    "identity_sha256": request.identity.identity_sha256,
                }
            ),
        )
        return_code = 1
        try:
            outcome = _execute_and_promote(
                request=request,
                plan=plan,
                canonical_method=canonical_method,
                compute_cap=compute_cap,
                verified_dataset=verified,
                artifact_root=artifact_root.absolute(),
                runs_dir=runs_dir,
                stage=stage,
                stage_identity=stage_identity,
                claim=claim,
            )
            return outcome
        except BaseException as error:
            if isinstance(error, KeyboardInterrupt):
                raise
            if isinstance(error, _PostPromotionError):
                raise
            if _has_complete_manifest_marker(stage, request.identity):
                raise
            if isinstance(error, subprocess.CalledProcessError):
                return_code = error.returncode
            elif isinstance(error, _ChildFailure):
                return_code = error.return_code
            try:
                diagnostic = failed_dir / (
                    f"{request.identity.run_id}-{_utc_stamp()}-{uuid.uuid4().hex[:8]}"
                )
                if stage.exists():
                    _verify_stage_owner(stage, claim, request.identity)
                    if (stage / "child-work.json").exists():
                        _finalize_diagnostic_evidence(
                            stage=stage,
                            artifact_root=artifact_root.absolute(),
                            claim=claim,
                        )
                    _write_file_durable(
                        stage / "failure.json",
                        canonical_json_bytes(
                            {
                                "schema": "run-failure-v1",
                                "error_type": type(error).__name__,
                                "message": str(error),
                                "cause": (
                                    str(error.__cause__)
                                    if error.__cause__ is not None
                                    else None
                                ),
                                "return_code": return_code,
                            }
                        ),
                    )
                    _rename_no_clobber(stage, diagnostic)
                else:
                    diagnostic = None
            except BaseException as diagnostic_error:
                error.add_note(
                    "Best-effort diagnostic finalization failed without "
                    "replacing the primary error: "
                    f"{type(diagnostic_error).__name__}: {diagnostic_error}"
                )
                raise error
    return RunOutcome("failed", None, diagnostic, return_code)


def _validate_request_dataset_root(
    request: RunRequest,
    artifact_root: Path,
) -> None:
    if not isinstance(artifact_root, Path):
        raise TypeError("artifact_root must be a Path")
    dataset_identity = request.identity.payload()[
        "dataset_identity_sha256"
    ]
    root = artifact_root.absolute()
    expected = root / "datasets" / dataset_identity
    if request.dataset_dir.absolute() != expected:
        raise ValueError(
            "request dataset must be under the canonical run artifact root"
        )
    discovery = discover_dataset_directories(root)
    if expected not in discovery.canonical_directories:
        raise ValueError("request dataset is not a canonical published dataset")
    verify_canonical_dataset_directory_discovery(discovery)


@dataclass
class _ChildFailure(Exception):
    return_code: int


class _PostPromotionError(RuntimeError):
    """A final identity directory exists and must never enter failed staging."""


class _PostPromotionDurabilityError(_PostPromotionError):
    pass


class _PostPromotionIntegrityError(_PostPromotionError):
    pass


@dataclass(frozen=True)
class _ComputeCap:
    wall_time_seconds: int
    peak_vram_bytes: int
    on_exceed: str


@dataclass
class _ComputeCapFailure(Exception):
    reason: str
    evidence: Mapping[str, object]


def _compute_cap(method: ResolvedMethod) -> _ComputeCap:
    cap = method.semantic_config.get("compute_cap")
    expected = {"wall_time_seconds", "peak_vram_bytes", "on_exceed"}
    if not isinstance(cap, Mapping) or set(cap) != expected:
        raise ValueError("compute cap must contain exactly the frozen fields")
    wall_time = cap["wall_time_seconds"]
    peak_vram = cap["peak_vram_bytes"]
    on_exceed = cap["on_exceed"]
    if type(wall_time) is not int or wall_time <= 0:
        raise TypeError("compute cap wall time must be a positive integer")
    if type(peak_vram) is not int or peak_vram <= 0:
        raise TypeError("compute cap peak VRAM must be a positive integer")
    if on_exceed != "ineligible-retain-artifacts":
        raise ValueError("compute cap on-exceed policy is unsupported")
    return _ComputeCap(wall_time, peak_vram, on_exceed)


def _process_vram_from_counter_records(
    records: list[tuple[str, int, int]],
    process_id: int,
) -> int | None:
    expected = re.compile(rf"^pid_{process_id}_.+_phys_[0-9]+$", re.ASCII)
    total = 0
    matched = False
    for instance_name, status, value in records:
        if type(instance_name) is not str or re.fullmatch(
            r"pid_[0-9]+_.+_phys_[0-9]+", instance_name, re.ASCII
        ) is None:
            raise ValueError("VRAM sampler returned a malformed instance")
        if not expected.fullmatch(instance_name):
            continue
        if status not in {0, 1}:
            raise ValueError("VRAM sampler returned invalid counter status")
        if type(value) is not int or value < 0:
            raise ValueError("VRAM sampler returned invalid dedicated usage")
        total += value
        matched = True
    return total if matched else None


def _is_exact_cpu_pilot_smoke(
    method: ResolvedMethod,
    plan: RunExecutionPlan,
) -> bool:
    resolution = plan.resolution_request
    return (
        resolution.requested_method_config_id == "default"
        and method.requested_method_config_id == "default"
        and method.method_config_id == "smoke-default-v1"
        and resolution.requested_execution_profile == "pilot-smoke-v1"
        and method.execution_profile == "controller-cpu-smoke-v1"
        and plan.config_resolved.get("phase_id") == "pilot-v1"
        and plan.requested_runtime_device == "cpu"
        and method.publication_eligible is False
        and method.selection_eligible is False
        and method.promotion_eligible is False
        and method.convergence_status == "smoke-only/not-convergence-assessed"
    )


def _validate_authoritative_request(
    request: RunRequest,
    plan: RunExecutionPlan,
):
    _validate_source_plan(plan)
    _validate_authoritative_python_plan(plan)
    if not isinstance(request.cell, ExperimentCell):
        raise TypeError("request cell must be an ExperimentCell")
    if not isinstance(request.dataset_dir, Path):
        raise TypeError("request dataset_dir must be a Path")
    if type(request.method) is not ResolvedMethod:
        raise TypeError("request method must be a ResolvedMethod")
    if type(plan.resolution_request) is not MethodResolutionRequest:
        raise TypeError("execution plan resolution request has an invalid type")
    canonical_method = resolve_method_semantics(
        plan.resolution_request.requested_method_id,
        method_config_id=(
            plan.resolution_request.requested_method_config_id
        ),
        base_config=plan.resolution_request.base_config,
        measurements_metadata=(
            plan.resolution_request.measurements_metadata
        ),
        execution_profile=(
            plan.resolution_request.requested_execution_profile
        ),
        registry_path=plan.registry_path,
    )
    if canonical_method != request.method:
        raise ValueError("request method does not match authoritative resolution")
    exact_cpu_pilot_smoke = _is_exact_cpu_pilot_smoke(canonical_method, plan)
    if not (
        request.cell.method_config_id
        == plan.resolution_request.requested_method_config_id
        == canonical_method.requested_method_config_id
    ) or (
        canonical_method.method_config_id
        != canonical_method.requested_method_config_id
        and not exact_cpu_pilot_smoke
    ):
        raise ValueError(
            "experiment cell method_config_id disagrees with method resolution"
        )
    if not canonical_method.execution_ready or canonical_method.execution_blockers:
        raise ValueError("method execution is not ready")
    if not canonical_method.promotion_eligible and not exact_cpu_pilot_smoke:
        raise ValueError("method is not promotion eligible")
    compute_cap = _compute_cap(canonical_method)
    runner_config = plan.config_resolved.get("runner_execution")
    expected_cap = {
        "wall_time_seconds": compute_cap.wall_time_seconds,
        "peak_vram_bytes": compute_cap.peak_vram_bytes,
        "on_exceed": compute_cap.on_exceed,
    }
    if (
        type(runner_config) is not dict
        or runner_config.get("compute_cap") != expected_cap
    ):
        raise ValueError("identity config disagrees with authoritative compute cap")
    payload = request.identity.payload()
    cell = request.cell
    expected_cell = {
        "scientific_contract_id": cell.scientific_contract_id,
        "scientific_contract_sha256": cell.scientific_contract_sha256,
        "method_id": cell.method,
        "target_id": cell.target,
        "motion_id": cell.motion,
        "seed": cell.seed,
    }
    if any(payload[name] != value for name, value in expected_cell.items()):
        raise ValueError("request identity does not match experiment cell")
    if cell.method != canonical_method.method_id:
        raise ValueError("experiment cell method does not match resolved method")
    verified = verify_dataset_directory(
        request.dataset_dir,
        expected_dataset_identity_sha256=payload["dataset_identity_sha256"],
        expected_dataset_manifest_sha256=(
            plan.expected_dataset_manifest_sha256
        ),
    )
    manifest = verified.manifest
    _validate_cell_campaign_and_acquisition(
        request.cell,
        plan,
        manifest,
    )
    expected_dataset_input_contract = build_dataset_input_contract(verified)
    if runner_config.get("dataset_input_contract") != (
        expected_dataset_input_contract
    ):
        raise ValueError(
            "identity config disagrees with authoritative dataset inputs"
        )
    if plan.source_snapshot is None:
        raise ValueError("authoritative runtime requires a verified source snapshot")
    authoritative_runtime, runtime_hashes = _authoritative_runtime_projection(
        plan.source_snapshot.root / "requirements-lock.txt",
        plan.source_snapshot.root
        / "docs"
        / "reproducibility"
        / "environment-lock.json",
        plan.requested_runtime_device,
    )
    if (
        dict(plan.runtime_metadata) != authoritative_runtime
        or runner_config.get("runtime_contract") != authoritative_runtime
        or payload["dependencies_sha256"]
        != runtime_hashes["dependencies_sha256"]
        or payload["environment_lock_sha256"]
        != runtime_hashes["environment_lock_sha256"]
    ):
        raise ValueError(
            "identity config disagrees with live authoritative runtime metadata"
        )
    validate_dataset_protocol_binding(
        manifest,
        scientific_contract_id=cell.scientific_contract_id,
        scientific_contract_sha256=cell.scientific_contract_sha256,
        target_id=cell.target,
        motion_id=cell.motion,
        seed=cell.seed,
        assets_sha256=payload["assets_sha256"],
    )
    generator_config = manifest["resolved_generator_config"]
    expected_assets = _identity_asset_mapping(generator_config["target"])
    if expected_assets != dict(plan.assets_sha256):
        raise ValueError("execution plan assets do not match dataset manifest")
    acquisition = verified.acquisition
    algorithm_seed = derive_algorithm_seed(
        cell_seed=request.cell.seed,
        dataset_identity_sha256=verified.dataset_identity_sha256,
        method_id=canonical_method.method_id,
        method_config_sha256=canonical_method.method_config_sha256,
    )
    _predicted_config, predicted_logical = _materialization_identity_documents(
        method=canonical_method,
        dataset_identity_sha256=verified.dataset_identity_sha256,
        measurements_file_sha256=verified.payload_evidence["measurements.npz"].sha256,
        expected_acquisition_spec=blind_acquisition_spec(acquisition),
        algorithm_seed=algorithm_seed,
        source_inventory=[dict(item) for item in plan.expected_source_inventory],
        requested_runtime_device=plan.requested_runtime_device,
    )
    predicted_logical_sha256 = hashlib.sha256(
        canonical_json_bytes(predicted_logical)
    ).hexdigest()
    if runner_config.get("materialization_logical_sha256") != predicted_logical_sha256:
        raise ValueError("identity config disagrees with materialization logical record")
    expected_method_info_contract = build_method_info_contract_v1(
        canonical_method,
        blind_acquisition_spec(acquisition),
    )
    if canonical_json_bytes(
        runner_config.get("method_info_contract")
    ) != canonical_json_bytes(expected_method_info_contract):
        raise ValueError(
            "identity config disagrees with authoritative method info contract"
        )
    checkpoint_hashes: dict[str, str] = {}
    if set(plan.checkpoint_store) != {
        item.logical_id for item in canonical_method.checkpoint_requirements
    }:
        raise ValueError("execution plan checkpoint locator set is incomplete")
    for requirement in canonical_method.checkpoint_requirements:
        if requirement.provenance_status not in {"verified", "complete"}:
            raise ValueError("checkpoint provenance is unresolved")
        path = plan.checkpoint_store[requirement.logical_id]
        if not isinstance(path, Path):
            raise ValueError("checkpoint locator is not a path")
        checkpoint_bytes = _read_stable_regular_bytes(
            path,
            noun="checkpoint locator",
        )
        if hashlib.sha256(checkpoint_bytes).hexdigest() != requirement.sha256:
            raise ValueError("checkpoint bytes do not match registry")
        checkpoint_hashes[requirement.logical_id] = requirement.sha256
    rebuilt = build_run_identity(
        execution_class="blind_method_child",
        scientific_contract_id=cell.scientific_contract_id,
        scientific_contract_sha256=cell.scientific_contract_sha256,
        method_id=canonical_method.method_id,
        target_id=cell.target,
        motion_id=cell.motion,
        seed=cell.seed,
        config_sha256=resolved_config_sha256(plan.config_resolved),
        dataset_identity_sha256=verified.dataset_identity_sha256,
        assets_sha256=plan.assets_sha256,
        checkpoints_sha256=checkpoint_hashes,
        code_commit=plan.code_commit,
        dirty_worktree=plan.dirty_worktree,
        source_tree_hash=plan.source_tree_hash,
        dependencies_sha256=plan.dependencies_sha256,
        environment_lock_sha256=plan.environment_lock_sha256,
        metric_version=plan.metric_version,
    )
    if rebuilt != request.identity or rebuilt != plan.identity:
        raise ValueError("request identity does not match authoritative inputs")
    return verified, canonical_method, compute_cap


def _validate_cell_campaign_and_acquisition(
    cell: ExperimentCell,
    plan: RunExecutionPlan,
    dataset_manifest: Mapping[str, object],
) -> None:
    if plan.source_snapshot is None:
        raise ValueError("campaign binding requires a verified source snapshot")
    protocol_root = plan.source_snapshot.root / "configs" / "protocols"
    matches: list[Mapping[str, object]] = []
    for path in sorted(protocol_root.glob("*.yaml")):
        document = load_protocol(path)
        if (
            document.get("document_kind") == "campaign"
            and document.get("campaign_id") == cell.campaign_id
        ):
            matches.append(document)
    if len(matches) != 1:
        raise ValueError("experiment cell campaign_id is not uniquely authoritative")
    campaign = matches[0]
    if (
        plan.resolution_request.requested_execution_profile
        != campaign["execution_profile"]
    ):
        raise ValueError(
            "execution plan execution_profile disagrees with authoritative campaign"
        )
    if plan.metric_version != campaign["metric_version"]:
        raise ValueError(
            "execution plan metric_version disagrees with authoritative campaign"
        )
    if cell not in expand_cells(campaign):
        raise ValueError(
            "experiment cell is not an exact member of the authoritative campaign"
        )
    configs = campaign["acquisition_configs"]
    if type(configs) is not dict or cell.acquisition_config_id not in configs:
        raise ValueError("experiment cell acquisition_config_id is unknown")
    acquisition_config = configs[cell.acquisition_config_id]
    if type(acquisition_config) is not dict:
        raise ValueError("authoritative acquisition config is malformed")
    generator = dataset_manifest.get("resolved_generator_config")
    if type(generator) is not dict:
        raise ValueError("dataset generator config is malformed")
    dimensions = generator.get("dimensions")
    acquisition = generator.get("acquisition")
    if type(dimensions) is not dict or type(acquisition) is not dict:
        raise ValueError("dataset acquisition projection is malformed")
    image_size = acquisition_config.get("image_size")
    if type(image_size) is not list or len(image_size) != 2:
        raise ValueError("authoritative acquisition image_size is malformed")
    expected_dimensions = {
        "H": image_size[0],
        "W": image_size[1],
        "T": acquisition_config.get("num_frames"),
        "K": acquisition_config.get("train_measurements"),
        "holdout_K": acquisition_config.get("holdout_measurements"),
    }
    expected_acquisition = {
        name: acquisition_config.get(name)
        for name in (
            "pattern_family",
            "pattern_values",
            "pattern_order",
            "time_assignment",
            "holdout_pattern_family",
            "snr_db",
            "noise_calibration_id",
        )
    }
    if dimensions != expected_dimensions or acquisition != expected_acquisition:
        raise ValueError(
            "dataset acquisition disagrees with authoritative acquisition_config_id"
        )


def _preflight_execution(
    plan: RunExecutionPlan,
    artifact_root: Path,
) -> None:
    if type(plan.minimum_free_bytes) is not int or plan.minimum_free_bytes < 0:
        raise ValueError("minimum_free_bytes must be a nonnegative integer")
    _preflight_runtime(plan)
    _preflight_disk_space(artifact_root, plan.minimum_free_bytes)


def _preflight_runtime(plan: RunExecutionPlan) -> None:
    _requested_device, cuda_index = _canonical_runtime_device(
        plan.requested_runtime_device
    )
    if cuda_index is not None:
        if not torch.cuda.is_available():
            raise ValueError("CUDA preflight failed")
        if cuda_index >= torch.cuda.device_count():
            raise ValueError("CUDA preflight index is unavailable")
    _validate_authoritative_python_plan(plan)
    if not plan.source_root.is_dir():
        raise ValueError("source root is missing")
    _validate_source_plan(plan)
    if cuda_index is not None:
        _windows_powershell_path()


def _validate_source_plan(plan: RunExecutionPlan) -> None:
    if plan.source_snapshot is None:
        if plan.expected_source_inventory or plan.expected_source_snapshot_sha256 is not None:
            raise ValueError("source snapshot evidence is incomplete")
        return
    snapshot = verify_source_snapshot_against_git(
        plan.source_snapshot,
        trusted_repo_root=_TRUSTED_REPO_ROOT,
        source_roots=_TRUSTED_SOURCE_ROOTS,
    )
    if (
        snapshot.root != plan.source_root
        or snapshot.commit != plan.code_commit
        or snapshot.snapshot_sha256 != plan.source_snapshot.snapshot_sha256
    ):
        raise ValueError("source snapshot disagrees with execution provenance")
    inventory, digest = selected_source_evidence(snapshot)
    if (
        canonical_json_bytes([dict(item) for item in inventory])
        != canonical_json_bytes(
            [dict(item) for item in plan.expected_source_inventory]
        )
        or digest != plan.expected_source_snapshot_sha256
    ):
        raise ValueError("source snapshot execution projection changed")
    runner_config = plan.config_resolved.get("runner_execution")
    if type(runner_config) is not dict or runner_config != {
        "schema": "runner-execution-identity-v1",
        "requested_runtime_device": plan.requested_runtime_device,
        "source_snapshot_sha256": snapshot.snapshot_sha256,
        "source_projection_sha256": digest,
        "compute_cap": runner_config.get("compute_cap") if type(runner_config) is dict else None,
        "materialization_logical_sha256": (
            runner_config.get("materialization_logical_sha256")
            if type(runner_config) is dict
            else None
        ),
        "method_info_contract": (
            runner_config.get("method_info_contract")
            if type(runner_config) is dict
            else None
        ),
        "dataset_input_contract": (
            runner_config.get("dataset_input_contract")
            if type(runner_config) is dict
            else None
        ),
        "runtime_contract": (
            runner_config.get("runtime_contract")
            if type(runner_config) is dict
            else None
        ),
        "python_executable_sha256": (
            runner_config.get("python_executable_sha256")
            if type(runner_config) is dict
            else None
        ),
    }:
        raise ValueError("identity config disagrees with device or source evidence")
    _validated_compute_cap_document(runner_config["compute_cap"])


def _validate_authoritative_python_plan(
    plan: RunExecutionPlan,
) -> tuple[Path, str, tuple[int, ...]]:
    canonical, digest, signature = _authoritative_python_executable_evidence()
    requested = _resolved_regular_file(
        plan.python_executable,
        noun="child Python executable",
        require_single_link=False,
    )
    if os.path.normcase(str(requested)) != os.path.normcase(str(canonical)):
        raise ValueError("child Python executable is not sys.executable")
    runner_config = plan.config_resolved.get("runner_execution")
    if (
        type(runner_config) is not dict
        or runner_config.get("python_executable_sha256") != digest
    ):
        raise ValueError(
            "child Python executable disagrees with the runtime identity contract"
        )
    return canonical, digest, signature


def _windows_powershell_path() -> Path:
    if os.name != "nt":
        raise OSError("Windows PowerShell is unavailable")
    get_directory = ctypes.WinDLL(
        "kernel32",
        use_last_error=True,
    ).GetSystemWindowsDirectoryW
    get_directory.argtypes = [wintypes.LPWSTR, wintypes.UINT]
    get_directory.restype = wintypes.UINT
    buffer = ctypes.create_unicode_buffer(32768)
    length = get_directory(buffer, len(buffer))
    if length == 0 or length >= len(buffer):
        raise ctypes.WinError(ctypes.get_last_error())
    path = (
        Path(buffer.value)
        / "System32"
        / "WindowsPowerShell"
        / "v1.0"
        / "powershell.exe"
    )
    _read_stable_regular_hardlinked_bytes(
        path,
        noun="Windows GPU counter sampler executable",
    )
    return path


def _read_stable_regular_hardlinked_bytes(path: Path, *, noun: str) -> bytes:
    resolved = _resolved_regular_file(
        path,
        noun=noun,
        require_single_link=False,
    )
    before = os.lstat(resolved)
    try:
        with resolved.open("rb") as stream:
            opened = os.fstat(stream.fileno())
            if (
                opened.st_dev,
                opened.st_ino,
                opened.st_size,
                opened.st_mtime_ns,
                opened.st_nlink,
            ) != (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_nlink,
            ):
                raise ValueError(f"{noun} changed while being opened")
            data = stream.read()
            after_handle = os.fstat(stream.fileno())
    except OSError as error:
        raise ValueError(f"cannot read {noun}") from error
    after_path = os.lstat(resolved)
    path_signature = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
        before.st_nlink,
    )
    handle_signature = (
        opened.st_dev,
        opened.st_ino,
        opened.st_size,
        opened.st_mtime_ns,
        opened.st_ctime_ns,
        opened.st_nlink,
    )
    if handle_signature != (
        after_handle.st_dev,
        after_handle.st_ino,
        after_handle.st_size,
        after_handle.st_mtime_ns,
        after_handle.st_ctime_ns,
        after_handle.st_nlink,
    ) or path_signature != (
        after_path.st_dev,
        after_path.st_ino,
        after_path.st_size,
        after_path.st_mtime_ns,
        after_path.st_ctime_ns,
        after_path.st_nlink,
    ):
        raise ValueError(f"{noun} changed while being read")
    return data


def _preflight_disk_space(artifact_root: Path, minimum_free_bytes: int) -> None:
    probe = artifact_root.absolute()
    while not probe.exists():
        if probe.parent == probe:
            raise ValueError("artifact root has no existing filesystem ancestor")
        probe = probe.parent
    if shutil.disk_usage(probe).free < minimum_free_bytes:
        raise ValueError("disk preflight failed")


def _execute_and_promote(
    *,
    request: RunRequest,
    plan: RunExecutionPlan,
    canonical_method: ResolvedMethod,
    compute_cap: _ComputeCap,
    verified_dataset,
    artifact_root: Path,
    runs_dir: Path,
    stage: Path,
    stage_identity,
    claim: Mapping[str, object],
) -> RunOutcome:
    if plan.source_snapshot is None:
        raise ValueError("source snapshot evidence is required for promotion")
    source_snapshot_bytes = _read_stable_regular_bytes(
        plan.source_snapshot.root / "source-snapshot.json",
        noun="promoted source snapshot manifest",
    )
    promoted_source_manifest = _load_source_snapshot_manifest(
        source_snapshot_bytes
    )
    if (
        promoted_source_manifest["commit"] != plan.code_commit
        or promoted_source_manifest["snapshot_sha256"]
        != plan.source_snapshot.snapshot_sha256
    ):
        raise ValueError("promoted source snapshot evidence changed")
    acquisition = verified_dataset.acquisition
    measurement_evidence = verified_dataset.payload_evidence["measurements.npz"]
    algorithm_seed = derive_algorithm_seed(
        cell_seed=request.cell.seed,
        dataset_identity_sha256=verified_dataset.dataset_identity_sha256,
        method_id=canonical_method.method_id,
        method_config_sha256=canonical_method.method_config_sha256,
    )
    work_parent = artifact_root / "work"
    _ensure_real_directory(work_parent)
    _require_same_filesystem(work_parent, runs_dir)
    work_id = uuid.uuid4().hex
    work = work_parent / work_id
    work_pointer = {
        "schema": "run-child-work-v1",
        "work_id": work_id,
        "owner_token": claim["owner_token"],
        "fence": claim["fence"],
    }
    _write_file_durable(
        stage / "child-work.json",
        canonical_json_bytes(work_pointer),
    )
    materialized = materialize_method_execution(
        canonical_method,
        resolution_request=plan.resolution_request,
        registry_path=plan.registry_path,
        stage_root=work,
        measurements_source=verified_dataset.dataset_dir / "measurements.npz",
        measurements_file_sha256=measurement_evidence.sha256,
        dataset_identity_sha256=verified_dataset.dataset_identity_sha256,
        expected_acquisition_spec=blind_acquisition_spec(acquisition),
        algorithm_seed=algorithm_seed,
        checkpoint_store=plan.checkpoint_store,
        python_executable=plan.python_executable,
        source_root=plan.source_root,
        requested_runtime_device=plan.requested_runtime_device,
    )
    logical_materialization = materialized.materialization_record["logical"]
    if not isinstance(logical_materialization, Mapping):
        raise ValueError("materialized logical record is malformed")
    if plan.source_snapshot is not None:
        observed_inventory = logical_materialization.get("source_inventory")
        observed_digest = logical_materialization.get("source_snapshot_sha256")
        if (
            canonical_json_bytes(observed_inventory)
            != canonical_json_bytes(
                [dict(item) for item in plan.expected_source_inventory]
            )
            or observed_digest != plan.expected_source_snapshot_sha256
        ):
            raise ValueError(
                "materialized source inventory disagrees with claimed snapshot"
            )
    logical_materialization_bytes = canonical_json_bytes(
        dict(logical_materialization)
    )
    runner_config = plan.config_resolved["runner_execution"]
    if hashlib.sha256(logical_materialization_bytes).hexdigest() != (
        runner_config["materialization_logical_sha256"]
    ):
        raise ValueError("materialized logical record disagrees with run identity")
    work_identity = _directory_identity(work, noun="child workspace")
    work_pointer = {
        **work_pointer,
        "work_device": work_identity.device,
        "work_inode": work_identity.inode,
    }
    _write_file_durable(
        work / ".runner-owner.json",
        canonical_json_bytes(work_pointer),
    )
    with (stage / "child-work.json").open("wb") as stream:
        stream.write(canonical_json_bytes(work_pointer))
        stream.flush()
        os.fsync(stream.fileno())
    started = _utc_now()
    config_bytes = canonical_json_bytes(dict(plan.config_resolved))
    _write_file_durable(stage / "resolved-config.json", config_bytes)
    running_lifecycle = {
        "schema": "run-lifecycle-v1",
        "state": "running",
        "identity_sha256": request.identity.identity_sha256,
        "owner_token": claim["owner_token"],
        "fence": claim["fence"],
    }
    _write_file_durable(
        stage / "lifecycle.json",
        canonical_json_bytes(running_lifecycle),
    )
    running_manifest = build_manifest(
        status="running",
        identity=request.identity,
        config_resolved=dict(plan.config_resolved),
        inputs={
            "measurements_file_sha256": measurement_evidence.sha256,
            "evaluation_truth_file_sha256": verified_dataset.payload_evidence[
                "evaluation-truth.npz"
            ].sha256,
            "dataset_manifest_sha256": verified_dataset.dataset_manifest_sha256,
        },
        runtime=dict(plan.runtime_metadata),
        execution={
            "command": list(logical_materialization["command_template"]),
            "started_at_utc": started,
            "ended_at_utc": None,
            "return_code": None,
            "runtime_seconds": 0.0,
            "peak_vram_bytes": 0,
        },
        measurement=dataset_measurement_record(verified_dataset.manifest),
        metrics=None,
        artifacts=[],
    )
    _write_file_durable(
        stage / "manifest.json",
        canonical_json_bytes(running_manifest),
    )
    _fsync_tree(stage)
    resource_evidence_path = work / "parent/logs/resource-sampling.json"
    running_resource_evidence_path = (
        work / "parent/logs/resource-sampling.running.json"
    )
    _write_file_durable(
        running_resource_evidence_path,
        canonical_json_bytes(
            _resource_evidence(
                status="running",
                compute_cap=compute_cap,
                runtime_seconds=0.0,
                sample_count=0,
                peak_vram_bytes=0,
                requested_runtime_device=plan.requested_runtime_device,
            )
        ),
    )
    with materialized.stdout_path.open("wb") as stdout_stream, materialized.stderr_path.open("wb") as stderr_stream:
        try:
            _validate_authoritative_python_plan(plan)
            return_code, resource_evidence = _run_method_child(
                argv=materialized.argv,
                cwd=materialized.cwd,
                env=dict(materialized.env),
                stdout_stream=stdout_stream,
                stderr_stream=stderr_stream,
                requested_runtime_device=plan.requested_runtime_device,
                compute_cap=compute_cap,
            )
            _validate_authoritative_python_plan(plan)
        except _ComputeCapFailure as error:
            _write_file_durable(
                resource_evidence_path,
                canonical_json_bytes(dict(error.evidence)),
            )
            running_resource_evidence_path.unlink()
            raise
        _write_file_durable(
            resource_evidence_path,
            canonical_json_bytes(resource_evidence),
        )
        running_resource_evidence_path.unlink()
        stdout_stream.flush()
        stderr_stream.flush()
        os.fsync(stdout_stream.fileno())
        os.fsync(stderr_stream.fileno())
    ended = _utc_now()
    runtime_seconds = resource_evidence["runtime_seconds"]
    if return_code != 0:
        raise _ChildFailure(return_code)
    audit = validate_audit_log(
        materialized.audit_log_path,
        expected_policy_sha256=materialized.audit_policy_sha256,
    )
    child_hashes = validate_method_child_outputs_v2(
        materialized.child_output_dir,
        expected_method=canonical_method,
        expected_acquisition=acquisition,
        expected_dataset_identity_sha256=verified_dataset.dataset_identity_sha256,
        expected_measurements_file_sha256=measurement_evidence.sha256,
        expected_algorithm_seed=algorithm_seed,
    )
    child_payloads: dict[str, bytes] = {}
    for name in ("reconstruction.npz", "method-info.json"):
        payload = _read_stable_regular_bytes(
            materialized.child_output_dir / name,
            noun=f"validated child output {name}",
        )
        if hashlib.sha256(payload).hexdigest() != child_hashes[name]:
            raise ValueError(
                f"validated child output bytes changed after validation: {name}"
            )
        child_payloads[name] = payload
    audit_bytes = _read_stable_regular_bytes(
        materialized.audit_log_path,
        noun="validated audit log",
    )
    if hashlib.sha256(audit_bytes).hexdigest() != audit["audit_log_sha256"]:
        raise ValueError("validated audit bytes changed after validation")
    evidence_payloads = {
        "outputs/stdout.log": _read_stable_regular_bytes(
            materialized.stdout_path,
            noun="method stdout",
        ),
        "outputs/stderr.log": _read_stable_regular_bytes(
            materialized.stderr_path,
            noun="method stderr",
        ),
        "evidence/audit.jsonl": audit_bytes,
        "evidence/audit-validation.json": canonical_json_bytes(dict(audit)),
        "evidence/resource-sampling.json": _read_stable_regular_bytes(
            resource_evidence_path,
            noun="resource sampling evidence",
        ),
        "evidence/materialization-logical.json": logical_materialization_bytes,
        "evidence/source-snapshot.json": source_snapshot_bytes,
    }
    outputs = stage / "outputs"
    evidence_dir = stage / "evidence"
    outputs.mkdir()
    evidence_dir.mkdir()
    _write_bound_snapshot(
        outputs / "reconstruction.npz",
        child_payloads["reconstruction.npz"],
        expected_sha256=child_hashes["reconstruction.npz"],
        noun="validated reconstruction",
    )
    _write_bound_snapshot(
        outputs / "method-info.json",
        child_payloads["method-info.json"],
        expected_sha256=child_hashes["method-info.json"],
        noun="validated method info",
    )
    for relative, payload in evidence_payloads.items():
        _write_bound_snapshot(
            stage / relative,
            payload,
            expected_sha256=hashlib.sha256(payload).hexdigest(),
            noun=relative,
        )
    reconstruction = load_reconstruction_v2(
        outputs / "reconstruction.npz"
    )
    metrics = _native_metrics(
        evaluate_video_global_affine(
            verified_dataset.truth.gt_frames,
            reconstruction.reconstruction,
        )
    )
    _validate_finite_json(metrics)
    metrics_bytes = canonical_json_bytes(metrics)
    _write_file_durable(outputs / "metrics.json", metrics_bytes)
    lifecycle = {
        "schema": "run-lifecycle-v1",
        "state": "complete",
        "identity_sha256": request.identity.identity_sha256,
        "owner_token": claim["owner_token"],
        "fence": claim["fence"],
    }
    lifecycle_bytes = canonical_json_bytes(lifecycle)
    _replace_file_durable(stage / "lifecycle.json", lifecycle_bytes)
    artifacts = []
    for role, relative, schema in _RUNNER_ARTIFACT_CONTRACT:
        path = stage / relative
        artifacts.append(
            {
                "role": role,
                "path": relative,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "schema_version": schema,
                "required": True,
            }
        )
    dataset_manifest = verified_dataset.manifest
    manifest = build_manifest(
        status="complete",
        identity=request.identity,
        config_resolved=dict(plan.config_resolved),
        inputs={
            "measurements_file_sha256": measurement_evidence.sha256,
            "evaluation_truth_file_sha256": verified_dataset.payload_evidence["evaluation-truth.npz"].sha256,
            "dataset_manifest_sha256": verified_dataset.dataset_manifest_sha256,
        },
        runtime=dict(plan.runtime_metadata),
        execution={
            "command": list(materialized.materialization_record["logical"]["command_template"]),
            "started_at_utc": started,
            "ended_at_utc": ended,
            "return_code": 0,
            "runtime_seconds": runtime_seconds,
            "peak_vram_bytes": resource_evidence["peak_vram_bytes"],
        },
        measurement=dataset_measurement_record(dataset_manifest),
        metrics={
            "version": plan.metric_version,
            "path": "outputs/metrics.json",
            "sha256": hashlib.sha256(metrics_bytes).hexdigest(),
        },
        artifacts=artifacts,
    )
    _remove_owned_work(artifact_root, work_id, work_pointer)
    (stage / "child-work.json").unlink()
    owner = stage / ".owner.json"
    owner.unlink()
    _fsync_tree(stage)
    _verify_claim(
        claim,
        path=(
            runs_dir.parent
            / ".claims"
            / f"{request.identity.identity_sha256}.json"
        ),
    )
    _replace_file_durable(
        stage / "manifest.json",
        canonical_json_bytes(manifest),
    )
    final_dir = runs_dir / request.identity.identity_sha256
    try:
        _promote_exact_directory_no_clobber(
            stage,
            final_dir,
            expected_device=stage_identity.device,
            expected_inode=stage_identity.inode,
        )
    except FileExistsError:
        try:
            winner = reusable_run(runs_dir.parent, request.identity)
        except (OSError, ValueError) as error:
            raise ValueError("promotion winner failed integrity validation") from error
        if winner is None:
            raise ValueError("promotion winner is not reusable")
        _cleanup_owned_final_stage(
            stage,
            runs_dir,
            claim,
            request.identity,
            stage_identity,
        )
        return RunOutcome("cached", winner, None, 0)
    try:
        _sync_directory(runs_dir)
    except Exception as sync_error:
        try:
            _strict_post_rename_revalidation(
                runs_dir.parent,
                request.identity,
                final_dir,
            )
        except _PostPromotionIntegrityError as integrity_error:
            integrity_error.add_note(
                "Directory sync also failed after atomic rename: "
                f"{type(sync_error).__name__}: {sync_error}"
            )
            raise
        raise _PostPromotionDurabilityError(
            "promoted run passed strict validation but directory durability "
            "sync failed"
        ) from sync_error
    _strict_post_rename_revalidation(
        runs_dir.parent,
        request.identity,
        final_dir,
    )
    return RunOutcome("complete", final_dir, None, 0)


def _strict_post_rename_revalidation(
    artifact_root: Path,
    identity: RunIdentity,
    final_dir: Path,
) -> Path:
    first_error: Exception | None = None
    for _attempt in range(2):
        try:
            winner = reusable_run(artifact_root, identity)
        except (OSError, ValueError) as error:
            first_error = first_error or error
            continue
        if winner == final_dir:
            return winner
        first_error = ValueError("promoted identity is not strictly reusable")
    raise _PostPromotionIntegrityError(
        "promoted run failed strict integrity validation"
    ) from first_error


def _run_method_child(
    *,
    argv: tuple[str, ...],
    cwd: Path,
    env: dict[str, str],
    stdout_stream,
    stderr_stream,
    requested_runtime_device: str,
    compute_cap: _ComputeCap,
) -> tuple[int, dict[str, object]]:
    _requested_device, cuda_index = _canonical_runtime_device(
        requested_runtime_device
    )
    sample_vram = cuda_index is not None
    method_job = _WindowsKillJob(active_process_limit=1)
    process = _create_suspended_in_job(
        argv,
        cwd=cwd,
        env=env,
        stdout=stdout_stream,
        stderr=stderr_stream,
        job=method_job,
    )
    sampler = None
    sampler_job = None
    sampler_queue: queue.Queue[tuple[str, object]] = queue.Queue()
    reader = None
    records: list[tuple[str, int, int]] = []
    sample_count = 0
    peak_vram_bytes = 0
    started = time.perf_counter()
    try:
        if sample_vram:
            try:
                sampler_job = _WindowsKillJob()
                sampler = _create_suspended_in_job(
                    [
                        str(_windows_powershell_path()),
                        "-NoLogo",
                        "-NoProfile",
                        "-NonInteractive",
                        "-Command",
                        _powershell_sampler_command(process.pid),
                    ],
                    cwd=cwd,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    job=sampler_job,
                    text=True,
                )
                assert sampler.stdout is not None
                reader = threading.Thread(
                    target=_read_sampler_lines,
                    args=(sampler.stdout, sampler_queue),
                    daemon=True,
                )
                reader.start()
                _resume_suspended_process(sampler.pid)
                _await_sampler_ready(sampler, sampler_queue)
            except (OSError, RuntimeError, ValueError) as error:
                raise _resource_failure(
                    f"VRAM sampler startup failed: {error}",
                    compute_cap,
                    started,
                    sample_count,
                    peak_vram_bytes,
                    requested_runtime_device,
                ) from error
        _resume_suspended_process(process.pid)
        started = time.perf_counter()
        while True:
            return_code = process.poll()
            if sample_vram:
                try:
                    sample_count, peak_vram_bytes = _drain_sampler_queue(
                        sampler_queue,
                        records,
                        process.pid,
                        sample_count,
                        peak_vram_bytes,
                        allow_eof=return_code is not None,
                        child_process=process,
                    )
                except ValueError as error:
                    raise _resource_failure(
                        f"VRAM sampler failed: {error}",
                        compute_cap,
                        started,
                        sample_count,
                        peak_vram_bytes,
                        requested_runtime_device,
                    ) from error
                if sampler is not None and sampler.poll() is not None:
                    return_code = process.poll()
                    if return_code is None:
                        raise _resource_failure(
                            "VRAM sampler exited before child",
                            compute_cap,
                            started,
                            sample_count,
                            peak_vram_bytes,
                            requested_runtime_device,
                        )
                if peak_vram_bytes > compute_cap.peak_vram_bytes:
                    raise _resource_failure(
                        "peak VRAM compute cap exceeded",
                        compute_cap,
                        started,
                        sample_count,
                        peak_vram_bytes,
                        requested_runtime_device,
                    )
            elapsed = time.perf_counter() - started
            if elapsed > compute_cap.wall_time_seconds:
                raise _resource_failure(
                    "wall time compute cap exceeded",
                    compute_cap,
                    started,
                    sample_count,
                    peak_vram_bytes,
                    requested_runtime_device,
                )
            if return_code is not None:
                child_runtime_seconds = time.perf_counter() - started
                if child_runtime_seconds > compute_cap.wall_time_seconds:
                    raise _resource_failure(
                        "wall time compute cap exceeded",
                        compute_cap,
                        started,
                        sample_count,
                        peak_vram_bytes,
                        requested_runtime_device,
                    )
                if sample_vram:
                    time.sleep(_VRAM_SAMPLING_INTERVAL_MS / 1000)
                    try:
                        sample_count, peak_vram_bytes = _drain_sampler_queue(
                            sampler_queue,
                            records,
                            process.pid,
                            sample_count,
                            peak_vram_bytes,
                            allow_eof=True,
                        )
                    except ValueError as error:
                        raise _resource_failure(
                            f"VRAM sampler failed: {error}",
                            compute_cap,
                            started,
                            sample_count,
                            peak_vram_bytes,
                            requested_runtime_device,
                        ) from error
                    assert sampler is not None
                    sampler_return_code = sampler.poll()
                    sampler_stopped_by_parent = sampler_return_code is None
                    if sampler_stopped_by_parent:
                        sampler.terminate()
                        try:
                            sampler.wait(timeout=5)
                        except subprocess.TimeoutExpired as error:
                            raise _resource_failure(
                                "VRAM sampler did not stop on parent request",
                                compute_cap,
                                started,
                                sample_count,
                                peak_vram_bytes,
                                requested_runtime_device,
                            ) from error
                    assert reader is not None
                    reader.join(timeout=5)
                    if reader.is_alive():
                        raise _resource_failure(
                            "VRAM sampler reader did not terminate",
                            compute_cap,
                            started,
                            sample_count,
                            peak_vram_bytes,
                            requested_runtime_device,
                        )
                    try:
                        sample_count, peak_vram_bytes = _drain_sampler_queue(
                            sampler_queue,
                            records,
                            process.pid,
                            sample_count,
                            peak_vram_bytes,
                            allow_eof=True,
                        )
                    except ValueError as error:
                        raise _resource_failure(
                            f"VRAM sampler failed: {error}",
                            compute_cap,
                            started,
                            sample_count,
                            peak_vram_bytes,
                            requested_runtime_device,
                        ) from error
                    if (
                        not sampler_stopped_by_parent
                        and sampler_return_code != 0
                    ):
                        raise _resource_failure(
                            "VRAM sampler failed with return code "
                            f"{sampler_return_code}",
                            compute_cap,
                            started,
                            sample_count,
                            peak_vram_bytes,
                            requested_runtime_device,
                        )
                    if sample_count == 0:
                        raise _resource_failure(
                            "VRAM sampler produced no valid sample",
                            compute_cap,
                            started,
                            sample_count,
                            peak_vram_bytes,
                            requested_runtime_device,
                        )
                    if peak_vram_bytes > compute_cap.peak_vram_bytes:
                        raise _resource_failure(
                            "peak VRAM compute cap exceeded",
                            compute_cap,
                            started,
                            sample_count,
                            peak_vram_bytes,
                            requested_runtime_device,
                        )
                evidence = _resource_evidence(
                    status="complete",
                    compute_cap=compute_cap,
                    runtime_seconds=child_runtime_seconds,
                    sample_count=sample_count,
                    peak_vram_bytes=peak_vram_bytes,
                    requested_runtime_device=requested_runtime_device,
                )
                return return_code, evidence
            time.sleep(0.05)
    finally:
        method_job.close()
        if sampler_job is not None:
            sampler_job.close()
        for child in (process, sampler):
            if child is not None:
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=5)
        if reader is not None:
            reader.join(timeout=5)
            if reader.is_alive():
                raise RuntimeError("VRAM sampler reader did not terminate")
        if sampler is not None:
            for stream in (sampler.stdout, sampler.stderr):
                if stream is not None:
                    stream.close()


def _read_sampler_lines(stream, destination: queue.Queue) -> None:
    try:
        for line in stream:
            destination.put(("line", line.rstrip("\r\n")))
    except BaseException as error:
        destination.put(("error", error))
    finally:
        destination.put(("eof", None))


def _powershell_sampler_command(process_id: int) -> str:
    if type(process_id) is not int or process_id <= 0:
        raise ValueError("VRAM sampler PID must be a positive integer")
    return (
        "& {\n"
        + _POWERSHELL_SAMPLER_SCRIPT
        + "\n} "
        + str(process_id)
        + " "
        + str(_VRAM_SAMPLING_INTERVAL_MS)
    )


def _await_sampler_ready(
    sampler: subprocess.Popen,
    source: queue.Queue,
) -> None:
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        try:
            kind, value = source.get(timeout=0.1)
        except queue.Empty:
            if sampler.poll() is not None:
                break
            continue
        if kind == "line" and value == "READY":
            return
        if kind == "error":
            raise ValueError("VRAM sampler output reader failed") from value
        raise ValueError("VRAM sampler failed before readiness")
    raise ValueError("VRAM sampler did not become ready")


def _drain_sampler_queue(
    source: queue.Queue,
    records: list[tuple[str, int, int]],
    process_id: int,
    sample_count: int,
    peak_vram_bytes: int,
    *,
    allow_eof: bool = False,
    child_process: subprocess.Popen | None = None,
) -> tuple[int, int]:
    while True:
        try:
            kind, value = source.get_nowait()
        except queue.Empty:
            return sample_count, peak_vram_bytes
        if kind == "error":
            raise ValueError("VRAM sampler output reader failed") from value
        if kind == "eof":
            child_has_exited = (
                child_process is not None and child_process.poll() is not None
            )
            if (allow_eof or child_has_exited) and not records:
                return sample_count, peak_vram_bytes
            raise ValueError("VRAM sampler output ended unexpectedly")
        if kind != "line" or type(value) is not str:
            raise ValueError("VRAM sampler output event is malformed")
        if value == "E":
            observed = _process_vram_from_counter_records(records, process_id)
            records.clear()
            if observed is not None:
                sample_count += 1
                peak_vram_bytes = max(peak_vram_bytes, observed)
            continue
        fields = value.split("\t")
        if len(fields) != 4 or fields[0] != "R":
            raise ValueError(
                f"VRAM sampler output line is malformed: {value!r}"
            )
        if re.fullmatch(r"[0-9]+", fields[2], re.ASCII) is None:
            raise ValueError("VRAM sampler counter status is malformed")
        if re.fullmatch(r"[0-9]+", fields[3], re.ASCII) is None:
            raise ValueError("VRAM sampler value is malformed")
        records.append((fields[1], int(fields[2]), int(fields[3])))


def _resource_evidence(
    *,
    status: str,
    compute_cap: _ComputeCap,
    runtime_seconds: float,
    sample_count: int,
    peak_vram_bytes: int,
    requested_runtime_device: str,
) -> dict[str, object]:
    return {
        "schema": "run-resource-sampling-v1",
        "status": status,
        "backend": (
            "windows-gpu-process-memory-dedicated-usage-v1"
            if _canonical_runtime_device(requested_runtime_device)[1] is not None
            else "cpu-no-vram-sampling-v1"
        ),
        "sampling_interval_ms": (
            _VRAM_SAMPLING_INTERVAL_MS
            if _canonical_runtime_device(requested_runtime_device)[1] is not None
            else 0
        ),
        "sample_count": sample_count,
        "peak_vram_bytes": peak_vram_bytes,
        "runtime_seconds": runtime_seconds,
        "requested_runtime_device": requested_runtime_device,
        "compute_cap": {
            "wall_time_seconds": compute_cap.wall_time_seconds,
            "peak_vram_bytes": compute_cap.peak_vram_bytes,
            "on_exceed": compute_cap.on_exceed,
        },
    }


def _resource_failure(
    reason: str,
    compute_cap: _ComputeCap,
    started: float,
    sample_count: int,
    peak_vram_bytes: int,
    requested_runtime_device: str,
) -> _ComputeCapFailure:
    return _ComputeCapFailure(
        reason,
        _resource_evidence(
            status=reason,
            compute_cap=compute_cap,
            runtime_seconds=time.perf_counter() - started,
            sample_count=sample_count,
            peak_vram_bytes=peak_vram_bytes,
            requested_runtime_device=requested_runtime_device,
        ),
    )


@contextmanager
def _claim_identity(
    claims_dir: Path,
    identity: RunIdentity,
) -> Iterator[Mapping[str, object]]:
    claims_identity = _directory_identity(claims_dir, noun="claims directory")
    lock_path = claims_dir / f"{identity.identity_sha256}.lock"
    metadata_path = claims_dir / f"{identity.identity_sha256}.json"
    with _open_claim_lock(lock_path) as stream:
        _lock_file(stream)
        try:
            _verify_directory_identity(claims_identity, noun="claims directory")
            _verify_claim_lock_handle(stream, lock_path)
            stream.seek(0, os.SEEK_END)
            if stream.tell() == 0:
                stream.write(b"\0")
                stream.flush()
                os.fsync(stream.fileno())
            previous_fence = 0
            stale_pattern = f"{identity.identity_sha256}.stale-*.json"
            for stale_path in sorted(claims_dir.glob(stale_pattern)):
                token = stale_path.name.removeprefix(
                    f"{identity.identity_sha256}.stale-"
                ).removesuffix(".json")
                if re.fullmatch(r"[0-9a-f]{32}", token) is None:
                    raise ValueError("stale claim filename is invalid")
                stale_claim = _validated_claim_metadata(stale_path, identity)
                previous_fence = max(previous_fence, stale_claim["fence"])
            if os.path.lexists(metadata_path):
                previous = _validated_claim_metadata(metadata_path, identity)
                previous_fence = max(previous_fence, previous["fence"])
                previous_info = os.lstat(metadata_path)
                _claim_metadata_barrier(metadata_path)
                _verify_directory_identity(
                    claims_identity,
                    noun="claims directory",
                )
                current_info = os.lstat(metadata_path)
                if _claim_metadata_signature(current_info) != _claim_metadata_signature(
                    previous_info
                ):
                    raise ValueError("claim metadata changed after stable read")
                stale = claims_dir / (
                    f"{identity.identity_sha256}.stale-{uuid.uuid4().hex}.json"
                )
                _rename_no_clobber(metadata_path, stale)
                if canonical_json_bytes(_read_unique_json(stale)) != canonical_json_bytes(
                    previous
                ):
                    raise ValueError("stale claim metadata changed during rename")
                _claim_rotation_barrier(stale, metadata_path)
            claim: dict[str, object] = {
                "schema": "run-claim-v1",
                "identity_sha256": identity.identity_sha256,
                "owner_token": uuid.uuid4().hex,
                "fence": previous_fence + 1,
                "pid": os.getpid(),
            }
            _verify_directory_identity(claims_identity, noun="claims directory")
            _write_file_durable(metadata_path, canonical_json_bytes(claim))
            created_info = os.lstat(metadata_path)
            yield claim
            _verify_claim(claim, path=metadata_path)
            _verify_directory_identity(claims_identity, noun="claims directory")
            current_info = os.lstat(metadata_path)
            if _claim_metadata_signature(current_info) != _claim_metadata_signature(
                created_info
            ):
                raise ValueError("claim metadata identity changed before cleanup")
            if os.name == "nt":
                _delete_windows_owned_path(
                    metadata_path,
                    expected_device=created_info.st_dev,
                    expected_inode=created_info.st_ino,
                    directory=False,
                )
            else:
                metadata_path.unlink()
            if os.path.lexists(metadata_path):
                raise ValueError("claim metadata remains after exact cleanup")
        finally:
            _unlock_file(stream)


def _claim_metadata_barrier(path: Path) -> None:
    """Private deterministic test hook after stable metadata read."""


def _claim_rotation_barrier(stale_path: Path, metadata_path: Path) -> None:
    """Private deterministic crash hook after active-to-stale rotation."""


def _validated_claim_metadata(
    path: Path,
    identity: RunIdentity,
) -> dict[str, object]:
    value = _read_unique_json(path)
    if type(value) is not dict or set(value) != {
        "schema",
        "identity_sha256",
        "owner_token",
        "fence",
        "pid",
    }:
        raise ValueError("claim metadata has an invalid exact shape")
    if (
        value["schema"] != "run-claim-v1"
        or value["identity_sha256"] != identity.identity_sha256
        or type(value["owner_token"]) is not str
        or re.fullmatch(r"[0-9a-f]{32}", value["owner_token"]) is None
        or type(value["fence"]) is not int
        or value["fence"] < 1
        or type(value["pid"]) is not int
        or value["pid"] < 1
    ):
        raise ValueError("claim metadata identity, token, fence, or pid is invalid")
    return value


def _claim_metadata_signature(info: os.stat_result) -> tuple[int, ...]:
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    if (
        not stat.S_ISREG(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or getattr(info, "st_file_attributes", 0) & reparse
        or info.st_nlink != 1
    ):
        raise ValueError("claim metadata must be an unlinked regular file")
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
        info.st_nlink,
    )


def _open_claim_lock(path: Path):
    _reject_linked_ancestors(path, noun="claim lock")
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOINHERIT", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as error:
        raise ValueError("cannot open claim lock without following links") from error
    try:
        opened = os.fstat(descriptor)
        observed = os.lstat(path)
        reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or stat.S_ISLNK(observed.st_mode)
            or getattr(observed, "st_file_attributes", 0) & reparse
            or not stat.S_ISREG(observed.st_mode)
            or observed.st_nlink != 1
            or (opened.st_dev, opened.st_ino)
            != (observed.st_dev, observed.st_ino)
        ):
            raise ValueError("claim lock must be an unlinked regular file")
        _reject_linked_ancestors(path, noun="claim lock")
        return os.fdopen(descriptor, "r+b", closefd=True)
    except BaseException:
        os.close(descriptor)
        raise


def _verify_claim_lock_handle(stream, path: Path) -> None:
    _reject_linked_ancestors(path, noun="claim lock")
    opened = os.fstat(stream.fileno())
    observed = os.lstat(path)
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    if (
        not stat.S_ISREG(opened.st_mode)
        or opened.st_nlink != 1
        or stat.S_ISLNK(observed.st_mode)
        or getattr(observed, "st_file_attributes", 0) & reparse
        or not stat.S_ISREG(observed.st_mode)
        or observed.st_nlink != 1
        or (opened.st_dev, opened.st_ino)
        != (observed.st_dev, observed.st_ino)
    ):
        raise ValueError("claim lock identity changed while acquiring lock")


def _lock_file(stream) -> None:
    if os.name == "nt":
        import msvcrt

        retryable = {errno.EACCES, errno.EAGAIN, errno.EDEADLK}
        while True:
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                if error.errno not in retryable:
                    raise
                time.sleep(0.1)
                continue
            return
    import fcntl

    fcntl.flock(stream.fileno(), fcntl.LOCK_EX)


def _unlock_file(stream) -> None:
    if os.name == "nt":
        import msvcrt

        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        return
    import fcntl

    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _verify_claim(
    expected: Mapping[str, object],
    *,
    path: Path | None = None,
) -> None:
    if path is None:
        raise TypeError("claim verification requires its metadata path")
    observed = _read_unique_json(path)
    if canonical_json_bytes(observed) != canonical_json_bytes(dict(expected)):
        raise ValueError("run claim owner token or fence was lost")


def _verify_stage_owner(
    stage: Path,
    claim: Mapping[str, object],
    identity: RunIdentity,
) -> None:
    expected = {
        "schema": "run-stage-owner-v1",
        "owner_token": claim["owner_token"],
        "fence": claim["fence"],
        "identity_sha256": identity.identity_sha256,
    }
    owner_path = stage / ".owner.json"
    if owner_path.exists():
        owner = _read_unique_json(owner_path)
        if canonical_json_bytes(owner) == canonical_json_bytes(expected):
            return
    lifecycle_path = stage / "lifecycle.json"
    if lifecycle_path.exists():
        lifecycle = _read_unique_json(lifecycle_path)
        if type(lifecycle) is dict and lifecycle == {
            "schema": "run-lifecycle-v1",
            "state": "complete",
            "identity_sha256": identity.identity_sha256,
            "owner_token": claim["owner_token"],
            "fence": claim["fence"],
        }:
            return
    raise ValueError("staging directory ownership changed")


def _has_complete_manifest_marker(stage: Path, identity: RunIdentity) -> bool:
    path = stage / "manifest.json"
    if not path.exists():
        return False
    try:
        manifest = _read_unique_json(path)
    except (OSError, ValueError):
        return False
    return (
        type(manifest) is dict
        and manifest.get("status") == "complete"
        and manifest.get("identity_sha256") == identity.identity_sha256
    )


def _remove_owned_work(
    artifact_root: Path,
    work_id: str,
    expected_owner: Mapping[str, object],
) -> None:
    if (
        type(work_id) is not str
        or len(work_id) != 32
        or any(character not in "0123456789abcdef" for character in work_id)
    ):
        raise ValueError("work_id must be 32 lowercase hexadecimal characters")
    work_parent = artifact_root.absolute() / "work"
    work = work_parent / work_id
    if work.parent != work_parent or work.name != work_id:
        raise ValueError("child work parent or name is invalid")
    marker = work / ".runner-owner.json"
    observed = _read_unique_json(marker)
    if canonical_json_bytes(observed) != canonical_json_bytes(
        dict(expected_owner)
    ):
        raise ValueError("child workspace ownership changed")
    work_device = expected_owner.get("work_device")
    work_inode = expected_owner.get("work_inode")
    if type(work_device) is not int or type(work_inode) is not int:
        raise ValueError("child workspace lacks pinned directory identity")
    identity = _directory_identity(work, noun="child workspace")
    if (identity.device, identity.inode) != (work_device, work_inode):
        raise ValueError("child workspace identity changed")
    _owned_cleanup_barrier(work)
    _cleanup_pinned_owned_tree(identity, noun="child workspace")


def _cleanup_owned_final_stage(
    stage: Path,
    runs_dir: Path,
    claim: Mapping[str, object],
    identity: RunIdentity,
    stage_identity,
) -> None:
    if stage.parent != runs_dir or not stage.name.startswith(
        f"{identity.identity_sha256}.tmp-"
    ):
        raise ValueError("final staging parent or name is invalid")
    _verify_directory_identity(stage_identity, noun="final staging directory")
    _verify_stage_owner(stage, claim, identity)
    _owned_cleanup_barrier(stage)
    _cleanup_pinned_owned_tree(stage_identity, noun="final staging directory")


def _owned_cleanup_barrier(path: Path) -> None:
    """Private deterministic test hook immediately before pinned cleanup."""


def _recover_stale_stages(
    *,
    runs_dir: Path,
    failed_dir: Path,
    claims_dir: Path,
    identity: RunIdentity,
    current_claim: Mapping[str, object],
) -> None:
    stale_claims: list[Mapping[str, object]] = []
    for path in claims_dir.glob(f"{identity.identity_sha256}.stale-*.json"):
        value = _read_unique_json(path)
        if (
            type(value) is not dict
            or value.get("schema") != "run-claim-v1"
            or value.get("identity_sha256") != identity.identity_sha256
            or type(value.get("owner_token")) is not str
            or type(value.get("fence")) is not int
        ):
            raise ValueError("stale claim evidence is malformed")
        stale_claims.append(value)
    candidates = list(
        runs_dir.glob(f"{identity.identity_sha256}.tmp-*")
    )
    for stage in candidates:
        suffix = stage.name.removeprefix(
            f"{identity.identity_sha256}.tmp-"
        )
        try:
            uuid.UUID(suffix)
        except ValueError as error:
            raise ValueError("stale run stage name has an invalid UUID") from error
        owner_match = None
        for stale in stale_claims:
            try:
                _verify_stage_owner(stage, stale, identity)
            except (OSError, ValueError):
                continue
            owner_match = stale
            break
        if owner_match is None:
            if (
                stale_claims
                and all(
                    stale["fence"] < current_claim["fence"]
                    for stale in stale_claims
                )
                and _is_interrupted_stage_owner_publication(stage)
            ):
                diagnostic = failed_dir / (
                    f"{identity.run_id}-recovered-{uuid.uuid4().hex}"
                )
                _rename_no_clobber(stage, diagnostic)
                continue
            raise ValueError("stale run stage has no fenced owner evidence")
        if (
            owner_match["owner_token"] == current_claim["owner_token"]
            or owner_match["fence"] >= current_claim["fence"]
        ):
            raise ValueError("stale run stage fencing is not older than current claim")
        diagnostic = failed_dir / (
            f"{identity.run_id}-recovered-{uuid.uuid4().hex}"
        )
        if (stage / "child-work.json").is_file():
            _finalize_diagnostic_evidence(
                stage=stage,
                artifact_root=runs_dir.parent,
                claim=owner_match,
            )
        _rename_no_clobber(stage, diagnostic)


def _is_interrupted_stage_owner_publication(stage: Path) -> bool:
    stage_identity = _directory_identity(
        stage,
        noun="interrupted run staging directory",
    )
    entries = list(stage.iterdir())
    _verify_directory_identity(
        stage_identity,
        noun="interrupted run staging directory",
    )
    if not entries:
        return True
    owner_path = stage / ".owner.json"
    if len(entries) != 1 or entries[0] != owner_path:
        return False
    try:
        _resolved_regular_file(
            owner_path,
            noun="interrupted stage owner marker",
        )
    except ValueError:
        return False
    try:
        _read_unique_json(owner_path)
    except (UnicodeError, ValueError):
        return True
    except OSError:
        return False
    return False


def _finalize_diagnostic_evidence(
    *,
    stage: Path,
    artifact_root: Path,
    claim: Mapping[str, object],
) -> None:
    pointer_path = stage / "child-work.json"
    expected_relatives = (
        "outputs/stdout.log",
        "outputs/stderr.log",
        "evidence/audit.jsonl",
        "evidence/resource-sampling.json",
    )
    if _reuse_finalized_diagnostic_evidence(
        stage=stage,
        artifact_root=artifact_root,
        claim=claim,
        expected_relatives=expected_relatives,
    ):
        return
    issues: list[str] = []
    try:
        pointer = _read_unique_json(pointer_path)
    except (OSError, ValueError) as error:
        _write_diagnostic_inventory(
            stage,
            present={},
            missing_expected=list(expected_relatives),
            retained=True,
            issues=[f"child work pointer is untrusted: {error}"],
        )
        return
    base_keys = {"schema", "work_id", "owner_token", "fence"}
    pinned_keys = base_keys | {"work_device", "work_inode"}
    pointer_trusted = True
    if type(pointer) is not dict or set(pointer) not in (base_keys, pinned_keys):
        pointer_trusted = False
    elif not (
        pointer["schema"] == "run-child-work-v1"
        and pointer["owner_token"] == claim["owner_token"]
        and pointer["fence"] == claim["fence"]
    ):
        pointer_trusted = False
    work_id = pointer.get("work_id") if type(pointer) is dict else None
    if (
        type(work_id) is not str
        or len(work_id) != 32
        or any(character not in "0123456789abcdef" for character in work_id)
    ):
        pointer_trusted = False
    if not pointer_trusted:
        _write_diagnostic_inventory(
            stage,
            present={},
            missing_expected=list(expected_relatives),
            retained=True,
            issues=["child work pointer token, fence, or path is untrusted"],
        )
        return
    assert type(work_id) is str
    work = artifact_root / "work" / work_id
    resource_evidence = work / "parent/logs/resource-sampling.json"
    if not resource_evidence.is_file():
        resource_evidence = (
            work / "parent/logs/resource-sampling.running.json"
        )
    sources = {
        "outputs/stdout.log": work / "parent/logs/stdout.log",
        "outputs/stderr.log": work / "parent/logs/stderr.log",
        "evidence/audit.jsonl": work / "parent/audit/file-opens.jsonl",
        "evidence/resource-sampling.json": resource_evidence,
    }
    descriptors: dict[str, dict[str, object]] = {}
    missing: list[str] = []
    for relative, source in sources.items():
        try:
            payload = _read_stable_regular_bytes(
                source,
                noun=f"diagnostic evidence {relative}",
            )
        except (OSError, ValueError) as error:
            missing.append(relative)
            if source.exists():
                issues.append(f"rejected {relative}: {error}")
            continue
        destination = stage / relative
        destination.parent.mkdir(exist_ok=True)
        if destination.exists():
            try:
                existing = _read_stable_regular_bytes(
                    destination,
                    noun=f"retained diagnostic evidence {relative}",
                )
            except (OSError, ValueError) as error:
                missing.append(relative)
                issues.append(f"rejected retained {relative}: {error}")
                continue
            if existing != payload:
                missing.append(relative)
                issues.append(f"retained {relative} disagrees with child workspace")
                continue
        else:
            _write_file_durable(destination, payload)
        descriptors[relative] = {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
    retained = work.exists()
    if retained:
        try:
            _remove_owned_work(artifact_root, work_id, pointer)
        except (OSError, ValueError) as error:
            issues.append(f"child workspace retained: {error}")
        else:
            retained = False
    _write_diagnostic_inventory(
        stage,
        present=descriptors,
        missing_expected=sorted(set(missing)),
        retained=retained,
        issues=issues,
    )
    _replace_file_durable(
        pointer_path,
        canonical_json_bytes({**pointer, "retained": retained}),
    )


def _reuse_finalized_diagnostic_evidence(
    *,
    stage: Path,
    artifact_root: Path,
    claim: Mapping[str, object],
    expected_relatives: tuple[str, ...],
) -> bool:
    inventory_path = stage / "diagnostic-artifacts.json"
    if not os.path.lexists(inventory_path):
        return False
    inventory = _read_unique_json(inventory_path)
    if type(inventory) is not dict or set(inventory) not in (
        {"schema", "present", "missing_expected", "retained"},
        {"schema", "present", "missing_expected", "retained", "issues"},
    ):
        raise ValueError("finalized diagnostic inventory has an invalid shape")
    present = inventory["present"]
    missing = inventory["missing_expected"]
    retained = inventory["retained"]
    issues = inventory.get("issues", [])
    if (
        inventory["schema"] != "run-diagnostic-artifacts-v1"
        or type(present) is not dict
        or type(missing) is not list
        or type(retained) is not bool
        or type(issues) is not list
        or any(type(issue) is not str for issue in issues)
        or any(type(relative) is not str for relative in missing)
        or len(missing) != len(set(missing))
    ):
        raise ValueError("finalized diagnostic inventory fields are invalid")
    expected = set(expected_relatives)
    if (
        set(present) & set(missing)
        or set(present) | set(missing) != expected
    ):
        raise ValueError("finalized diagnostic inventory coverage is invalid")
    for relative, descriptor in present.items():
        if (
            type(relative) is not str
            or type(descriptor) is not dict
            or set(descriptor) != {"sha256", "size_bytes"}
            or type(descriptor["sha256"]) is not str
            or _SHA256.fullmatch(descriptor["sha256"]) is None
            or type(descriptor["size_bytes"]) is not int
            or descriptor["size_bytes"] < 0
        ):
            raise ValueError("finalized diagnostic descriptor is invalid")
        payload = _read_stable_regular_bytes(
            stage / relative,
            noun=f"finalized diagnostic evidence {relative}",
        )
        if (
            len(payload) != descriptor["size_bytes"]
            or hashlib.sha256(payload).hexdigest() != descriptor["sha256"]
        ):
            raise ValueError("finalized diagnostic evidence changed")

    pointer_path = stage / "child-work.json"
    pointer = _read_unique_json(pointer_path)
    base_keys = {"schema", "work_id", "owner_token", "fence"}
    pinned_keys = base_keys | {"work_device", "work_inode"}
    if type(pointer) is not dict or set(pointer) not in (
        base_keys,
        pinned_keys,
        base_keys | {"retained"},
        pinned_keys | {"retained"},
    ):
        raise ValueError("finalized child work pointer has an invalid shape")
    work_id = pointer["work_id"]
    if (
        pointer["schema"] != "run-child-work-v1"
        or pointer["owner_token"] != claim["owner_token"]
        or pointer["fence"] != claim["fence"]
        or type(work_id) is not str
        or re.fullmatch(r"[0-9a-f]{32}", work_id, re.ASCII) is None
        or (
            set(pointer) in (pinned_keys, pinned_keys | {"retained"})
            and (
                type(pointer["work_device"]) is not int
                or type(pointer["work_inode"]) is not int
            )
        )
    ):
        raise ValueError("finalized child work pointer fields are invalid")
    if os.path.lexists(artifact_root / "work" / work_id) != retained:
        raise ValueError("finalized child work retention state changed")
    if "retained" in pointer:
        if type(pointer["retained"]) is not bool or pointer["retained"] != retained:
            raise ValueError("finalized child work retention marker is invalid")
    else:
        _replace_file_durable(
            pointer_path,
            canonical_json_bytes({**pointer, "retained": retained}),
        )
    return True


def _write_diagnostic_inventory(
    stage: Path,
    *,
    present: Mapping[str, Mapping[str, object]],
    missing_expected: list[str],
    retained: bool,
    issues: list[str],
) -> None:
    inventory: dict[str, object] = {
        "schema": "run-diagnostic-artifacts-v1",
        "present": dict(present),
        "missing_expected": missing_expected,
        "retained": retained,
    }
    if issues:
        inventory["issues"] = issues
    _replace_file_durable(
        stage / "diagnostic-artifacts.json",
        canonical_json_bytes(inventory),
    )


def _filesystem_device(path: Path) -> int:
    return os.stat(path).st_dev


def _require_same_filesystem(left: Path, right: Path) -> None:
    if _filesystem_device(left) != _filesystem_device(right):
        raise ValueError("child work and final stage must share one filesystem volume")


def _validate_finite_json(value: object, path: str = "$") -> None:
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite metric")
        return
    if type(value) is list:
        for index, child in enumerate(value):
            _validate_finite_json(child, f"{path}[{index}]")
        return
    if type(value) is dict:
        for key, child in value.items():
            if type(key) is not str:
                raise TypeError("metric keys must be exact strings")
            _validate_finite_json(child, f"{path}.{key}")
        return
    raise TypeError(f"{path} contains unsupported metric type")


def _native_metrics(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if type(value) is dict:
        return {key: _native_metrics(child) for key, child in value.items()}
    if type(value) is list:
        return [_native_metrics(child) for child in value]
    return value


def _read_unique_json(path: Path) -> object:
    raw = _read_stable_regular_bytes(path, noun="runner JSON evidence")
    text = raw.decode("utf-8", errors="strict")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key!r}")
            result[key] = value
        return result

    return json.loads(
        text,
        object_pairs_hook=unique,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant: {value}")
        ),
    )


def _ensure_real_directory(path: Path) -> None:
    _reject_linked_ancestors(path, noun="artifact directory")
    if not path.exists():
        try:
            path.mkdir()
        except FileExistsError:
            pass
    info = os.lstat(path)
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    if (
        not stat.S_ISDIR(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or getattr(info, "st_file_attributes", 0) & reparse
    ):
        raise ValueError(f"artifact directory is linked or not a directory: {path}")
    _reject_linked_ancestors(path, noun="artifact directory")
    _resolved = _directory_identity(path, noun="artifact directory")
    _verify_directory_identity(_resolved, noun="artifact directory")


def _write_file_durable(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _replace_file_durable(path: Path, payload: bytes) -> None:
    domain = {"lifecycle.json": "lc", "manifest.json": "mf"}.get(
        path.name,
        "rp",
    )
    temporary: Path | None = None
    for _attempt in range(16):
        token = secrets.token_hex(6)
        if type(token) is not str or re.fullmatch(r"[0-9a-f]{12}", token) is None:
            raise ValueError("durable replacement token is invalid")
        candidate = path.with_name(f".{domain}-{token}.tmp")
        if os.name == "nt" and len(str(candidate.absolute())) >= 260:
            raise ValueError(
                "durable replacement temporary path exceeds Win32 path policy"
            )
        try:
            _write_file_durable(candidate, payload)
        except FileExistsError:
            continue
        temporary = candidate
        break
    if temporary is None:
        raise FileExistsError("could not allocate a unique replacement temporary")
    try:
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_bound_snapshot(
    path: Path,
    payload: bytes,
    *,
    expected_sha256: str,
    noun: str,
) -> None:
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError(f"{noun} snapshot hash disagrees before write")
    _write_file_durable(path, payload)
    written = _read_stable_regular_bytes(path, noun=f"staged {noun}")
    if written != payload or hashlib.sha256(written).hexdigest() != expected_sha256:
        raise ValueError(f"{noun} snapshot changed during staged write")


def _fsync_tree(root: Path) -> None:
    for current, directories, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in [*directories, *files]:
            info = os.lstat(current_path / name)
            reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & reparse:
                raise ValueError("run staging tree contains a link or reparse point")
        if os.name != "nt":
            for name in files:
                with (current_path / name).open("rb") as stream:
                    os.fsync(stream.fileno())
        _sync_directory(current_path)


def _sync_directory(path: Path) -> None:
    if os.name == "nt":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        create_file = kernel32.CreateFileW
        create_file.argtypes = [
            ctypes.c_wchar_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
        ]
        create_file.restype = ctypes.c_void_p
        flush = kernel32.FlushFileBuffers
        flush.argtypes = [ctypes.c_void_p]
        flush.restype = ctypes.c_int
        close = kernel32.CloseHandle
        close.argtypes = [ctypes.c_void_p]
        handle = create_file(str(path), 0x40000000, 0x7, None, 3, 0x02000000, None)
        invalid = ctypes.c_void_p(-1).value
        if handle in (None, invalid):
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            if not flush(handle):
                raise ctypes.WinError(ctypes.get_last_error())
        finally:
            close(handle)
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_no_clobber(source: Path, destination: Path) -> None:
    if _WINDOWS_ATOMIC_RENAME:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        move = kernel32.MoveFileExW
        move.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32]
        move.restype = ctypes.c_int
        if move(str(source), str(destination), _MOVEFILE_WRITE_THROUGH):
            return
        code = ctypes.get_last_error()
        if code in {80, 183}:
            raise FileExistsError(code, "destination already exists")
        raise ctypes.WinError(code)
    raise RuntimeError("atomic no-clobber rename is supported only on Windows")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
