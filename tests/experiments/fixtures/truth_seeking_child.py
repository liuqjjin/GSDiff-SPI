"""Adversarial child used to pressure-test the procedural audit boundary."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


def _argument(name: str) -> str | None:
    try:
        return sys.argv[sys.argv.index(name) + 1]
    except (ValueError, IndexError):
        return None


def _expect_denied(operation) -> None:
    try:
        operation()
    except PermissionError:
        print("DENIED-CAUGHT")
        return
    raise RuntimeError("governed operation unexpectedly succeeded")


def _write_v2_outputs(output_dir: Path, request_path: Path) -> None:
    import numpy as np

    from gsdiff.data._artifact_identity import array_descriptor
    from gsdiff.data._artifact_models import SPIAcquisitionData
    from gsdiff.experiments.child_outputs import (
        MethodChildResult,
        write_method_child_outputs_v2,
    )
    from gsdiff.experiments.execution import (
        load_materialized_method_request,
    )

    request = load_materialized_method_request(request_path)
    arrays = {
        "patterns": np.ones((2, 2, 2), dtype=np.float32),
        "measurements": np.ones(2, dtype=np.float32),
        "frame_indices": np.arange(2, dtype=np.int64),
        "time_grid": np.arange(2, dtype=np.float64),
    }
    acquisition = SPIAcquisitionData(
        dataset_identity_sha256=request.dataset_identity_sha256,
        **arrays,
        holdout_patterns=None,
        holdout_measurements=None,
        holdout_frame_indices=None,
        H=2,
        W=2,
        T=2,
        K=2,
        holdout_K=0,
        acquisition={
            "pattern_family": "bernoulli",
            "pattern_values": [0, 1],
            "pattern_order": "sequential",
            "time_assignment": "uniform",
            "holdout_pattern_family": "bernoulli",
            "noise_convention": "absolute-gaussian-sigma",
            "noise_sigma_absolute": 0.0,
        },
        array_descriptors={
            name: array_descriptor(array) for name, array in arrays.items()
        },
    )
    result = MethodChildResult(
        method_id="dgi",
        reconstruction=np.ones((2, 2, 2), dtype=np.float32),
        estimated_motion_trajectory=None,
        dgi=None,
        info={
            "parameter_count": 0,
            "native_iteration_unit": "pass",
            "native_iteration_budget": 1,
            "convergence_status": request.method.convergence_status,
            "selected_hyperparameters": None,
            "selection": None,
            "checkpoint_hashes": [],
        },
        history=(),
    )
    hashes = write_method_child_outputs_v2(
        output_dir,
        method=request.method,
        acquisition=acquisition,
        measurements_file_sha256=request.measurements_file_sha256,
        algorithm_seed=request.algorithm_seed,
        result=result,
        child_started_at_utc="2026-07-28T00:00:00Z",
        child_finished_at_utc="2026-07-28T00:00:01Z",
    )
    print(
        "V2-HASHES="
        + json.dumps(
            dict(hashes),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )


def _emit_mutation_event(
    action: str,
    target: Path,
) -> None:
    if action == "audit-chown":
        sys.audit("os.chown", str(target), 1, 1, -1)
    elif action == "audit-chflags":
        sys.audit("os.chflags", str(target), 0)
    elif action == "audit-setxattr":
        sys.audit("os.setxattr", str(target), "user.gsdiff", b"x", 0)
    elif action == "audit-removexattr":
        sys.audit("os.removexattr", str(target), "user.gsdiff")
    elif action == "audit-mknod":
        sys.audit("os.mknod", str(target), 0o600, 0, -1)
    elif action == "audit-unknown-mutation":
        sys.audit("os.future_filesystem_mutation", str(target))
    elif action == "audit-chown-malformed":
        sys.audit("os.chown", str(target))
    elif action == "audit-chflags-malformed":
        sys.audit("os.chflags", str(target))
    elif action == "audit-setxattr-malformed":
        sys.audit("os.setxattr", str(target))
    elif action == "audit-removexattr-malformed":
        sys.audit("os.removexattr", str(target))
    elif action in {"audit-chown-dir-fd", "audit-mknod-dir-fd"}:
        if action == "audit-chown-dir-fd":
            sys.audit("os.chown", str(target.name), 1, 1, 17)
        else:
            sys.audit("os.mknod", str(target.name), 0o600, 0, 17)
    elif action in {
        "audit-chown-fd",
        "audit-setxattr-fd",
        "audit-removexattr-fd",
    }:
        descriptor = os.open(
            target,
            os.O_RDONLY | getattr(os, "O_BINARY", 0),
        )
        try:
            if action == "audit-chown-fd":
                sys.audit("os.chown", descriptor, 1, 1, -1)
            elif action == "audit-setxattr-fd":
                sys.audit(
                    "os.setxattr",
                    descriptor,
                    "user.gsdiff",
                    b"x",
                    0,
                )
            else:
                sys.audit(
                    "os.removexattr",
                    descriptor,
                    "user.gsdiff",
                )
        finally:
            os.close(descriptor)
    else:
        raise RuntimeError(f"unknown synthetic mutation action: {action}")


def _trigger_reentry(
    *,
    action: str,
    safe_path: Path,
    forbidden_path: Path,
) -> None:
    class ReentrantPath:
        def __fspath__(self) -> str:
            try:
                if action == "reentry-open":
                    leaked = forbidden_path.read_text(
                        encoding="utf-8",
                        errors="strict",
                    )
                    print(f"REENTRY-LEAK={leaked}")
                elif action == "reentry-process":
                    subprocess.run(
                        [
                            sys.executable,
                            "-c",
                            (
                                "from pathlib import Path;"
                                f"Path({str(forbidden_path)!r})"
                                ".write_text('nested-process',"
                                "encoding='utf-8')"
                            ),
                        ],
                        check=True,
                        shell=False,
                    )
                else:
                    raise RuntimeError(
                        f"unknown re-entry action: {action}"
                    )
            except PermissionError:
                print("REENTRY-DENIED-CAUGHT")
            return str(safe_path)

    try:
        sys.audit("open", ReentrantPath(), "r", 0)
    except PermissionError:
        print("REENTRY-OUTER-DENIED-CAUGHT")


def main() -> None:
    action = _argument("--audit-action")
    target_text = _argument("--target")
    second_text = _argument("--target2")
    target = Path(target_text) if target_text is not None else None
    second = Path(second_text) if second_text is not None else None
    expect_denied = "--expect-denied" in sys.argv

    def perform() -> None:
        if action == "noop":
            print("盲态验证")
            print("盲态验证", file=sys.stderr)
        elif action == "read":
            assert target is not None
            print(target.read_bytes().hex())
        elif action == "write-read":
            assert target is not None
            target.write_text("盲态验证", encoding="utf-8", errors="strict")
            print(target.read_text(encoding="utf-8", errors="strict"))
        elif action == "os-open-read":
            assert target is not None
            descriptor = os.open(
                target,
                os.O_RDONLY | getattr(os, "O_BINARY", 0),
            )
            try:
                print(os.read(descriptor, 4096).hex())
            finally:
                os.close(descriptor)
        elif action == "os-open-write":
            assert target is not None
            descriptor = os.open(
                target,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_TRUNC
                | getattr(os, "O_BINARY", 0),
                0o600,
            )
            try:
                os.write(descriptor, "盲态验证".encode("utf-8"))
            finally:
                os.close(descriptor)
        elif action == "os-open-temporary":
            assert target is not None
            descriptor = os.open(
                target,
                os.O_RDONLY
                | os.O_TEMPORARY
                | getattr(os, "O_BINARY", 0),
            )
            os.close(descriptor)
        elif action == "write-v2-outputs":
            assert target is not None and second is not None
            _write_v2_outputs(target, second)
        elif action == "truncate-fd":
            assert target is not None
            descriptor = os.open(
                target,
                os.O_RDWR | getattr(os, "O_BINARY", 0),
            )
            try:
                os.truncate(descriptor, 0)
            finally:
                os.close(descriptor)
        elif action == "winapi-create-file":
            assert target is not None
            import _winapi

            handle = _winapi.CreateFile(
                str(target),
                0x40000000,
                0,
                _winapi.NULL,
                1,
                0x80,
                _winapi.NULL,
            )
            _winapi.CloseHandle(handle)
        elif action == "winapi-create-junction":
            assert target is not None and second is not None
            import _winapi

            _winapi.CreateJunction(str(target), str(second))
        elif action in {"reentry-open", "reentry-process"}:
            assert target is not None and second is not None
            _trigger_reentry(
                action=action,
                safe_path=target,
                forbidden_path=second,
            )
        elif action is not None and action.startswith("audit-"):
            assert target is not None
            _emit_mutation_event(action, target)
        elif action == "chown":
            assert target is not None
            os.chown(target, 1, 1)
        elif action == "chflags":
            assert target is not None
            os.chflags(target, 0)
        elif action == "setxattr":
            assert target is not None
            os.setxattr(target, "user.gsdiff", b"x")
        elif action == "removexattr":
            assert target is not None
            os.removexattr(target, "user.gsdiff")
        elif action == "mknod":
            assert target is not None
            os.mknod(target)
        elif action == "listdir":
            assert target is not None
            os.listdir(target)
        elif action == "scandir":
            assert target is not None
            with os.scandir(target) as entries:
                list(entries)
        elif action == "chdir":
            assert target is not None
            os.chdir(target)
        elif action == "subprocess":
            subprocess.Popen(
                [sys.executable, "-c", "print('nested')"],
                shell=False,
            )
        elif action == "system":
            os.system("echo nested")
        elif action == "spawn":
            os.spawnv(
                os.P_NOWAIT,
                sys.executable,
                [sys.executable, "-c", "print('nested')"],
            )
        elif action == "mkdir":
            assert target is not None
            os.mkdir(target)
        elif action == "remove":
            assert target is not None
            os.remove(target)
        elif action == "rmdir":
            assert target is not None
            os.rmdir(target)
        elif action == "rename":
            assert target is not None and second is not None
            os.replace(target, second)
        elif action == "truncate":
            assert target is not None
            os.truncate(target, 0)
        elif action == "chmod":
            assert target is not None
            os.chmod(target, 0o600)
        elif action == "utime":
            assert target is not None
            os.utime(target)
        elif action == "symlink":
            assert target is not None and second is not None
            os.symlink(target, second)
        elif action == "hardlink":
            assert target is not None and second is not None
            os.link(target, second)
        elif action == "import-fresh":
            assert target_text is not None
            __import__(target_text)
        elif action == "import-truth":
            try:
                from gsdiff.data import load_evaluation_truth
            except (ImportError, ModuleNotFoundError):
                print("TRUTH-LOADER-UNAVAILABLE")
            else:
                raise RuntimeError(
                    f"truth loader unexpectedly available: "
                    f"{load_evaluation_truth!r}"
                )
        elif action == "sys-path":
            print(
                "SYSPATH="
                + json.dumps(
                    sys.path,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
        else:
            raise RuntimeError(f"unknown audit action: {action!r}")

    if expect_denied:
        _expect_denied(perform)
    else:
        perform()


if __name__ == "__main__":
    main()
