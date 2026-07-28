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
