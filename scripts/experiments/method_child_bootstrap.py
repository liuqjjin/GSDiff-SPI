"""Install the stdlib audit hook before exposing staged or runtime imports.

This bootstrap intentionally runs under ``-I -S -B -X utf8``.  It never calls
``site.main`` or ``site.addsitedir`` and never reads or executes ``.pth``
files.  Runtime package roots are appended as plain, validated ``sys.path``
entries after the hook is installed.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import stat
import sys


_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_MAX_POLICY_BYTES = 4 * 1024 * 1024
_ENTRYPOINTS = frozenset({"train.py", "scripts/run_baselines.py"})


def _parse_bootstrap_arguments(
    values: list[str],
) -> tuple[Path, Path, str, tuple[str, ...]]:
    try:
        delimiter = values.index("--")
    except ValueError as error:
        raise ValueError("bootstrap arguments require an exact -- delimiter") from error
    trusted = values[:delimiter]
    if (
        len(trusted) != 6
        or trusted[0] != "--policy"
        or trusted[2] != "--code-root"
        or trusted[4] != "--entrypoint"
    ):
        raise ValueError("bootstrap trusted arguments have an invalid shape")
    entrypoint = trusted[5]
    if entrypoint not in _ENTRYPOINTS:
        raise ValueError("bootstrap entrypoint is not approved")
    return (
        Path(trusted[1]),
        Path(trusted[3]),
        entrypoint,
        tuple(values[delimiter + 1 :]),
    )


def _install_controlled_sys_path(
    policy: object,
    code_root: Path,
) -> tuple[str, ...]:
    """Install literal import roots without importing ``site`` or reading pth."""
    if not isinstance(policy, dict):
        try:
            policy = dict(policy)
        except (TypeError, ValueError) as error:
            raise ValueError("audit policy must be mapping-like") from error
    runtime_value = policy.get("python_runtime_root")
    site_values = policy.get("runtime_site_package_roots")
    if (
        type(runtime_value) is not str
        or type(site_values) not in (list, tuple)
    ):
        raise ValueError("audit policy runtime import roots are invalid")
    runtime_root = _real_directory(
        Path(runtime_value), noun="Python runtime root"
    )
    code = _real_directory(code_root, noun="staged code root")

    retained: list[Path] = []
    for entry in sys.path:
        if type(entry) is not str or not entry:
            raise ValueError("initial sys.path contains an unsafe entry")
        candidate = Path(os.path.abspath(entry))
        if not (
            _path_is_within(candidate, runtime_root)
            or _path_is_within(candidate, code)
        ):
            raise ValueError(
                f"initial sys.path escapes runtime/source closure: {candidate}"
            )
        retained.append(candidate)

    site_roots: list[Path] = []
    for value in site_values:
        if type(value) is not str or not value:
            raise ValueError("runtime site-packages entry is invalid")
        candidate = _real_directory(
            Path(value), noun="runtime site-packages root"
        )
        if not _path_is_within(candidate, runtime_root):
            raise ValueError(
                "runtime site-packages root escapes Python runtime"
            )
        site_roots.append(candidate)

    ordered = [code, *retained, *site_roots]
    final: list[str] = []
    seen: set[str] = set()
    for candidate in ordered:
        if not (
            _path_is_within(candidate, runtime_root)
            or _path_is_within(candidate, code)
        ):
            raise ValueError("controlled sys.path closure check failed")
        key = os.path.normcase(os.path.normpath(str(candidate)))
        if key not in seen:
            seen.add(key)
            final.append(str(candidate))
    sys.path[:] = final
    return tuple(final)


def main() -> None:
    policy_arg, code_arg, entrypoint, child_arguments = (
        _parse_bootstrap_arguments(sys.argv[1:])
    )
    code_root = _real_directory(code_arg, noun="staged code root")
    stage_root = code_root.parent
    expected_policy = stage_root / "parent" / "audit" / "policy.json"
    policy_path = _regular_file(policy_arg, noun="audit policy")
    if _path_key(policy_path) != _path_key(expected_policy):
        raise ValueError("bootstrap policy path is not the canonical stage policy")
    audit_module_path = _regular_file(
        code_root / "gsdiff" / "experiments" / "audit.py",
        noun="staged audit module",
    )
    entrypoint_path = _regular_file(
        code_root.joinpath(*Path(entrypoint).parts),
        noun="staged method entrypoint",
    )

    policy_raw = _read_bounded_file(
        policy_path, max_bytes=_MAX_POLICY_BYTES, noun="audit policy"
    )
    try:
        policy_document = json.loads(
            policy_raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("audit policy is not unique strict JSON") from error

    spec = importlib.util.spec_from_file_location(
        "_gsdiff_staged_method_audit",
        audit_module_path,
    )
    if spec is None or spec.loader is None:
        raise ValueError("cannot construct staged audit module loader")
    audit_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit_module)
    policy = audit_module.validate_audit_policy(policy_document)

    expected_audit_log = (
        stage_root / "parent" / "audit" / "file-opens.jsonl"
    )
    audit_log_path = _regular_file(
        Path(policy["audit_log_path"]), noun="audit log"
    )
    if _path_key(audit_log_path) != _path_key(expected_audit_log):
        raise ValueError("audit log path is not the canonical parent-owned log")
    log_fd = _open_empty_append_descriptor(audit_log_path)
    boundary = None
    try:
        policy_sha256 = hashlib.sha256(policy_raw).hexdigest()
        boundary = audit_module.build_audit_boundary(
            policy,
            log_fd=log_fd,
            policy_sha256=policy_sha256,
        )
        sys.addaudithook(boundary)
        boundary.record_hook_installed()

        status = "error"
        try:
            sys.dont_write_bytecode = True
            sys.stdout.reconfigure(
                encoding="utf-8", errors="strict", newline="\n"
            )
            sys.stderr.reconfigure(
                encoding="utf-8", errors="strict", newline="\n"
            )
            _install_controlled_sys_path(dict(policy), code_root)
            sys.argv = [entrypoint, *child_arguments]
            runpy.run_path(str(entrypoint_path), run_name="__main__")
            status = "success"
        except SystemExit as error:
            status = "success" if error.code in (None, 0) else "error"
            raise
        finally:
            boundary.record_finished(status=status)
            os.fsync(log_fd)
    finally:
        os.close(log_fd)


def _real_directory(path: Path, *, noun: str) -> Path:
    lexical = Path(os.path.abspath(path))
    _reject_linked_ancestors(lexical, noun=noun)
    try:
        info = os.lstat(lexical)
    except OSError as error:
        raise ValueError(f"cannot inspect {noun}") from error
    if _is_link_or_reparse(info) or not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"{noun} must be a real directory")
    return lexical.resolve(strict=True)


def _regular_file(path: Path, *, noun: str) -> Path:
    lexical = Path(os.path.abspath(path))
    _reject_linked_ancestors(lexical, noun=noun)
    try:
        info = os.lstat(lexical)
    except OSError as error:
        raise ValueError(f"cannot inspect {noun}") from error
    if (
        _is_link_or_reparse(info)
        or not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 1
    ):
        raise ValueError(f"{noun} must be an unlinked regular file")
    return lexical.resolve(strict=True)


def _read_bounded_file(path: Path, *, max_bytes: int, noun: str) -> bytes:
    before = os.lstat(path)
    try:
        with path.open("rb") as stream:
            opened = os.fstat(stream.fileno())
            if _cross_stat_signature(before) != _cross_stat_signature(opened):
                raise ValueError(f"{noun} changed while being opened")
            raw = stream.read(max_bytes + 1)
            after_handle = os.fstat(stream.fileno())
    except OSError as error:
        raise ValueError(f"cannot read {noun}") from error
    if len(raw) > max_bytes:
        raise ValueError(f"{noun} exceeds byte bound")
    after_path = os.lstat(path)
    if (
        _stat_signature(opened) != _stat_signature(after_handle)
        or _stat_signature(before) != _stat_signature(after_path)
    ):
        raise ValueError(f"{noun} changed while being read")
    return raw


def _open_empty_append_descriptor(path: Path) -> int:
    before = os.lstat(path)
    if before.st_size != 0:
        raise ValueError("audit log must be empty before bootstrap")
    flags = (
        os.O_WRONLY
        | os.O_APPEND
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError("cannot safely open audit log") from error
    try:
        opened = os.fstat(descriptor)
        if _cross_stat_signature(before) != _cross_stat_signature(opened):
            raise ValueError("audit log changed while being opened")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _reject_linked_ancestors(path: Path, *, noun: str) -> None:
    current = Path(os.path.abspath(path))
    while True:
        if os.path.lexists(current):
            try:
                info = os.lstat(current)
            except OSError as error:
                raise ValueError(f"cannot inspect {noun}: {current}") from error
            if _is_link_or_reparse(info):
                raise ValueError(
                    f"{noun} contains a symlink or reparse point: {current}"
                )
        parent = current.parent
        if parent == current:
            return
        current = parent


def _is_link_or_reparse(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & _REPARSE_POINT
    )


def _path_is_within(candidate: Path, root: Path) -> bool:
    candidate_key = _path_key(candidate)
    root_key = _path_key(root)
    try:
        return os.path.commonpath((candidate_key, root_key)) == root_key
    except ValueError:
        return False


def _path_key(path: Path) -> str:
    return os.path.normcase(os.path.normpath(str(Path(path).absolute())))


def _stat_signature(
    info: os.stat_result,
) -> tuple[int, int, int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
        info.st_nlink,
    )


def _cross_stat_signature(
    info: os.stat_result,
) -> tuple[int, int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_nlink,
    )


def _unique_json_object(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, child in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key!r}")
        value[key] = child
    return value


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant rejected: {value}")


if __name__ == "__main__":
    main()
