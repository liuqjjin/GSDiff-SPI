"""Stdlib-only procedural audit boundary for strict method children.

This boundary governs CPython audit events emitted by Python and cooperating
stdlib code.  It is not an adversarial operating-system sandbox: native
extensions or direct native system calls can operate below ``sys.addaudithook``.
The parent therefore treats the log as procedural evidence, never as a claim
reported by the child.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import stat
import sys
import threading
from types import MappingProxyType


_SHA256 = re.compile(r"[0-9a-f]{64}\Z", flags=re.ASCII)
_UTC_TIMESTAMP = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z\Z",
    flags=re.ASCII,
)
_DISPLAY_PATH_SENTINEL = re.compile(
    r"<(?:file-descriptor:-?\d+|unsupported:[^<>:]+)>\Z",
    flags=re.ASCII,
)
_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_OPEN_ACCESS_MODE_MASK = getattr(
    os,
    "O_ACCMODE",
    os.O_WRONLY | os.O_RDWR,
)
_MUTATING_OPEN_FLAGS = (
    os.O_CREAT
    | os.O_TRUNC
    | os.O_APPEND
    | getattr(os, "O_EXCL", 0)
    | getattr(os, "O_TEMPORARY", 0)
    | getattr(os, "O_TMPFILE", 0)
)
_POLICY_KEYS = frozenset(
    {
        "schema",
        "audit_log_path",
        "exact_read_paths",
        "read_roots",
        "write_roots",
        "chdir_roots",
        "python_runtime_root",
        "windows_system_read_root",
        "runtime_site_package_roots",
        "logged_unrelated_events",
    }
)
_PROCESS_EVENTS = frozenset(
    {
        "subprocess.Popen",
        "os.system",
        "os.posix_spawn",
        "os.posix_spawnp",
        "os.exec",
        "os.spawn",
        "os.fork",
        "os.forkpty",
        "os.startfile",
        "os.startfile/2",
        "pty.spawn",
    }
)
_SINGLE_WRITE_EVENTS = frozenset(
    {
        "os.remove",
        "os.rmdir",
        "os.mkdir",
        "os.truncate",
        "os.chmod",
        "os.chown",
        "os.chflags",
        "os.utime",
        "os.setxattr",
        "os.removexattr",
    }
)
_SINGLE_WRITE_EVENT_ARITIES = MappingProxyType(
    {
        "os.remove": 2,
        "os.rmdir": 2,
        "os.mkdir": 3,
        "os.truncate": 2,
        "os.chmod": 3,
        "os.chown": 4,
        "os.chflags": 2,
        "os.utime": 4,
        "os.setxattr": 4,
        "os.removexattr": 2,
    }
)
_TWO_WRITE_EVENTS = frozenset({"os.rename"})
_ALWAYS_DENIED_LINK_EVENTS = frozenset({"os.link", "os.symlink"})
_DIRECT_WINDOWS_FILESYSTEM_EVENT_ARITIES = MappingProxyType(
    {
        "_winapi.CreateFile": 5,
        "_winapi.CreateJunction": 2,
        "_winapi.CopyFile2": 3,
    }
)
_SOCKET_EVENT_ARITIES = MappingProxyType(
    {
        "socket.__new__": 4,
        "socket.bind": 2,
        "socket.connect": 2,
        "socket.getaddrinfo": 5,
        "socket.gethostbyaddr": 1,
        "socket.gethostbyname": 1,
        "socket.gethostname": 0,
        "socket.getnameinfo": 1,
        "socket.getservbyname": 2,
        "socket.getservbyport": 2,
        "socket.sendto": 2,
    }
)
_ONE_PATH_EVENT_ARITIES = MappingProxyType(
    {
        "os.listdir": 1,
        "os.scandir": 1,
        "os.add_dll_directory": 1,
        "os.chdir": 1,
    }
)
_ALWAYS_DENIED_LINK_EVENT_ARITIES = MappingProxyType(
    {
        "os.link": 4,
        "os.symlink": 3,
    }
)
_ENVIRONMENT_EVENT_ARITIES = MappingProxyType(
    {
        "os.putenv": 2,
        "os.unsetenv": 1,
    }
)
_COPY_FILE2_ALLOWED_FLAGS = 0
_MAX_AUDIT_LOG_BYTES = 128 * 1024 * 1024
_BASE_AUDIT_FIELD_RULES = MappingProxyType(
    {
        "sequence": ("nonnegative-int",),
        "timestamp_utc": ("utc-timestamp",),
        "operation": ("nonempty-str",),
        "decision": ("decision",),
    }
)
_RESERVED_MALFORMED_PATHS = frozenset(
    {
        "<directory-descriptor>",
        "<invalid-access-intent>",
        "<invalid-arguments>",
        "<invalid-open-arguments>",
        "<invalid-two-path-operation>",
    }
)


def _audit_schema_variant(
    **additional_rules: tuple[object, ...],
) -> Mapping[str, tuple[object, ...]]:
    return MappingProxyType(
        {
            **_BASE_AUDIT_FIELD_RULES,
            **additional_rules,
        }
    )


def _build_audit_record_schemas() -> Mapping[
    tuple[str, str],
    tuple[Mapping[str, tuple[object, ...]], ...],
]:
    schemas: dict[
        tuple[str, str],
        tuple[Mapping[str, tuple[object, ...]], ...],
    ] = {}

    def register(
        operation: str,
        decision: str,
        *variants: Mapping[str, tuple[object, ...]],
    ) -> None:
        key = (operation, decision)
        if key in schemas or not variants:
            raise RuntimeError("duplicate or empty audit schema registration")
        schemas[key] = tuple(variants)

    path = ("path-display",)
    text = ("str",)
    nonempty = ("nonempty-str",)
    integer = ("int",)
    nonnegative_integer = ("nonnegative-int",)
    invalid_arguments = ("const", "<invalid-arguments>")

    register(
        "hook-installed",
        "allow",
        _audit_schema_variant(policy_sha256=("sha256",)),
    )
    register(
        "bootstrap-finished",
        "allow",
        _audit_schema_variant(status=("status",)),
    )
    register(
        "audit-reentry",
        "deny",
        _audit_schema_variant(reentered_operation=nonempty),
    )
    register(
        "audit-socket-poisoned",
        "deny",
        _audit_schema_variant(denied_operation=nonempty),
    )
    for operation, expected_arity in _SOCKET_EVENT_ARITIES.items():
        register(
            operation,
            "deny",
            _audit_schema_variant(
                expected_arity=("const", expected_arity),
                observed_arity=nonnegative_integer,
            ),
        )
    register(
        "socket-unknown",
        "deny",
        _audit_schema_variant(
            source_operation=("unknown-socket-operation",),
            observed_arity=nonnegative_integer,
        ),
    )
    register(
        "open",
        "allow",
        _audit_schema_variant(
            resolved_path=path,
            access=("access",),
        ),
    )
    register(
        "open",
        "deny",
        _audit_schema_variant(resolved_path=path),
        _audit_schema_variant(
            resolved_path=("const", "<invalid-open-arguments>")
        ),
    )

    one_path_operations = (
        set(_ONE_PATH_EVENT_ARITIES)
        | set(_SINGLE_WRITE_EVENTS)
    )
    for operation in sorted(one_path_operations):
        malformed = (
            "<directory-descriptor>"
            if operation in _SINGLE_WRITE_EVENTS
            else "<invalid-arguments>"
        )
        register(
            operation,
            "allow",
            _audit_schema_variant(resolved_path=path),
        )
        register(
            operation,
            "deny",
            _audit_schema_variant(resolved_path=path),
            _audit_schema_variant(
                resolved_path=("const", malformed)
            ),
        )

    for operation in sorted(_TWO_WRITE_EVENTS):
        register(
            operation,
            "allow",
            _audit_schema_variant(
                resolved_path=path,
                destination_path=path,
            ),
        )
        register(
            operation,
            "deny",
            _audit_schema_variant(
                resolved_path=path,
                destination_path=path,
            ),
            _audit_schema_variant(
                resolved_path=(
                    "const",
                    "<invalid-two-path-operation>",
                )
            ),
        )

    for operation in sorted(_ALWAYS_DENIED_LINK_EVENTS):
        register(
            operation,
            "deny",
            _audit_schema_variant(
                resolved_path=path,
                destination_path=path,
            ),
            _audit_schema_variant(resolved_path=invalid_arguments),
        )

    register(
        "_winapi.CopyFile2",
        "allow",
        _audit_schema_variant(
            resolved_path=path,
            destination_path=path,
            flags=("const", _COPY_FILE2_ALLOWED_FLAGS),
        ),
    )
    register(
        "_winapi.CopyFile2",
        "deny",
        _audit_schema_variant(
            resolved_path=path,
            destination_path=path,
            flags=integer,
        ),
        _audit_schema_variant(
            resolved_path=path,
            destination_path=path,
            flags=("const", "<invalid>"),
        ),
        _audit_schema_variant(resolved_path=invalid_arguments),
    )
    register(
        "_winapi.CreateFile",
        "deny",
        _audit_schema_variant(
            resolved_path=path,
            desired_access=integer,
            creation_disposition=integer,
        ),
        _audit_schema_variant(resolved_path=invalid_arguments),
        _audit_schema_variant(
            resolved_path=("const", "<invalid-access-intent>")
        ),
    )
    register(
        "_winapi.CreateJunction",
        "deny",
        _audit_schema_variant(
            resolved_path=path,
            destination_path=path,
        ),
        _audit_schema_variant(resolved_path=invalid_arguments),
    )

    for operation in sorted(_PROCESS_EVENTS):
        register(
            operation,
            "deny",
            _audit_schema_variant(command_class=("operation",)),
        )
    register(
        "process-unknown",
        "deny",
        _audit_schema_variant(
            source_operation=("unknown-process-operation",)
        ),
    )
    register(
        "os-unknown",
        "deny",
        _audit_schema_variant(
            source_operation=("unknown-os-operation",),
            resolved_path=text,
        ),
    )
    for operation in sorted(_ENVIRONMENT_EVENT_ARITIES):
        register(
            operation,
            "allow",
            _audit_schema_variant(),
        )
        register(
            operation,
            "deny",
            _audit_schema_variant(resolved_path=invalid_arguments),
        )
    return MappingProxyType(schemas)


_AUDIT_RECORD_SCHEMAS = _build_audit_record_schemas()


def _audit_field_matches(
    rule: tuple[object, ...],
    value: object,
    event: Mapping[str, object],
) -> bool:
    kind = rule[0]
    if kind == "str":
        return type(value) is str
    if kind == "nonempty-str":
        return type(value) is str and bool(value)
    if kind == "int":
        return type(value) is int
    if kind == "nonnegative-int":
        return type(value) is int and value >= 0
    if kind == "sha256":
        return type(value) is str and _SHA256.fullmatch(value) is not None
    if kind == "utc-timestamp":
        if (
            type(value) is not str
            or _UTC_TIMESTAMP.fullmatch(value) is None
        ):
            return False
        try:
            datetime(
                int(value[0:4]),
                int(value[5:7]),
                int(value[8:10]),
                int(value[11:13]),
                int(value[14:16]),
                int(value[17:19]),
                int(value[20:26]),
                tzinfo=timezone.utc,
            )
        except ValueError:
            return False
        return True
    if kind == "decision":
        return value in {"allow", "deny"}
    if kind == "status":
        return value in {"success", "error"}
    if kind == "access":
        return value in {"read", "write", "read-write"}
    if kind == "path-display":
        if (
            type(value) is not str
            or value in _RESERVED_MALFORMED_PATHS
        ):
            return False
        if value.startswith("<") or value.endswith(">"):
            return _DISPLAY_PATH_SENTINEL.fullmatch(value) is not None
        return True
    if kind == "operation":
        return (
            type(value) is str
            and value == event.get("operation")
        )
    if kind == "const":
        expected = rule[1]
        return type(value) is type(expected) and value == expected
    if kind == "unknown-socket-operation":
        return (
            type(value) is str
            and value.startswith("socket.")
            and value not in _SOCKET_EVENT_ARITIES
        )
    if kind == "unknown-process-operation":
        return (
            type(value) is str
            and value not in _PROCESS_EVENTS
            and _is_process_event(value)
        )
    if kind == "unknown-os-operation":
        return (
            type(value) is str
            and value.startswith("os.")
            and (value, "allow") not in _AUDIT_RECORD_SCHEMAS
            and (value, "deny") not in _AUDIT_RECORD_SCHEMAS
        )
    raise RuntimeError(f"unknown audit schema rule: {kind!r}")


def _validate_audit_event_schema(event: Mapping[str, object]) -> None:
    operation = event.get("operation")
    decision = event.get("decision")
    variants = _AUDIT_RECORD_SCHEMAS.get((operation, decision))
    if variants is None:
        raise ValueError("audit log record schema is unknown")
    for variant in variants:
        if set(event) != set(variant):
            continue
        if all(
            _audit_field_matches(rule, event[field], event)
            for field, rule in variant.items()
        ):
            return
    raise ValueError("audit log record schema mismatch")


def _canonical_audit_event_bytes(event: Mapping[str, object]) -> bytes:
    return json.dumps(
        event,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def validate_audit_policy(value: object) -> Mapping[str, object]:
    """Validate and freeze the exact policy consumed by the bootstrap."""
    if type(value) is not dict or set(value) != _POLICY_KEYS:
        raise ValueError("audit policy has an invalid top-level shape")
    if value["schema"] != "method-audit-policy-v1":
        raise ValueError("audit policy schema mismatch")
    _validate_json_native(value)
    scalar_paths = (
        "audit_log_path",
        "python_runtime_root",
        "windows_system_read_root",
    )
    for name in scalar_paths:
        _require_absolute_path(value[name], f"audit policy {name}")
    list_paths = (
        "exact_read_paths",
        "read_roots",
        "write_roots",
        "chdir_roots",
        "runtime_site_package_roots",
    )
    for name in list_paths:
        paths = value[name]
        if type(paths) is not list:
            raise ValueError(f"audit policy {name} must be a list")
        normalized: set[str] = set()
        for item in paths:
            path = _require_absolute_path(
                item, f"audit policy {name} entry"
            )
            key = os.path.normcase(os.path.normpath(path))
            if key in normalized:
                raise ValueError(f"audit policy {name} contains duplicates")
            normalized.add(key)
    if not value["exact_read_paths"]:
        raise ValueError("audit policy exact_read_paths must not be empty")
    if not value["read_roots"] or not value["write_roots"]:
        raise ValueError("audit policy roots must not be empty")
    logged = value["logged_unrelated_events"]
    if type(logged) is not list:
        raise ValueError(
            "audit policy logged_unrelated_events must be a list"
        )
    if logged:
        raise ValueError(
            "audit policy logged_unrelated_events must be empty"
        )
    runtime_root = str(value["python_runtime_root"])
    for site_root in value["runtime_site_package_roots"]:
        if not _path_is_within(str(site_root), runtime_root):
            raise ValueError(
                "runtime site-packages root escapes Python runtime"
            )
    return _freeze_json(value)


def build_audit_boundary(
    policy: Mapping[str, object],
    *,
    log_fd: int,
    policy_sha256: str,
) -> "_AuditBoundary":
    """Build a callable hook that writes only through a pre-opened descriptor."""
    if not isinstance(policy, Mapping):
        raise TypeError("audit policy must be a mapping")
    validated = validate_audit_policy(_thaw_json(policy))
    if type(log_fd) is not int or log_fd < 0:
        raise ValueError("audit log descriptor must be a nonnegative integer")
    if type(policy_sha256) is not str or _SHA256.fullmatch(
        policy_sha256
    ) is None:
        raise ValueError("audit policy hash must be canonical SHA-256")
    _disable_platform_hostname_lookup()
    return _AuditBoundary(validated, log_fd, policy_sha256)


class _AuditBoundary:
    def __init__(
        self,
        policy: Mapping[str, object],
        log_fd: int,
        policy_sha256: str,
    ) -> None:
        self._log_fd = log_fd
        self._policy_sha256 = policy_sha256
        self._sequence = 0
        self._lock = threading.Lock()
        self._local = threading.local()
        self._poisoned = False
        self._poisoned_event: str | None = None
        self._poisoned_kind: str | None = None
        self._exact_reads = tuple(
            str(path) for path in policy["exact_read_paths"]
        )
        self._read_roots = tuple(
            str(path) for path in policy["read_roots"]
        )
        self._write_roots = tuple(
            str(path) for path in policy["write_roots"]
        )
        self._chdir_roots = tuple(
            str(path) for path in policy["chdir_roots"]
        )

    def __call__(self, event: str, arguments: tuple[object, ...]) -> None:
        if getattr(self._local, "active", False):
            if _is_governed_event(event):
                self._poison_reentry(event)
                raise PermissionError(
                    f"audit policy denied governed re-entry: {event}"
                )
            return
        if self._poisoned:
            if _is_governed_event(event):
                raise PermissionError(
                    "audit boundary is poisoned after governed re-entry"
                )
            return
        self._local.active = True
        try:
            if event.startswith("socket."):
                self._audit_socket(event, arguments)
            elif event == "open":
                self._audit_open(arguments)
            elif event in {
                "os.listdir",
                "os.scandir",
                "os.add_dll_directory",
            }:
                if len(arguments) != _ONE_PATH_EVENT_ARITIES[event]:
                    self._deny_path(event, "<invalid-arguments>")
                raw = arguments[0]
                allowed, display = self._authorize_path(
                    raw,
                    exact_paths=(),
                    roots=self._read_roots,
                )
                self._decide_path(event, allowed, display)
            elif event == "os.chdir":
                if len(arguments) != _ONE_PATH_EVENT_ARITIES[event]:
                    self._deny_path(event, "<invalid-arguments>")
                raw = arguments[0]
                allowed, display = self._authorize_path(
                    raw, exact_paths=(), roots=self._chdir_roots
                )
                self._decide_path(event, allowed, display)
            elif event in _SINGLE_WRITE_EVENTS:
                raw = arguments[0] if arguments else object()
                if (
                    len(arguments) != _SINGLE_WRITE_EVENT_ARITIES[event]
                    or _uses_directory_descriptor(event, arguments)
                ):
                    self._deny_path(event, "<directory-descriptor>")
                allowed, display = self._authorize_path(
                    raw,
                    exact_paths=(),
                    roots=self._write_roots,
                    reject_unsafe_write_leaf=True,
                )
                self._decide_path(event, allowed, display)
            elif event in _TWO_WRITE_EVENTS:
                if len(arguments) != 4 or _uses_directory_descriptor(
                    event, arguments
                ):
                    self._deny_path(event, "<invalid-two-path-operation>")
                first_allowed, first = self._authorize_path(
                    arguments[0],
                    exact_paths=(),
                    roots=self._write_roots,
                    reject_unsafe_write_leaf=True,
                )
                second_allowed, second = self._authorize_path(
                    arguments[1],
                    exact_paths=(),
                    roots=self._write_roots,
                    reject_unsafe_write_leaf=True,
                )
                if first_allowed and second_allowed:
                    self.record(
                        event,
                        decision="allow",
                        resolved_path=first,
                        destination_path=second,
                    )
                else:
                    self.record(
                        event,
                        decision="deny",
                        resolved_path=first,
                        destination_path=second,
                    )
                    raise PermissionError(
                        f"audit policy denied {event}: {first} -> {second}"
                    )
            elif event in _ALWAYS_DENIED_LINK_EVENTS:
                if (
                    len(arguments)
                    != _ALWAYS_DENIED_LINK_EVENT_ARITIES[event]
                ):
                    self._deny_path(event, "<invalid-arguments>")
                first = _display_path(arguments[0])
                second = _display_path(arguments[1])
                self.record(
                    event,
                    decision="deny",
                    resolved_path=first,
                    destination_path=second,
                )
                raise PermissionError(f"audit policy denied {event}")
            elif event == "_winapi.CopyFile2":
                self._audit_copy_file2(arguments)
            elif event in _DIRECT_WINDOWS_FILESYSTEM_EVENT_ARITIES:
                if (
                    len(arguments)
                    != _DIRECT_WINDOWS_FILESYSTEM_EVENT_ARITIES[event]
                ):
                    self._deny_path(event, "<invalid-arguments>")
                first = _display_path(arguments[0])
                if event == "_winapi.CreateJunction":
                    second = _display_path(arguments[1])
                    self.record(
                        event,
                        decision="deny",
                        resolved_path=first,
                        destination_path=second,
                    )
                else:
                    desired_access = arguments[1]
                    creation_disposition = arguments[3]
                    if (
                        type(desired_access) is not int
                        or type(creation_disposition) is not int
                    ):
                        self._deny_path(event, "<invalid-access-intent>")
                    self.record(
                        event,
                        decision="deny",
                        resolved_path=first,
                        desired_access=desired_access,
                        creation_disposition=creation_disposition,
                    )
                raise PermissionError(f"audit policy denied {event}")
            elif event in _PROCESS_EVENTS:
                self.record(
                    event,
                    decision="deny",
                    command_class=event,
                )
                raise PermissionError(
                    f"audit policy denied nested process event {event}"
                )
            elif _is_process_event(event):
                self.record(
                    "process-unknown",
                    decision="deny",
                    source_operation=event,
                )
                raise PermissionError(
                    f"audit policy denied unknown process event {event}"
                )
            elif event in _ENVIRONMENT_EVENT_ARITIES:
                if len(arguments) != _ENVIRONMENT_EVENT_ARITIES[event]:
                    self._deny_path(event, "<invalid-arguments>")
                self.record(event, decision="allow")
            elif event.startswith("os."):
                raw = arguments[0] if arguments else object()
                display = _display_path(raw)
                self.record(
                    "os-unknown",
                    decision="deny",
                    source_operation=event,
                    resolved_path=display,
                )
                raise PermissionError(
                    f"audit policy denied unknown OS event {event}: "
                    f"{display}"
                )
            if self._poisoned:
                raise PermissionError(
                    "audit boundary is poisoned after governed re-entry"
                )
        finally:
            self._local.active = False

    def _poison_reentry(self, event: str) -> None:
        if not self._poisoned:
            self._poisoned_event = event
            self._poisoned_kind = "reentry"
            self._poisoned = True

    def _poison_socket(self, event: str) -> None:
        if not self._poisoned:
            self._poisoned_event = event
            self._poisoned_kind = "socket"
            self._poisoned = True

    def record(
        self,
        operation: str,
        *,
        decision: str,
        **fields: object,
    ) -> None:
        if type(operation) is not str or not operation:
            raise ValueError("audit operation must be a nonempty string")
        if decision not in {"allow", "deny"}:
            raise ValueError("audit decision must be allow or deny")
        with self._lock:
            event = {
                "sequence": self._sequence,
                "timestamp_utc": _utc_now(),
                "operation": operation,
                "decision": decision,
                **fields,
            }
            _validate_audit_event_schema(event)
            self._sequence += 1
            raw = _canonical_audit_event_bytes(event) + b"\n"
            _write_all(self._log_fd, raw)

    def record_hook_installed(self) -> None:
        self.record(
            "hook-installed",
            decision="allow",
            policy_sha256=self._policy_sha256,
        )

    def record_finished(self, *, status: str) -> None:
        if status not in {"success", "error"}:
            raise ValueError("bootstrap status must be success or error")
        if self._poisoned:
            poisoned_event = (
                self._poisoned_event
                if self._poisoned_event is not None
                else "<unknown>"
            )
            if self._poisoned_kind == "socket":
                self.record(
                    "audit-socket-poisoned",
                    decision="deny",
                    denied_operation=poisoned_event,
                )
            else:
                self.record(
                    "audit-reentry",
                    decision="deny",
                    reentered_operation=poisoned_event,
                )
            status = "error"
        self.record(
            "bootstrap-finished",
            decision="allow",
            status=status,
        )

    def _audit_socket(
        self,
        event: str,
        arguments: tuple[object, ...],
    ) -> None:
        expected_arity = _SOCKET_EVENT_ARITIES.get(event)
        if expected_arity is None:
            self.record(
                "socket-unknown",
                decision="deny",
                source_operation=event,
                observed_arity=len(arguments),
            )
        else:
            self.record(
                event,
                decision="deny",
                expected_arity=expected_arity,
                observed_arity=len(arguments),
            )
        self._poison_socket(event)
        raise PermissionError(f"audit policy denied socket event {event}")

    def _audit_open(self, arguments: tuple[object, ...]) -> None:
        if len(arguments) != 3:
            self._deny_path("open", "<invalid-open-arguments>")
        raw_path, mode, flags = arguments
        if type(raw_path) is int:
            self._deny_path(
                "open", f"<file-descriptor:{raw_path}>"
            )
        try:
            read_requested, write_requested = _open_access(mode, flags)
        except ValueError:
            self._deny_path("open", _display_path(raw_path))
        displays: list[str] = []
        allowed = True
        if read_requested:
            read_allowed, display = self._authorize_path(
                raw_path,
                exact_paths=self._exact_reads,
                roots=self._read_roots,
            )
            displays.append(display)
            allowed = allowed and read_allowed
        if write_requested:
            write_allowed, display = self._authorize_path(
                raw_path,
                exact_paths=(),
                roots=self._write_roots,
                reject_unsafe_write_leaf=True,
            )
            displays.append(display)
            allowed = allowed and write_allowed
        if not read_requested and not write_requested:
            allowed = False
            displays.append(_display_path(raw_path))
        display = displays[-1]
        if allowed:
            self.record(
                "open",
                decision="allow",
                resolved_path=display,
                access=(
                    "read-write"
                    if read_requested and write_requested
                    else "write"
                    if write_requested
                    else "read"
                ),
            )
            return
        self._deny_path("open", display)

    def _audit_copy_file2(
        self,
        arguments: tuple[object, ...],
    ) -> None:
        event = "_winapi.CopyFile2"
        if len(arguments) != _DIRECT_WINDOWS_FILESYSTEM_EVENT_ARITIES[event]:
            self._deny_path(event, "<invalid-arguments>")
        raw_source, raw_destination, flags = arguments
        if (
            type(raw_source) is not str
            or type(raw_destination) is not str
            or type(flags) is not int
            or flags != _COPY_FILE2_ALLOWED_FLAGS
        ):
            source = (
                raw_source
                if type(raw_source) is str
                else f"<unsupported:{type(raw_source).__name__}>"
            )
            destination = (
                raw_destination
                if type(raw_destination) is str
                else f"<unsupported:{type(raw_destination).__name__}>"
            )
            self.record(
                event,
                decision="deny",
                resolved_path=source,
                destination_path=destination,
                flags=flags if type(flags) is int else "<invalid>",
            )
            raise PermissionError(f"audit policy denied {event}")
        read_allowed, source = self._authorize_path(
            raw_source,
            exact_paths=self._exact_reads,
            roots=self._read_roots,
        )
        write_allowed, destination = self._authorize_path(
            raw_destination,
            exact_paths=(),
            roots=self._write_roots,
            reject_unsafe_write_leaf=True,
        )
        allowed = (
            read_allowed
            and write_allowed
            and _is_regular_single_link_file(source)
            and _is_safe_copy_destination(destination)
        )
        self.record(
            event,
            decision="allow" if allowed else "deny",
            resolved_path=source,
            destination_path=destination,
            flags=flags,
        )
        if not allowed:
            raise PermissionError(
                f"audit policy denied {event}: {source} -> {destination}"
            )

    def _authorize_path(
        self,
        raw_path: object,
        *,
        exact_paths: tuple[str, ...],
        roots: tuple[str, ...],
        reject_unsafe_write_leaf: bool = False,
    ) -> tuple[bool, str]:
        try:
            candidate, traversed = _lexical_path(raw_path)
        except (OSError, TypeError, ValueError):
            return False, _display_path(raw_path)
        candidate_text = str(candidate)
        if not _matches_policy_path(
            candidate_text,
            exact_paths=exact_paths,
            roots=roots,
        ):
            # Do not stat or resolve an already-denied external target.  This
            # is important for forbidden network/workspace paths.
            return False, candidate_text
        try:
            _reject_reparse_traversal(traversed)
            if reject_unsafe_write_leaf:
                _reject_unsafe_write_leaf(candidate)
            resolved = str(Path(candidate_text).resolve(strict=False))
        except (OSError, RuntimeError, ValueError):
            return False, candidate_text
        return (
            _matches_policy_path(
                resolved, exact_paths=exact_paths, roots=roots
            ),
            resolved,
        )

    def _decide_path(
        self, operation: str, allowed: bool, display: str
    ) -> None:
        if allowed:
            self.record(
                operation,
                decision="allow",
                resolved_path=display,
            )
            return
        self._deny_path(operation, display)

    def _deny_path(self, operation: str, display: str) -> None:
        self.record(
            operation,
            decision="deny",
            resolved_path=display,
        )
        raise PermissionError(
            f"audit policy denied {operation}: {display}"
        )


def validate_audit_log(
    path: Path,
    *,
    expected_policy_sha256: str,
) -> Mapping[str, object]:
    """Parse and independently validate a completed child audit log."""
    if not isinstance(path, Path):
        raise TypeError("audit log path must be a Path")
    if (
        type(expected_policy_sha256) is not str
        or _SHA256.fullmatch(expected_policy_sha256) is None
    ):
        raise ValueError("expected policy hash must be canonical SHA-256")
    raw = _read_regular_unlinked_file(
        path,
        max_bytes=_MAX_AUDIT_LOG_BYTES,
        noun="audit log",
    )
    if not raw or not raw.endswith(b"\n"):
        raise ValueError("audit log is empty or truncated")
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError("audit log is not strict UTF-8") from error
    raw_lines = text.splitlines()
    if not raw_lines or any(not line for line in raw_lines):
        raise ValueError("audit log contains an empty record")
    events: list[dict[str, object]] = []
    for line in raw_lines:
        try:
            value = json.loads(
                line,
                object_pairs_hook=_unique_json_object,
                parse_constant=_reject_json_constant,
            )
        except (json.JSONDecodeError, UnicodeError, ValueError) as error:
            raise ValueError("audit log contains malformed JSON") from error
        if type(value) is not dict:
            raise ValueError("audit log event must be an exact object")
        if line.encode("utf-8") != _canonical_audit_event_bytes(value):
            raise ValueError(
                "audit log record is not canonical JSON bytes"
            )
        events.append(value)
    for expected_sequence, event in enumerate(events):
        _validate_audit_event_schema(event)
        if event["sequence"] != expected_sequence:
            raise ValueError(
                "audit log sequence is duplicate, missing, or out of order"
            )
    headers = [
        event
        for event in events
        if event.get("operation") == "hook-installed"
    ]
    first = events[0]
    if (
        len(headers) != 1
        or first is not headers[0]
        or first.get("decision") != "allow"
        or first.get("policy_sha256") != expected_policy_sha256
    ):
        raise ValueError("audit log policy hash or header mismatch")
    terminal = [
        event
        for event in events
        if event.get("operation") == "bootstrap-finished"
    ]
    if (
        len(terminal) != 1
        or events[-1] is not terminal[0]
        or terminal[0].get("decision") != "allow"
        or terminal[0].get("status") != "success"
    ):
        raise ValueError("audit log terminal event is missing or unsuccessful")
    if any(event.get("decision") == "deny" for event in events):
        raise ValueError("audit log contains a denied event")
    return MappingProxyType(
        {
            "schema": "validated-method-audit-log-v1",
            "policy_sha256": expected_policy_sha256,
            "audit_log_sha256": hashlib.sha256(raw).hexdigest(),
            "event_count": len(events),
            "terminal_status": "success",
        }
    )


def _open_access(mode: object, flags: object) -> tuple[bool, bool]:
    if mode is not None:
        if type(mode) is not str or not mode:
            raise ValueError("open mode is invalid")
        write = any(marker in mode for marker in "wax+")
        read = mode.startswith("r") or "+" in mode
        if not read and not write:
            raise ValueError("open mode has no governed access")
        return read, write
    if type(flags) is not int:
        raise ValueError("open flags are invalid")
    access_mode = flags & _OPEN_ACCESS_MODE_MASK
    read = access_mode in {os.O_RDONLY, os.O_RDWR}
    write = access_mode in {os.O_WRONLY, os.O_RDWR} or bool(
        flags & _MUTATING_OPEN_FLAGS
    )
    return read, write


def _disable_platform_hostname_lookup() -> None:
    """Keep Windows platform probes deterministic and audit-safe."""

    def no_hostname(default: str = "") -> str:
        return default

    def no_system_command(
        system: str = "",
        release: str = "",
        version: str = "",
        supported_platforms: object = (),
    ) -> tuple[str, str, str]:
        del supported_platforms
        return system, release, version

    platform._uname_cache = None
    platform._node = no_hostname
    platform._syscmd_ver = no_system_command
    if os.name != "nt":
        return

    runtime = sys.version.casefold()
    machine = (
        "AMD64"
        if "amd64" in runtime
        else "ARM64"
        if "(arm64)" in runtime
        else "ARM"
        if "(arm)" in runtime
        else ""
    )
    sanitized = platform.uname_result(
        "Windows",
        "",
        "",
        "",
        machine,
    )
    sanitized.__dict__["processor"] = ""

    def sanitized_uname() -> object:
        return sanitized

    platform._uname_cache = sanitized
    platform.uname = sanitized_uname


def _lexical_path(raw_path: object) -> tuple[Path, tuple[Path, ...]]:
    if isinstance(raw_path, int):
        raise TypeError("file descriptors are not paths")
    if isinstance(raw_path, os.PathLike):
        raw_path = os.fspath(raw_path)
    if type(raw_path) not in {str, bytes}:
        raise TypeError("unsupported path type")
    text = os.fsdecode(raw_path)
    if "\x00" in text:
        raise ValueError("path contains NUL")
    drive, _ = os.path.splitdrive(text)
    if drive and not os.path.isabs(text):
        raise ValueError("drive-relative paths are unsupported")
    combined = text if os.path.isabs(text) else os.path.join(os.getcwd(), text)
    drive, tail = os.path.splitdrive(combined)
    if os.name == "nt":
        separator_pattern = r"[\\/]+"
        anchor = drive + os.sep
    else:
        separator_pattern = r"/+"
        anchor = os.sep
    if not os.path.isabs(combined):
        raise ValueError("path did not become absolute")
    current = Path(anchor)
    traversed: list[Path] = [current]
    for component in re.split(separator_pattern, tail.lstrip("\\/")):
        if not component or component == ".":
            continue
        if os.name == "nt" and ":" in component:
            raise ValueError(
                "Windows alternate data stream path component rejected"
            )
        if component == "..":
            current = current.parent
            traversed.append(current)
            continue
        current = current / component
        traversed.append(current)
    return Path(os.path.normpath(str(current))), tuple(traversed)


def _reject_reparse_traversal(paths: tuple[Path, ...]) -> None:
    for path in paths:
        if not os.path.lexists(path):
            continue
        try:
            info = os.lstat(path)
        except OSError as error:
            raise ValueError(f"cannot inspect lexical path: {path}") from error
        if stat.S_ISLNK(info.st_mode) or bool(
            getattr(info, "st_file_attributes", 0) & _REPARSE_POINT
        ):
            raise ValueError(
                f"lexical path contains a symlink or reparse point: {path}"
            )


def _reject_unsafe_write_leaf(path: Path) -> None:
    if not os.path.lexists(path):
        return
    try:
        info = os.lstat(path)
    except OSError as error:
        raise ValueError(f"cannot inspect write target: {path}") from error
    if stat.S_ISDIR(info.st_mode):
        return
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError(
            f"write target must be a single-link regular file: {path}"
        )


def _is_regular_single_link_file(path: str) -> bool:
    try:
        info = os.lstat(path)
    except OSError:
        return False
    return stat.S_ISREG(info.st_mode) and info.st_nlink == 1


def _is_safe_copy_destination(path: str) -> bool:
    if not os.path.lexists(path):
        return True
    return _is_regular_single_link_file(path)


def _matches_policy_path(
    candidate: str,
    *,
    exact_paths: tuple[str, ...],
    roots: tuple[str, ...],
) -> bool:
    normalized = os.path.normcase(os.path.normpath(candidate))
    if any(
        normalized == os.path.normcase(os.path.normpath(path))
        for path in exact_paths
    ):
        return True
    return any(_path_is_within(candidate, root) for root in roots)


def _path_is_within(candidate: str, root: str) -> bool:
    candidate_key = os.path.normcase(os.path.normpath(candidate))
    root_key = os.path.normcase(os.path.normpath(root))
    try:
        return os.path.commonpath((candidate_key, root_key)) == root_key
    except ValueError:
        return False


def _uses_directory_descriptor(
    event: str, arguments: tuple[object, ...]
) -> bool:
    indexes = {
        "os.remove": (1,),
        "os.rmdir": (1,),
        "os.mkdir": (2,),
        "os.chmod": (2,),
        "os.chown": (3,),
        "os.utime": (3,),
        "os.rename": (2, 3),
    }.get(event, ())
    return any(
        index < len(arguments)
        and arguments[index] not in {None, -1}
        for index in indexes
    )


def _is_process_event(event: str) -> bool:
    lowered = event.lower()
    return (
        event in _PROCESS_EVENTS
        or event.startswith("subprocess.")
        or event.startswith("os.spawn")
        or event.startswith("os.exec")
        or event.startswith("os.fork")
        or "createprocess" in lowered
    )


def _is_governed_event(event: str) -> bool:
    return (
        event == "open"
        or event.startswith("os.")
        or event.startswith("socket.")
        or event in _DIRECT_WINDOWS_FILESYSTEM_EVENT_ARITIES
        or _is_process_event(event)
    )


def _display_path(value: object) -> str:
    if type(value) is int:
        return f"<file-descriptor:{value}>"
    try:
        return os.fsdecode(os.fspath(value))
    except (TypeError, ValueError, OSError):
        return f"<unsupported:{type(value).__name__}>"


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise RuntimeError("audit log descriptor stopped accepting bytes")
        view = view[written:]


def _require_absolute_path(value: object, noun: str) -> str:
    if (
        type(value) is not str
        or not value
        or "\x00" in value
        or not os.path.isabs(value)
    ):
        raise ValueError(f"{noun} must be an absolute exact path")
    return value


def _validate_json_native(value: object, path: str = "$") -> None:
    if value is None or type(value) in {str, bool, int}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number")
        return
    if type(value) is list:
        for index, child in enumerate(value):
            _validate_json_native(child, f"{path}[{index}]")
        return
    if type(value) is dict:
        for key, child in value.items():
            if type(key) is not str:
                raise ValueError(f"{path} contains a non-string key")
            _validate_json_native(child, f"{path}.{key}")
        return
    raise ValueError(f"{path} contains a non-JSON-native value")


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


def _read_regular_unlinked_file(
    path: Path,
    *,
    max_bytes: int,
    noun: str,
) -> bytes:
    lexical = Path(os.path.abspath(path))
    _, traversed = _lexical_path(str(lexical))
    _reject_reparse_traversal(traversed)
    try:
        before = os.lstat(lexical)
    except OSError as error:
        raise ValueError(f"cannot inspect {noun}") from error
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_size > max_bytes
    ):
        raise ValueError(f"{noun} must be a bounded unlinked regular file")
    try:
        with lexical.open("rb") as stream:
            opened = os.fstat(stream.fileno())
            if _cross_stat_signature(before) != _cross_stat_signature(opened):
                raise ValueError(f"{noun} changed while being opened")
            raw = stream.read(max_bytes + 1)
            after_handle = os.fstat(stream.fileno())
    except OSError as error:
        raise ValueError(f"cannot read {noun}") from error
    if len(raw) > max_bytes:
        raise ValueError(f"{noun} exceeds byte bound")
    if _stat_signature(opened) != _stat_signature(after_handle):
        raise ValueError(f"{noun} changed while being read")
    try:
        after_path = os.lstat(lexical)
    except OSError as error:
        raise ValueError(f"cannot re-inspect {noun}") from error
    if _stat_signature(before) != _stat_signature(after_path):
        raise ValueError(f"{noun} path changed while being read")
    return raw


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


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                str(key): _freeze_json(child)
                for key, child in value.items()
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(child) for child in value)
    return value


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _thaw_json(child)
            for key, child in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_thaw_json(child) for child in value]
    return value
