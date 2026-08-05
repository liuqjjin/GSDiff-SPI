"""Handle-bound cleanup for a recursively inventoried private Windows tree."""

from __future__ import annotations

from dataclasses import dataclass
import ctypes
import os
from pathlib import Path
import stat

from .execution import _reject_linked_ancestors, _verify_directory_identity


if os.name == "nt":
    from ctypes import wintypes

    class _WindowsFileId128(ctypes.Structure):
        _fields_ = [("identifier", ctypes.c_ubyte * 16)]

    class _WindowsFileIdInfo(ctypes.Structure):
        _fields_ = [
            ("volume_serial_number", ctypes.c_ulonglong),
            ("file_id", _WindowsFileId128),
        ]

    class _WindowsDispositionInfoEx(ctypes.Structure):
        _fields_ = [("flags", wintypes.DWORD)]


@dataclass(frozen=True)
class _PinnedTreeEntry:
    path: Path
    device: int
    inode: int
    directory: bool


@dataclass
class _PinnedWindowsDeleteHandle:
    entry: _PinnedTreeEntry
    handle: object
    kernel32: object


def cleanup_pinned_owned_tree(identity, *, noun: str) -> None:
    """Delete only the exact pinned tree, or retain everything on mismatch."""
    if os.name != "nt":
        raise ValueError(
            "handle-bound owned-tree cleanup is supported only on Windows"
        )
    _verify_directory_identity(identity, noun=noun)
    entries = _capture_pinned_tree_entries(
        identity.path,
        expected_device=identity.device,
        expected_inode=identity.inode,
        noun=noun,
    )
    _verify_directory_identity(identity, noun=noun)
    root_entry = _PinnedTreeEntry(
        identity.path,
        identity.device,
        identity.inode,
        True,
    )
    pinned: list[_PinnedWindowsDeleteHandle] = []
    try:
        for entry in (*entries, root_entry):
            pinned.append(_open_windows_delete_handle(entry, noun=noun))
        _pinned_delete_barrier(tuple(item.entry for item in pinned))
        for item in pinned:
            _verify_windows_delete_handle(item, noun=noun)
        for item in pinned:
            _dispose_windows_delete_handle(item)
            _close_windows_delete_handle(item)
    finally:
        for item in pinned:
            _close_windows_delete_handle(item)
    if os.path.lexists(identity.path):
        raise ValueError(f"{noun} remains after handle-bound deletion")


def _pinned_delete_barrier(entries: tuple[_PinnedTreeEntry, ...]) -> None:
    """Private deterministic hook after all delete handles are pinned."""


def _open_windows_delete_handle(
    entry: _PinnedTreeEntry,
    *,
    noun: str,
) -> _PinnedWindowsDeleteHandle:
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
    flags = 0x00200000
    if entry.directory:
        flags |= 0x02000000
    handle = create_file(
        str(entry.path),
        0x00010000 | 0x00000080,
        0x1,
        None,
        3,
        flags,
        None,
    )
    invalid = ctypes.c_void_p(-1).value
    if handle in (None, invalid):
        raise ValueError(f"cannot pin {noun} entry for deletion") from ctypes.WinError(
            ctypes.get_last_error()
        )
    pinned = _PinnedWindowsDeleteHandle(entry, handle, kernel32)
    try:
        _verify_windows_delete_handle(pinned, noun=noun)
    except BaseException:
        _close_windows_delete_handle(pinned)
        raise
    return pinned


def _verify_windows_delete_handle(
    pinned: _PinnedWindowsDeleteHandle,
    *,
    noun: str,
) -> None:
    get_information = pinned.kernel32.GetFileInformationByHandleEx
    get_information.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_uint32,
    ]
    get_information.restype = ctypes.c_int
    file_id_info = _WindowsFileIdInfo()
    if not get_information(
        pinned.handle,
        18,
        ctypes.byref(file_id_info),
        ctypes.sizeof(file_id_info),
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    handle_identity = (
        int(file_id_info.volume_serial_number),
        int.from_bytes(
            bytes(file_id_info.file_id.identifier),
            byteorder="little",
        ),
    )
    expected = (pinned.entry.device, pinned.entry.inode)
    if handle_identity != expected:
        raise ValueError(f"{noun} entry handle identity changed")
    try:
        path_info = os.lstat(pinned.entry.path)
    except OSError as error:
        raise ValueError(f"{noun} entry path changed while pinned") from error
    if (
        (path_info.st_dev, path_info.st_ino) != expected
        or stat.S_ISDIR(path_info.st_mode) != pinned.entry.directory
    ):
        raise ValueError(f"{noun} entry path changed while pinned")


def _dispose_windows_delete_handle(
    pinned: _PinnedWindowsDeleteHandle,
) -> None:
    set_information = pinned.kernel32.SetFileInformationByHandle
    set_information.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_uint32,
    ]
    set_information.restype = ctypes.c_int
    disposition = _WindowsDispositionInfoEx(flags=0x1 | 0x10)
    if not set_information(
        pinned.handle,
        21,
        ctypes.byref(disposition),
        ctypes.sizeof(disposition),
    ):
        raise ctypes.WinError(ctypes.get_last_error())


def _close_windows_delete_handle(pinned: _PinnedWindowsDeleteHandle) -> None:
    if pinned.handle is not None:
        handle = pinned.handle
        pinned.handle = None
        pinned.kernel32.CloseHandle(handle)


def _capture_pinned_tree_entries(
    root: Path,
    *,
    expected_device: int,
    expected_inode: int,
    noun: str,
) -> tuple[_PinnedTreeEntry, ...]:
    _reject_linked_ancestors(root, noun=noun)
    root_info = os.lstat(root)
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    if (
        not stat.S_ISDIR(root_info.st_mode)
        or stat.S_ISLNK(root_info.st_mode)
        or getattr(root_info, "st_file_attributes", 0) & reparse
        or (root_info.st_dev, root_info.st_ino)
        != (expected_device, expected_inode)
    ):
        raise ValueError(f"{noun} identity changed before recursive cleanup")
    captured: list[_PinnedTreeEntry] = []

    def visit(directory: Path) -> None:
        before = os.lstat(directory)
        if (
            not stat.S_ISDIR(before.st_mode)
            or stat.S_ISLNK(before.st_mode)
            or getattr(before, "st_file_attributes", 0) & reparse
            or before.st_dev != expected_device
        ):
            raise ValueError(
                f"{noun} contains a linked or foreign directory"
            )
        with os.scandir(directory) as iterator:
            names = sorted(entry.name for entry in iterator)
        for name in names:
            child = directory / name
            _reject_linked_ancestors(child, noun=noun)
            info = os.lstat(child)
            if (
                stat.S_ISLNK(info.st_mode)
                or getattr(info, "st_file_attributes", 0) & reparse
                or info.st_dev != expected_device
            ):
                raise ValueError(
                    f"{noun} contains a linked or reparse entry"
                )
            if stat.S_ISDIR(info.st_mode):
                visit(child)
                current = os.lstat(child)
                if (current.st_dev, current.st_ino) != (
                    info.st_dev,
                    info.st_ino,
                ):
                    raise ValueError(
                        f"{noun} directory changed during inventory"
                    )
                captured.append(
                    _PinnedTreeEntry(child, info.st_dev, info.st_ino, True)
                )
            elif stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
                captured.append(
                    _PinnedTreeEntry(child, info.st_dev, info.st_ino, False)
                )
            else:
                raise ValueError(
                    f"{noun} contains a non-regular or multiply-linked entry"
                )
        after = os.lstat(directory)
        if (after.st_dev, after.st_ino) != (
            before.st_dev,
            before.st_ino,
        ):
            raise ValueError(f"{noun} directory changed during inventory")

    visit(root)
    final_root = os.lstat(root)
    if (final_root.st_dev, final_root.st_ino) != (
        expected_device,
        expected_inode,
    ):
        raise ValueError(f"{noun} identity changed during inventory")
    return tuple(captured)
