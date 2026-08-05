"""Normalized read/write path-role separation for experiment CLIs."""

from __future__ import annotations

from dataclasses import dataclass
import errno
from itertools import combinations
import os
from pathlib import Path
from typing import Literal


class PathRoleError(ValueError):
    """One or more CLI paths do not form disjoint read/write roles."""


@dataclass(frozen=True)
class PathRole:
    """One named filesystem path and its intended access role."""

    name: str
    path: Path
    access: Literal["read", "write"]

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name:
            raise TypeError("path role name must be a nonempty exact string")
        if not isinstance(self.path, Path):
            raise TypeError("path role path must be a Path")
        if self.access not in {"read", "write"}:
            raise ValueError("path role access must be exactly read or write")


def require_disjoint_path_roles(*roles: PathRole) -> None:
    """Reject normalized overlap whenever at least one path is writable.

    Read-only aliases are allowed. A write role must be distinct from every
    other role at both lexical/real-path and existing-file identity levels.
    This is an argument-contract check, not a handle-bound race defence.
    """

    if not roles:
        raise TypeError("at least one path role is required")
    if any(type(role) is not PathRole for role in roles):
        raise TypeError("path roles must be exact PathRole values")
    names = [role.name for role in roles]
    if len(names) != len(set(names)):
        raise PathRoleError("path role names must be unique")
    normalized = {
        role.name: _normalized_path(role.path, noun=role.name) for role in roles
    }
    for left, right in combinations(roles, 2):
        if left.access == right.access == "read":
            continue
        if _paths_overlap(
            left.path,
            normalized[left.name],
            right.path,
            normalized[right.name],
        ):
            raise PathRoleError(
                "path role overlap: "
                f"{left.name} ({left.access}) and "
                f"{right.name} ({right.access})"
            )


def _normalized_path(path: Path, *, noun: str) -> Path:
    try:
        resolved = path.resolve(strict=False)
    except (OSError, RuntimeError) as error:
        raise PathRoleError(f"cannot normalize path role {noun}") from error
    normalized = os.path.normcase(os.path.normpath(str(resolved)))
    result = Path(normalized)
    if not result.is_absolute():
        raise PathRoleError(f"path role {noun} did not normalize to an absolute path")
    return result


def _paths_overlap(
    left_original: Path,
    left: Path,
    right_original: Path,
    right: Path,
) -> bool:
    if left == right or left in right.parents or right in left.parents:
        return True
    return _same_existing_file(left_original, right_original)


def _same_existing_file(left: Path, right: Path) -> bool:
    try:
        return os.path.samefile(left, right)
    except OSError as error:
        if isinstance(error, FileNotFoundError) or error.errno in {
            errno.ENOENT,
            errno.ENOTDIR,
        }:
            return False
        raise PathRoleError("cannot compare existing path-role identities") from error


__all__ = ["PathRole", "PathRoleError", "require_disjoint_path_roles"]
