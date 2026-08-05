"""Shared lexical policy for path components consumed on Windows."""

from __future__ import annotations


_RESERVED_WINDOWS_COMPONENT_STEMS = frozenset(
    {"con", "prn", "aux", "nul", "conin$", "conout$"}
    | {f"com{number}" for number in range(1, 10)}
    | {f"lpt{number}" for number in range(1, 10)}
    | {f"com{suffix}" for suffix in "¹²³"}
    | {f"lpt{suffix}" for suffix in "¹²³"}
)


def windows_component_collision_key(component: str) -> str:
    """Validate one Win32 path component and return its collision key."""
    if type(component) is not str or not component:
        raise ValueError("path contains an empty Windows component")
    if component.rstrip(" .") != component:
        raise ValueError("path has an ambiguous Windows component")
    if any(
        ord(character) < 32 or character in '<>:"|?*'
        for character in component
    ):
        raise ValueError("path contains a Windows-unsafe component")
    normalized = component.casefold()
    stem = normalized.split(".", 1)[0]
    if stem in _RESERVED_WINDOWS_COMPONENT_STEMS:
        raise ValueError("path contains a reserved Windows component")
    return normalized
