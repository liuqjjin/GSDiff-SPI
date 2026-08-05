"""Deprecated translation wrapper for versioned experiment campaigns."""

from __future__ import annotations

import sys

if __name__ == "__main__" and not sys.flags.isolated:
    sys.stderr.write(
        "campaign execution refused: isolated-python-required\n"
    )
    raise SystemExit(2)

from pathlib import Path
import re
import os
import subprocess

REPO_ROOT = Path(__file__).resolve().parents[1]
_TRUSTED_ROOT = str(REPO_ROOT)
_TRUSTED_KEY = os.path.normcase(_TRUSTED_ROOT)
_RETAINED_PATHS = []
for _candidate in sys.path:
    try:
        _candidate_key = os.path.normcase(
            str(Path(_candidate or os.curdir).resolve())
        )
    except OSError:
        _RETAINED_PATHS.append(_candidate)
        continue
    if _candidate_key != _TRUSTED_KEY:
        _RETAINED_PATHS.append(_candidate)
sys.path[:] = [_TRUSTED_ROOT, *_RETAINED_PATHS]


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    print(
        "DEPRECATED: use scripts/experiments/run_campaign.py --protocol <versioned-yaml>",
        file=sys.stderr,
    )
    if "--campaign" in arguments:
        if "--protocol" in arguments:
            print("deprecated wrapper accepts one campaign selector", file=sys.stderr)
            return 2
        try:
            index = arguments.index("--campaign")
            campaign_id = arguments[index + 1]
        except (IndexError, ValueError):
            print("deprecated wrapper requires a versioned --campaign", file=sys.stderr)
            return 2
        if re.fullmatch(r"[a-z0-9][a-z0-9_-]*-v[0-9]+", campaign_id) is None:
            print("deprecated wrapper requires a versioned --campaign", file=sys.stderr)
            return 2
        arguments[index : index + 2] = [
            "--protocol",
            str(REPO_ROOT / "configs/protocols" / f"{campaign_id}.yaml"),
        ]
    if arguments.count("--protocol") != 1:
        print("deprecated wrapper requires --protocol", file=sys.stderr)
        return 2
    try:
        value = arguments[arguments.index("--protocol") + 1]
    except (IndexError, ValueError):
        print("deprecated wrapper requires --protocol", file=sys.stderr)
        return 2
    if re.fullmatch(r"[a-z0-9][a-z0-9_-]*-v[0-9]+\.yaml", Path(value).name) is None:
        print("deprecated wrapper accepts only a versioned --protocol", file=sys.stderr)
        return 2
    allowed = {
        "--protocol",
        "--artifact-root",
        "--device",
        "--checkpoint",
        "--minimum-free-bytes",
    }
    if any(item.startswith("--") and item not in allowed for item in arguments):
        print("deprecated wrapper rejects free-form scientific arguments; use --protocol", file=sys.stderr)
        return 2
    completed = subprocess.run(
        [
            str(Path(sys.executable).resolve()),
            "-I",
            "-B",
            "-X",
            "utf8",
            str(
                (
                    REPO_ROOT
                    / "scripts"
                    / "experiments"
                    / "run_campaign.py"
                ).resolve()
            ),
            *arguments,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    sys.stdout.write(completed.stdout)
    sys.stderr.write(completed.stderr)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
