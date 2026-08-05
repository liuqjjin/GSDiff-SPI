"""Deprecated translation wrapper for versioned experiment campaigns."""

import sys

if __name__ == "__main__" and not sys.flags.isolated:
    sys.stderr.write(
        "campaign execution refused: isolated-python-required\n"
    )
    raise SystemExit(2)

from pathlib import Path
import os

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

from scripts.run_eval_matrix import main


if __name__ == "__main__":
    raise SystemExit(main())
