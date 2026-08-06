"""Preflight or train the fixed diffusion-prior-v2 candidate."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gsdiff.prior.training_v2 import training_cli  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Preflight or train the fixed diffusion prior v2 candidate."
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="run the required real-CUDA one-batch cycle without durable writes",
    )
    args = parser.parse_args()
    return training_cli(preflight_only=args.preflight_only)


if __name__ == "__main__":
    raise SystemExit(main())
