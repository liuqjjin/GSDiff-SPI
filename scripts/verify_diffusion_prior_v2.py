"""Independently verify and publish diffusion-prior-v2 provenance."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gsdiff.prior.training_v2 import verification_cli  # noqa: E402


def main() -> int:
    argparse.ArgumentParser(
        description="Independently verify the fixed diffusion prior v2 candidate."
    ).parse_args()
    return verification_cli()


if __name__ == "__main__":
    raise SystemExit(main())
