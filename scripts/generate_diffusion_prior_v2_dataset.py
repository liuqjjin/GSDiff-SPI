"""Generate or strictly reuse the fixed diffusion-prior-v2 dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gsdiff.prior.training_v2 import dataset_cli  # noqa: E402


def main() -> int:
    argparse.ArgumentParser(
        description="Generate the fixed target-disjoint diffusion prior v2 dataset."
    ).parse_args()
    return dataset_cli()


if __name__ == "__main__":
    raise SystemExit(main())
