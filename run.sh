#!/bin/bash
# Reconstruction (default = ADMM + diffusion prior at the tuned operating point)
python train.py --config configs/default.yaml

# SGD baseline
python train.py --config configs/default.yaml --solver sgd

# Multi-seed run (mean ± std, GT-free-selected)
python scripts/run_multiseed.py --config configs/default.yaml --seeds 7 11 42 --name default
