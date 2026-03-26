
# GSDiff-SPI Project Context

## What this project does
Dynamic single-pixel imaging reconstruction using:
- 2D Gaussian Splatting for scene representation
- SE(2) rigid-body motion model
- ADMM with TV regularization (Phase 1)
- Future: 3DTV prior next:spatiotemporal diffusion prior (Phase 2)

## Architecture
```
gsdiff/
├── scene/gaussian2d.py    # Canonical 2D Gaussian rendering (differentiable)
├── motion/se2.py          # SE(2) transform of Gaussian params (Theorem 3.1)
├── forward/spi.py         # Physics: scene+motion → video → measurements
├── prior/tv.py            # Chambolle TV proximal operator
├── solver/admm.py         # ADMM outer loop (Boyd 2011 scaled form)
├── solver/sgd.py          # Direct Adam baseline
├── data/simulation.py     # Synthetic data generation (multiple motion types)
├── data/patterns.py       # Bernoulli / Gaussian / S-matrix patterns
├── data/dgi.py            # DGI baseline reconstruction
└── utils.py               # Seed, config, metrics, I/O
```

## Critical math conventions
- ADMM signs follow Boyd et al. 2011 §3.1.1 (SCALED form):
  - θ-step target: z − u (NOT z + u)
  - z-step input: R(θ) + u (NOT R(θ) − u)
  - u-step: u ← u + R(θ) − z
- Measurements are z-scored INDEPENDENTLY (pred and target each normalized separately)
- All losses use mean reduction (NOT sum)
- SE(2) rotation center is ((H-1)/2, (W-1)/2) to match scipy.ndimage.rotate

## How to run
```bash
python train.py --solver admm    # ADMM with TV prior
python train.py --solver sgd     # SGD baseline
```

## Code style
- PyTorch tensors with explicit shape comments
- Config via YAML (configs/default.yaml)
- No magic numbers: all hyperparams in config
- Every module must be replaceable (for Phase 2 extensions)

## Current status
- Phase 1 complete: ADMM/TV working, PSNR ~15-22 dB on 28×28 test images
- SGD baseline working: PSNR ~17-22 dB
- Next: Phase 2 = replace TV with other prior (3DTV or spatiotemporal diffusion prior (STEP/DAPS))
