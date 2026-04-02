# GSDiff-SPI: Dynamic Single-Pixel Imaging via 2D Gaussian Splatting

A framework for reconstructing dynamic scenes from single-pixel camera measurements using **2D Gaussian Splatting (2DGS)** and a **shared SE(2) rigid-body motion model**, optimized jointly by ADMM with TV regularization or direct Adam (SGD).

---

## Overview

Single-pixel imaging (SPI) recovers a 2D image from a sequence of scalar inner-product measurements against structured illumination patterns. When the scene is **moving**, each measurement captures a different temporal snapshot, making reconstruction a coupled space-time inverse problem.

**Key challenge**: K = 2500 measurements must recover T × H × W ≈ 81,920 unknowns — a severely underdetermined system further corrupted by noise.

**Our approach**:
1. Represent the canonical scene as a sum of M oriented 2D Gaussians (compact, differentiable).
2. Model inter-frame motion as a shared SE(2) rigid-body transform — only 2–3 motion parameters regardless of scene complexity.
3. Jointly optimize scene and motion via gradient-based solvers with total variation regularization.

---

## Method

### 2D Gaussian Scene Representation

The canonical scene is parameterized as:

```
s(u) = Σ_{m=1}^{M}  a_m · exp( -½ (u - μ_m)ᵀ Σ_m⁻¹ (u - μ_m) )
```

Each Gaussian m is described by 6 learnable parameters:

| Parameter | Description | Parameterization |
|-----------|-------------|-----------------|
| a_m | Amplitude | softplus(raw_amp) > 0 |
| μ_m ∈ ℝ² | Spatial center (y, x) | unconstrained |
| (sy, sx) | Semi-axes | exp(log_scale) > 0 |
| θ_m | Orientation angle | unconstrained |

The covariance is `Σ_m = R(θ_m) diag(sy², sx²) R(θ_m)ᵀ`.
Rendering is **exact** and **fully differentiable** — no rasterization approximation.

### SE(2) Motion Model

All M Gaussians share a single rigid-body transform parameterized by total displacement **v** = [v_y, v_x] and rotation angle ω over the time interval [0, 1]:

```
μ_m(t) = R(ωt)(μ_m − c) + c + vt
Σ_m(t) = R(ωt) Σ_m R(ωt)ᵀ
```

where c = ((H−1)/2, (W−1)/2) is the image center. A 2D Gaussian under SE(2) remains Gaussian, so the transformed frame is rendered analytically with no interpolation artifacts.

### SPI Forward Model

The k-th scalar measurement is the Frobenius inner product between the k-th illumination pattern P_k and the frame at time t_{f(k)}:

```
y_k = ⟨P_k, I_{f(k)}⟩ = Σ_{i,j} P_k[i,j] · I_{f(k)}[i,j]
```

Patterns are random U[0,1] matrices. Frame assignment: `frame_idx[k] = k // ⌈K/T⌉`.

### Loss Function

Both solvers minimize a z-score normalized data fidelity term plus spatial/temporal TV regularization:

```
min_θ   ½ · MSE(zscore(ŷ), zscore(y))  +  λ · TV(V)
```

Z-score normalization (applied independently to prediction and target) removes the large DC offset of SPI measurements and makes the loss scale-invariant. SNR is defined relative to the AC variance of the measurements (`sig_pow = np.var(y)`).

**Optional 3D TV**: `TV3D(V) = Σ_{t,i,j} √((α·ΔtV)² + (ΔyV)² + (ΔxV)²)` couples temporal frames and suppresses frame-to-frame flicker.

### ADMM Solver

ADMM splits data fidelity from TV regularization via auxiliary variable z:

```
L_ρ(θ, z, u) = f(θ) + g(z) + (ρ/2)‖R(θ) − z + u‖²
```

**θ-step** (n_inner Adam steps — decoupled data + soft-TV + consistency):
```
target = z − u     (Boyd 2011 scaled form)
min_θ  f(θ) + λ_soft·TV(R(θ)) + (ρ/2)‖R(θ) − target‖²
```

**z-step** (exact TV denoising via Chambolle dual projection):
```
z = prox_{λ/ρ · TV}(R(θ) + u)
```

**u-step** (dual variable update):
```
u ← u + R(θ) − z
```

The z-step uses 2D Chambolle (`TVPrior`) or 3D isotropic Chambolle (`TVPrior3D`, step size `τ = 1/(8 + 4α²)`). The ADMM framework is designed so that Phase 2 (diffusion prior) only requires replacing the `proximal()` method — all other components stay unchanged.

### SGD Baseline

Direct end-to-end Adam optimization (no variable splitting):

```
loss = ½ MSE(zscore(ŷ), zscore(y)) + λ · TV(V)
```

Scene and motion parameters are updated jointly, with separate learning rates (motion lr = 15× scene lr) and CosineAnnealingLR decay.

---

## Installation

```bash
git clone <repo>
cd gsdiff_spi
pip install -r requirements.txt
```

**Dependencies**: `torch`, `numpy`, `scipy`, `scikit-image`, `Pillow`, `matplotlib`, `pyyaml`

---

## Quick Start

```bash
# Run ADMM solver (default config)
python train.py

# Run SGD baseline
python train.py --solver sgd

# Custom config
python train.py --config configs/default.yaml --solver admm
```

Results are written to `output_dir` (default `./results/9`):
- `comparison.png` — GT / DGI / reconstruction side-by-side for selected frames
- `recon_video.gif` — animated reconstruction
- `gt_video.gif` — animated ground truth
- `loss.png` — convergence curve
- `y_comparison.png` — raw and z-scored measurement fit
- `results.json` — PSNR, velocity error, runtime
- `checkpoint.pt` — scene and motion state dicts

---

## Configuration

All hyperparameters are set in `configs/default.yaml`. Key options:

```yaml
seed: 42

scene:
  num_gaussians: 500        # M: number of 2D Gaussians
  init_scale: 1.5           # initial Gaussian spread (pixels)
  init_mode: random         # random | dgi (DGI amplitude warm-start)

motion:
  enable_rotation: false    # true: also learn omega

data:
  image_size: [64, 64]
  num_frames: 20            # T
  num_patterns: 2500        # K
  pattern_type: random      # bernoulli | gaussian | random | s_matrix
  snr_db: 30
  shape: "assets/tank.png"  # built-in: "7" | "L" | "T" | "circle" | image path
  motion_type: custom_se2   # custom_se2 | translation | rotation | shear | ...
  gt_velocity: [8, 8]       # [v_y, v_x] total pixels over t∈[0,1]
  gt_omega: 0.0             # radians total

solver:
  type: admm                # admm | sgd

  # SGD hyperparameters
  lr_scene: 0.9e-2
  lr_motion: 15.0e-2
  sgd_steps: 2500
  tv_weight: 0.005

  # ADMM hyperparameters
  num_outer: 80             # outer iterations
  num_inner: 50             # inner gradient steps per outer (80×50 = 4000 total)
  admm_n_warmup: 80         # warmup iters (skip z/u steps; = num_outer → full warmup)
  rho: 0.1
  rho_growth: 1.05
  admm_tv_weight: 0.005     # Chambolle TV weight (active only after warmup)
  admm_soft_tv_weight: 0.005
  admm_lr_scene: 0.9e-2
  admm_lr_motion: 15.0e-2

  # 3D TV (optional)
  use_3dtv: true            # true: add temporal TV to both solvers
  temporal_tv_weight: 0.1   # α: temporal vs spatial balance (α<1 → weaker temporal)

output_dir: ./results/9
```

---

## Project Structure

```
gsdiff_spi/
├── train.py                     Main training/evaluation script
├── configs/
│   └── default.yaml             All hyperparameters
├── gsdiff/
│   ├── scene/
│   │   └── gaussian2d.py        2D Gaussian scene (render, differentiable)
│   ├── motion/
│   │   └── se2.py               SE(2) rigid-body motion model
│   ├── forward/
│   │   └── spi.py               SPI measurement model
│   ├── prior/
│   │   └── tv.py                TVPrior (2D Chambolle) + TVPrior3D (3D isotropic)
│   ├── solver/
│   │   ├── admm.py              ADMM solver (Boyd 2011 scaled form)
│   │   └── sgd.py               Direct Adam baseline
│   ├── data/
│   │   ├── simulation.py        Synthetic dynamic SPI data generation
│   │   ├── patterns.py          Illumination pattern generators
│   │   └── dgi.py               DGI (differential ghost imaging) baseline
│   └── utils.py                 Seed, metrics, I/O utilities
└── assets/
    └── tank.png                 Default test image
```

---

## Benchmark Results

Configuration: 64×64 image, T=20 frames, K=2500 patterns, SNR=30 dB, random patterns, translation motion v=[8,8] px, seed=42.

| Method | PSNR (dB) | Velocity Error (px) | Steps |
|--------|-----------|---------------------|-------|
| DGI (motion-blurred) | ~6.7 | — | — |
| SGD | ~23.5 | ~[0.0, 0.3] | 2500 |
| ADMM (full warmup) | ~23.75 | ~[0.0, 0.3] | 4000 |

**Note**: With `admm_n_warmup = num_outer`, ADMM runs in full-warmup mode (Chambolle z-step inactive), making it equivalent to SGD with a larger step budget. Reducing `admm_n_warmup` below `num_outer` activates the Chambolle z-step and full ADMM behavior.

---

## Extending to Phase 2 (Diffusion Prior)

The ADMM architecture is designed for plug-and-play prior replacement. To swap TV for a video diffusion prior:

1. Create `gsdiff/prior/diffusion.py` implementing:
   ```python
   class DiffusionPrior:
       def proximal(self, x: Tensor, weight: float) -> Tensor: ...
       def energy(self, x: Tensor) -> float: ...
   ```
2. In `train.py`, replace `TVPrior(...)` with `DiffusionPrior(...)`.
3. No other files change.

---

## Citation

If you use this code, please cite:

```bibtex
@misc{gsdiff_spi,
  title   = {GSDiff-SPI: Dynamic Single-Pixel Imaging via 2D Gaussian Splatting},
  author  = {},
  year    = {2026},
  url     = {}
}
```

---

## References

- Boyd, S., Parikh, N., Chu, E., Peleato, B., & Eckstein, J. (2011). Distributed optimization and statistical learning via the alternating direction method of multipliers. *Foundations and Trends in Machine Learning*, 3(1), 1–122.
- Chambolle, A. (2004). An algorithm for total variation minimization and applications. *Journal of Mathematical Imaging and Vision*, 20(1–2), 89–97.
- Kerbl, B., Kopanas, G., Leimkühler, T., & Drettakis, G. (2023). 3D Gaussian splatting for real-time radiance field rendering. *ACM Transactions on Graphics*, 42(4).
