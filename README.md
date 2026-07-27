# GSDiff-SPI: Dynamic Single-Pixel Imaging via 2D Gaussian Splatting

A framework for reconstructing dynamic scenes from single-pixel camera measurements using **2D Gaussian Splatting (2DGS)** and a **shared SE(2) rigid-body motion model**, optimized jointly by ADMM. The ADMM z-step accepts either an exact TV proximal (Chambolle) or a learned video diffusion denoiser plugged in PnP-ADMM style (with an independent σ annealing schedule). A direct Adam (SGD) solver is provided as a baseline.

---

## Overview

Single-pixel imaging (SPI) recovers a 2D image from a sequence of scalar inner-product measurements against structured illumination patterns. When the scene is **moving**, each measurement captures a different temporal snapshot, making reconstruction a coupled space-time inverse problem.

**Key challenge**: K = 2500 measurements must recover T × H × W ≈ 81,920 unknowns — a severely underdetermined system further corrupted by noise.

**Our approach**:
1. Represent the canonical scene as a sum of M oriented 2D Gaussians (compact, differentiable).
2. Model inter-frame motion as a shared SE(2) rigid-body transform — only 2–3 motion parameters regardless of scene complexity.
3. Jointly optimize scene and motion via ADMM, with the z-step pluggable between an exact TV proximal and a video diffusion denoiser (PnP-ADMM).

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

Both solvers use a z-score normalized data fidelity term. Their differentiable θ-step
regularizer retains the historical componentwise anisotropic mean:

```
TVθ(V) = mean|ΔyV| + mean|ΔxV| + α·mean|ΔtV|
```

Z-score normalization (applied independently to prediction and target) removes the large DC offset of SPI measurements and makes the loss scale-invariant. SNR is defined relative to the AC variance of the measurements (`sig_pow = np.var(y)`).

The TV z-step proximal and its reported energy instead use the pointwise isotropic sum
`TVz(V) = Σ_{t,i,j} √((α·ΔtV)² + (ΔyV)² + (ΔxV)²)`. These are deliberately
different objectives; `use_3dtv` adds their respective temporal terms.

### ADMM Solver

ADMM splits data fidelity from the prior via auxiliary variable z and the augmented Lagrangian (Boyd 2011 scaled form):

```
L_ρ(θ, z, u) = f(θ) + g(z) + (ρ/2)‖R(θ) − z + u‖²
```

**θ-step** (n_inner Adam steps — decoupled data + soft-TV + consistency):
```
target = z − u
min_θ  f(θ) + λ_soft·TVθ(R(θ)) + (ρ/2)‖R(θ) − target‖²
```

**z-step** — chooses *one* of the two priors below:

- **TV mode** (`prior_type: tv`) — exact Chambolle dual projection:
  ```
  z = prox_{(λ/ρ)·TVz}(R(θ) + u)
  ```
  2D per-frame (`TVPrior`) or 3D isotropic (`TVPrior3D`, step size `τ = 1/(8 + 4α²)`).

- **Diffusion (PnP) mode** (`prior_type: diffusion`) — learned Gaussian denoiser:
  ```
  z = D_σ_k(R(θ) + u)
  D_σ(v) = v − σ · ε_θ(v, σ)        (single-step Tweedie)
  ```
  with σ chosen by an **independent log-linear annealing schedule** `σ_start → σ_end` over
  `num_outer − admm_n_warmup` z-step calls. This is **plug-and-play**: the denoiser is
  substituted into the proximal slot without committing to an explicit `g(z)`. There is
  *no* `g(z)` whose proximal equals `D_σ`; convergence relies on standard PnP arguments
  (Venkatakrishnan et al. 2013, Ryu et al. 2019), not on strong convexity.

**u-step** (dual variable update, identical for both modes):
```
u ← u + R(θ) − z
```

**Why σ is *not* derived from `1/√ρ` or `√(λ/ρ)`**: those mappings tie σ either to a parameter
of a different prior (TV's `λ`) or to the ADMM penalty `ρ` directly. In our setup the latter
puts σ outside the network's training distribution, and the former collapses σ to ~0.03–0.14
where the denoiser is essentially the identity, making the diffusion prior empirically
indistinguishable from a mild TV prior. The independent schedule keeps σ inside
`[σ_min=0.002, σ_max=0.5]` and produces meaningful denoising in the early outer iterations.

**Warmup**: `admm_n_warmup` outer iterations skip both the consistency term and the z/u updates,
letting the velocity converge before the prior turns on. With `admm_n_warmup = num_outer` ADMM
degenerates to SGD with `num_outer × num_inner` steps and the z-step never fires.

### SGD Baseline

Direct end-to-end Adam optimization (no variable splitting):

```
loss = ½ MSE(zscore(ŷ), zscore(y)) + λ · TVθ(V)
```

Scene and motion parameters are updated jointly, with separate learning rates (motion lr = 15× scene lr) and CosineAnnealingLR decay.

### Diffusion Prior — UNet3D and training

The diffusion prior is a lightweight pixel-space video DDPM:

- **Network** (`gsdiff/prior/unet3d.py`): 3D UNet, ~2.8 M params, channels [32, 64, 128], FiLM-style σ conditioning via `log(σ)` → sinusoidal embedding → MLP. Input/output `[B, 1, T, H, W]`.
- **Objective**: ε-prediction at log-uniform σ ∈ [σ_min, σ_max] (defaults 0.002, 0.5):
  ```
  L = E_{x₀, σ, ε} ‖ε_θ(x₀ + σ·ε, σ) − ε‖²
  ```
- **Inference inside ADMM**: single-step Tweedie `D_σ(v) = v − σ·ε_θ(v, σ)` (default), or
  `denoise_steps`-step DDIM from `σ` down to `σ_min` for `denoise_steps > 1`.
- **Training data** (`scripts/generate_video_dataset.py`): synthetic SE(2) videos generated by
  the same `gsdiff.data.simulation` pipeline used for evaluation, swept over a small set of
  source images, velocities, and rotation rates. Output is a single `.pt` file containing a
  `[N, T, H, W]` float32 tensor in `[0, 1]`.

---

## Results

Default benchmark: 64×64 canonical scene, T = 20 frames, K = 2500 random patterns
(≈ 3 % per-frame sampling), shared SE(2) motion (translation **v** = [8, 8] px + rotation
ω = 0.3 rad), SNR 25 dB. All numbers are the mean ± std over 3 seeds {7, 11, 42}, with
hyperparameters selected on a **ground-truth-free** held-out measurement residual (PSNR is
reported, never used for selection).

| Configuration | PSNR (dB) | Velocity error (px) |
|---|---|---|
| DGI (motion-blurred lower bound) | ~7 | — |
| SGD (direct Adam, 2D TV) | 22.1 | [0.05, 0.38] |
| ADMM + 2D TV | 24.4 | [0.06, 0.27] |
| ADMM + 3D TV | 24.5 | [0.04, 0.27] |
| ADMM + diffusion prior (PnP) | 27.6 | [0.01, 0.02] |
| **+ content-adaptive init + ρ-continuation + M = 1000** | **35.8 ± 0.3** | **[0.02, 0.03]** |

The final operating point recovers the SE(2) motion to ≈ 0.03 px translation and ≈ 0.002 rad
rotation, and reconstructs each frame to ~35 dB from only 125 measurements per frame.

**Verified ablation findings** (each 3-seed, GT-free-selected):

- **True ADMM beats direct SGD** by ~3 dB once the warmup/transition is correct.
- **The diffusion (PnP) z-step beats the exact TV proximal** by ~1.3 dB at SNR 25.
- **ADMM beats its HQS reduction (u ≡ 0) only with the diffusion prior** (+1 dB) — the dual /
  Bregman memory corrects the learned denoiser's systematic bias; with the exact TV prox HQS
  ties ADMM. This is the mechanistic justification for keeping the dual variable.
- **Content-adaptive Gaussian initialization** (centers ∝ |∇DGI|) removes a translation-only
  failure basin (11 → 25 dB) and is the single largest robustness fix.
- **Bernoulli {0,1} patterns** outperform U[0,1] random by ~2 dB; **coefficient-ordered
  orthogonal bases** (Hadamard / Fourier / S-matrix) are structurally mismatched to the
  measurement-domain z-score loss and fail regardless of temporal ordering — a first-principles
  limitation documented in `THEORY.md`.
- **Rejected on evidence** (kept off by default): re-noised PnP z-step (−2.9 dB), σ_end → 0.02
  (−3 dB), latent/VAE priors, adaptive ρ.

Classical and learned **baselines** (Monin-style translation compensation, GIDC / GIDC-3DTV,
a matched-DOF INR + SE(2) control, and CS lower bounds) and a **multi-scene × multi-motion**
comparison are included in the evaluation suite; see `configs/` and `scripts/`.

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

### Reconstruction

```bash
# Run ADMM solver (default config — prior chosen by configs/default.yaml)
python train.py

# Run SGD baseline
python train.py --solver sgd

# Custom config
python train.py --config configs/default.yaml --solver admm
```

The prior is selected by `solver.prior_type` in the YAML — `tv` or `diffusion`. The diffusion mode requires a trained UNet checkpoint pointed to by `solver.diffusion_prior.checkpoint`.

### Train the diffusion prior (one-time, before using `prior_type: diffusion`)

```bash
# 1. Generate the synthetic SE(2) video dataset (~5000 videos by default)
python scripts/generate_video_dataset.py
#   → data/video_dataset.pt   ({"videos": [N, T, H, W] float32 in [0,1]})

# 2. Train the UNet3D ε-prediction network
python scripts/train_diffusion_prior.py --config configs/diffusion_prior.yaml
#   → checkpoints/diffusion_prior.pt
```

### Outputs

Results are written to `output_dir`:
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
  num_gaussians: 1000       # M: number of 2D Gaussians (capacity was the binding constraint)
  init_scale: 1.5           # initial Gaussian spread (pixels)
  init_mode: dgi_adaptive   # random | dgi | dgi_adaptive (centers ∝ |∇DGI|, amps from intensity)

motion:
  enable_rotation: true     # also learn omega
  poly_degree: 1            # 2 → accelerated SE(2): d(t)=v·t+a·t², angle(t)=ω·t+β·t²
  enable_affine: false      # true → A(t)=R(angle)·(I+t·L_sym): scale/shear (exact transport)

data:
  image_size: [64, 64]
  num_frames: 20            # T
  num_patterns: 2500        # K
  pattern_type: random      # random | bernoulli | gaussian | hadamard_cc/walsh/natural
                            #   | fourier | s_matrix (twin-prime H,W) | s_matrix_m (2^n grids)
  pattern_order: sequential # sequential | stratified | random  (temporal schedule of ranked patterns)
  snr_db: 25
  noise_sigma_abs: null     # detector-referred σ (overrides snr_db; use for cross-pattern-family fairness)
  time_assignment_mode: uniform   # uniform | interpolation
  shape: "assets/tank.png"  # built-in: "7" | "L" | "T" | "circle" | image path
  motion_type: custom_se2
  motion_mode: 2
  gt_velocity: [8, 8]       # [v_y, v_x] total pixels over t∈[0,1]
  gt_omega: 0.3             # radians total
  holdout_extra: 250        # non-invasive GT-free eval set (training unaffected; for model selection)

solver:
  type: admm                # admm | sgd
  loss_norm: zscore         # zscore | target_std

  # SGD hyperparameters
  lr_scene: 0.9e-2
  lr_motion: 15.0e-2
  sgd_steps: 2500
  tv_weight: 0.005

  # ADMM hyperparameters
  num_outer: 80             # outer iterations
  num_inner: 50             # inner gradient steps per outer (80×50 = 4000 total)
  admm_n_warmup: 20         # warmup iters; MUST be < num_outer for the prior to ever fire
  rho: 0.1
  rho_growth: 1.1           # monotone ρ continuation (Chan 2017); 1.05→1.1 was worth +5 dB
  admm_tv_weight: 0.005     # only used by TV prior (ignored by diffusion prior)
  admm_soft_tv_weight: 0.006  # soft TV inside θ-step (do NOT set to 0)
  admm_lr_scene: 0.9e-2
  admm_lr_motion: 15.0e-2
  hqs: false                # true → u≡0 (HQS ablation)

  # 3D TV (isotropic sum in the z-step; anisotropic mean in the θ-step)
  use_3dtv: true
  temporal_tv_weight: 0.05  # α: temporal vs spatial balance

  # ── Prior selection ────────────────────────────────────────
  prior_type: diffusion     # tv | diffusion

  diffusion_prior:          # only used when prior_type=diffusion
    checkpoint: checkpoints/diffusion_prior.pt
    denoise_steps: 1        # 1 = single-step Tweedie, 3-5 = multi-step DDIM
    clamp_range: [0.0, 1.0]
    sigma_start: 0.3        # σ at the FIRST z-step (after warmup)
    sigma_end:   0.05       # σ at the LAST z-step

output_dir: ./results/default_run
```

`sigma_start` and `sigma_end` define the log-linear σ annealing schedule that the diffusion prior follows over its `num_outer − admm_n_warmup` z-step calls. Both values are clamped to `[σ_min, σ_max]` of the network's noise schedule (defaults `0.002` and `0.5`). The full set of optional knobs (accelerated/affine motion, pattern families and temporal scheduling, HQS ablation, RED-diff, non-invasive holdout) is documented in `CLAUDE.md`; every one defaults off and reproduces the historical behaviour bit-exactly.

---

## Project Structure

```
gsdiff_spi/
├── train.py                          Main reconstruction script
├── configs/
│   ├── default.yaml                  Reconstruction config (solver + prior selection)
│   └── diffusion_prior.yaml          UNet3D / DDPM training config
├── scripts/
│   ├── generate_video_dataset.py     Build [N,T,H,W] training set for the diffusion prior
│   └── train_diffusion_prior.py      Train UNet3D with ε-prediction loss (EMA optional)
├── gsdiff/
│   ├── scene/
│   │   └── gaussian2d.py             2D Gaussian scene (render, differentiable)
│   ├── motion/
│   │   └── se2.py                    SE(2) rigid-body motion model
│   ├── forward/
│   │   └── spi.py                    SPI measurement model
│   ├── prior/
│   │   ├── tv.py                     TVPrior (2D Chambolle) + TVPrior3D (3D isotropic)
│   │   ├── diffusion.py              DiffusionPrior (PnP denoiser, σ annealing schedule)
│   │   ├── unet3d.py                 Lightweight 3D UNet ε-prediction network
│   │   └── noise_schedule.py         EDM-style log-linear σ schedule
│   ├── solver/
│   │   ├── admm.py                   ADMM solver (Boyd 2011 scaled form)
│   │   └── sgd.py                    Direct Adam baseline
│   ├── data/
│   │   ├── simulation.py             Synthetic dynamic SPI data generation
│   │   ├── patterns.py               Illumination pattern generators
│   │   └── dgi.py                    DGI (differential ghost imaging) baseline
│   └── utils.py                      Seed, metrics, I/O utilities
└── assets/
    └── tank.png                      Default test image
```

---

## Notes on the Diffusion Prior (PnP-ADMM)

A few non-obvious points worth keeping in mind when using `prior_type: diffusion`:

- **It is PnP, not strict proximal.** There is no explicit `g(z)` such that the denoiser
  `D_σ` equals `prox_{g/ρ}`. The denoiser is *substituted* into the proximal slot following
  the Plug-and-Play ADMM template. Convergence guarantees are weaker than in the TV/Chambolle
  case; expect to tune `sigma_start`, `sigma_end`, and `denoise_steps` empirically.

- **σ is set by an independent annealing schedule, not by `1/√ρ` or `√(λ/ρ)`.** The latter
  mappings either ignore that the network was trained on `σ ∈ [σ_min, σ_max]` or collapse
  σ to a regime where the denoiser is essentially the identity (which empirically makes the
  diffusion prior indistinguishable from a mild TV prior). The schedule used here is
  `σ_k = exp((1 − k/N)·log σ_start + (k/N)·log σ_end)` over `N = num_outer − admm_n_warmup`
  z-step calls.

- **`admm_tv_weight` is ignored** when `prior_type: diffusion`. The `weight` argument of
  `proximal()` is kept only for interface compatibility with `TVPrior`.

- **`admm_n_warmup` must be `< num_outer`** for the diffusion prior to ever fire. With
  `admm_n_warmup = num_outer` the z-step never executes and ADMM degenerates to SGD with a
  larger step budget.

- **`admm_soft_tv_weight` should not be 0.** Removing the soft TV inside the θ-step measurably
  degrades reconstruction quality, regardless of which z-step prior is chosen.

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
- Venkatakrishnan, S. V., Bouman, C. A., & Wohlberg, B. (2013). Plug-and-play priors for model based reconstruction. *IEEE GlobalSIP*.
- Ryu, E. K., Liu, J., Wang, S., Chen, X., Wang, Z., & Yin, W. (2019). Plug-and-play methods provably converge with properly trained denoisers. *ICML*.
- Karras, T., Aittala, M., Aila, T., & Laine, S. (2022). Elucidating the design space of diffusion-based generative models. *NeurIPS* (EDM).
- Robbins, H. (1956). An empirical Bayes approach to statistics. (Tweedie's formula.)
