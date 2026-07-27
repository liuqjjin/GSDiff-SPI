# GSDiff-SPI Project Context

## What This Project Does

Dynamic single-pixel imaging (SPI) reconstruction using:
- **2D Gaussian Splatting (2DGS)** for differentiable, compact scene representation
- **SE(2) rigid-body motion model** — shared translation + rotation across all Gaussians
- **ADMM** as the main solver, with two interchangeable z-step priors:
  - **TV prior** (`prior/tv.py`): exact Chambolle proximal operator (2D or 3D isotropic)
  - **Diffusion prior** (`prior/diffusion.py`): pixel-space video DDPM denoiser plugged into the
    z-step in **PnP-ADMM** style — i.e. `z = D_σ(R(θ) + u)` with an *independent* σ annealing
    schedule. This is **not** the closed-form proximal of an explicit `g(z)`; it's the standard
    Plug-and-Play substitution where a learned Gaussian denoiser replaces `prox_{g/ρ}`.
- **Direct Adam (SGD)** as the baseline solver
- Optional **3D TV** (spatial + temporal) for both solvers via `use_3dtv: true`

---

## Algorithm Principles

### 1. 2D Gaussian Splatting (2DGS) — Scene Representation

The canonical scene is a sum of M oriented 2D Gaussians:

```
s(u) = Σ_m  a_m · exp(-½ (u - μ_m)ᵀ Σ_m⁻¹ (u - μ_m))
```

Each Gaussian m has 6 parameters:

| Parameter | Storage | Constraint |
|-----------|---------|-----------|
| Amplitude a_m | `raw_amps` → softplus | a_m > 0 |
| Center μ_m = [y_m, x_m] | `centers` [M,2] | free |
| Scale (sy, sx) | `log_scales` [M,2] | exp(log_s) > 0 |
| Rotation angle θ_m | `angles` [M] | free |

Covariance matrix: `Σ_m = R(θ_m) · diag(sy², sx²) · R(θ_m)ᵀ`

**Rendering pipeline** (`gaussian2d.py`):
1. Build pixel grid `u ∈ ℝ^{H×W×2}`
2. Compute precision matrices `Σ_m⁻¹` via matrix inverse
3. `quad[m,n] = diff^T Σ^{-1} diff` — Mahalanobis distance per pixel
4. `img = Σ_m a_m · exp(-½ · quad_m)`
5. ReLU for non-negativity → `[1,1,H,W]`

**Key property**: Fully differentiable w.r.t. all 6 parameters.
**Complexity**: O(M × H × W) per frame — dense, NOT rasterization-based.
**Parameter count**: M=500 Gaussians × 6 = 3000 scene DOF + 2–3 motion DOF.

---

### 2. SE(2) Motion Model (`se2.py`)

All Gaussians share a single rigid-body SE(2) transformation at time t ∈ [0,1]:

```
μ_m(t) = R(ω·t) · (μ_m − c) + c + v·t
Σ_m(t) = R(ω·t) · Σ_m · R(ω·t)ᵀ
```

- `c = ((H-1)/2, (W-1)/2)` — rotation center, matches `scipy.ndimage.rotate`
- `v = [v_y, v_x]` — total pixel displacement over t ∈ [0,1] (learnable)
- `ω` — total rotation angle in radians (learnable, optional)
- `R(α) = [[cos α, -sin α], [sin α, cos α]]`

Initialized at `v=0, ω=0`. All M Gaussians share these 2–3 DOF.

**Key**: A Gaussian under SE(2) remains Gaussian — no pixel-space warping needed, only μ and Σ are updated. Gradients are exact.

---

### 3. SPI Forward Model (`spi.py`)

Single-pixel camera physics: one scalar measurement per illumination pattern.

```
y_k = ⟨P_k, I_{f(k)}⟩ = Σ_{i,j} P_k[i,j] · I_{f(k)}[i,j]
```

Full pipeline:
```
(scene params, motion params)
    ↓  SE(2) transform → render each frame
video V ∈ ℝ^{T×1×H×W}
    ↓  inner product with patterns
ŷ ∈ ℝ^K
```

Frame assignment: `ppf = ⌈K/T⌉`, `frame_idx[k] = k // ppf`.

**Measurement statistics** (critical for SNR interpretation):
- DC mean: `E[y_k] ≈ H·W · 0.5 · μ_pixel` — large, carries NO structural info
- AC variation: `std[y_k]` — small, carries ALL useful info
- SNR must be defined relative to AC variance: `sig_pow = np.var(signal)` (not DC mean)

---

### 4. Loss Function — Z-Score Normalization

Both solvers use z-score normalized data fidelity:

```
f(θ) = ½ · MSE(zscore(ŷ), zscore(y))
zscore(x) = (x - mean(x)) / (std(x) + ε)
```

Properties:
- **Scale-invariant** and **DC-invariant**: removes the large DC offset automatically
- ŷ and y are normalized **independently** (not relative to each other)

The code deliberately uses two distinct TV definitions:

**Differentiable θ-step regularizer (SGD and ADMM soft TV)** — componentwise
anisotropic mean:
```
TVθ(V) = mean|ΔyV| + mean|ΔxV| + α·mean|ΔtV|
```

**TV z-step proximal and reported energy** — pointwise isotropic sum:
```
TVz(V) = Σ_{t,i,j} √((α·ΔtV)² + (ΔyV)² + (ΔxV)²)
```
Here `α = temporal_tv_weight`; `use_3dtv: true` enables the temporal component.
The reductions and norms are intentionally not interchangeable.

---

### 5. SGD Solver (`solver/sgd.py`)

Direct end-to-end Adam optimization over all parameters θ = {scene, motion}.

```
Initialize: scene (random Gaussians), motion (v=0, ω=0)
Adam: [scene: lr_scene=0.9e-2], [motion: lr_motion=15.0e-2]
CosineAnnealingLR: lr → 0.1·lr over N=sgd_steps

For i = 1..N:
    ŷ, V = fwd(patterns, frame_idx, t_grid)
    loss = f(θ) + λ_TV · TVθ(V)
    loss.backward()
    clip_grad_norm_(params, 5.0)
    optimizer.step(); scheduler.step()
```

**Key hyperparameters**:
- `lr_motion` must be large (15× scene lr) — velocity has few DOF but needs fast convergence
- CosineAnnealingLR is critical: high lr early for velocity, low lr late for refinement

---

### 6. ADMM Solver (`solver/admm.py`)

Decouples data fidelity from the prior via variable splitting.
Auxiliary variable z (video domain), constraint R(θ) = z via augmented Lagrangian.

**Augmented Lagrangian (Boyd 2011 scaled form)**:
```
L_ρ(θ, z, u) = f(θ) + λ_soft·TVθ(R(θ)) + g(z) + (ρ/2)‖R(θ) - z + u‖²
```
where `f(θ)` is the data fidelity defined above. The prior `g(z)` is:
- **TV mode**: `g(z) = λ·TVz(z)`, the pointwise isotropic sum solved by Chambolle dual projection.
- **Diffusion (PnP) mode**: no explicit `g(z)` — the proximal step is *replaced* by a Gaussian
  video denoiser `D_σ(·)`, following the standard Plug-and-Play ADMM template (Venkatakrishnan
  et al. 2013). The denoiser's σ is chosen by an **independent annealing schedule**, not
  derived from `1/√ρ` or `√(λ/ρ)` (see §7.2 below for the rationale).

**Three-step iteration**:
```
θ-step (n_inner Adam steps):
    min_θ  f(θ) + λ_soft·TVθ(R(θ)) + (ρ/2)‖R(θ) - (z - u)‖²
    target = z − u   ← Boyd sign convention (CRITICAL)

z-step:
    TV mode:        z = prox_{(λ/ρ)·TVz}(R(θ) + u)       # exact Chambolle
    PnP (diffusion): z = D_σ_k(R(θ) + u)                  # learned denoiser, σ from schedule
    input = R(θ) + u   ← Boyd sign convention (CRITICAL)

u-step:
    u ← u + R(θ) − z
```

**Warmup mechanism**:
- During warmup (`outer_iter ≤ admm_n_warmup`): skip `(ρ/2)‖·‖²` and z/u updates entirely.
  The θ-step is pure data fidelity + soft TV, identical to the SGD loss.
- Allows velocity to converge without z=0 anchoring the scene to zero.
- **Transition iter** `n_warmup + 1`: θ-step still runs without the consistency term (z is not yet
  valid), then `z_step` and `u_step` execute to initialize z and u. From `n_warmup + 2` onward,
  full ADMM with `target = z − u`.
- The diffusion prior counts its z-step calls and anneals σ over `n_outer − n_warmup` steps. The
  schedule is initialized via `prior.set_n_steps(n_zsteps)` from `train.py`.

**`admm_n_warmup` regimes**:
- `admm_n_warmup = num_outer` → z-step never runs → ADMM ≡ SGD with `num_outer × num_inner` steps.
- `admm_n_warmup < num_outer` → true ADMM after warmup → prior actually contributes.

**Persistent optimizer**: Adam state preserved across all outer iterations.
**CosineAnnealingLR**: `T_max = num_outer × num_inner` — matches SGD total budget.

---

### 7. Priors

#### 7.1 TV Priors (`prior/tv.py`)

**TVPrior (2D)**: pointwise isotropic-sum Chambolle dual projection, frame-by-frame.
- Dual variable: `p ∈ ℝ^{H×W×2}`, step size `τ = 1/8`
- `proximal(x, weight)` solves `prox_{weight·TV}(x)` exactly. ADMM passes `weight = λ_TV/ρ`.

**TVPrior3D**: Isotropic 3D TV, processes full video jointly.
- `TV3D(V) = Σ_{t,i,j} √((α·ΔtV)² + (ΔyV)² + (ΔxV)²)`
- Dual variable: `p ∈ ℝ^{T×H×W×3}`, step size `τ = 1/(8 + 4α²)` (from spectral norm bound)
- Enabled by `use_3dtv: true`, tuned via `temporal_tv_weight`

#### 7.2 Diffusion Prior (`prior/diffusion.py`) — PnP-ADMM with σ annealing

A pixel-space video diffusion model (`UNet3D` in `prior/unet3d.py`) acts as a **learned Gaussian
denoiser** in the ADMM z-step:

```
z = D_σ(R(θ) + u)        single-step Tweedie:  D_σ(v) = v − σ·ε_θ(v, σ)
                          multi-step DDIM:      iterative denoising from σ down to σ_min
```

**Why this is PnP, not a strict proximal**: in PnP-ADMM the denoiser is *substituted* into the
algorithmic slot occupied by `prox_{g/ρ}` without committing to an explicit `g(z)`. There is
*no* `g(z)` such that the denoiser exactly equals its proximal operator. Convergence relies on
empirical observations / contraction-style results (Ryu et al. 2019), not strong-convex analysis.

**σ schedule (CRITICAL design choice)**: σ is chosen by an **independent log-linear schedule**

```
σ_k = exp((1 − k/N)·log(σ_start) + (k/N)·log(σ_end))     k = 0, 1, …, N−1
N = num_outer − admm_n_warmup
```

**not** by the seemingly-natural mapping `σ = √(weight) = √(tv_weight/ρ)`. The latter ties
`σ` to the TV regularization coefficient (which has nothing to do with diffusion prior strength)
and pushes σ into a regime (`~0.03–0.14`) where the denoiser is essentially the identity,
making the diffusion prior indistinguishable from mild TV. The independent schedule
(default `σ_start=0.3 → σ_end=0.05`) keeps σ inside the network's training distribution
`[σ_min=0.002, σ_max=0.5]` and gives meaningful denoising in the early outer iterations.

**The `weight` argument is still in the signature** (`proximal(x, weight)`) for interface
compatibility with `TVPrior`, but it is **ignored** by `DiffusionPrior`.

**Counting and bookkeeping**:
- `train.py` computes `n_zsteps = num_outer − admm_n_warmup` and calls
  `prior.set_n_steps(n_zsteps)` once after construction.
- Each `proximal()` call increments `_call_count`, advancing the σ schedule.
- The current σ is logged each outer iteration (next to ρ).

**Energy monitor**: `DiffusionPrior.energy(x)` returns spatial TV of `x` purely as a *monitoring*
quantity (so the existing ADMM logging path keeps working). It has no role in optimization.

#### 7.3 UNet3D score network (`prior/unet3d.py`)

Lightweight 3D UNet, ~2.8 M parameters:
- Input/output: `[B, 1, T, H, W]` (single-channel grayscale video, batch-first)
- Encoder: Conv3d(1→32) → Down(32→64) → Down(64→128)
- Middle: ResBlock3D(128)
- Decoder: Up(128+128→64) → Up(64+64→32) → Conv3d(32→1)
- Each ResBlock3D has FiLM-style σ conditioning: `log(σ)` → sinusoidal embed → MLP → (scale, shift)
- ε-prediction objective; conversion to Tweedie inside `DiffusionPrior._tweedie`

#### 7.4 Noise schedule (`prior/noise_schedule.py`)

EDM-style log-linear schedule used **only for training and for clamping σ during inference**:

```
σ(t) = exp((1−t)·log(σ_min) + t·log(σ_max)),  t ∈ [0,1]
```

Defaults: `σ_min=0.002`, `σ_max=0.5`. The inference-time σ schedule of `DiffusionPrior` is
clamped into `[σ_min, σ_max]` so the network is never queried outside its training distribution.

---

### 8. Full Training Pipeline

```
1. Generate data:
   - Load/create canonical image (e.g. tank.png)
   - Generate T GT frames via SE(2) motion (scipy.ndimage)
   - Generate K random patterns U[0,1]
   - Compute clean measurements y_k = ⟨P_k, I_{f(k)}⟩
   - Add noise: std = sqrt(np.var(y) / 10^(snr_db/10))  ← AC variance

2. DGI baseline: motion-blurred single image (lower bound on PSNR)

3. Optional DGI warm-start (init_mode='dgi'):
   Scale Gaussian amplitudes so render() mean ≈ DGI mean

4. Solve (SGD or ADMM):
   - Joint optimization: scene (M Gaussians) + motion (v, ω)
   - 2500 steps (SGD) or 4000 steps (ADMM, 80×50)

5. Evaluate:
   - Per-frame PSNR vs GT frames
   - Velocity error |v_est - v_gt|
   - Save comparison figures, GIFs, results.json, checkpoint.pt
```

---

## Architecture

```
gsdiff/
├── scene/gaussian2d.py     Canonical 2D Gaussian rendering (differentiable)
├── motion/se2.py           SE(2) transform of Gaussian params
├── forward/spi.py          Physics: scene+motion → video → measurements
├── prior/
│   ├── tv.py               TVPrior (2D Chambolle) + TVPrior3D (3D isotropic)
│   ├── diffusion.py        DiffusionPrior — PnP denoiser w/ independent σ annealing
│   ├── unet3d.py           Lightweight 3D UNet ε-prediction network (~2.8M params)
│   └── noise_schedule.py   EDM-style log-linear σ schedule (training + inference clamp)
├── solver/admm.py          ADMM outer loop (Boyd 2011 scaled form)
├── solver/sgd.py           Direct Adam baseline
├── data/simulation.py      Synthetic data generation (multiple motion types)
├── data/patterns.py        Bernoulli / Gaussian / random / S-matrix patterns
├── data/dgi.py             DGI baseline reconstruction
└── utils.py                Seed, config, metrics, I/O

scripts/
├── generate_video_dataset.py  Build [N,T,H,W] training set for the diffusion prior
└── train_diffusion_prior.py   Train UNet3D with ε-prediction loss (EMA optional)

configs/
├── default.yaml               Main config (solver + prior + diffusion_prior block)
└── diffusion_prior.yaml       Diffusion-prior training config (UNet, noise, optimizer)
```

The prior is a plug-in: `train.py` instantiates either `TVPrior(3D)` or `DiffusionPrior` based
on `solver.prior_type`, and `ADMMSolver` calls `prior.proximal(...)` without knowing which one
it received. Only `train.py` needs to know about the diffusion-specific `set_n_steps()` call.

---

## Critical Math Conventions

- ADMM signs follow **Boyd et al. 2011 §3.1.1 (scaled form)**:
  - θ-step target: `z − u` (NOT `z + u`)
  - z-step input: `R(θ) + u` (NOT `R(θ) − u`)
  - u-step: `u ← u + R(θ) − z`
- Measurements z-scored **independently** (pred and target each normalized separately)
- Differentiable θ-step TV uses a componentwise **mean**; the TV z-step
  proximal/energy uses a pointwise isotropic **sum**
- SE(2) rotation center: `((H-1)/2, (W-1)/2)` to match `scipy.ndimage.rotate`
- SNR: `sig_pow = np.var(signal)` — AC variance, NOT DC mean

---

## How to Run

### Reconstruction (`train.py`)

```bash
python train.py                              # ADMM (default config)
python train.py --solver sgd                 # SGD baseline
python train.py --config configs/default.yaml --solver admm
```

The prior is selected by `solver.prior_type` in the YAML — `tv` or `diffusion`. The diffusion
mode requires a trained UNet checkpoint at `solver.diffusion_prior.checkpoint` (see below).

### Training the diffusion prior (one-time)

```bash
# 1. Generate the training dataset (~5000 SE(2) videos by default)
python scripts/generate_video_dataset.py
#   → data/video_dataset.pt   ({"videos": [N, T, H, W] float32 in [0,1]})

# 2. Train the UNet3D ε-prediction network
python scripts/train_diffusion_prior.py --config configs/diffusion_prior.yaml
#   → checkpoints/diffusion_prior.pt
```

After the checkpoint exists, set `solver.prior_type: diffusion` in `configs/default.yaml`
and run `python train.py` as usual.

### Key configuration knobs

```yaml
solver:
  type: admm
  prior_type: diffusion         # tv | diffusion
  num_outer: 80
  num_inner: 50
  admm_n_warmup: 40             # MUST be < num_outer for diffusion to ever fire
  rho: 0.1
  rho_growth: 1.05
  admm_tv_weight: 0.005         # only used by TV prior
  admm_soft_tv_weight: 0.006    # soft TV inside θ-step (do NOT set to 0)
  use_3dtv: true
  temporal_tv_weight: 0.05

  diffusion_prior:              # only used when prior_type=diffusion
    checkpoint: checkpoints/diffusion_prior.pt
    denoise_steps: 1            # 1 = single-step Tweedie, 3-5 = multi-step DDIM
    clamp_range: [0.0, 1.0]
    sigma_start: 0.3            # σ at the FIRST z-step (after warmup)
    sigma_end:   0.05           # σ at the LAST z-step
```

`sigma_start` and `sigma_end` define the log-linear σ annealing schedule over
`num_outer − admm_n_warmup` z-step calls. Both values are clamped to `[σ_min, σ_max]` of the
training schedule (`0.002` and `0.5`).

### 2026-07 upgrade knobs (all default-off; defaults reproduce historical runs bit-exactly)

```yaml
data:
  pattern_type: hadamard_cc   # + hadamard_walsh | hadamard_natural | fourier | s_matrix_m
                              #   (s_matrix now ASSERTS twin-prime H,W; use s_matrix_m for 2^n grids)
  pattern_order: sequential   # stratified | random — temporal display schedule of ranked patterns;
                              #   frame_idx[k]=k//ppf makes display order the time axis (interacts with motion)
  holdout_mod: 0              # 10 → k%10==holdout_offset measurements excluded from training;
  holdout_offset: 7           #   GT-free holdout_residual reported in results.json (model selection)
  gt_accel: null              # [ay,ax] — accelerated SE(2) ground truth (custom_se2 explicit branch)
  gt_beta: null               # angular acceleration
  noise_sigma_abs: null       # detector-referred absolute noise σ (overrides snr_db).
                              #   REQUIRED for cross-pattern-family benchmarks: the AC-variance
                              #   snr_db convention is family-unfair (ordered bases' var(y) is
                              #   dominated by a few huge low-freq coefficients → same snr_db =
                              #   ~4x the absolute noise vs random patterns; measured 1.77 vs 0.45)
motion:
  poly_degree: 1              # 2 → d(t)=v·t+a·t², angle(t)=ω·t+β·t² (exact transport)
  enable_affine: false        # true → A(t)=R(angle)·(I+t·L_sym), L symmetric 3-DOF (exact transport)
solver:
  hqs: false                  # true → u≡0 (HQS ablation: isolates the ADMM dual's contribution)
  red_weight: 0.0             # SGD only: >0 adds RED-diff regularizer red_weight·½‖V−D_σ(V)‖²
                              #   (uses solver.diffusion_prior.checkpoint; σ anneals over sgd_steps)
  freeze_motion: false        # SGD only: true → static-scene fit (motion-blur lower-bound baseline)
  diffusion_prior:
    renoise: false            # true → z-step input re-noised: z=D_σ(v+σε) (on-manifold query)
    ddim_spacing: linear      # log → log-spaced DDIM σ ladder (consistent w/ the annealing schedule)
```

Multi-seed protocol: `python scripts/run_multiseed.py --config <yaml> --seeds 7 11 42 --name <exp>
[--override dotted.key=val ...]` → `results/<exp>/summary.json` (mean±std).
Hyperparameter search: `python scripts/autoresearch.py --base <yaml> --moves-set tv|diffusion|common`
— coordinate descent, accepts ONLY on multi-(motion×seed)-mean GT-free holdout, never on PSNR.
Full design rationale: `UPGRADE_PLAN.md`.

---

## Code Style

- PyTorch tensors with explicit shape comments
- Config via YAML (`configs/default.yaml`); no magic numbers in code
- Each module replaceable (for Phase 2: swap TV → diffusion prior)
- Differentiable θ-step losses use mean reduction; the TV z-step prior/energy uses an isotropic sum.
  Gradient clipping is 5.0 for stability.

---

## Current Status

- ADMM + TV (2D/3D) and diffusion (PnP) priors working; SGD baseline working.
- **Tuned operating point (configs/default.yaml, 2026-07)**: M=1000 Gaussians, `init_mode: dgi_adaptive`,
  `admm_n_warmup: 20`, `rho: 0.1`, `rho_growth: 1.1`, SNR 25, diffusion Tweedie σ 0.3→0.05.
  Reaches **35.8 ± 0.3 dB** on the tank SE(2) scene (3 seeds, GT-free-selected). Trajectory from the
  historical baseline: 27.6 → 32.9 (adaptive init + ρ-continuation) → 35.8 (M=1000).
- Hyperparameters confirmed locally optimal by `scripts/autoresearch.py` (coordinate descent, GT-free
  acceptance); only `num_gaussians 500→1000` was accepted.
- SNR calibrated via `np.var(signal)` (AC-based); `data.noise_sigma_abs` available for cross-pattern-family
  fairness.

### Implementation Notes (verified by experiment)

- **ADMM transition bug fixed**: `step()` has an `is_transition` flag for iteration `n_warmup+1`.
  On that iteration the θ-step runs without the consistency term (z not yet valid), then z_step and
  u_step run to initialize z/u. From `n_warmup+2` onward, full ADMM with proper `target = z − u`.
- **True ADMM outperforms SGD** once the transition fix is applied and `admm_n_warmup < num_outer`.
- **`init_mode: dgi_adaptive` is essential for the translation-only regime**: random init collapses
  ~2/3 seeds at SNR 25 (motion diverges during warmup, ~11 dB); adaptive init (centers ∝ |∇DGI|,
  amps from local intensity) fixes it (→25 dB) and is neutral-or-better with rotation.
- **`rho_growth: 1.1` beats 1.05** by ~5 dB at the tuned point (monotone ρ continuation, Chan 2017).
- **HQS (`hqs: true`, u≡0) ties ADMM with the TV prox but loses ~1 dB with the diffusion prior** — the
  dual/Bregman memory corrects the learned denoiser's bias. Keep the dual variable.
- **Model selection: use `data.holdout_extra` (non-invasive eval set), NEVER `holdout_mod`.** Removing
  in-training measurements destabilizes the bimodal joint scene+motion basin (27→10 dB). autoresearch
  selects on a per-motion-median residual with a collapse-count guard.
- **Pattern family**: Bernoulli {0,1} > random U[0,1] by ~2 dB; ordered orthogonal bases (Hadamard /
  Fourier / S-matrix) fail structurally under the z-score loss regardless of temporal schedule.
- **`temporal_tv_weight` has an optimal range**: empirically `0.05` works best; `0.02` is too weak
  (insufficient temporal smoothing), `0.08` is too strong (over-smooths across frames).
- **`admm_soft_tv_weight` must not be set to 0**: removing the soft TV in the θ-step causes a
  noticeable drop in reconstruction quality. Keep it at `~0.005–0.006`.

### Algorithm audit (2026-07, finalization) — the "two losses cancel" question

A multi-agent adversarial audit (2DGS render + SE(2) transport + SPI forward + ADMM +
diffusion PnP + loss) plus per-iteration empirical tracing found **no correctness bug in the
core algorithm** — Boyd ADMM signs, SE(2) covariance transport, the SPI forward inner product,
and the independent z-score loss are all verified correct.

- **The "两条 loss 取消" observation is NOT a bug — it is two distinct, benign phenomena:**
  1. *Warmup plotting artifact.* During warmup `z ≡ 0`, so the curve plotted as "primal
     residual" (`prim_res = MSE(video, z)`) is actually the video's raw energy `mean(video²)`,
     not a residual. It *rises* as the scene fills in while data fidelity *falls* → the two
     curves form a spurious "X", then `prim_res` drops off a cliff at the transition iter when
     `z` is initialized to ≈`video`. Fixed: `save_loss_curve` now NaN-masks the warmup span and
     overlays the true consistency term `(ρ/2)‖R(θ)-(z-u)‖²`; the console logs `consist`/`u_norm`.
  2. *Healthy post-warmup trade-off.* After warmup the data-fidelity loss **rises** (~2×,
     0.0013→0.0026 on tank) while the consistency term rises and `ρ` grows to ~30. This is the
     diffusion prior correctly trading measurement-fit for reconstruction quality. Per-iter PSNR
     tracing proves it is beneficial: PSNR climbs **monotonically 24→36.3 dB** over exactly this
     phase, peaking at the final outer iter (`peak − final = +0.00 dB`). **Do not "fix" the
     rising data loss** and **do not cap ρ** — an explicit rho-cap ablation would have frozen
     PSNR ~3 dB lower; the monotone ρ-continuation is load-bearing (consistent with `rho_growth
     1.1 > 1.05`). `num_outer=80` lands right at the PSNR peak.
- **Fixes applied from the audit** (all low-severity, behavior-preserving at the operating point):
  motion `__init__.py` docstring rotation-center `H/2→(H-1)/2`; `save_loss_curve` warmup mask +
  consistency overlay; console `consist`/`u_norm`; deleted dead `SE2Motion._R`; `init_from_image`
  softplus-inverse hardened to the overflow-free identity `y+log(-expm1(-y))`; `DiffusionPrior._n_steps`
  now `None`-sentinel and `proximal()` raises if `set_n_steps` was skipped (was a silent σ-collapse).
- **Considered and rejected on evidence:** ρ-cap / residual-balancing (refuted by the PSNR
  trajectory); top-k Gaussian culling (changes measured `y`, non-result-preserving); `order=3` GT
  interpolation (introduces ringing the non-negative Gaussian model also cannot match).

### Diffusion-prior Implementation Notes

- **Do NOT derive σ from `tv_weight/ρ`**. An earlier version computed
  `σ = √(tv_weight/ρ)` inside `DiffusionPrior.proximal`, which collapsed σ to ~0.03–0.14 and made
  the diffusion prior empirically indistinguishable from a mild TV prior. The fix is the
  independent log-linear annealing schedule (`sigma_start → sigma_end`) described in §7.2.
- **`set_n_steps` must be called before the first ADMM step.** `train.py` does this with
  `n_zsteps = num_outer − admm_n_warmup` immediately after constructing `DiffusionPrior`.
  Skipping `set_n_steps()` leaves `_n_steps = None` and `proximal()` raises `RuntimeError`; it no
  longer silently runs a one-step schedule.
- **σ is clamped into the training range** `[σ_min, σ_max]` of `NoiseSchedule` so the network is
  never queried out-of-distribution. If you want larger σ values, retrain the UNet with a larger
  `sigma_max` in `configs/diffusion_prior.yaml` *before* changing `sigma_start`.
- **`weight` argument to `DiffusionPrior.proximal` is ignored** (kept only for interface
  compatibility with `TVPrior`). Changing `admm_tv_weight` has no effect when
  `prior_type: diffusion`.
- **PnP, not strict proximal**: there is no explicit `g(z)` such that `D_σ` is its proximal.
  Convergence guarantees are weaker than the TV/Chambolle case; expect to tune `sigma_start`,
  `sigma_end`, and `denoise_steps` empirically per scene.
