# GSDiff-SPI Project Context

## What This Project Does

Dynamic single-pixel imaging (SPI) reconstruction using:
- **2D Gaussian Splatting (2DGS)** for differentiable, compact scene representation
- **SE(2) rigid-body motion model** — shared translation + rotation across all Gaussians
- **ADMM with TV regularization** as the main solver (exact proximal operator via Chambolle)
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

TV regularization options:

**Spatial TV (2D)**:
```
TV(V) = mean|V[:,:,1:,:] - V[:,:,:-1,:]| + mean|V[:,:,:,1:] - V[:,:,:,:-1]|
```

**3D TV (spatial + temporal)** when `use_3dtv: true`:
```
TV3D(V) = Σ_{t,i,j} √((α·ΔtV)² + (ΔyV)² + (ΔxV)²)
```
where `α = temporal_tv_weight`.

---

### 5. SGD Solver (`solver/sgd.py`)

Direct end-to-end Adam optimization over all parameters θ = {scene, motion}.

```
Initialize: scene (random Gaussians), motion (v=0, ω=0)
Adam: [scene: lr_scene=0.9e-2], [motion: lr_motion=15.0e-2]
CosineAnnealingLR: lr → 0.1·lr over N=sgd_steps

For i = 1..N:
    ŷ, V = fwd(patterns, frame_idx, t_grid)
    loss = f(θ) + λ_TV · TV(V)
    loss.backward()
    clip_grad_norm_(params, 5.0)
    optimizer.step(); scheduler.step()
```

**Key hyperparameters**:
- `lr_motion` must be large (15× scene lr) — velocity has few DOF but needs fast convergence
- CosineAnnealingLR is critical: high lr early for velocity, low lr late for refinement

---

### 6. ADMM Solver (`solver/admm.py`)

Decouples data fidelity from TV regularization via variable splitting.
Auxiliary variable z (video domain), constraint R(θ) = z via augmented Lagrangian.

**Augmented Lagrangian (Boyd 2011 scaled form)**:
```
L_ρ(θ, z, u) = f(θ) + g(z) + (ρ/2)‖R(θ) - z + u‖²
```
where `f(θ)` = data fidelity + soft TV, `g(z) = λ·TV(z)` (Chambolle exact).

**Three-step iteration**:
```
θ-step (n_inner Adam steps):
    min_θ  f(θ) + λ_soft·TV(R(θ)) + (ρ/2)‖R(θ) - (z - u)‖²
    target = z − u   ← Boyd sign convention (CRITICAL)

z-step (Chambolle proximal — exact TV denoising):
    z = prox_{g/ρ}(R(θ) + u)
    input = R(θ) + u   ← Boyd sign convention (CRITICAL)

u-step:
    u ← u + R(θ) − z
```

**Warmup mechanism** (current config: `admm_n_warmup = num_outer = 80`):
- During warmup: skip `(ρ/2)‖·‖²` and z/u updates
- Allows velocity to converge without z=0 anchoring the scene to zero
- With `n_warmup = n_outer = 80`: Chambolle z-step NEVER executes → ADMM ≈ SGD with 4000 steps

**Why ADMM ≈ SGD currently**: All 80 outer iterations are warmup. The θ-step uses identical soft TV loss as SGD. Only difference: 4000 total gradient steps vs SGD's 2500.

**Persistent optimizer**: Adam state preserved across all outer iterations.
**CosineAnnealingLR**: `T_max = num_outer × num_inner` — matches SGD total budget.

---

### 7. TV Priors (`prior/tv.py`)

**TVPrior (2D)**: Chambolle dual projection, frame-by-frame.
- Dual variable: `p ∈ ℝ^{H×W×2}`, step size `τ = 1/8`

**TVPrior3D**: Isotropic 3D TV, processes full video jointly.
- `TV3D(V) = Σ_{t,i,j} √((α·ΔtV)² + (ΔyV)² + (ΔxV)²)`
- Dual variable: `p ∈ ℝ^{T×H×W×3}`, step size `τ = 1/(8 + 4α²)` (from spectral norm bound)
- Enabled by `use_3dtv: true`, tuned via `temporal_tv_weight`

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
├── scene/gaussian2d.py    Canonical 2D Gaussian rendering (differentiable)
├── motion/se2.py          SE(2) transform of Gaussian params
├── forward/spi.py         Physics: scene+motion → video → measurements
├── prior/tv.py            TVPrior (2D Chambolle) + TVPrior3D (3D isotropic)
├── solver/admm.py         ADMM outer loop (Boyd 2011 scaled form)
├── solver/sgd.py          Direct Adam baseline
├── data/simulation.py     Synthetic data generation (multiple motion types)
├── data/patterns.py       Bernoulli / Gaussian / random / S-matrix patterns
├── data/dgi.py            DGI baseline reconstruction
└── utils.py               Seed, config, metrics, I/O
```

**Phase 2 extension point**: swap `prior/tv.py` for `prior/diffusion.py` — implement `proximal(x, weight)` interface, no other files change.

---

## Critical Math Conventions

- ADMM signs follow **Boyd et al. 2011 §3.1.1 (scaled form)**:
  - θ-step target: `z − u` (NOT `z + u`)
  - z-step input: `R(θ) + u` (NOT `R(θ) − u`)
  - u-step: `u ← u + R(θ) − z`
- Measurements z-scored **independently** (pred and target each normalized separately)
- All losses use **mean** reduction (NOT sum)
- SE(2) rotation center: `((H-1)/2, (W-1)/2)` to match `scipy.ndimage.rotate`
- SNR: `sig_pow = np.var(signal)` — AC variance, NOT DC mean

---

## How to Run

```bash
python train.py                              # ADMM (default)
python train.py --solver sgd                 # SGD baseline
python train.py --config configs/default.yaml --solver admm
```

---

## Code Style

- PyTorch tensors with explicit shape comments
- Config via YAML (`configs/default.yaml`); no magic numbers in code
- Each module replaceable (for Phase 2: swap TV → diffusion prior)
- All losses use mean reduction; grad clipping at 5.0 for stability

---

## Current Status

- Phase 1 complete: ADMM + 2D/3D TV working, SGD baseline working
- Benchmark (`snr_db=30`, `vel=[8,8]`, `64×64`, `seed=42`):
  - SGD: PSNR ~23.5 dB, est vel ~[8.0, 7.7]
  - ADMM (full-warmup, 4000 steps): PSNR ~23.75 dB
  - DGI baseline: ~6.7 dB
- SNR correctly calibrated via `np.var(signal)` (AC-based)
- Next: Phase 2 = replace TV with spatiotemporal diffusion prior (STEP/DAPS)
