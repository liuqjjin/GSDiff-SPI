"""Dynamic SPI data simulation.

Supports motion types: translation, rotation, shear, swirl,
translation_and_rotation, custom_se2.
Supports pattern types: bernoulli, gaussian, random, s_matrix.
Supports custom images via load_image().
"""
import numpy as np
from dataclasses import dataclass
from PIL import Image
from skimage import transform
import os
from .patterns import generate_patterns, apply_pattern_order


@dataclass
class SPIData:
    canonical: np.ndarray       # [H,W]
    gt_frames: np.ndarray       # [T,H,W]
    patterns: np.ndarray        # [K,H,W]
    measurements: np.ndarray    # [K]
    frame_idx: np.ndarray       # [K] int
    t_grid: np.ndarray          # [T]
    # GT motion params (for evaluation)
    gt_velocity: np.ndarray     # [2]
    gt_omega: float
    motion_type: str
    H: int; W: int; T: int; K: int
    snr_db: float
    # Optional NON-INVASIVE eval set (extra measurements never shown to the
    # solver; training data is byte-identical whether or not this exists).
    # Always random U[0,1] patterns from RandomState(seed+9999) — a neutral,
    # family-independent generalization probe.
    eval_patterns: np.ndarray = None      # [Ke,H,W]
    eval_measurements: np.ndarray = None  # [Ke]
    eval_frame_idx: np.ndarray = None     # [Ke]


# ─── Image loading ──────────────────────────────────────────
def load_image(image_path, dim=(28, 28)):
    """Load and resize a grayscale image to [0,1]."""
    if os.path.exists(image_path):
        img = Image.open(image_path).convert('L')
        img = np.asarray(img, dtype=np.float64) / 255.0
        img = transform.resize(img, dim, anti_aliasing=True)
    else:
        raise FileNotFoundError(f"Image not found: {image_path}")
    return img.astype(np.float32)


def make_test_image(shape='7', H=28, W=28):
    """Built-in test images: '7', 'L', 'T', 'circle'."""
    img = np.zeros((H, W), np.float64)
    if shape == '7':
        cy = H // 4
        img[cy-1:cy+2, W//4:3*W//4] = 0.9
        for i in range(H//4, 3*H//4):
            j = int(3*W//4 - (i - H//4) * 0.6)
            j = max(W//6, min(j, W-3))
            img[i:i+2, j:j+3] = 0.9
    elif shape == 'L':
        img[H//5:4*H//5, W//4:W//4+3] = 0.9
        img[4*H//5-2:4*H//5+1, W//4:3*W//4] = 0.9
    elif shape == 'T':
        img[H//5:H//5+3, W//5:4*W//5] = 0.9
        img[H//5:4*H//5, W//2-1:W//2+2] = 0.9
    elif shape == 'circle':
        cy, cx, r = H//2, W//2, min(H, W)//4
        yy, xx = np.ogrid[:H, :W]
        img[(yy - cy)**2 + (xx - cx)**2 <= r**2] = 0.9
    else:
        raise ValueError(f"Unknown shape: {shape}")
    return img.astype(np.float32)


# ─── Motion transforms (from your code) ─────────────────────
def _object_transform(obj, dim, translate_x, translate_y, angle_deg, scale):
    """Apply similarity transform: scale → rotate → shift (around center)."""
    h, w = dim
    # Resize + pad to target dim
    sf = min(h / obj.shape[0], w / obj.shape[1])
    obj = transform.resize(obj, (int(obj.shape[0]*sf), int(obj.shape[1]*sf)), anti_aliasing=True)
    pad_h = (h - obj.shape[0]) // 2
    pad_w = (w - obj.shape[1]) // 2
    obj = np.pad(obj, ((pad_h, h - obj.shape[0] - pad_h),
                       (pad_w, w - obj.shape[1] - pad_w)))

    cs = transform.SimilarityTransform(translation=[-w/2, -h/2])
    ts = transform.SimilarityTransform(scale=scale)
    tr = transform.SimilarityTransform(rotation=np.deg2rad(angle_deg))
    ci = transform.SimilarityTransform(translation=[w/2, h/2])
    tt = transform.SimilarityTransform(translation=[translate_x, translate_y])
    total = cs + ts + tr + ci + tt
    return transform.warp(obj, total.inverse, mode='constant', cval=0)


def _object_transform_affine(obj, dim, tx, ty, angle_deg, scale, shear_deg):
    """Affine transform with shear."""
    h, w = dim
    sf = min(h / obj.shape[0], w / obj.shape[1])
    obj = transform.resize(obj, (int(obj.shape[0]*sf), int(obj.shape[1]*sf)), anti_aliasing=True)
    pad_h = (h - obj.shape[0]) // 2
    pad_w = (w - obj.shape[1]) // 2
    obj = np.pad(obj, ((pad_h, h - obj.shape[0] - pad_h),
                       (pad_w, w - obj.shape[1] - pad_w)))

    cs = transform.SimilarityTransform(translation=[-w/2, -h/2])
    ta = transform.AffineTransform(
        scale=scale, rotation=np.deg2rad(angle_deg),
        shear=np.deg2rad(shear_deg),
        translation=[tx / max(scale, 1e-8), ty / max(scale, 1e-8)])
    ci = transform.SimilarityTransform(translation=[w/2, h/2])
    return transform.warp(obj, (cs + ta + ci).inverse, mode='constant', cval=0)


def _object_transform_swirl(obj, dim, scale, strength, radius):
    """Swirl deformation."""
    h, w = dim
    sf = min(h / obj.shape[0], w / obj.shape[1])
    obj = transform.resize(obj, (int(obj.shape[0]*sf), int(obj.shape[1]*sf)), anti_aliasing=True)
    pad_h = (h - obj.shape[0]) // 2
    pad_w = (w - obj.shape[1]) // 2
    obj = np.pad(obj, ((pad_h, h - obj.shape[0] - pad_h),
                       (pad_w, w - obj.shape[1] - pad_w)))
    cs = transform.SimilarityTransform(translation=[-w/2, -h/2])
    ts = transform.SimilarityTransform(scale=scale)
    ci = transform.SimilarityTransform(translation=[w/2, h/2])
    obj = transform.warp(obj, (cs + ts + ci).inverse)
    return transform.swirl(obj, center=(w/2, h/2), strength=strength,
                           radius=radius, mode='constant')


# ─── Motion sequence generation ──────────────────────────────
def simulate_motion(img, motion_type, num_frames, speed_factor=1.0, dim=None):
    """Generate motion sequence.

    Parameters
    ----------
    img          : [H,W] source image
    motion_type  : "translation", "rotation", "shear", "swirl",
                   "translation_and_rotation", "custom_se2"
    num_frames   : T
    speed_factor : amplitude multiplier
    dim          : (H, W) output dimension

    Returns
    -------
    frames : [T, H, W] numpy array
    """
    if dim is None:
        dim = img.shape
    h, w = dim
    frames = []

    if motion_type == "translation":
        for t in np.linspace(0, 1, num_frames):
            tx = t * 10 * speed_factor
            ty = t * 0  # only x-direction
            frames.append(_object_transform(img, dim, tx, ty, 0, 0.9))

    elif motion_type == "rotation":
        for t in np.linspace(0, 1, num_frames):
            angle = t * 180 * speed_factor
            frames.append(_object_transform(img, dim, 0, 0, angle, 0.9))

    elif motion_type == "shear":
        for t in np.linspace(0, 1, num_frames):
            shear = t * 30 * speed_factor
            frames.append(_object_transform_affine(img, dim, 0, 0, 0, 0.9, shear))

    elif motion_type == "swirl":
        for t in np.linspace(0, 1, num_frames):
            strength = t * 5 * speed_factor
            frames.append(_object_transform_swirl(img, dim, 0.9, strength, radius=70))

    elif motion_type == "translation_and_rotation":
        for t in np.linspace(0, 1, num_frames):
            tx = t * 10 * speed_factor
            ty = t * 10 * speed_factor
            angle = t * 180 * speed_factor
            frames.append(_object_transform(img, dim, tx, ty, angle, 0.9))

    elif motion_type == "custom_se2":
        # Pure SE(2): rotation + translation using scipy (matches our SE2Motion model)
        from scipy.ndimage import rotate as nd_rotate, shift as nd_shift
        # speed_factor encodes (total_vy, total_vx, total_omega_rad)
        # Default: use provided speed_factor as uniform scale
        vy, vx = speed_factor * 3.0, speed_factor * 3.0
        omega = speed_factor * 0.3  # radians
        for t in np.linspace(0, 1, num_frames):
            f = nd_rotate(img, np.degrees(omega * t), reshape=False, order=1, mode='constant')
            f = nd_shift(f, [vy * t, vx * t], order=1, mode='constant')
            frames.append(np.clip(f, 0, 1))

    else:
        raise ValueError(f"Unknown motion type: {motion_type}")

    return np.array(frames, dtype=np.float32)


# ─── Noise ───────────────────────────────────────────────────
def _add_noise(signal, snr_db, rng, sigma_abs=None):
    """AC-variance SNR convention (default) or detector-referred absolute σ.

    sigma_abs, when set, takes precedence over snr_db. The variance convention
    is NOT transferable across pattern families: for coefficient-ordered bases
    (Hadamard/Fourier/S-matrix) var(y) is dominated by a few huge low-frequency
    coefficients, so the same snr_db implies a far larger absolute noise floor
    than for random patterns (e.g. 64x64 tank scene @25dB: σ=1.77 vs 0.45) and
    drowns the coefficient tail. Cross-family benchmarks must use sigma_abs
    (same detector ⇒ same absolute noise; multiplex advantages then show up
    naturally instead of being erased by the convention).
    """
    if sigma_abs is not None and sigma_abs > 0:
        return signal + rng.normal(0, sigma_abs, signal.shape)
    if snr_db <= 0: return signal
    sig_pow = np.var(signal)
    noise_pow = sig_pow / (10 ** (snr_db / 10))
    return signal + rng.normal(0, np.sqrt(noise_pow), signal.shape)


# ─── Main data generation ────────────────────────────────────
def _measure_interpolated_np(patterns, gt_frames, K, T):
    """Numpy counterpart of SPIForwardModel.measure_interpolated (must stay identical).

    t_k = k/(K-1),  u_k = t_k*(T-1),  m_k = floor(u_k),  alpha_k = u_k - m_k.
    Boundary (m_k == T-1): m_k = T-2, alpha_k = 1.0.
    y[k] = (1-alpha_k)*<patterns[k], gt_frames[m_k]> + alpha_k*<patterns[k], gt_frames[m_k+1]>
    """
    meas = np.zeros(K, dtype=np.float64)
    for k in range(K):
        t_k   = k / max(K - 1, 1)
        u_k   = t_k * (T - 1)
        m_k   = int(np.floor(u_k))
        a_k   = u_k - m_k
        if m_k >= T - 1:          # boundary: only k == K-1 when K>1
            m_k, a_k = T - 2, 1.0
        f_interp = (1.0 - a_k) * gt_frames[m_k] + a_k * gt_frames[m_k + 1]
        meas[k]  = np.sum(patterns[k] * f_interp)
    return meas


def generate_spi_data(
    H=28, W=28, T=10, K=784,
    pattern_type="bernoulli",
    motion_type="custom_se2",
    speed_factor=1.0,
    snr_db=30.0,
    seed=42,
    shape="7",                 # built-in shape OR path to image
    motion_mode=2,             # 1: K frames, 2: fixed T frames
    # For custom_se2: explicit velocity/omega (overrides speed_factor)
    gt_velocity=None,          # [vy, vx] pixels total
    gt_omega=None,             # radians total
    gt_accel=None,             # [ay, ax] pixels·t⁻² (accelerated SE(2), optional)
    gt_beta=None,              # radians·t⁻² (angular acceleration, optional)
    noise_sigma_abs=None,      # detector-referred absolute noise σ (overrides snr_db)
    holdout_extra=0,           # Ke extra eval measurements (non-invasive GT-free metric)
    time_assignment_mode="uniform",   # uniform | interpolation
    pattern_order="sequential",       # sequential | stratified | random
) -> SPIData:
    """Generate complete dynamic SPI dataset.

    Supports all motion types from your simulation code,
    plus 'custom_se2' which matches the SE(2) motion model exactly.
    """
    rng = np.random.RandomState(seed)

    # 1. Load/create canonical image
    if os.path.exists(str(shape)):
        canonical = load_image(shape, (H, W))
    else:
        canonical = make_test_image(shape, H, W)

    # 2. Generate motion frames
    if motion_type == "custom_se2" and gt_velocity is not None:
        # Direct SE(2) using scipy (matches our reconstruction model)
        from scipy.ndimage import rotate as nd_rotate, shift as nd_shift
        vel = np.array(gt_velocity, dtype=np.float64)
        omega = gt_omega if gt_omega is not None else 0.0
        acc = np.array(gt_accel, dtype=np.float64) if gt_accel is not None else np.zeros(2)
        beta = gt_beta if gt_beta is not None else 0.0
        t_grid = np.linspace(0, 1, T)
        gt_frames = np.zeros((T, H, W), dtype=np.float64)
        for i, t in enumerate(t_grid):
            angle_t = omega * t + beta * t * t
            shift_t = vel * t + acc * t * t
            f = nd_rotate(canonical.astype(np.float64), np.degrees(angle_t),
                          reshape=False, order=1, mode='constant', cval=0)
            f = nd_shift(f, shift_t, order=1, mode='constant', cval=0)
            gt_frames[i] = np.clip(f, 0, 1)
    else:
        # Use simulate_motion (your old code's motion types)
        if gt_velocity is None:
            gt_velocity = [speed_factor * 3.0, speed_factor * 3.0]
        if gt_omega is None:
            gt_omega = 0.0
        vel = np.array(gt_velocity, dtype=np.float64)
        omega = gt_omega

        if motion_mode == 1:
            gt_frames = simulate_motion(canonical, motion_type, K, speed_factor, (H, W))
            T = K  # override T
        else:
            gt_frames = simulate_motion(canonical, motion_type, T, speed_factor, (H, W))

        t_grid = np.linspace(0, 1, gt_frames.shape[0])
        T = gt_frames.shape[0]

    if 't_grid' not in locals():
        t_grid = np.linspace(0, 1, T)

    # 3. Patterns (family rank order), then temporal display schedule
    patterns = generate_patterns(H, W, K, pattern_type, seed)
    patterns = apply_pattern_order(patterns, pattern_order, T, seed)

    # 4. Frame index: pattern k → frame f(k)
    ppf = max(1, int(np.ceil(K / T)))
    frame_idx = np.clip(np.arange(K) // ppf, 0, T - 1).astype(np.int64)

    # 5. Measurements
    if time_assignment_mode == "uniform":
        # Original uniform logic (unchanged)
        meas = np.zeros(K, dtype=np.float64)
        for k in range(K):
            meas[k] = np.sum(patterns[k] * gt_frames[frame_idx[k]])
    elif time_assignment_mode == "interpolation":
        meas = _measure_interpolated_np(patterns, gt_frames, K, T)
    else:
        raise ValueError(f"Unknown time_assignment_mode: {time_assignment_mode}")

    # 6. Noise
    meas = _add_noise(meas, snr_db, rng, sigma_abs=noise_sigma_abs).astype(np.float32)

    # 7. Optional non-invasive eval set (own RNG stream — steps 1-6 unaffected).
    #    In-training holdout (removing measurements) was found to destabilize
    #    the knife-edge joint scene+motion basin; this probe set does not.
    ev_pat = ev_meas = ev_fidx = None
    if holdout_extra and holdout_extra > 0 and time_assignment_mode == "uniform":
        Ke = int(holdout_extra)
        rng_e = np.random.RandomState(seed + 9999)
        ev_pat = rng_e.rand(Ke, H, W).astype(np.float32)
        ev_fidx = np.clip((np.arange(Ke) * T) // Ke, 0, T - 1).astype(np.int64)
        ev_meas = np.array([np.sum(ev_pat[j] * gt_frames[ev_fidx[j]])
                            for j in range(Ke)], dtype=np.float64)
        ev_meas = _add_noise(ev_meas, snr_db, rng_e,
                             sigma_abs=noise_sigma_abs).astype(np.float32)

    return SPIData(
        eval_patterns=ev_pat, eval_measurements=ev_meas, eval_frame_idx=ev_fidx,
        canonical=canonical, gt_frames=gt_frames.astype(np.float32),
        patterns=patterns, measurements=meas,
        frame_idx=frame_idx, t_grid=t_grid.astype(np.float32),
        gt_velocity=np.array(gt_velocity, dtype=np.float32),
        gt_omega=float(omega), motion_type=motion_type,
        H=H, W=W, T=T, K=K, snr_db=snr_db)
