"""Utilities: seeding, config, metrics, I/O."""
import os, random, yaml, numpy as np, torch
from types import SimpleNamespace

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def get_device():
    d = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if d.type == "cuda": print(f"  GPU: {torch.cuda.get_device_name(0)}")
    return d

def _to_ns(d):
    if isinstance(d, dict): return SimpleNamespace(**{k: _to_ns(v) for k, v in d.items()})
    if isinstance(d, list): return [_to_ns(x) for x in d]
    return d

def load_config(path):
    with open(path) as f: return _to_ns(yaml.safe_load(f))

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def psnr_legacy_60db(pred, target):
    """Historical PSNR helper with a 60-dB exact-recovery sentinel."""
    mse = np.mean((pred - target)**2)
    if mse < 1e-12: return 60.0
    return 10 * np.log10(1.0 / mse)


def normalize_01_legacy_minmax(x):
    """Historical min-max normalization, including its constant-image policy."""
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-8: return np.zeros_like(x)
    return (x - lo) / (hi - lo)


# Backward-compatible names used throughout visualization and older scripts.
psnr_fn = psnr_legacy_60db
normalize_01 = normalize_01_legacy_minmax

def to_native(obj):
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, list): return [to_native(x) for x in obj]
    if isinstance(obj, dict): return {k: to_native(v) for k, v in obj.items()}
    return obj

def save_gif(frames_np, path, fps=10):
    import imageio
    imgs = [(normalize_01(np.clip(f, 0, None)) * 255).astype(np.uint8) for f in frames_np]
    imageio.mimsave(path, imgs, duration=int(1000/fps), loop=0)
