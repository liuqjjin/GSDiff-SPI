"""SPI measurement pattern generation."""
import numpy as np


def generate_patterns(H, W, K, pattern_type="bernoulli", seed=0):
    """Generate K patterns of size [K, H, W]."""
    rng = np.random.RandomState(seed)
    if pattern_type == "bernoulli":
        return rng.randint(0, 2, (K, H, W)).astype(np.float32)
    elif pattern_type == "gaussian":
        return rng.randn(K, H, W).astype(np.float32)
    elif pattern_type == "random":
        return rng.rand(K, H, W).astype(np.float32)
    elif pattern_type == "s_matrix":
        return _s_matrix_patterns(H, W, K)
    else:
        raise ValueError(f"Unknown pattern type: {pattern_type}")


def _legendre(a, p):
    ls = pow(a, (p - 1) // 2, p)
    return -1 if ls == p - 1 else ls


def _make_S0(p=59, q=61):
    N = p * q
    qr_p = {x for x in range(1, p) if _legendre(x, p) == 1}
    qr_q = {x for x in range(1, q) if _legendre(x, q) == 1}
    f = np.zeros(N, np.int8); g = np.zeros(N, np.int8)
    for x in range(N):
        f[x] = 0 if x % p == 0 else (1 if x % p in qr_p else -1)
        g[x] = 0 if x % q == 0 else (1 if x % q in qr_q else -1)
    S0 = np.ones(N, np.float32)
    S0[(f == g) | (g == 0)] = 0.0
    return S0


def _s_matrix_patterns(H, W, K):
    p, q = H, W; N = p * q
    assert K <= N, f"K={K} > N={N}"
    S0 = _make_S0(p, q)
    i = np.arange(p)[:, None]; j = np.arange(q)[None, :]
    base = i + p * j
    ks = np.arange(K)[:, None, None]
    return S0[(ks + base[None]) % N].astype(np.float32)
