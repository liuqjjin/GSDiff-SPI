"""SPI measurement pattern generation.

Families
--------
random / bernoulli / gaussian : i.i.d. random patterns (seeded)
hadamard_natural / hadamard_walsh / hadamard_cc :
    Separable Walsh-Hadamard basis. Row (a,b) of H_{HW} = H_H (x) H_W reshaped
    to [H,W] is the outer product outer(H_H[a], H_W[b]), so ordering metrics
    have closed forms from the per-axis sequencies s(a), s(b):
      natural : Sylvester index a*W + b
      walsh   : 2D sequency  s(a) + s(b)           (zigzag)
      cc      : cake-cutting block count (s(a)+1)*(s(b)+1)   [Yu 2019]
    The all-ones DC row is EXCLUDED (zero gradient under the z-score loss).
    Entries are +/-1 (differential acquisition convention: costs 2x display
    time on real hardware; documented, not modeled).
fourier : real 3-step phase-shifting cosine patterns 0.5+0.5*cos(2pi(u x/W
    + v y/H) + phi), phi in {0, 2pi/3, 4pi/3}; one conjugate representative
    per frequency, ordered by radius (low frequency first); each 3-phase
    triplet is temporally adjacent. DC (0,0) excluded.
s_matrix : cyclic S-matrix. Twin-prime construction requires H, W to be
    twin primes (e.g. 59x61) — asserted, since Legendre symbols are invalid
    for composite moduli. For power-of-two grids use s_matrix_m.
s_matrix_m : cyclic S-matrix from a maximal-length LFSR (m-sequence),
    N = 2^n - 1. Valid when H*W == 2^n (one dead pixel at the last position)
    or H*W == 2^n - 1 (exact).

Temporal scheduling (apply_pattern_order)
-----------------------------------------
frame_idx[k] = k // ppf makes the pattern DISPLAY ORDER the time axis, so for
deterministically ordered families the schedule interacts with motion:
  sequential : slot k shows rank k (frame 1 sees only the lowest-rank
               patterns, late frames only high ranks)
  stratified : frame f's slots show ranks {f, f+T, f+2T, ...} — every frame
               samples the whole ranked spectrum (recommended for joint
               scene+motion recovery)
  random     : seeded random permutation of the ranks (control)
"""
import numpy as np


# ─── Random families ─────────────────────────────────────────
def _random_patterns(H, W, K, pattern_type, seed):
    rng = np.random.RandomState(seed)
    if pattern_type == "bernoulli":
        return rng.randint(0, 2, (K, H, W)).astype(np.float32)
    if pattern_type == "gaussian":
        return rng.randn(K, H, W).astype(np.float32)
    return rng.rand(K, H, W).astype(np.float32)   # "random"


# ─── Walsh-Hadamard family ───────────────────────────────────
def _hadamard_1d(n):
    """Sylvester Hadamard matrix H_n, n a power of 2. [n,n] int8."""
    assert n > 0 and (n & (n - 1)) == 0, f"Hadamard size {n} is not a power of 2"
    Hm = np.array([[1]], dtype=np.int8)
    while Hm.shape[0] < n:
        Hm = np.block([[Hm, Hm], [Hm, -Hm]]).astype(np.int8)
    return Hm


def _sequency(Hm):
    """Sign changes per row. [n] int."""
    return (np.diff(Hm.astype(np.int16), axis=1) != 0).sum(axis=1)


def _hadamard_patterns(H, W, K, ordering):
    Hh, Hw = _hadamard_1d(H), _hadamard_1d(W)
    sh, sw = _sequency(Hh), _sequency(Hw)
    a, b = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    a, b = a.ravel(), b.ravel()
    if ordering == "natural":
        key = a * W + b
    elif ordering == "walsh":
        key = (sh[a] + sw[b]) * (H * W) + sh[a]          # zigzag, tie-break s(a)
    elif ordering == "cc":
        blocks = (sh[a] + 1) * (sw[b] + 1)
        key = blocks * (H * W) + (sh[a] + sw[b])          # tie-break 2D sequency
    else:
        raise ValueError(f"Unknown hadamard ordering: {ordering}")
    order = np.argsort(key, kind="stable")
    order = order[~((a[order] == 0) & (b[order] == 0))]   # drop DC row
    assert K <= len(order), f"K={K} > {len(order)} available Hadamard patterns"
    idx = order[:K]
    # outer(H_H[a], H_W[b]) for each selected (a,b)
    return np.einsum("kh,kw->khw", Hh[a[idx]].astype(np.float32),
                     Hw[b[idx]].astype(np.float32))


# ─── Fourier family (3-step phase shifting) ──────────────────
def _fourier_patterns(H, W, K):
    fv = np.fft.fftfreq(H, d=1.0 / H).astype(int)         # signed integer freqs
    fu = np.fft.fftfreq(W, d=1.0 / W).astype(int)
    freqs = [(v, u) for v in fv for u in fu
             if (v > 0 or (v == 0 and u > 0))]            # conjugate half-plane, no DC
    freqs.sort(key=lambda t: (t[0] ** 2 + t[1] ** 2, np.arctan2(t[0], t[1])))
    n_freq = int(np.ceil(K / 3))
    assert n_freq <= len(freqs), f"K={K} needs {n_freq} freqs > {len(freqs)} available"
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    pats = np.empty((K, H, W), np.float32)
    k = 0
    for (v, u) in freqs[:n_freq]:
        base = 2 * np.pi * (u * xx / W + v * yy / H)
        for phi in (0.0, 2 * np.pi / 3, 4 * np.pi / 3):
            if k == K:
                break
            pats[k] = 0.5 + 0.5 * np.cos(base + phi)
            k += 1
    return pats


# ─── Cyclic S-matrices ───────────────────────────────────────
def _is_prime(n):
    if n < 2:
        return False
    return all(n % d for d in range(2, int(n ** 0.5) + 1))


def _legendre(a, p):
    ls = pow(a, (p - 1) // 2, p)
    return -1 if ls == p - 1 else ls


def _make_S0(p, q):
    """Twin-prime difference-set sequence, length N = p*q, values {0,1}."""
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
    if not (_is_prime(p) and _is_prime(q) and q == p + 2):
        raise ValueError(
            f"s_matrix (twin-prime) requires H, W to be twin primes "
            f"(e.g. 5x7, 11x13, 29x31, 59x61); got {H}x{W}. "
            f"For power-of-two grids use pattern_type: s_matrix_m.")
    assert K <= N, f"K={K} > N={N}"
    S0 = _make_S0(p, q)
    i = np.arange(p)[:, None]; j = np.arange(q)[None, :]
    base = i + p * j
    ks = np.arange(K)[:, None, None]
    return S0[(ks + base[None]) % N].astype(np.float32)


# Primitive polynomial tap exponents over GF(2), x^n + ... + 1
_PRIMITIVE_TAPS = {
    2: (2, 1), 3: (3, 2), 4: (4, 3), 5: (5, 3), 6: (6, 5), 7: (7, 6),
    8: (8, 6, 5, 4), 9: (9, 5), 10: (10, 7), 11: (11, 9), 12: (12, 6, 4, 1),
    13: (13, 4, 3, 1), 14: (14, 5, 3, 1), 15: (15, 14), 16: (16, 15, 13, 4),
}


def _lfsr_run(n, taps):
    N = 2 ** n - 1
    state = (1 << n) - 1                      # any non-zero seed
    seq = np.empty(N, np.int8)
    for i in range(N):
        seq[i] = state & 1
        fb = 0
        for t in taps:                        # tap t → bit (n - t), Fibonacci右移约定
            fb ^= (state >> (n - t)) & 1
        state = (state >> 1) | (fb << (n - 1))
    return seq, state


def _m_sequence(n):
    """Maximal-length LFSR bit sequence, period N = 2^n - 1, values {0,1}.

    Tries the tabulated taps and their reciprocal-polynomial counterpart
    (the shift-direction convention differs between references); a maximal
    sequence must return to the seed after N steps and be balanced.
    """
    taps = _PRIMITIVE_TAPS[n]
    recip = tuple(sorted({n} | {n - t for t in taps if t != n}, reverse=True))
    for cand in (taps, recip):
        seq, end_state = _lfsr_run(n, cand)
        if end_state == (1 << n) - 1 and seq.sum() == 2 ** (n - 1):
            return seq
    raise AssertionError(f"no maximal LFSR found for n={n}")


def _m_seq_s_matrix_patterns(H, W, K):
    HW = H * W
    n = HW.bit_length() - 1
    if HW == 2 ** n:                          # e.g. 64x64=4096 → N=4095, 1 dead pixel
        N, dead = HW - 1, 1
    elif HW == 2 ** (HW.bit_length()) - 1:    # e.g. 63x65=4095 exact
        N, dead = HW, 0
        n = HW.bit_length()
    else:
        raise ValueError(
            f"s_matrix_m requires H*W == 2^n or 2^n - 1; got {H}x{W}={HW}")
    if n not in _PRIMITIVE_TAPS:
        raise ValueError(f"No primitive polynomial tabulated for n={n}")
    assert K <= N, f"K={K} > N={N}"
    S0 = _m_sequence(n).astype(np.float32)
    ks = np.arange(K)[:, None]
    rows = S0[(ks + np.arange(N)[None]) % N]          # [K, N] cyclic shifts
    if dead:
        rows = np.concatenate([rows, np.zeros((K, 1), np.float32)], axis=1)
    return rows.reshape(K, H, W)


# ─── Dispatch ────────────────────────────────────────────────
def generate_patterns(H, W, K, pattern_type="bernoulli", seed=0):
    """Generate K patterns of size [K, H, W], in the family's rank order."""
    if pattern_type in ("bernoulli", "gaussian", "random"):
        return _random_patterns(H, W, K, pattern_type, seed)
    if pattern_type.startswith("hadamard"):
        ordering = pattern_type.split("_", 1)[1] if "_" in pattern_type else "cc"
        return _hadamard_patterns(H, W, K, ordering)
    if pattern_type == "fourier":
        return _fourier_patterns(H, W, K)
    if pattern_type == "s_matrix":
        return _s_matrix_patterns(H, W, K)
    if pattern_type == "s_matrix_m":
        return _m_seq_s_matrix_patterns(H, W, K)
    raise ValueError(f"Unknown pattern type: {pattern_type}")


# ─── Temporal scheduling ─────────────────────────────────────
def apply_pattern_order(patterns, order, T, seed=0):
    """Reorder the display schedule of rank-ordered patterns. [K,H,W] → [K,H,W].

    order: 'sequential' (identity) | 'stratified' (frame f shows ranks
    {f, f+T, ...}) | 'random' (seeded permutation).
    """
    K = patterns.shape[0]
    if order == "sequential":
        return patterns
    if order == "random":
        perm = np.random.RandomState(seed).permutation(K)
    elif order == "stratified":
        ppf = max(1, int(np.ceil(K / T)))
        f = np.clip(np.arange(K) // ppf, 0, T - 1)
        j = np.arange(K) - f * ppf
        r = f + j * T
        # slot k shows rank r_k; rank-compress r into a valid permutation
        perm = np.argsort(np.argsort(r, kind="stable"), kind="stable")
    else:
        raise ValueError(f"Unknown pattern_order: {order}")
    return patterns[perm]
