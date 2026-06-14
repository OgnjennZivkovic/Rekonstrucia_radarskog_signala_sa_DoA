"""Turn the raw CSV into arrays every model can consume.

The reconstruction unit is one pulse. For each pulse we build:
    X    (n, 3M) : masked signal (2M real/imag, hidden antennas zeroed) + mask (M)
    Y    (n, 2M) : the full signal (real/imag) -- the target
    mask (n, M)  : 1 = observed, 0 = masked (the antennas to predict)

Masks are random PER PULSE but fixed (seeded) so every member sees the same
masked inputs -- essential for a fair comparison and for stacking.

Splitting is by scenario. Normalization stats come from train only. The test
split is also kept grouped by scenario (complex, de-normalized) so the DoA
metric can run MUSIC per scenario.
"""

import numpy as np
from config import Config
from data import load_csv


def _complex_to_real(sig):
    """(..., M) complex -> (..., M, 2) float32."""
    return np.stack([sig.real, sig.imag], axis=-1).astype(np.float32)


def _make_masks(n, M, mask_min, mask_max, seed):
    rng = np.random.default_rng(seed)
    counts = rng.integers(mask_min, mask_max + 1, size=n)
    masks = np.ones((n, M), dtype=np.float32)
    for i in range(n):
        idx = rng.choice(M, size=int(counts[i]), replace=False)
        masks[i, idx] = 0.0
    return masks


def _augment(sig, cfg, seed):
    """Train-only augmentation on complex pulses (n, M). Returns augmented copy."""
    rng = np.random.default_rng(seed)
    out = sig.copy()
    if cfg.aug_phase_rot:
        phi = rng.uniform(0, 2 * np.pi, size=(out.shape[0], 1))
        out = out * np.exp(1j * phi)
    if cfg.aug_mirror:
        flip = rng.random(out.shape[0]) < 0.5
        out[flip] = np.conjugate(out[flip, ::-1])
    return out


def build_arrays(cfg: Config):
    signals, angles = load_csv(cfg.data_csv, cfg.M, cfg.P)      # (n_scen, P, M) complex
    n_scen = signals.shape[0]

    if cfg.per_pulse_power_norm:
        power = np.mean(np.abs(signals) ** 2, axis=-1, keepdims=True)
        signals = signals / np.sqrt(power + 1e-12)

    rng = np.random.default_rng(cfg.seed)
    order = rng.permutation(n_scen)
    n_test = int(n_scen * cfg.test_frac)
    n_val = int(n_scen * cfg.val_frac)
    test_idx = order[:n_test]
    val_idx = order[n_test:n_test + n_val]
    train_idx = order[n_test + n_val:]

    def pulses_of(idx):
        return signals[idx].reshape(-1, cfg.M)                 # (n*P, M) complex

    tr_c = pulses_of(train_idx)
    if cfg.aug_phase_rot or cfg.aug_mirror:
        tr_c = _augment(tr_c, cfg, cfg.seed + 99)
    va_c = pulses_of(val_idx)
    te_c = pulses_of(test_idx)

    tr = _complex_to_real(tr_c)                                # (n, M, 2)
    va = _complex_to_real(va_c)
    te = _complex_to_real(te_c)

    mean = tr.mean(axis=(0, 1), keepdims=True)                 # (1,1,2)
    std = tr.std(axis=(0, 1), keepdims=True) + 1e-8

    def pack(real_arr, seed):
        x = (real_arr - mean) / std                            # normalize (n, M, 2)
        m = _make_masks(x.shape[0], cfg.M, cfg.mask_min, cfg.mask_max, seed)
        masked = x * m[:, :, None]
        X = np.concatenate([masked.reshape(len(x), -1), m], axis=1).astype(np.float32)  # (n,3M)
        Y = x.reshape(len(x), -1).astype(np.float32)           # (n,2M)
        return X, Y, m

    Xtr, Ytr, Mtr = pack(tr, cfg.seed + 1)
    Xva, Yva, Mva = pack(va, cfg.seed + 2)
    Xte, Yte, Mte = pack(te, cfg.seed + 3)

    stats = {"mean": mean, "std": std}

    # test scenarios kept grouped (complex, de-normalized) for DoA
    test_scen = {
        "signals": signals[test_idx],                          # (n_test_scen, P, M) complex
        "angles": [angles[i] for i in test_idx],
        "n_scen": len(test_idx),
    }

    splits = {
        "train": (Xtr, Ytr, Mtr),
        "val": (Xva, Yva, Mva),
        "test": (Xte, Yte, Mte),
    }
    return splits, stats, test_scen


def cap_rows(X, Y, cap, seed=0):
    """Subsample rows for heavy members."""
    if cap is None or X.shape[0] <= cap:
        return X, Y
    rng = np.random.default_rng(seed)
    sel = rng.choice(X.shape[0], size=cap, replace=False)
    return X[sel], Y[sel]
