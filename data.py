"""Load the radar CSV, group pulses into scenarios, and serve masked samples.

The reconstruction unit is a single pulse: a length-M complex vector (the M
antenna readings). A random subset of antennas is "masked" (hidden), and the
model must predict the hidden values. Splitting is done by *scenario* so pulses
from one scenario never straddle the train/val/test boundary.

A complex signal is represented as a real tensor of shape (M, 2):
column 0 = real part, column 1 = imaginary part.
"""

import re
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

_NUM = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def parse_complex(s):
    """'(0.15+0.21j)' -> complex. Tolerant of stray spaces/parentheses."""
    return complex(str(s).strip().replace(" ", ""))


def parse_angles(s):
    """'[58 -57 21]' -> np.array([58., -57., 21.])."""
    return np.array([float(x) for x in _NUM.findall(str(s))], dtype=np.float32)


def load_csv(path, M, P):
    """Read the CSV into per-scenario signal blocks (with .npz caching).

    Parsing ~512k rows x 29 complex strings is slow, so after the first load
    the parsed arrays are cached next to the CSV as `<path>.cache.npz` and
    reused on later runs (cache is rebuilt if the CSV is newer).

    Returns:
        signals: (n_scenarios, P, M) complex64
        angles:  list of np.ndarray (variable length, one per scenario)
    """
    import os
    import pandas as pd

    cache = path + ".cache.npz"
    if os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(path):
        z = np.load(cache, allow_pickle=True)
        sig = z["sig"]
        ang_per_row = list(z["angles"])
    else:
        df = pd.read_csv(path)
        ant_cols = [f"ant_{i+1}" for i in range(M)]
        raw = df[ant_cols].to_numpy()                       # (rows, M) object/str
        sig = np.vectorize(parse_complex)(raw).astype(np.complex64)
        ang_per_row = df["angle"].astype(str).tolist()
        np.savez_compressed(cache, sig=sig, angles=np.array(ang_per_row, dtype=object))
        print(f"[data] parsed {sig.shape[0]} rows; cached to {cache}")

    n_rows = sig.shape[0]
    if n_rows % P != 0:
        raise ValueError(
            f"{n_rows} rows is not a multiple of P={P}. "
            f"Set P to the true pulses-per-scenario for this file."
        )
    n_scen = n_rows // P
    signals = sig.reshape(n_scen, P, M)
    angles = [parse_angles(ang_per_row[s * P]) for s in range(n_scen)]
    return signals, angles


def complex_to_real(signals):
    """(..., M) complex -> (..., M, 2) float32 (real, imag)."""
    return np.stack([signals.real, signals.imag], axis=-1).astype(np.float32)


class MaskedAntennaDataset(Dataset):
    """Each item is one pulse with a random set of antennas masked out.

    __getitem__ returns:
        masked_signal: (M, 2) float32  -- observed antennas, masked ones zeroed
        mask:          (M,)   float32  -- 1.0 = observed, 0.0 = masked (to predict)
        target:        (M, 2) float32  -- the full, unmasked signal (ground truth)
    """

    def __init__(self, pulses, cfg, mean, std, train=True):
        # pulses: (n_pulses, M, 2) float32, already normalized outside or here
        self.x = (pulses - mean) / std
        self.cfg = cfg
        self.train = train
        self.M = cfg.M
        # for reproducible val/test masks, pre-seed a generator per item
        self._rng = np.random.default_rng(cfg.seed)

    def __len__(self):
        return self.x.shape[0]

    def _make_mask(self, idx):
        if self.train:
            rng = np.random
        else:
            rng = np.random.default_rng(self.cfg.seed + idx)  # fixed mask per eval item
        n_mask = rng.integers(self.cfg.mask_min, self.cfg.mask_max + 1) \
            if hasattr(rng, "integers") else rng.randint(self.cfg.mask_min, self.cfg.mask_max + 1)
        idxs = rng.choice(self.M, size=int(n_mask), replace=False) \
            if hasattr(rng, "choice") else rng.choice(self.M, int(n_mask), replace=False)
        mask = np.ones(self.M, dtype=np.float32)
        mask[idxs] = 0.0
        return mask

    def __getitem__(self, idx):
        target = self.x[idx]                       # (M, 2)
        mask = self._make_mask(idx)                # (M,)
        masked = target * mask[:, None]            # zero the hidden antennas
        return (
            torch.from_numpy(masked),
            torch.from_numpy(mask),
            torch.from_numpy(target),
        )


def build_dataloaders(cfg):
    """Load CSV, split by scenario, normalize, and wrap in DataLoaders.

    Returns train_loader, val_loader, test_loader, stats(dict with mean/std).
    """
    signals, _angles = load_csv(cfg.data_csv, cfg.M, cfg.P)   # (n_scen, P, M) complex
    n_scen = signals.shape[0]

    if cfg.per_pulse_power_norm:
        # scale every pulse to unit average power (applied to ALL splits;
        # it's deterministic preprocessing, not augmentation)
        power = np.mean(np.abs(signals) ** 2, axis=-1, keepdims=True)  # (n_scen, P, 1)
        signals = signals / np.sqrt(power + 1e-12)

    rng = np.random.default_rng(cfg.seed)
    order = rng.permutation(n_scen)
    n_test = int(n_scen * cfg.test_frac)
    n_val = int(n_scen * cfg.val_frac)
    test_idx = order[:n_test]
    val_idx = order[n_test:n_test + n_val]
    train_idx = order[n_test + n_val:]

    def gather(idx):
        block = signals[idx]                       # (n, P, M)
        block = block.reshape(-1, cfg.M)           # (n*P, M) pulses
        return complex_to_real(block)              # (n*P, M, 2)

    train_pulses = gather(train_idx)
    val_pulses = gather(val_idx)
    test_pulses = gather(test_idx)

    # normalization stats from TRAIN only (per real/imag channel)
    mean = train_pulses.mean(axis=(0, 1), keepdims=True)   # (1,1,2)
    std = train_pulses.std(axis=(0, 1), keepdims=True) + 1e-8

    train_ds = MaskedAntennaDataset(train_pulses, cfg, mean, std, train=True)
    val_ds = MaskedAntennaDataset(val_pulses, cfg, mean, std, train=False)
    test_ds = MaskedAntennaDataset(test_pulses, cfg, mean, std, train=False)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    stats = {"mean": mean, "std": std}
    return train_loader, val_loader, test_loader, stats
