# Ucitavanje podataka, maskiranje, podela, predprocesiranje, agumentacije, ...

import re
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

_NUM = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?") # priprema za citanje vrednosti iz data seta


def parse_complex(s):# konvertuje sve csv podatke u stringove bez razmaka
    
    return complex(str(s).strip().replace(" ", ""))


def parse_angles(s): # prebacuje uglove u np.array
    return np.array([float(x) for x in _NUM.findall(str(s))], dtype=np.float32)


def load_csv(path, M, P): # vrsi kesiranje rezultata i ucitava csv
    
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
        raw = df[ant_cols].to_numpy()                       
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


def complex_to_real(signals): # razdvaja kompleksni signal na imaginarni i realni
    
    return np.stack([signals.real, signals.imag], axis=-1).astype(np.float32)


class MaskedAntennaDataset(Dataset): #pravi maske za antene pre treniranja
    

    def __init__(self, pulses, cfg, mean, std, train=True):
        self.x = (pulses - mean) / std # z-score normalizacija
        self.cfg = cfg
        self.train = train
        self.M = cfg.M
        self._rng = np.random.default_rng(cfg.seed)

    def __len__(self): # broj uzoraka koje imamo
        return self.x.shape[0]

    def _make_mask(self, idx): # pravi masku 
        if self.train: # random maska za svaki slucaj
            rng = np.random
        else: # fiksna maska za sve slucajeve
            rng = np.random.default_rng(self.cfg.seed + idx)  
        #Koliko se antena maskira
        n_mask = rng.integers(self.cfg.mask_min, self.cfg.mask_max + 1) \
            if hasattr(rng, "integers") else rng.randint(self.cfg.mask_min, self.cfg.mask_max + 1)
        #Koje antene sakrivamo
        idxs = rng.choice(self.M, size=int(n_mask), replace=False) \
            if hasattr(rng, "choice") else rng.choice(self.M, int(n_mask), replace=False)
        mask = np.ones(self.M, dtype=np.float32)
        mask[idxs] = 0.0
        return mask

    def __getitem__(self, idx): # uzima skup i vrsi maskiranje nad podacima
        target = self.x[idx]                       # (M, 2)
        mask = self._make_mask(idx)                # (M,)
        masked = target * mask[:, None]            # zero the hidden antennas
        return (
            torch.from_numpy(masked),
            torch.from_numpy(mask),
            torch.from_numpy(target),
        )


def build_dataloaders(cfg): # Ucitava podatke

    signals, _angles = load_csv(cfg.data_csv, cfg.M, cfg.P)   # ne ucitava podatke o uglovima
    n_scen = signals.shape[0]

    if cfg.per_pulse_power_norm: # snaga po svim pulsevima se ujednacuje
        power = np.mean(np.abs(signals) ** 2, axis=-1, keepdims=True)  # (n_scen, P, 1)
        signals = signals / np.sqrt(power + 1e-12)

    rng = np.random.default_rng(cfg.seed)
    order = rng.permutation(n_scen)
    # odredjuje koliko scenarija se prosledjuje kom delu
    n_test = int(n_scen * cfg.test_frac)
    n_val = int(n_scen * cfg.val_frac)
    test_idx = order[:n_test]
    val_idx = order[n_test:n_test + n_val]
    train_idx = order[n_test + n_val:]

    def gather(idx): # skuplja pulseve i scenarije u jednu matricu
        block = signals[idx]                       # (n, P, M)
        block = block.reshape(-1, cfg.M)           # (n*P, M) pulses
        return complex_to_real(block)              # (n*P, M, 2)

    train_pulses = gather(train_idx)
    val_pulses = gather(val_idx)
    test_pulses = gather(test_idx)

    #Racunaju se normalizovane statistike
    mean = train_pulses.mean(axis=(0, 1), keepdims=True)   # (1,1,2)
    std = train_pulses.std(axis=(0, 1), keepdims=True) + 1e-8

    #Kreiraju se data set objekti
    train_ds = MaskedAntennaDataset(train_pulses, cfg, mean, std, train=True)
    val_ds = MaskedAntennaDataset(val_pulses, cfg, mean, std, train=False)
    test_ds = MaskedAntennaDataset(test_pulses, cfg, mean, std, train=False)

    #Ucitavanje podataka, train nasumicno, kako se ne bi upamtila sekvenca
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    stats = {"mean": mean, "std": std} # sluzi za denormalizacuju pri vizuelizaciji
    return train_loader, val_loader, test_loader, stats
