"""The metric set used for every model, so comparisons are one-to-one.

Reconstruction metrics (computed on the MASKED antennas, in physical units):
  - NMSE (ratio + dB)         : relative error power
  - Explained Variance Score  : fraction of variance explained
  - MAE on |error|            : typical error magnitude (robust)
  - mean abs phase error (deg): domain metric -- is the phase preserved?
  - complex correlation       : shape agreement, scale-independent
  - P95 / max error           : tail behaviour
  - NMSE by number masked     : how error grows with missing antennas

DoA metric (per scenario, via MUSIC):
  - mean angle error and detection rate, computed on the RECONSTRUCTED array
    and on the TRUE array, so you see how much reconstruction degraded the
    angle information rather than how good MUSIC is.
"""

import numpy as np


def _denorm(arr, stats):
    """(n, 2M) -> (n, M, 2) physical complex-as-real."""
    M = stats["mean"].shape[-2] if stats["mean"].ndim == 3 else arr.shape[1] // 2
    n = arr.shape[0]
    a = arr.reshape(n, -1, 2)
    mean = stats["mean"].reshape(1, 1, 2)
    std = stats["std"].reshape(1, 1, 2)
    return a * std + mean


def reconstruction_metrics(Yhat, Y, mask, stats):
    """Yhat, Y: (n, 2M); mask: (n, M) 1=observed 0=masked."""
    yh = _denorm(Yhat, stats)           # (n, M, 2)
    yt = _denorm(Y, stats)
    ch = yh[..., 0] + 1j * yh[..., 1]   # (n, M) complex
    ct = yt[..., 0] + 1j * yt[..., 1]
    hidden = (mask == 0)                # (n, M) bool

    e = (ch - ct)[hidden]               # complex errors on masked entries
    t = ct[hidden]
    err_mag = np.abs(e)

    nmse = (np.abs(e) ** 2).sum() / ((np.abs(t) ** 2).sum() + 1e-12)

    # explained variance on stacked real+imag scalars
    es = np.concatenate([e.real, e.imag])
    ts = np.concatenate([t.real, t.imag])
    evs = 1.0 - es.var() / (ts.var() + 1e-12)

    mae = err_mag.mean()
    # phase error (deg), wrapped, weighted away from near-zero magnitudes
    dphi = np.angle(ch[hidden] * np.conj(ct[hidden]))
    phase_err = np.abs(np.degrees(dphi)).mean()

    corr = np.abs(np.sum(np.conj(ch[hidden]) * ct[hidden])) / (
        np.sqrt((np.abs(ch[hidden]) ** 2).sum() * (np.abs(ct[hidden]) ** 2).sum()) + 1e-12)

    p95 = np.percentile(err_mag, 95)
    emax = err_mag.max()

    # NMSE by number of masked antennas
    counts = (mask == 0).sum(axis=1).astype(int)
    by_count = {}
    for c in np.unique(counts):
        rows = counts == c
        hh = hidden[rows]
        ee = (ch[rows] - ct[rows])[hh]
        tt = ct[rows][hh]
        by_count[int(c)] = float((np.abs(ee) ** 2).sum() / ((np.abs(tt) ** 2).sum() + 1e-12))

    return {
        "NMSE": float(nmse),
        "NMSE_dB": float(10 * np.log10(nmse + 1e-12)),
        "ExplVar": float(evs),
        "MAE": float(mae),
        "PhaseErrDeg": float(phase_err),
        "ComplexCorr": float(corr),
        "P95": float(p95),
        "MaxErr": float(emax),
        "NMSE_by_masked": by_count,
    }


# ----------------------------- DoA (MUSIC) --------------------------------
def _steering(angle_deg, M, d=0.5):
    m = np.arange(M)
    return np.exp(-1j * 2 * np.pi * d * m * np.sin(np.deg2rad(angle_deg)))


def music_angles(X_scen, K, M, scan):
    """X_scen: (P, M) complex pulses. Return K estimated angles (deg)."""
    angles, _ = _music_angles_and_spectrum(X_scen, K, M, scan)
    return angles


def _music_angles_and_spectrum(X_scen, K, M, scan):
    """Returns (angles_deg, spectrum) where spectrum is the full MUSIC pseudo-spectrum."""
    R = (X_scen.conj().T @ X_scen) / X_scen.shape[0]      # (M, M)
    w, V = np.linalg.eigh(R)
    En = V[:, :M - K]
    EnEn = En @ En.conj().T
    spec = np.empty(scan.size)
    for i, a in enumerate(scan):
        s = _steering(a, M)
        spec[i] = 1.0 / np.real(s.conj() @ EnEn @ s).clip(1e-12)
    # pick K highest local peaks
    peaks = [i for i in range(1, scan.size - 1)
             if spec[i] > spec[i - 1] and spec[i] >= spec[i + 1]]
    peaks.sort(key=lambda i: spec[i], reverse=True)
    sel = sorted(scan[peaks[:K]]) if peaks else []
    return np.array(sel, dtype=float), spec


def _match_error(est, true):
    """Greedy nearest-match mean abs error and per-target hits (<2 deg)."""
    true = list(np.sort(true))
    est = list(np.sort(est))
    errs = []
    for t in true:
        if not est:
            errs.append(60.0)         # missed target -> large penalty
            continue
        j = int(np.argmin([abs(t - e) for e in est]))
        errs.append(abs(t - est.pop(j)))
    errs = np.array(errs)
    return errs.mean(), float((errs < 2.0).mean())


def doa_metric(reconstruct_fn, test_scen, cfg, stats, n_eval=None):
    """reconstruct_fn(masked_signal_real (P,M,2), mask (P,M)) -> recon (P,M,2) normalized.

    We mask each scenario's pulses with the same per-pulse scheme, reconstruct,
    then run MUSIC on the reconstructed array vs the true array.
    Returns dict with angle error / detection rate for both.
    """
    scan = np.linspace(-70, 70, 1401)
    M = cfg.M
    mean = stats["mean"].reshape(1, M, 2) if stats["mean"].shape[-2] == M else stats["mean"].reshape(1, 1, 2)
    std = stats["std"].reshape(mean.shape)

    n_eval = n_eval or min(cfg.doa_eval_scenarios, test_scen["n_scen"])
    rng = np.random.default_rng(cfg.seed + 7)

    rec_errs, rec_hits, true_errs, true_hits = [], [], [], []
    angles_true, angles_recon, angles_true_ref = [], [], []
    spectrum_data = {}  # captured for the first scenario
    for s in range(n_eval):
        sig = test_scen["signals"][s]                 # (P, M) complex
        K = len(test_scen["angles"][s])
        truth = test_scen["angles"][s]

        real = np.stack([sig.real, sig.imag], axis=-1).astype(np.float32)  # (P,M,2)
        xn = (real - mean) / std
        # per-pulse masks
        m = np.ones((sig.shape[0], M), dtype=np.float32)
        for p in range(sig.shape[0]):
            c = rng.integers(cfg.mask_min, cfg.mask_max + 1)
            m[p, rng.choice(M, int(c), replace=False)] = 0.0
        masked = xn * m[:, :, None]

        rec = reconstruct_fn(masked, m)                # (P,M,2) normalized
        rec = rec * std + mean
        rec_c = rec[..., 0] + 1j * rec[..., 1]

        est_r, spec_r = _music_angles_and_spectrum(rec_c, K, M, scan)
        est_t, spec_t = _music_angles_and_spectrum(sig, K, M, scan)
        er, hr = _match_error(est_r, truth)
        et, ht = _match_error(est_t, truth)
        rec_errs.append(er); rec_hits.append(hr)
        true_errs.append(et); true_hits.append(ht)
        angles_true.append(np.sort(truth).tolist())
        angles_recon.append(np.sort(est_r).tolist())
        angles_true_ref.append(np.sort(est_t).tolist())

        if s == 0:
            spectrum_data = {
                "scan": scan,
                "recon": spec_r,
                "true": spec_t,
                "angles": np.sort(truth).tolist(),
            }

    return {
        "DoA_err_recon_deg": float(np.mean(rec_errs)),
        "DoA_hit_recon": float(np.mean(rec_hits)),
        "DoA_err_true_deg": float(np.mean(true_errs)),
        "DoA_hit_true": float(np.mean(true_hits)),
        "n_scenarios": n_eval,
        "angles_true": angles_true,
        "angles_recon": angles_recon,
        "angles_true_ref": angles_true_ref,
        "spectrum_scan": spectrum_data.get("scan", []),
        "spectrum_recon": spectrum_data.get("recon", []),
        "spectrum_true": spectrum_data.get("true", []),
        "spectrum_angles": spectrum_data.get("angles", []),
    }
