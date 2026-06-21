"""Per-scenario MMSE (covariance-based) reconstruction.

Unlike the per-pulse members, this uses ALL pulses of a scenario at once: from
the observed antennas across the scenario it estimates the spatial covariance R,
then fills each pulse's masked antennas with the linear MMSE (conditional-mean)
estimate

    x_hat[masked] = R[masked, obs] @ inv(R[obs, obs]) @ x[obs].

R is estimated with the available-case (pairwise) estimator so per-pulse masking
is handled: each pair (i, j) is averaged only over the pulses where both are
observed, then a small diagonal loading keeps R[obs, obs] invertible.

Everything works in the same (P, M, 2) real/imag layout the rest of the project
uses, so it can feed the DoA metric and the reconstruction metrics directly.
"""

import numpy as np


def estimate_covariance(z, observed, ridge=5e-2):
    """z: (P, M) complex; observed: (P, M) bool. Returns R: (M, M) Hermitian PSD.

    Available-case (pairwise) estimate, then projected onto the PSD cone (negative
    eigenvalues clipped to zero) because the pairwise estimator is not PSD in
    general, plus a diagonal loading so R[obs, obs] stays well-conditioned."""
    Z = np.where(observed, z, 0.0).astype(np.complex128)        # masked entries -> 0
    O = observed.astype(np.float64)
    num = Z.T @ Z.conj()                                        # (M,M): sum_p z_i conj(z_j)
    cnt = O.T @ O                                               # (M,M): #pulses both observed
    R = num / np.maximum(cnt, 1.0)
    R = 0.5 * (R + R.conj().T)                                  # force Hermitian
    w, V = np.linalg.eigh(R)                                    # PSD projection
    w = np.clip(w.real, 0.0, None)
    R = (V * w) @ V.conj().T
    load = ridge * (w.max() + 1e-12)
    R += load * np.eye(R.shape[0])
    return R


def mmse_reconstruct(masked_signal, mask, ridge=5e-2):
    """masked_signal: (P, M, 2) with masked antennas zeroed; mask: (P, M) 1=observed.
    Returns reconstruction (P, M, 2): observed antennas passed through, masked filled by MMSE."""
    P, M, _ = masked_signal.shape
    z = masked_signal[..., 0] + 1j * masked_signal[..., 1]      # (P, M) complex
    observed = mask > 0.5
    R = estimate_covariance(z, observed, ridge)

    out = z.copy()
    for p in range(P):
        o = np.where(observed[p])[0]
        m = np.where(~observed[p])[0]
        if m.size == 0 or o.size == 0:
            continue
        Roo = R[np.ix_(o, o)]
        Rmo = R[np.ix_(m, o)]
        # solve Roo w = x_o, then x_m = Rmo w
        w = np.linalg.solve(Roo, z[p, o])
        out[p, m] = Rmo @ w

    return np.stack([out.real, out.imag], axis=-1).astype(np.float32)


def make_mmse_reconstruct_fn(cfg, ridge=5e-2):
    """reconstruct_fn(masked_signal (P,M,2), mask (P,M)) -> (P,M,2), for doa_metric."""
    def fn(masked_signal, mask):
        return mmse_reconstruct(masked_signal, mask, ridge)
    return fn
