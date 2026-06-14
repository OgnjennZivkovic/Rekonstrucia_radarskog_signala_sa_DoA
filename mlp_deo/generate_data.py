"""Generate a synthetic radar dataset in the SAME format as the real one.

Each row is one pulse:  angle , ant_1 , ... , ant_29
  - `angle` is a bracketed list of the target bearings, e.g. "[58 -57 21]"
  - `ant_k` is a complex number written like "(0.15+0.21j)"
Consecutive blocks of `P` rows share an angle list and form one scenario.

This lets you run the whole pipeline before plugging in the real CSV, and it
mirrors the simulated conditions described in the dataset (ULA, multiple
targets, additive noise at a fixed SNR).

Usage:
    python generate_data.py                 # uses Config defaults
    python generate_data.py --out data/radar.csv --scenarios 2000 --pulses 256
"""

import os
import argparse
import numpy as np

from config import default_config


def steering_vector(angle_deg, n_elements, d):
    """Phase each ULA element sees for a plane wave from `angle_deg`."""
    m = np.arange(n_elements)
    return np.exp(-1j * 2 * np.pi * d * m * np.sin(np.deg2rad(angle_deg)))


def simulate_scenario(cfg, rng):
    """Return (angles, signal) for one scenario.

    signal has shape (P, M) complex: P pulses, M antennas.
    """
    k = rng.integers(cfg.min_targets, cfg.max_targets + 1)
    angles = rng.uniform(cfg.angle_min, cfg.angle_max, size=k)

    A = np.column_stack([steering_vector(a, cfg.M, cfg.d_over_lambda) for a in angles])  # M x K
    # random complex source amplitudes, one column per pulse
    S = (rng.standard_normal((k, cfg.P)) + 1j * rng.standard_normal((k, cfg.P))) / np.sqrt(2)
    clean = (A @ S).T  # P x M

    sig_power = np.mean(np.abs(clean) ** 2)
    noise_power = sig_power / (10 ** (cfg.snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (
        rng.standard_normal((cfg.P, cfg.M)) + 1j * rng.standard_normal((cfg.P, cfg.M))
    )
    return angles, clean + noise


def angles_to_str(angles):
    return "[" + " ".join(f"{a:.4f}" for a in angles) + "]"


def main():
    cfg = default_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=cfg.data_csv)
    ap.add_argument("--scenarios", type=int, default=cfg.n_scenarios)
    ap.add_argument("--pulses", type=int, default=cfg.P)
    ap.add_argument("--seed", type=int, default=cfg.seed)
    args = ap.parse_args()

    cfg.P = args.pulses
    rng = np.random.default_rng(args.seed)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    header = "angle," + ",".join(f"ant_{i+1}" for i in range(cfg.M))

    with open(args.out, "w") as f:
        f.write(header + "\n")
        for s in range(args.scenarios):
            angles, signal = simulate_scenario(cfg, rng)
            astr = angles_to_str(angles)
            for p in range(cfg.P):
                cells = [astr] + [str(complex(round(v.real, 10), round(v.imag, 10)))
                                  for v in signal[p]]
                f.write(",".join(cells) + "\n")
            if (s + 1) % 50 == 0:
                print(f"  ...{s + 1}/{args.scenarios} scenarios")

    print(f"Wrote {args.scenarios} scenarios x {cfg.P} pulses to {args.out}")


if __name__ == "__main__":
    main()
