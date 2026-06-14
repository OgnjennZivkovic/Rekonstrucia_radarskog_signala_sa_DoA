"""Use a trained model to fill in missing antennas.

Programmatic use:
    from infer import Reconstructor
    r = Reconstructor("checkpoints/model.pt")
    full = r.reconstruct(observed_complex_vector, missing_indices)

Quick demo on a random pulse from a CSV:
    python infer.py --ckpt checkpoints/model.pt --data data/radar.csv --mask 3 7 11
"""

import argparse
import numpy as np
import torch

from config import Config
from model import ReconMLP


class Reconstructor:
    def __init__(self, ckpt_path, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.cfg = Config(**ckpt["config"])
        self.model = ReconMLP(self.cfg).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.mean = ckpt["mean"]            # (1,1,2)
        self.std = ckpt["std"]

    @torch.no_grad()
    def reconstruct(self, signal_complex, missing_indices):
        """signal_complex: length-M complex array (values at missing indices ignored).
        missing_indices: iterable of antenna indices to predict.
        Returns a length-M complex array with the missing entries filled in.
        """
        M = self.cfg.M
        x = np.stack([np.real(signal_complex), np.imag(signal_complex)], axis=-1).astype(np.float32)
        mean = self.mean[0]                 # (1,2)
        std = self.std[0]
        x = (x - mean) / std

        mask = np.ones(M, dtype=np.float32)
        mask[list(missing_indices)] = 0.0
        x_masked = x * mask[:, None]

        xm = torch.from_numpy(x_masked).unsqueeze(0).to(self.device)
        mk = torch.from_numpy(mask).unsqueeze(0).to(self.device)
        recon = self.model(xm, mk)[0].cpu().numpy()      # (M,2) normalized
        recon = recon * std + mean

        out = signal_complex.astype(np.complex64).copy()
        for i in missing_indices:
            out[i] = recon[i, 0] + 1j * recon[i, 1]
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/model.pt")
    ap.add_argument("--data", default="data/radar.csv")
    ap.add_argument("--mask", type=int, nargs="+", default=[3, 7, 11],
                    help="antenna indices to hide then reconstruct")
    args = ap.parse_args()

    from data import load_csv
    r = Reconstructor(args.ckpt)
    signals, _ = load_csv(args.data, r.cfg.M, r.cfg.P)
    pulse = signals[0, 0]                                # first pulse of first scenario

    truth = pulse.copy()
    filled = r.reconstruct(pulse, args.mask)

    print(f"Hid antennas: {args.mask}\n")
    print(f"{'ant':>4} {'true':>26} {'reconstructed':>26}")
    for i in args.mask:
        print(f"{i:>4} {str(truth[i]):>26} {str(filled[i]):>26}")


if __name__ == "__main__":
    main()
