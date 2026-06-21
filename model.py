#Karakteristike za MLP

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return x + self.net(x)


class ReconMLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.M = cfg.M
        in_dim = 2 * cfg.M + cfg.M     # masked signal (2M) + mask (M)
        out_dim = 2 * cfg.M
        self.input = nn.Linear(in_dim, cfg.hidden)
        self.blocks = nn.ModuleList(
            [ResidualBlock(cfg.hidden, cfg.dropout) for _ in range(cfg.n_blocks)]
        )
        self.head = nn.Sequential(nn.LayerNorm(cfg.hidden), nn.Linear(cfg.hidden, out_dim))

    def forward(self, masked_signal, mask):
        B = masked_signal.shape[0]
        flat = masked_signal.reshape(B, -1)              # (B, 2M)
        x = torch.cat([flat, mask], dim=1)               # (B, 3M)
        x = self.input(x)
        for blk in self.blocks:
            x = blk(x)
        out = self.head(x)                               # (B, 2M)
        return out.reshape(B, self.M, 2)


def masked_reconstruction_loss(recon, target, mask, obs_weight=0.1): # funkcija gubitaka, kaznjava samo maskirane delove
    
    err = (recon - target) ** 2                          # (B, M, 2)
    err = err.sum(dim=-1)                                # (B, M) per-antenna squared error

    hidden = (1.0 - mask)                                # pravi matricu samo sa maskiranim antenama
    observed = mask

    hidden_loss = (err * hidden).sum() / (hidden.sum() + 1e-8)
    observed_loss = (err * observed).sum() / (observed.sum() + 1e-8)
    return hidden_loss + obs_weight * observed_loss, hidden_loss.detach() # obs sluzi da odrzi mrezu samo na dobrom putu


@torch.no_grad()
def nmse_on_masked(recon, target, mask): # nmse na maskiranim antenama
    """Normalized MSE over the hidden antennas only (lower is better)."""
    err = ((recon - target) ** 2).sum(dim=-1)            # (B, M)
    pwr = (target ** 2).sum(dim=-1)                      # (B, M)
    hidden = (1.0 - mask)
    return (err * hidden).sum() / ((pwr * hidden).sum() + 1e-8)
