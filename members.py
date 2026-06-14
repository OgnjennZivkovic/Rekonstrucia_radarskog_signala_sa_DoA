"""Ensemble members with a common fit/predict interface.

    member.fit(X, Y)            # X (n, 3M), Y (n, 2M)
    yhat = member.predict(X)    # (n, 2M)

All members solve the SAME job: map the masked signal + mask to the full
signal. The MLP member wraps the torch ReconMLP and can also save a
reconstruction figure per epoch (set .figdir and .stats before fit).
"""

import os
import numpy as np

from sklearn.linear_model import LinearRegression, Ridge, HuberRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor

from features import cap_rows


class SklearnMember:
    def __init__(self, name, estimator, cap=None, seed=0):
        self.name = name
        self.est = estimator
        self.cap = cap
        self.seed = seed

    def fit(self, X, Y):
        Xc, Yc = cap_rows(X, Y, self.cap, self.seed)
        self.est.fit(Xc, Yc)
        return self

    def predict(self, X):
        return np.asarray(self.est.predict(X), dtype=np.float32)


class MLPMember:
    """ReconMLP trained to minimize MSE on the masked antennas."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.name = "mlp"
        self.model = None
        self.device = None
        self.figdir = None     # set externally to enable per-epoch figures
        self.stats = None

    def fit(self, X, Y):
        import torch
        from model import ReconMLP

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        M = self.cfg.M
        self.model = ReconMLP(self.cfg).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr,
                               weight_decay=self.cfg.weight_decay)

        Xt = torch.from_numpy(X)
        Yt = torch.from_numpy(Y)
        mask_all = Xt[:, 2 * M:]
        n = Xt.shape[0]
        bs = self.cfg.batch_size

        # fixed sample for the per-epoch figure
        fixed = (X[0, :2 * M].copy(), X[0, 2 * M:].copy(), Y[0].copy()) if self.figdir else None

        for ep in range(1, self.cfg.mlp_member_epochs + 1):
            print(f"  [mlp] epoch {ep}/{self.cfg.mlp_member_epochs}")
            self.model.train()
            perm = torch.randperm(n)
            for s in range(0, n, bs):
                b = perm[s:s + bs]
                xb = Xt[b].to(self.device)
                yb = Yt[b].to(self.device)
                mb = mask_all[b].to(self.device)
                masked_signal = xb[:, :2 * M].reshape(-1, M, 2)
                recon = self.model(masked_signal, mb).reshape(-1, 2 * M)
                err = ((recon - yb) ** 2).reshape(-1, M, 2).sum(-1)
                hidden = (1.0 - mb)
                loss = (err * hidden).sum() / (hidden.sum() + 1e-8) \
                    + self.cfg.obs_loss_weight * (err * mb).sum() / (mb.sum() + 1e-8)
                opt.zero_grad()
                loss.backward()
                opt.step()

            if self.figdir is not None and self.stats is not None:
                self._save_fig(fixed, ep)
        return self

    def _save_fig(self, fixed, ep):
        import torch
        from viz import plot_reconstruction
        masked_signal, mask, target = fixed
        M = self.cfg.M
        self.model.eval()
        with torch.no_grad():
            ms = torch.from_numpy(masked_signal).reshape(1, M, 2).to(self.device)
            mk = torch.from_numpy(mask).reshape(1, M).to(self.device)
            rec = self.model(ms, mk).reshape(-1).cpu().numpy()
        path = os.path.join(self.figdir, f"epoch_{ep:03d}.png")
        plot_reconstruction(rec, target, mask, self.stats, path,
                            f"MLP member - epoch {ep}")

    def predict(self, X):
        import torch
        M = self.cfg.M
        self.model.eval()
        out = []
        Xt = torch.from_numpy(X)
        with torch.no_grad():
            for s in range(0, Xt.shape[0], 4096):
                xb = Xt[s:s + 4096].to(self.device)
                masked_signal = xb[:, :2 * M].reshape(-1, M, 2)
                mb = xb[:, 2 * M:]
                rec = self.model(masked_signal, mb).reshape(-1, 2 * M)
                out.append(rec.cpu().numpy())
        return np.concatenate(out, axis=0).astype(np.float32)

    def to_cpu(self):
        if self.model is not None:
            self.model.to("cpu")
            self.device = "cpu"


def build_member(name, cfg):
    cap = cfg.tree_sample_cap
    if name == "linear":
        return SklearnMember("linear", LinearRegression())
    if name == "ridge":
        return SklearnMember("ridge", Ridge(alpha=cfg.ridge_alpha))
    if name == "huber":
        return SklearnMember(
            "huber",
            MultiOutputRegressor(HuberRegressor(epsilon=cfg.huber_epsilon, max_iter=200)),
            cap=cap, seed=cfg.seed)
    if name == "rf":
        return SklearnMember(
            "rf",
            RandomForestRegressor(n_estimators=cfg.rf_estimators, max_depth=cfg.rf_max_depth,
                                  n_jobs=-1, random_state=cfg.seed, verbose=1),
            cap=cap, seed=cfg.seed)
    if name == "knn":
        return SklearnMember("knn", KNeighborsRegressor(n_neighbors=cfg.knn_neighbors),
                             cap=cap, seed=cfg.seed)
    if name == "xgb":
        from xgboost import XGBRegressor
        try:
            est = XGBRegressor(n_estimators=cfg.xgb_estimators, max_depth=cfg.xgb_max_depth,
                               tree_method="hist", multi_strategy="multi_output_tree",
                               n_jobs=-1, random_state=cfg.seed, verbosity=1)
            return SklearnMember("xgb", est, cap=cap, seed=cfg.seed)
        except TypeError:
            base = XGBRegressor(n_estimators=cfg.xgb_estimators, max_depth=cfg.xgb_max_depth,
                                tree_method="hist", n_jobs=-1, random_state=cfg.seed,
                                verbosity=1)
            return SklearnMember("xgb", MultiOutputRegressor(base), cap=cap, seed=cfg.seed)
    if name == "mlp":
        return MLPMember(cfg)
    raise ValueError(f"unknown member '{name}'")
