#Treniranje modela, stack

import os
import time
import numpy as np
import joblib
from sklearn.linear_model import Ridge

from members import build_member, MLPMember


class Stacker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.members = [build_member(n, cfg) for n in cfg.members]
        self.meta = Ridge(alpha=cfg.meta_alpha)
        self.stats = None

    def _meta_features(self, preds, mask): # prima feture za meta model
        n, M = mask.shape
        hidden = (mask == 0) # samo maskirane vrednosti
        feats = [preds[m.name].reshape(n, M, 2)[hidden] for m in self.members]
        return np.concatenate(feats, axis=1), hidden # svi clanovi jedan feature vektor

    def fit(self, splits, stats=None, mlp_figdir=None): # pozivanje treninga kompletnog ansambla
        Xtr, Ytr, _ = splits["train"]
        Xva, Yva, Mva = splits["val"]
        self.stats = stats

        self.timings = {}
        for mem in self.members:
            if isinstance(mem, MLPMember) and mlp_figdir is not None:
                mem.figdir = mlp_figdir
                mem.stats = stats
            print(f"[ensemble] starting member '{mem.name}'...")
            t = time.time()
            mem.fit(Xtr, Ytr)
            elapsed = time.time() - t
            self.timings[mem.name] = elapsed
            print(f"[ensemble] finished member '{mem.name}' in {elapsed:.1f}s")
        #Svaki clan daje predikciju za validacioni skup
        preds_va = {m.name: m.predict(Xva) for m in self.members}
        #Konkateniraju se kao feature vektori + uzimaju samo maskirane vrednosti
        F, hidden = self._meta_features(preds_va, Mva)
        #Ridge regresija uvi na maskiranim pozicijama
        M = self.cfg.M
        self.meta.fit(F, Yva.reshape(-1, M, 2)[hidden])
        print("  trained meta-model (Ridge) on masked val entries")
        return self

    def member_predictions(self, X): # vraca dict predikcja za clanove
        return {m.name: m.predict(X) for m in self.members}

    def ensemble_predict(self, X, preds=None): # predikcija celog ansambla
        M = self.cfg.M
        n = X.shape[0]
        mask = X[:, 2 * M:]
        if preds is None: # ako predikcije nisu vec pozvane za onda ih poziva
            preds = self.member_predictions(X)
        full = X[:, :2 * M].reshape(n, M, 2).copy()
        F, hidden = self._meta_features(preds, mask)
        if F.shape[0] > 0:
            full[hidden] = self.meta.predict(F).astype(np.float32)
        return full.reshape(n, 2 * M) # vraca rekonstruisan signal

    def predict(self, X): # alijas za ensemble_predict
        return self.ensemble_predict(X)

    def save(self, path): # cuva ceo ansambl 
        for mem in self.members:           # store torch model on CPU for portability
            if isinstance(mem, MLPMember):
                mem.figdir = None
                mem.to_cpu()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump(self, path)
        print(f"  saved trained ensemble -> {os.path.abspath(path)}")


def load_stacker(path): # ucitava istrenirani ansambl
    return joblib.load(path)


def make_reconstruct_fn(predictor, cfg): # pravi funkciju za rekonstrukciju koja se moze evaluirati 
    M = cfg.M

    def fn(masked_signal, mask):
        X = np.concatenate([masked_signal.reshape(len(mask), -1), mask], axis=1).astype(np.float32)
        return predictor.predict(X).reshape(len(mask), M, 2)

    return fn
