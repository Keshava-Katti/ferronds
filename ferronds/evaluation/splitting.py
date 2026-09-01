"""Purged splitting and ridge penalty selection, identical for every model"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from ferronds.analog.macromodels import FeDWeightBank, LinearReadout

DEFAULT_LAMBDAS = (1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1)

def purge_gap(horizon: int, taus_ms, fs_hz: float, n_tau: float = 5.0) -> int:
    return int(horizon + n_tau*max(taus_ms)*1e-3*fs_hz)

@dataclass(frozen=True)
class Split:
    train: slice
    test: slice
    horizon: int
    purge: int

def make_split(n_samples: int, horizon: int, purge: int, train_frac: float = 0.5) -> Split:
    sp = int(n_samples*train_frac)
    n_train = sp - horizon - purge
    return Split(slice(0, n_train), slice(sp, n_samples - horizon), horizon, purge)

def xy(features, signal, sl: Split, which="train"):
    if which == "train":
        n = sl.train.stop
        return features[:n], np.asarray(signal)[sl.horizon:sl.horizon + n]
    return features[sl.test], np.asarray(signal)[sl.test.start + sl.horizon:
                                                 sl.test.stop + sl.horizon]

def purged_blocked_folds(n: int, purge: int, k: int = 5, min_train=100, min_val=50):
    edges = np.linspace(0, n, k + 1).astype(int)
    for i in range(k):
        a, b = edges[i], edges[i + 1]
        val = np.arange(a, b)
        keep = np.ones(n, bool)
        keep[max(0, a - purge):min(n, b + purge)] = False
        tr = np.where(keep)[0]
        if len(tr) >= min_train and len(val) >= min_val:
            yield tr, val

def select_lambda(X, y, purge, fit_predict, lambdas=DEFAULT_LAMBDAS, k=5):
    best, best_lam = np.inf, lambdas[0]
    for lam in lambdas:
        scores = [np.mean((fit_predict(X[tr], y[tr], X[va], lam) - y[va])**2)
                  for tr, va in purged_blocked_folds(len(X), purge, k)]
        if scores and np.mean(scores) < best:
            best, best_lam = float(np.mean(scores)), lam
    return best_lam, best


def _fit_predict_factory(bank_weights: FeDWeightBank, quantize: bool):
    def fp(Xtr, ytr, Xva, lam):
        ro = LinearReadout(bank=bank_weights).fit_ridge(Xtr, ytr, lam=lam)
        return ro.predict(Xva, quantize=quantize)
    return fp
