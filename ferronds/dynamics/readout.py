"""A multivariate nonlinear readout that maps onto the ferrodiode crossbar"""

from __future__ import annotations
import numpy as np
from ferronds.analog.macromodels import FeDWeightBank, Rails


def _relu(z):  return np.maximum(z, 0.0)
def _drelu(z): return (z > 0).astype(z.dtype)


class MappableReadout:

    def __init__(self, n_in, hidden=(32,), weight_bank=None, rails=None,
                 quantize=True, seed=0):
        rng = np.random.default_rng(seed)
        self.bank = weight_bank or FeDWeightBank(signed=True, ideal=True)
        self.rails = rails if rails is not None else Rails()
        self.quantize = quantize
        self.n_in = n_in
        self.hidden = tuple(hidden)
        dims = [n_in] + list(self.hidden) + [1]
        self.W = [rng.normal(0, np.sqrt(2.0/dims[i]), (dims[i], dims[i+1]))
                  for i in range(len(dims)-1)]
        self.b = [np.zeros(dims[i+1]) for i in range(len(dims)-1)]
        self._m = [np.zeros_like(w) for w in self.W] + [np.zeros_like(v) for v in self.b]
        self._v = [np.zeros_like(w) for w in self.W] + [np.zeros_like(v) for v in self.b]
        self._t = 0

    def _q(self, W):
        return self.bank.quantize(W) if self.quantize else W

    def _clip(self, z, sigma):
        if not self.rails.enabled:
            return z, np.ones_like(z)
        L = self.rails.limit(sigma=sigma)
        if not np.isfinite(L):
            return z, np.ones_like(z)
        return np.clip(z, -L, L), (np.abs(z) < L).astype(z.dtype)

    def _scale(self, F):
        return (F - self._mu)/self._sd

    def forward(self, F, cache=False):
        h = self._scale(F)
        Wq = [self._q(w) for w in self.W]
        acts, masks, hs = [], [], [h]
        for i, (w, bb) in enumerate(zip(Wq, self.b)):
            a = h @ w + bb
            a, m = self._clip(a, self._s[i])
            acts.append(a); masks.append(m)
            h = _relu(a) if i < len(Wq) - 1 else a
            hs.append(h)
        if cache:
            self._c = (Wq, acts, masks, hs)
        return h[:, 0]

    def fit(self, F, y, epochs=800, batch=256, lr=1e-2, l2=1e-5,
            val_frac=0.2, purge=0, seed=0, patience=60, max_steps=25000,
            verbose=False):
        rng = np.random.default_rng(seed)
        n = len(F); n_val = int(val_frac*n); tr_hi = max(1, n - n_val - purge)
        Ftr, ytr = F[:tr_hi], y[:tr_hi]
        Fva, yva = F[n - n_val:], y[n - n_val:]
        self._mu = Ftr.mean(0)
        self._sd = Ftr.std(0); self._sd[self._sd < 1e-12] = 1.0
        self._s = [1.0]*len(self.W)
        h = self._scale(Ftr)
        for i, w in enumerate(self.W):
            a = h @ self._q(w) + self.b[i]
            self._s[i] = float(np.std(a)) or 1.0
            h = _relu(a) if i < len(self.W) - 1 else a
        self._s[-1] = float(np.std(ytr)) or 1.0

        best, best_state, bad, steps = np.inf, None, 0, 0
        for ep in range(epochs):
            idx = rng.permutation(len(Ftr))
            for k in range(0, len(idx), batch):
                j = idx[k:k+batch]
                self._step(Ftr[j], ytr[j], lr, l2)
                steps += 1
            if steps >= max_steps:
                epochs = ep + 1
            v = float(np.mean((self.forward(Fva) - yva)**2))
            if v < best - 1e-9:
                best, bad = v, 0
                best_state = ([w.copy() for w in self.W], [q.copy() for q in self.b])
            else:
                bad += 1
                if bad >= patience:
                    break
            if steps >= max_steps:
                break
            if verbose and ep % 50 == 0:
                print(f"Epoch {ep}: val {v:.5f}")
        if best_state is not None:
            self.W, self.b = best_state
        self.val_mse_ = best
        return self

    def _step(self, F, y, lr, l2):
        p = self.forward(F, cache=True)
        Wq, acts, masks, hs = self._c
        L = len(self.W)
        g = ((2.0/len(F))*(p - y))[:, None]*masks[-1]
        gW, gb = [None]*L, [None]*L
        for i in range(L-1, -1, -1):
            gW[i] = hs[i].T @ g + l2*self.W[i]
            gb[i] = g.sum(0)
            if i > 0:
                g = (g @ Wq[i].T)*_drelu(acts[i-1])*masks[i-1]
        self._t += 1
        params = self.W + self.b
        grads = gW + gb
        for k, (par, gr) in enumerate(zip(params, grads)):
            self._m[k] = 0.9*self._m[k] + 0.1*gr
            self._v[k] = 0.999*self._v[k] + 0.001*(gr*gr)
            mh = self._m[k]/(1 - 0.9**self._t); vh = self._v[k]/(1 - 0.999**self._t)
            par -= lr*mh/(np.sqrt(vh) + 1e-8)
        return

    def n_devices(self) -> int:
        return sum(self.bank.n_devices(w.size) for w in self.W)

    def n_nonlinear_elements(self) -> int:
        return sum(self.hidden)
