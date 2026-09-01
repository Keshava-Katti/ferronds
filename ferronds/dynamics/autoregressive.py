"""Closed-loop autoregressive prediction for the chaotic-function protocol"""

from __future__ import annotations
import numpy as np
from ferronds.analog.macromodels import ResonatorMacro, IntegratorMacro, FeDWeightBank, LinearReadout

# --------------------------------------------------------- streaming front end
class StatefulFrontEnd:
    def __init__(self, bank, taus_ms, dt, nonlinearity=None, n_rates=3, drive=1.0):
        self.nl = nonlinearity; self.n_rates = n_rates; self.drive = drive
        self.dt = dt; self.n_res = len(bank); self.taus = list(taus_ms)
        self.rb, self.ra = [], []
        for m in bank:
            w = 2*np.pi*m.f0_hz; K = 2.0/dt
            a0 = K*K + 2*m.zeta*w*K + w*w
            self.rb.append(np.array([1., 2., 1.])/a0)
            self.ra.append(np.array([2*w*w - 2*K*K, K*K - 2*m.zeta*w*K + w*w])/a0)
        self.alpha_s = np.exp(-dt/(0.75e-3))
        self.alpha_m = np.array([np.exp(-dt/(t*1e-3)) for t in self.taus])
        from ferronds.analog import nonlinearity as _NL
        self._rates = np.linspace(_NL.A_MIN, _NL.A_MAX, n_rates)*_NL.STAGE1_SLOPE*drive
        self._nch = self.n_res*len(self.taus)*(n_rates if nonlinearity == "exp" else 1)
        self.gain = np.ones(self._nch); self.offset = np.zeros(self._nch)
        self.reset()

    def reset(self):
        n = self.n_res
        self.xh = np.zeros((n, 2)); self.yh = np.zeros((n, 2))
        self.s = np.zeros(n); self.u = np.zeros((len(self.taus), n))

    def step(self, x):
        y = np.empty(self.n_res)
        for i in range(self.n_res):
            b, a = self.rb[i], self.ra[i]
            y[i] = b[0]*x + b[1]*self.xh[i,0] + b[2]*self.xh[i,1] \
                   - a[0]*self.yh[i,0] - a[1]*self.yh[i,1]
            self.xh[i,1] = self.xh[i,0]; self.xh[i,0] = x
            self.yh[i,1] = self.yh[i,0]; self.yh[i,0] = y[i]
        self.s = self.alpha_s*self.s + (1 - self.alpha_s)*y
        for j, am in enumerate(self.alpha_m):
            self.u[j] = am*self.u[j] + (1 - am)*self.s
        v = self.u.reshape(-1)
        if self.nl == "exp":
            v = np.exp(np.clip(np.outer(v, self._rates), -20, 20)).reshape(-1)
        return (v - self.offset)*self.gain

    def calibrate(self, x_seq):
        self.reset(); self.gain = np.ones(self._nch); self.offset = np.zeros(self._nch)
        F = np.stack([self.step(v) for v in x_seq])
        self.offset = F.mean(0)
        sd = F.std(0); sd[sd < 1e-12] = 1.0
        self.gain = 1.0/sd
        return (F - self.offset)*self.gain

# --------------------------------------------------------------------- rollout
def run_closed_loop(x, split_frac=0.5, bank=None, taus_ms=None, weight_bank=None,
                    lambdas=(1e-6,1e-5,1e-4,1e-3,1e-2,1e-1), quantize=True,
                    n_val_frac=0.2, nonlinearity=None, drive=1.0, n_rates=3,
                    dt=1.0/2000.0):
    from ferronds.dynamics.features import build_bank, default_taus
    bank = bank if bank is not None else build_bank()
    taus_ms = taus_ms if taus_ms is not None else default_taus()
    weight_bank = weight_bank or FeDWeightBank(signed=True, ideal=True)
    x = np.asarray(x, float); sp = int(len(x)*split_frac)

    fe = StatefulFrontEnd(bank, taus_ms, dt, nonlinearity, n_rates, drive)
    Ftr = fe.calibrate(x[:sp])
    Xtr, ytr = Ftr[:-1], x[1:sp]
    mu, sg = float(ytr.mean()), float(ytr.std())
    sg = sg if sg > 0 else 1.0

    nv = int(n_val_frac*len(Xtr))
    best, blam = np.inf, lambdas[0]
    for lam in lambdas:
        ro = LinearReadout(bank=weight_bank).fit_ridge(
            Xtr[:-nv], (ytr[:-nv] - mu)/sg, lam=lam)
        v = _free_run(bank, taus_ms, dt, fe, ro, x[:len(Xtr)-nv], nv, quantize, mu, sg)
        e = np.mean((v - ytr[-nv:])**2)
        if np.isfinite(e) and e < best: best, blam = e, lam

    ro = LinearReadout(bank=weight_bank).fit_ridge(Xtr, (ytr - mu)/sg, lam=blam)
    pred = _free_run(bank, taus_ms, dt, fe, ro, x[:sp], len(x)-sp, quantize, mu, sg)
    return dict(pred=pred, true=x[sp:sp+len(pred)], lam=blam, split=sp,
                target_mean=mu, target_std=sg,
                _internals=dict(readout=ro, front_end=fe, bank=bank,
                                taus_ms=taus_ms, dt=dt, quantize=quantize,
                                warmup=x[:sp], n_steps=len(x)-sp,
                                free_run=_free_run))

def _free_run(bank, taus_ms, dt, src, readout, warmup, n_steps, quantize,
              mu=0.0, sg=1.0):
    fe = StatefulFrontEnd(bank, taus_ms, dt, src.nl, src.n_rates, src.drive)
    fe.gain = src.gain; fe.offset = src.offset
    f = None
    for v in warmup: f = fe.step(v)
    out = np.empty(n_steps); cur = warmup[-1]
    for k in range(n_steps):
        p = float(readout.predict(f[None, :], quantize=quantize)[0])*sg + mu
        p = float(np.clip(p, -5.0, 5.0))
        out[k] = p
        f = fe.step(p); cur = p
    return out
