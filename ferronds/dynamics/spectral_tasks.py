"""Tasks whose target is a property of the spectrum: fault power and frequency tracking"""

from __future__ import annotations
import numpy as np
from dataclasses import replace

from ferronds.dynamics import features
from ferronds.dynamics.spectral import RFBank
from ferronds.analog.macromodels import IntegratorMacro
from ferronds.data.signals import Timebase, make_hard_signal, signal_seed, _ou, NOISE_SIGMA

TASKS = ("freq_track", "band_power")


def make_band_power(tb: Timebase, rng: np.random.Generator):
    N, dt = tb.n_samples, tb.dt
    lo, hi = tb.band_lo_hz, tb.band_hi_hz
    f_shaft = rng.uniform(lo + 5, lo + 25)
    f_fault = rng.uniform(0.55*hi, 0.9*hi)
    tau_a = rng.uniform(0.3, 0.7)
    a = 0.10 + 0.08*np.abs(_ou(N, tau_a, dt, rng))
    t = tb.t
    shaft = np.sin(2*np.pi*f_shaft*t + rng.uniform(0, 2*np.pi))
    fault = a*np.sin(2*np.pi*f_fault*t + rng.uniform(0, 2*np.pi))
    x = shaft + fault + rng.normal(0, 3*NOISE_SIGMA, N)
    return x.astype(float), (a**2).astype(float)


def make_task(task: str, tb: Timebase, seed: int):
    rng = np.random.default_rng(signal_seed(task, seed))
    if task == "freq_track":
        x, st = make_hard_signal("wander_sine", tb, rng, return_state=True)
        return x, st["inst_freq_hz"]
    if task == "band_power":
        return make_band_power(tb, rng)
    raise ValueError(task)


MODES = ("ReZ", "ReZ2_noint", "ReZ2", "absZ2", "absZ2_int")


TAU_DETECTOR_MS = 18.16


def front_end(x, tb, mode, n_res=16, zeta=0.05, tau_ms=None):
    bank = features.build_bank(n_res, (tb.band_lo_hz, tb.band_hi_hz), zeta)
    if mode.startswith("absZ2"):
        Z = RFBank(list(bank), tb.dt).response(x)
        R = Z.real**2 + Z.imag**2
    elif mode.startswith("ReZ2"):
        Z = RFBank(list(bank), tb.dt).response(x)
        R = Z.real**2
    else:
        R = np.stack([m.response(x, tb.dt) for m in bank])
    R = R/(R.std(axis=1, keepdims=True) + 1e-12)
    if mode in ("ReZ", "ReZ2", "absZ2_int"):
        tm = TAU_DETECTOR_MS if tau_ms is None else tau_ms
        R = np.stack([IntegratorMacro(tau_m_ms=tm).response(r, tb.dt) for r in R])
    return R.T
