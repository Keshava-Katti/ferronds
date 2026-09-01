"""Signal generators and the timebase every task shares"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.signal import chirp as _chirp, square as _square, sawtooth as _sawtooth

NOISE_SIGMA = 0.05 

# -------------------------------------------------------------------- timebase
@dataclass(frozen=True)
class Timebase:
    fs_hz: float = 2000.0
    f0_hz: float = 35.0
    n_periods: int = 100
    horizon_periods: float = 0.5
    band_lo_hz: float = 30.0
    band_hi_hz: float = 200.0

    @property
    def dt(self): return 1.0/self.fs_hz
    @property
    def duration_s(self): return self.n_periods/self.f0_hz
    @property
    def n_samples(self): return int(round(self.duration_s*self.fs_hz))
    @property
    def horizon(self): return int(round(self.horizon_periods*self.fs_hz/self.f0_hz))
    @property
    def t(self): return np.arange(self.n_samples)*self.dt

    def harmonics_in_band(self, k_max=12):
        return [k for k in range(1, k_max+1)
                if self.band_lo_hz <= k*self.f0_hz <= self.band_hi_hz]

    def describe(self):
        return (f"fs={self.fs_hz/1000:g} kHz, f0={self.f0_hz:g} Hz, "
                f"{self.n_periods} periods ({self.duration_s:.3f} s, {self.n_samples} samples), "
                f"H={self.horizon_periods:g} period ({self.horizon} samples, "
                f"{1e3*self.horizon/self.fs_hz:.1f} ms), band {self.band_lo_hz:g}-{self.band_hi_hz:g} Hz")

SIGNAL_CLASS = {
    "Noisy Sine":   "periodic",
    "Noisy Square": "periodic",
    "AM Sine":      "periodic",
    "Chirp":        "quasi-periodic",
    "Env. Sine":     "quasi-periodic",
    "Composite":    "quasi-periodic",
}
KINDS = list(SIGNAL_CLASS)

HARD_CLASS = {
    "Wander Sine":      "aperiodic",
    "Wander AM":        "aperiodic",
    "Jitter Impulse":   "aperiodic",
    "Driven Resonance": "stochastic",
}
HARD_KINDS = list(HARD_CLASS)
ALL_CLASS = {**SIGNAL_CLASS, **HARD_CLASS}
ALL_KINDS = KINDS + HARD_KINDS

_SNAKE_ALIASES = {k.lower().replace(". ", "_").replace(" ", "_"): k for k in ALL_KINDS}


def canonical_kind(kind: str) -> str:
    return _SNAKE_ALIASES.get(kind, kind)


def signal_seed(kind: str, seed: int) -> int:
    import zlib
    return 1000*int(seed) + zlib.crc32(kind.encode()) % 997

def _ou(n, tau_s, dt, rng):
    a = np.exp(-dt/tau_s)
    s = np.sqrt(1 - a*a)
    x = np.empty(n); x[0] = rng.normal()
    for k in range(1, n):
        x[k] = a*x[k-1] + s*rng.normal()
    return x

# ------------------------------------------------------------------ generators
def make_hard_signal(kind: str, tb: Timebase, rng: np.random.Generator,
                     return_state: bool = False):
    kind = canonical_kind(kind)
    t, N, dt = tb.t, tb.n_samples, tb.dt
    lo, hi = tb.band_lo_hz, tb.band_hi_hz
    state = {}

    if kind in ("Wander Sine", "Wander AM"):
        f_mid = 0.5*(lo + hi)
        f_dev = rng.uniform(8.0, 14.0)            
        tau = rng.uniform(0.4, 0.8)                   
        f = np.clip(f_mid + f_dev*_ou(N, tau, dt, rng), lo + 2, hi - 2)
        ph = 2*np.pi*np.cumsum(f)*dt + rng.uniform(0, 2*np.pi)
        s = np.sin(ph)
        state.update(inst_freq_hz=f, phase_rad=ph, env=np.ones(N))
        if kind == "Wander AM":
            env = np.clip(1.0 + 0.7*_ou(N, rng.uniform(0.05, 0.15), dt, rng), 0.15, None)
            s = env*s; sc = np.max(np.abs(s)); s = s/sc
            state.update(env=env/sc)
        s = s + rng.normal(0, NOISE_SIGMA, N)

    elif kind == "Jitter Impulse":
        f_res = rng.uniform(0.55*hi, 0.9*hi)       
        zeta = rng.uniform(0.02, 0.05)
        f_rep = rng.uniform(1.6*lo, 2.6*lo)            
        jit = rng.uniform(0.02, 0.05)                  
        s = np.zeros(N)
        k = rng.integers(0, int(tb.fs_hz/f_rep))
        while k < N:
            n_ring = min(N - k, int(6/(zeta*2*np.pi*f_res)*tb.fs_hz))
            tt = np.arange(n_ring)*dt
            s[k:k+n_ring] += (rng.uniform(0.7, 1.3)
                              * np.exp(-zeta*2*np.pi*f_res*tt)*np.sin(2*np.pi*f_res*tt))
            k += max(1, int(round(tb.fs_hz/f_rep*(1 + jit*rng.normal()))))
        s = s/np.max(np.abs(s)) + rng.normal(0, NOISE_SIGMA, N)

    elif kind == "Driven Resonance":
        from scipy.signal import butter, lfilter, sosfilt, butter as _b
        f_res = rng.uniform(1.4*lo, 0.7*hi)
        q = rng.uniform(6.0, 16.0)
        drive = rng.normal(0, 1.0, N)
        sos = _b(4, [max(lo*0.6, 1.0)/(tb.fs_hz/2), min(hi*1.4, tb.fs_hz/2*0.95)/(tb.fs_hz/2)],
                 "bandpass", output="sos")
        drive = sosfilt(sos, drive)
        w = 2*np.pi*f_res; z = 1.0/(2*q); K = 2.0/dt
        a0 = K*K + 2*z*w*K + w*w
        s = lfilter(np.array([1., 2., 1.])/a0,
                    np.array([a0, 2*w*w - 2*K*K, K*K - 2*z*w*K + w*w])/a0, drive)
        s = s/np.std(s)

    s = s.astype(float)
    return (s, state) if return_state else s

def oracle_wander(tb: Timebase, state: dict, horizon: int) -> np.ndarray:
    f, ph, env = state["inst_freq_hz"], state["phase_rad"], state["env"]
    dt = tb.dt
    return env*np.sin(ph + 2*np.pi*f*horizon*dt)

def make_signal(kind: str, tb: Timebase, rng: np.random.Generator) -> np.ndarray:
    kind = canonical_kind(kind)
    t, T, f0 = tb.t, tb.duration_s, tb.f0_hz
    N = tb.n_samples
    ph = rng.uniform(0, 2*np.pi)
    if kind == "Noisy Sine":
        s = np.sin(2*np.pi*f0*t + ph) + rng.normal(0, NOISE_SIGMA, N)
    elif kind == "Noisy Square":
        s = _square(2*np.pi*f0*t + ph) + rng.normal(0, NOISE_SIGMA, N)
    elif kind == "AM Sine":
        carrier = rng.uniform(f0, 2*f0)
        mod = rng.uniform(f0/7, f0/3.5)
        s = (1 + 0.5*np.sin(2*np.pi*mod*t + rng.uniform(0, 2*np.pi))) * \
            np.sin(2*np.pi*carrier*t + ph)
        s = s/np.max(np.abs(s))
    elif kind == "Chirp":
        s = _chirp(t, f0=rng.uniform(tb.band_lo_hz, tb.band_lo_hz + 10),
                   f1=rng.uniform(tb.band_hi_hz - 50, tb.band_hi_hz),
                   t1=T, method="linear")
    elif kind == "Env. Sine":
        carrier = rng.uniform(f0, 2*f0)
        s = np.exp(-((t - T/2)**2)/(2*(T/5)**2))*np.sin(2*np.pi*carrier*t + ph)
    elif kind == "Composite":
        a = np.sin(2*np.pi*f0*t[:N//3] + ph)
        b = _square(2*np.pi*f0*t[N//3:2*N//3] + rng.uniform(0, 2*np.pi))
        c = _sawtooth(2*np.pi*f0*t[2*N//3:] + rng.uniform(0, 2*np.pi))
        s = np.concatenate([a, b, c])
    return s.astype(float)
