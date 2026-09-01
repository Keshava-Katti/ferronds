"""The oscillator bank as a short-time Fourier transform"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from ferronds.analog.macromodels import ResonatorMacro

# ----------------------------------------------- pole parameters and bandwidth
def rf_params(macro: ResonatorMacro, dt: float) -> tuple[float, float]:
    w = 2*np.pi*macro.f0_hz; z = macro.zeta; K = 2.0/dt
    a0 = K*K + 2*z*w*K + w*w
    a1 = 2*w*w - 2*K*K
    a2 = K*K - 2*z*w*K + w*w
    lam = float(np.sqrt(a2/a0))
    cos_th = float(-a1/(2*np.sqrt(a0*a2)))
    return lam, float(np.arccos(np.clip(cos_th, -1.0, 1.0)))

def zeta_for_pole_radius(lam: float, f0_hz: float, dt: float) -> float:
    w = 2*np.pi*f0_hz; K = 2.0/dt
    S = K*K + w*w; P = 2*w*K
    return float(S*(1 - lam*lam)/(P*(1 + lam*lam)))

def bilinear_warped_damped_hz(macro: ResonatorMacro, dt: float) -> float:
    return rf_params(macro, dt)[1]/(2*np.pi*dt)

def window_length_samples(lam: float) -> float:
    return float(-1.0/np.log(lam)) if 0 < lam < 1 else np.inf

def equivalent_noise_bandwidth_hz(lam: float, fs: float) -> float:
    return float(fs*(1.0 - lam)/(1.0 + lam))

# -------------------------------------------------------------------- filter bank
@dataclass
class RFBank:
    macros: list
    dt: float

    def __post_init__(self):
        p = [rf_params(m, self.dt) for m in self.macros]
        self.lam = np.array([q[0] for q in p])
        self.theta = np.array([q[1] for q in p])

    @property
    def fs(self) -> float:
        return 1.0/self.dt

    @property
    def f0_hz(self) -> np.ndarray:
        return self.theta/(2*np.pi*self.dt)

    def response(self, x) -> np.ndarray:
        from scipy.signal import lfilter
        x = np.asarray(x, float)
        pole = self.lam*np.exp(1j*self.theta)
        return np.stack([lfilter([1.0 + 0j], [1.0, -p], x) for p in pole])

    def unrolled(self, x, k: int, t: int, n_terms: int | None = None) -> complex:
        x = np.asarray(x, float)
        n_terms = t + 1 if n_terms is None else min(n_terms, t + 1)
        n = np.arange(n_terms)
        return complex(np.sum(np.exp(1j*n*self.theta[k])*self.lam[k]**n * x[t - n]))

    def window(self, k: int, n: int = 200) -> np.ndarray:
        return self.lam[k]**np.arange(n)

    def enbw_hz(self) -> np.ndarray:
        return np.array([equivalent_noise_bandwidth_hz(l, self.fs) for l in self.lam])

    def q_factor(self) -> np.ndarray:
        return self.f0_hz/self.enbw_hz()

    def reconstruct(self, Z) -> np.ndarray:
        n = Z.shape[1]
        acc = np.real(Z).sum(0)
        wgrid = 2*np.pi*np.fft.rfftfreq(n, self.dt)*self.dt
        H = np.zeros(len(wgrid), complex)
        for lam, th in zip(self.lam, self.theta):
            e = np.exp(-1j*wgrid)
            H += 0.5*(1.0/(1 - lam*np.exp(1j*th)*e) + 1.0/(1 - lam*np.exp(-1j*th)*e))
        A = np.fft.rfft(acc)
        keep = np.abs(H) > 1e-3*np.max(np.abs(H))
        out = np.zeros_like(A)
        out[keep] = A[keep]/H[keep]
        return np.fft.irfft(out, n)

# ----------------------------------------------------------- bank constructors
def build_constant_q_bank(n_resonators=32, band=(30.0, 200.0), zeta=0.15, L_H=1.0):
    freqs = np.logspace(*np.log10(band), n_resonators)
    return [ResonatorMacro.for_target(f, zeta, L_H=L_H) for f in freqs]

def build_constant_bw_bank(n_resonators=32, band=(30.0, 200.0), enbw_hz=12.0,
                           fs=2000.0, L_H=1.0):
    lam = (fs - enbw_hz)/(fs + enbw_hz)
    freqs = np.linspace(*band, n_resonators)
    return [ResonatorMacro.for_target(f, zeta_for_pole_radius(lam, f, 1.0/fs), L_H=L_H)
            for f in freqs]

def build_hybrid_bank(n_resonators=48, band=(30.0, 200.0), zeta=0.15,
                      enbw_hz=12.0, fs=2000.0, L_H=1.0, split=0.5):
    n_bw = int(round(split*n_resonators))
    return (build_constant_bw_bank(n_bw, band, enbw_hz, fs, L_H)
            + build_constant_q_bank(n_resonators - n_bw, band, zeta, L_H))


def tap_fractions(macros) -> np.ndarray:
    return np.array([100*m.tap_n for m in macros])

# ------------------------------------------- reference transform and agreement
def exponential_window_stft(x, freqs_hz, fs, lam):
    x = np.asarray(x, float); n = len(x)
    lam = np.atleast_1d(lam)
    if lam.size == 1:
        lam = np.full(len(freqs_hz), float(lam))
    out = np.empty((len(freqs_hz), n), complex)
    for k, f in enumerate(freqs_hz):
        th = 2*np.pi*f/fs
        b = np.array([1.0])
        a = np.array([1.0, -lam[k]*np.exp(1j*th)])
        from scipy.signal import lfilter
        out[k] = lfilter(b, a, x)
    return out

def spectrogram_agreement(A, B):
    a = np.abs(np.asarray(A)).ravel(); b = np.abs(np.asarray(B)).ravel()
    r = float(np.corrcoef(a, b)[0, 1])
    rel = float(np.linalg.norm(a - b)/(np.linalg.norm(b) + 1e-300))
    return dict(correlation=r, relative_error=rel)

def reconstruction_correlation(x, x_hat, skip=None):
    x = np.asarray(x, float); x_hat = np.asarray(x_hat, float)
    n = min(len(x), len(x_hat))
    s = skip if skip is not None else n//10
    return float(np.corrcoef(x[s:n], x_hat[s:n])[0, 1])
