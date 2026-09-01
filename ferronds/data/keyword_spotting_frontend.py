"""FerroNDS and log-mel front ends for keyword spotting"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from scipy.signal import stft as _stft

from ferronds.dynamics import features
from ferronds.dynamics import spectral
from ferronds.analog.macromodels import IntegratorMacro

SPEECH_BAND_HZ = (100.0, 4000.0)


@dataclass
class KWSFrontEnd:
    n_channels: int = 32
    band_hz: tuple = SPEECH_BAND_HZ
    zeta: float = 0.05
    tau_ms: float = 8.0
    fs_hz: float = 16000.0
    hop_ms: float = 10.0
    log_compress: bool = True

    def __post_init__(self):
        self.bank = features.build_bank(self.n_channels, self.band_hz, self.zeta)
        self.rf = spectral.RFBank(list(self.bank), 1.0/self.fs_hz)

    @property
    def hop(self) -> int:
        return int(round(self.hop_ms*1e-3*self.fs_hz))

    def enbw_hz(self):
        return self.rf.enbw_hz()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        dt = 1.0/self.fs_hz
        Z = self.rf.response(np.asarray(x, float))
        P = Z.real**2 + Z.imag**2
        if self.tau_ms:
            P = np.stack([IntegratorMacro(tau_m_ms=self.tau_ms).response(p, dt) for p in P])
        P = P[:, ::self.hop]
        if self.log_compress:
            P = np.log(P + 1e-12)
        return P.T

    def n_devices_for(self, n_classes: int, hidden: int | None = None,
                      n_in: int | None = None) -> int:
        n_in = 2*self.n_channels if n_in is None else int(n_in)
        if hidden is None:
            return 2*(n_in*n_classes)
        return 2*(n_in*hidden + hidden*n_classes)


@dataclass
class MelReference:
    n_channels: int = 32
    band_hz: tuple = SPEECH_BAND_HZ
    fs_hz: float = 16000.0
    n_fft: int = 400
    hop: int = 160

    def __call__(self, x: np.ndarray) -> np.ndarray:
        f, _t, S = _stft(x, fs=self.fs_hz, nperseg=self.n_fft,
                         noverlap=self.n_fft - self.hop)
        P = np.abs(S)**2
        lo, hi = self.band_hz
        m = lambda h: 2595*np.log10(1 + h/700)
        edges = np.linspace(m(lo), m(hi), self.n_channels + 2)
        hz = 700*(10**(edges/2595) - 1)
        out = np.empty((self.n_channels, P.shape[1]))
        for i in range(self.n_channels):
            sel = (f >= hz[i]) & (f < hz[i + 2])
            out[i] = P[sel].sum(0) if sel.any() else 0.0
        return np.log(out + 1e-12).T
