"""Behavioural macros for FerroNDS analog blocks"""

from __future__ import annotations
from dataclasses import dataclass, field, replace
import numpy as np
from scipy.signal import lfilter


# ---------------------------------------------------------------- supply rails
V_RAIL_LO = 0.6
V_RAIL_HI = 1.8


@dataclass(frozen=True)
class Rails:
    v_lo: float = V_RAIL_LO
    v_hi: float = V_RAIL_HI
    headroom_sigma: float | None = 3.0
    gain_v_per_unit: float | None = None
    enabled: bool = True

    @classmethod
    def off(cls) -> "Rails":
        return cls(enabled=False)

    @property
    def v_ref(self) -> float:
        return 0.5*(self.v_lo + self.v_hi)

    @property
    def swing_v(self) -> float:
        return min(self.v_ref - self.v_lo, self.v_hi - self.v_ref)

    MIN_SAMPLES_TO_INFER = 32

    def limit(self, x=None, sigma: float | None = None) -> float:
        if self.gain_v_per_unit is not None:
            return self.swing_v/self.gain_v_per_unit
        if self.headroom_sigma is None:
            return np.inf
        if sigma is None:
            a = np.asarray(x, float) if x is not None else None
            if a is None or a.size < self.MIN_SAMPLES_TO_INFER:
                raise ValueError(
                    f"sigma required: {0 if a is None else a.size} samples, "
                    f"need {self.MIN_SAMPLES_TO_INFER}")
            sigma = float(np.std(a))
        return self.headroom_sigma*max(sigma, 1e-12)

    def clip(self, x, sigma: float | None = None):
        if not self.enabled:
            return x
        L = self.limit(x, sigma)
        return x if not np.isfinite(L) else np.clip(x, -L, L)

    def to_volts(self, x, sigma: float = 1.0):
        L = self.limit(sigma=sigma)
        g = self.swing_v/L if np.isfinite(L) and L > 0 else 0.0
        return self.v_ref + g*np.asarray(x, float)

    def gain_v_per_unit_at(self, sigma: float = 1.0) -> float:
        L = self.limit(sigma=sigma)
        return float(self.swing_v/L) if np.isfinite(L) and L > 0 else 0.0

    def clipped_fraction(self, x, sigma: float | None = None) -> float:
        if not self.enabled:
            return 0.0
        L = self.limit(x, sigma)
        if not np.isfinite(L):
            return 0.0
        return float(np.mean(np.abs(np.asarray(x, float)) > L))

    def with_headroom(self, headroom_sigma: float) -> "Rails":
        return replace(self, headroom_sigma=headroom_sigma, gain_v_per_unit=None,
                       enabled=True)


DEFAULT_RAILS = Rails()


# -------------------------------------------------- measured ferrodiode states

# Exponential fits I = G exp(A V) over [6.45, 7.45] V window for 8
# selected states. (Kim et al., ACS Nano 18(24):15925 (2024), Fig. 5(c))

# Coefficients fitted in this work; underlying I-V sweeps are Kim et al.'s
# and available from corresponding author on reasonable request
REAL_A = np.array([1.341, 1.300, 1.226, 1.187, 1.129, 1.055, 1.007, 0.897])
REAL_G = np.array([5.57e-13, 5.29e-13, 6.17e-13, 4.16e-13,
                   3.70e-13, 5.19e-13, 6.46e-13, 1.14e-12])

VIN_RANGE  = (0.0, 3.3)
FED_WINDOW = (6.45, 7.45)


@dataclass
class FeDWeightBank:
    signed: bool = True
    ideal: bool = True
    rails: Rails = field(default_factory=Rails)

    def saturate(self, y, sigma: float | None = None):
        return self.rails.clip(y, sigma)

    def slopes(self) -> np.ndarray:
        if self.ideal:
            return np.linspace(REAL_A[0], REAL_A[-1], len(REAL_A))
        return REAL_A.copy()

    def weight_levels(self) -> np.ndarray:
        m = self.slopes()
        w = np.subtract.outer(m, m).ravel() if self.signed else m
        w = np.unique(np.round(w, 12))
        return w / np.max(np.abs(w))

    def quantize(self, W: np.ndarray) -> np.ndarray:
        lv = self.weight_levels()
        s = np.max(np.abs(W)) / np.max(np.abs(lv)) if np.max(np.abs(W)) > 0 else 1.0
        idx = np.abs(W[..., None]/s - lv).argmin(-1)
        return lv[idx] * s

    def n_devices(self, n_weights: int) -> int:
        return 2 * n_weights if self.signed else n_weights


# -------------------------------------------------------- band-pass oscillator

# Fitted, one free parameter; effective loss resistance at the tap
_R_LOSS_OHM = 36.9

@dataclass
class ResonatorMacro:
    L_H: float = 0.1
    C_eq_F: float = 6.33e-6
    tap_n: float = 0.05
    R_loss_ohm: float = _R_LOSS_OHM
    dcr_ohm: float = 0.0
    rails: Rails = field(default_factory=Rails)

    @property
    def z0(self) -> float:
        return float(np.sqrt(self.L_H / self.C_eq_F))

    @property
    def f0_hz(self) -> float:
        return 1.0 / (2*np.pi*np.sqrt(self.L_H * self.C_eq_F))

    @property
    def zeta(self) -> float:
        return (self.tap_n**2/2)*self.z0/self.R_loss_ohm + self.dcr_ohm/(2*self.z0)

    @classmethod
    def for_target(cls, f0_hz: float, zeta: float, L_H: float = 0.1, dcr_ohm: float = 0.0,
                   rails: Rails | None = None):
        C_eq = 1.0/((2*np.pi*f0_hz)**2 * L_H)
        z0 = np.sqrt(L_H/C_eq)
        z_res = zeta - dcr_ohm/(2*z0)
        if z_res <= 0:
            raise ValueError(f"zeta {zeta:.4f} from DCR alone")
        n = float(np.sqrt(2*z_res*_R_LOSS_OHM/z0))
        if not 0 < n <= 0.975:
            raise ValueError(f"tap fraction {n:.3f} outside 2.5-97.5%")
        return cls(L_H=L_H, C_eq_F=C_eq, tap_n=n, dcr_ohm=dcr_ohm,
                   rails=rails if rails is not None else Rails())

    def components(self) -> dict:
        n = self.tap_n
        return dict(L_H=self.L_H, C1_uF=self.C_eq_F/(1-n)*1e6,
                    C2_uF=self.C_eq_F/n*1e6, tap_pct=100*n,
                    f0_hz=self.f0_hz, zeta=self.zeta, Q=1/(2*self.zeta))

    def response(self, x: np.ndarray, dt: float, sigma: float | None = None) -> np.ndarray:
        w = 2*np.pi*self.f0_hz; z = self.zeta; K = 2.0/dt
        a0 = K*K + 2*z*w*K + w*w
        b = np.array([1.0, 2.0, 1.0])/a0
        a = np.array([a0, 2*w*w - 2*K*K, K*K - 2*z*w*K + w*w])/a0
        return self.rails.clip(lfilter(b, a, x), sigma)


# ------------------------------------------------------------ leaky integrator
TAU_S_MS = 0.75
TAU_M_MS = 4.54
K_SPAN   = (0.984, 1.445)

@dataclass
class IntegratorMacro:
    tau_m_ms: float = TAU_M_MS
    tau_s_ms: float = TAU_S_MS
    gain: float = 1.0
    include_kernel: bool = True
    rails: Rails = field(default_factory=Rails)

    @staticmethod
    def _lp(x, tau_s, dt):
        a = np.exp(-dt/tau_s)
        return lfilter([1-a], [1.0, -a], x)

    def response(self, x: np.ndarray, dt: float, sigma: float | None = None) -> np.ndarray:
        if self.include_kernel:
            x = self.rails.clip(self._lp(x, self.tau_s_ms*1e-3, dt), sigma)
        return self.rails.clip(self.gain*self._lp(x, self.tau_m_ms*1e-3, dt), sigma)


# ------------------------------------------------------------ crossbar readout
@dataclass
class LinearReadout:
    bank: FeDWeightBank = field(default_factory=FeDWeightBank)
    w: np.ndarray | None = None
    bias: float = 0.0
    out_sigma: float | None = None

    def fit_ridge(self, X: np.ndarray, y: np.ndarray, lam: float = 1e-4):
        A = np.hstack([X, np.ones((len(X), 1))])
        n = A.shape[1]
        sol = np.linalg.solve(A.T@A + lam*len(A)*np.eye(n), A.T@y)
        self.w, self.bias = sol[:-1], sol[-1]
        self.out_sigma = float(np.std(A@sol))
        return self

    def predict(self, X: np.ndarray, quantize: bool = False,
                sigma: float | None = None) -> np.ndarray:
        w = self.bank.quantize(self.w) if quantize else self.w
        if sigma is None:
            sigma = self.out_sigma
        return self.bank.saturate(X@w + self.bias, sigma)

    def n_devices(self) -> int:
        return self.bank.n_devices(len(self.w))
