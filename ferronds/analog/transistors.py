"""Non-linear elements as circuits"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

# ------------------------------------------------------------------- constants
K_B = 1.380649e-23
Q_E = 1.602176634e-19


def thermal_voltage(T_kelvin: float = 300.0) -> float:
    return K_B*T_kelvin/Q_E


def _softplus(u):
    u = np.asarray(u, float)
    return np.where(u > 30.0, u, np.log1p(np.exp(np.clip(u, -60.0, 30.0))))


def _sigmoid(u):
    u = np.clip(np.asarray(u, float), -60.0, 60.0)
    return 1.0/(1.0 + np.exp(-u))


# --------------------------------------------------------------------- devices
@dataclass
class NMOSRectifier:
    v_th: float = 0.0
    n: float = 1.3
    T: float = 300.0
    i_s: float = 1.0

    @property
    def u_t(self) -> float:
        return thermal_voltage(self.T)

    @property
    def knee_v(self) -> float:
        return 2*self.n*self.u_t

    def current(self, v_gs):
        return self.i_s*_softplus((np.asarray(v_gs, float) - self.v_th)/self.knee_v)**2

    def gm(self, v_gs):
        u = (np.asarray(v_gs, float) - self.v_th)/self.knee_v
        return self.i_s*2.0*_softplus(u)*_sigmoid(u)/self.knee_v


@dataclass
class DiffPair:
    n: float = 1.3
    T: float = 300.0
    v_ov: float = 0.2
    regime: str = "subthreshold"

    @property
    def u_t(self) -> float:
        return thermal_voltage(self.T)

    @property
    def v_sat(self) -> float:
        if self.regime == "subthreshold":
            return 2*2*self.n*self.u_t
        return float(np.sqrt(2)*self.v_ov)

    def transfer(self, v_id):
        v = np.asarray(v_id, float)
        if self.regime == "subthreshold":
            return np.tanh(v/(2*self.n*self.u_t))
        lim = np.sqrt(2)*self.v_ov
        vc = np.clip(v, -lim, lim)
        out = (vc/(np.sqrt(2)*self.v_ov))*np.sqrt(np.maximum(1 - vc**2/(4*self.v_ov**2), 0.0))
        return np.where(np.abs(v) >= lim, np.sign(v)*1.0, out/np.max([
            np.max(np.abs((lim/(np.sqrt(2)*self.v_ov))*np.sqrt(max(1 - lim**2/(4*self.v_ov**2), 0.0)))), 1e-12]))

    def dtransfer(self, v_id, h=1e-6):
        v = np.asarray(v_id, float)
        if self.regime == "subthreshold":
            s = 2*self.n*self.u_t
            return (1.0 - np.tanh(v/s)**2)/s
        return (self.transfer(v + h) - self.transfer(v - h))/(2*h)


# ------------------------------------------------------------ feature channels
def channel_scale_v(headroom_sigma: float = 3.0, swing_v: float = 0.6) -> float:
    return swing_v/headroom_sigma


def nmos_channel(theta_sigma=0.0, headroom_sigma=3.0, n=1.3, T=300.0):
    s = channel_scale_v(headroom_sigma)
    dev = NMOSRectifier(v_th=theta_sigma*s, n=n, T=T)
    z = np.random.default_rng(0).normal(size=20000)*s
    norm = float(dev.current(z).std()) or 1.0
    return (lambda f, th=0.0: dev.current(np.asarray(f, float)*s)/norm,
            lambda f, th=0.0: dev.gm(np.asarray(f, float)*s)*s/norm)


def diffpair_channel(atten=1.0, headroom_sigma=3.0, n=1.3, T=300.0, regime="subthreshold"):
    s = channel_scale_v(headroom_sigma)*atten
    dev = DiffPair(n=n, T=T, regime=regime)
    z = np.random.default_rng(0).normal(size=20000)*s
    norm = float(dev.transfer(z).std()) or 1.0
    return (lambda f, th=0.0: dev.transfer(np.asarray(f, float)*s)/norm,
            lambda f, th=0.0: dev.dtransfer(np.asarray(f, float)*s)*s/norm)


def compare_to_ideal(headroom_sigma=3.0, n=1.3, T=300.0, span_sigma=3.0):
    x = np.linspace(-span_sigma, span_sigma, 2001)
    s = channel_scale_v(headroom_sigma)
    out = {}

    def fit_err(a, b):
        A = np.stack([a, np.ones_like(a)], 1)
        c = np.linalg.lstsq(A, b, rcond=None)[0]
        r = A@c - b
        return float(np.sqrt(np.mean(r**2))/(b.std() + 1e-12)), float(np.max(np.abs(r))/(b.std() + 1e-12))

    nm = NMOSRectifier(n=n, T=T)
    i_nmos = nm.current(x*s)
    out["1 NMOS vs ideal ReLU"] = fit_err(i_nmos, np.maximum(x, 0.0))
    out["1 NMOS vs ideal square-law rectifier"] = fit_err(i_nmos, np.where(x > 0, x**2, 0.0))

    for at, lab in [(1.0, "1.0"), (0.3, "0.3"), (0.1, "0.1")]:
        dp = DiffPair(n=n, T=T)
        y = dp.transfer(x*s*at)
        out[f"diff pair vs ideal tanh, attenuation {lab}"] = fit_err(y, np.tanh(x))
        out[f"diff pair vs sign(x), attenuation {lab}"] = fit_err(y, np.sign(x))
    out["_knee_mV"] = 1e3*nm.knee_v
    out["_channel_scale_V_per_sigma"] = s
    return out


if __name__ == "__main__":
    nm = NMOSRectifier()
    print(f"Thermal voltage (300 K): {1e3*thermal_voltage():.2f} mV")
    print(f"NMOS turn-on width: {1e3*nm.knee_v:.1f} mV")
    print(f"Channel scale (3 sigma): {1e3*channel_scale_v():.0f} mV/sigma")
    print(f"Diff pair soft region: +/-{1e3*DiffPair().v_sat:.0f} mV")
    r = compare_to_ideal()
    for k, v in r.items():
        if k.startswith("_"):
            continue
        print(f"{k}: RMS {v[0]:.3f}, max {v[1]:.3f}")
