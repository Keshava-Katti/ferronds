"""Circuit non-idealities as samplers and analytic sensitivities"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from ferronds.analog.macromodels import REAL_A, REAL_G, FeDWeightBank
from ferronds.analog.transistors import thermal_voltage, K_B, Q_E

V_READ = 6.95
T_REF = 300.0


_p = np.polyfit(REAL_A, np.log(REAL_G), 1)
LNG_SLOPE, LNG_INTERCEPT = float(_p[0]), float(_p[1])
LNG_RESID_SD = float(np.std(np.log(REAL_G) - np.polyval(_p, REAL_A)))
CORR_A_LNG = float(np.corrcoef(REAL_A, np.log(REAL_G))[0, 1])
CV_A_STATES = float(REAL_A.std()/REAL_A.mean())


# ---------------------------------------------- programming and process spread
@dataclass
class ProgrammingVariation:
    sigma_A: float = 0.02
    correlated: bool = True

    def draw(self, state_idx: np.ndarray, rng: np.random.Generator):
        A0 = REAL_A[state_idx]
        lnG0 = np.log(REAL_G[state_idx])
        A = A0*(1.0 + rng.normal(0.0, self.sigma_A, A0.shape))
        if self.correlated:
            resid_sd = self.sigma_A*(LNG_RESID_SD/CV_A_STATES)
            lnG = lnG0 + LNG_SLOPE*(A - A0) + rng.normal(0.0, resid_sd, A.shape)
        else:
            lnG = lnG0 + rng.normal(0.0, self.sigma_A, A.shape)
        return A, lnG


K_EV = 8.617333262e-5


def _beta_of(A0, v0: float = V_READ):
    return (np.asarray(A0, float) - 1.0/v0)/(0.5/np.sqrt(v0))


@dataclass
class ProcessVariation:
    cv_d: float = 0.02
    cv_eps: float = 0.0
    sigma_phi_ev: float = 0.0
    T: float = T_REF

    def draw(self, state_idx: np.ndarray, rng: np.random.Generator):
        A0 = REAL_A[state_idx]
        lnG0 = np.log(REAL_G[state_idx])
        b0 = _beta_of(A0)
        dd = rng.normal(0.0, self.cv_d, A0.shape) if self.cv_d else 0.0
        de = rng.normal(0.0, self.cv_eps, A0.shape) if self.cv_eps else 0.0
        dphi = (rng.normal(0.0, self.sigma_phi_ev, A0.shape)
                if self.sigma_phi_ev else 0.0)
        db = -0.5*(dd + de)*b0
        A = A0 + db*(0.5/np.sqrt(V_READ))
        lnG = lnG0 - dd + db*(np.sqrt(V_READ)/2) - dphi/(K_EV*self.T)
        return A, lnG

    def implied_sigma_A(self) -> float:
        A0 = float(REAL_A.mean())
        return float(FIELD_FRACTION_OF_A*0.5*np.hypot(self.cv_d, self.cv_eps))


FIELD_FRACTION_OF_A = 1.0 - (1.0/V_READ)/float(REAL_A.mean())
# Kim Fig. 3(g), digitised
BARRIER_EV_AT_10V = 0.62


# ----------------------------------------------------------------- temperature
def A_scale(T_kelvin: float, T_ref: float = T_REF) -> float:
    f = FIELD_FRACTION_OF_A
    return (1.0 - f) + f*(T_ref/T_kelvin)


def A_scale_digitised(T_kelvin: float, T_ref: float = T_REF) -> float:
    return float(np.exp(-0.0115*(T_kelvin - T_ref)))


def temperature_is_calibratable() -> str:
    return temperature_is_calibratable.__doc__


# ------------------------------------------------------------------- retention

# Retention windows from Kim Fig. 5; 300 s is ten reads at 30 s, 5e4 s is
# two-state on/off 737 point; exponent is fitted to LRS trace, which
# rises 4.00x from 10 s to 5e4 s as power law R ~ t^0.1635
RETENTION_MULTISTATE_S = 300.0
RETENTION_TWOSTATE_S = 5.0e4
RETENTION_T0_S = 10.0

RETENTION_EXPONENT = 0.1635


def retention_current_factor(t_s, exponent: float = RETENTION_EXPONENT,
                             t0_s: float = RETENTION_T0_S):
    return (np.asarray(t_s, float)/t0_s)**(-exponent)


def retention_is_an_offset() -> str:
    return retention_is_an_offset.__doc__


# ----------------------------------------------------------------------- noise
def shot_noise_a(i_read_a: float = 3.0e-9, bw_hz: float = 1.0e4) -> float:
    return float(np.sqrt(2*Q_E*i_read_a*bw_hz))


def ktc_noise_v(c_f: float = 0.01e-6, T: float = T_REF) -> float:
    return float(np.sqrt(K_B*T/c_f))


# Datasheet input-referred noise; LT6003 325 nV/rtHz and 12 fA/rtHz,
# LTC2068 90 nV/rtHz and 35 fA/rtHz, both at f <= 100 Hz
AMPS = {200: dict(name="LT6003", en_v_rthz=325e-9, in_a_rthz=12e-15),
        10_000: dict(name="LTC2068", en_v_rthz=90e-9, in_a_rthz=35e-15)}
N_STAGES = 5


def noise_floors(f_hz: int = 10_000, i_read_a: float = 3.0e-9,
                 c_f: float = 0.01e-6, swing_v: float = 0.6,
                 n_stages: int = N_STAGES) -> dict:
    amp = AMPS[f_hz]
    bw = float(f_hz)
    shot = shot_noise_a(i_read_a, bw)
    i_amp = amp["in_a_rthz"]*np.sqrt(bw)
    i_tot = float(np.hypot(shot, i_amp))
    v_amp = amp["en_v_rthz"]*np.sqrt(bw)*np.sqrt(n_stages)
    v_ktc = ktc_noise_v(c_f)
    return dict(amp=amp["name"], bw_hz=bw,
                shot_a=shot, shot_frac=shot/i_read_a,
                amp_i_a=i_amp, amp_i_frac=i_amp/i_read_a,
                current_total_a=i_tot, current_total_frac=i_tot/i_read_a,
                amp_v_v=v_amp, amp_v_frac=v_amp/swing_v,
                ktc_v=v_ktc, ktc_frac=v_ktc/swing_v,
                margin_vs_matching_spec=0.02/(i_tot/i_read_a))


# ----------------------------------------- read current and energy sensitivity
def read_current_a(A, lnG, v_read: float = V_READ):
    return np.exp(np.asarray(lnG, float) + np.asarray(A, float)*v_read)


def read_current_spread(sigma_A: float, v_read: float = V_READ) -> dict:
    A0 = float(REAL_A.mean())
    log_sd = A0*v_read*sigma_A
    return dict(sigma_A=sigma_A, log_sd=log_sd,
                current_cv=float(np.sqrt(np.exp(log_sd**2) - 1.0)),
                mean_shift=float(np.exp(0.5*log_sd**2)))


def energy_sensitivity(sigma_A: float, f_hz: int = 10_000, design=None) -> dict:
    from ferronds.analog import power as P
    d = design if design is not None else P.Design(
        "frozen band_power", n_res=16, n_channels=16, n_weights=16, n_integrators=16)
    br = P.breakdown_w(d, f_hz)
    total, fed = br["total_w"], br["fed_w"]
    cur = read_current_spread(sigma_A)
    return dict(fed_w=fed, total_w=total, fed_fraction=fed/total,
                current_cv=cur["current_cv"],
                total_power_cv=cur["current_cv"]*fed/total)


if __name__ == "__main__":
    print(f"ln G locus: {LNG_SLOPE:.3f} A {LNG_INTERCEPT:+.2f}")
    print(f"Residual sd: {LNG_RESID_SD:.3f}")
    print(f"corr(A, ln G): {CORR_A_LNG:+.3f}")
    print(f"Field-dependent fraction of A: {FIELD_FRACTION_OF_A:.3f}")
    print(f"Barrier (10 V): {BARRIER_EV_AT_10V} eV")
    for dT in (20, 40, 58):
        print(f"A scale (+{dT} K): {A_scale(T_REF+dT):.3f} bound, "
              f"{A_scale_digitised(T_REF+dT):.3f} digitised")
    nf = noise_floors()
    print(f"Shot noise: {nf['shot_a']:.2e} A")
    print(f"Shot noise fraction: {100*nf['shot_frac']:.4f}%")
    print(f"kT/C noise: {nf['ktc_v']:.2e} V")
    print(f"kT/C noise fraction: {100*nf['ktc_frac']:.5f}%")
    print(f"Amplifier: {nf['amp']}")
    for s in (0.01, 0.02, 0.05):
        c = read_current_spread(s)
        print(f"sigma_A {100*s:.1f}%: current CV {100*c['current_cv']:.1f}%, "
              f"mean shift {100*(c['mean_shift']-1):+.2f}%")
