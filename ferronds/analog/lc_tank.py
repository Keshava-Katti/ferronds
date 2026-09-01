"""Maps oscillator macro onto real L and C values"""

from __future__ import annotations
import numpy as np
from ferronds.analog.macromodels import ResonatorMacro
from ferronds.dynamics import spectral


# ------------------------------------------------------- impedance and damping
def characteristic_impedance(macro: ResonatorMacro) -> float:
    return float(np.sqrt(macro.L_H/macro.C_eq_F))


def damping_from_series_R(macro: ResonatorMacro, r_series_ohm: float) -> float:
    return float(r_series_ohm/(2*characteristic_impedance(macro)))


def damping_from_shunt_R(macro: ResonatorMacro, r_shunt_ohm: float) -> float:
    return float(characteristic_impedance(macro)/(2*r_shunt_ohm))


# ------------------------------------------ quadrature from integrator tap
def quadrature_by_integration(macro: ResonatorMacro, tau_s: float, dt: float):
    w = 2*np.pi*macro.f0_hz
    ideal = 1.0/(1j*w)
    leaky = tau_s/(1.0 + 1j*w*tau_s)
    ratio = leaky/ideal
    return float(np.abs(ratio)/np.abs(ratio)), float(np.degrees(np.angle(ratio)))


def integrator_phase_error_deg(f0_hz: float, tau_s: float) -> float:
    w = 2*np.pi*f0_hz
    return float(90.0 - np.degrees(np.arctan(w*tau_s)))


def quadrature_conditioning(phase_error_deg: float) -> float:
    sep = np.radians(90.0 - abs(phase_error_deg))
    return float(1.0/max(np.sin(sep), 1e-12))


def quadrature_options(macro: ResonatorMacro, dt: float,
                       r_series_ohm=1.0, r_shunt_ohm=1e6, tau_s=10e-3) -> dict:
    z0 = characteristic_impedance(macro)
    d_series = damping_from_series_R(macro, r_series_ohm)
    d_shunt = damping_from_shunt_R(macro, r_shunt_ohm)
    return dict(
        f0_hz=macro.f0_hz, zeta=macro.zeta, z0_ohm=z0,
        series_sense=dict(r_ohm=r_series_ohm, dzeta=d_series,
                          pct_of_zeta=100*d_series/macro.zeta),
        shunt_load=dict(r_ohm=r_shunt_ohm, dzeta=d_shunt,
                        pct_of_zeta=100*d_shunt/macro.zeta),
        integrator=dict(tau_s=tau_s,
                        phase_err_deg=integrator_phase_error_deg(macro.f0_hz, tau_s),
                        dzeta=d_shunt, pct_of_zeta=100*d_shunt/macro.zeta),
    )


# ------------------------------------------------------------ component values
def components(macro: ResonatorMacro) -> tuple[float, float, float]:
    n = macro.tap_n
    return macro.L_H, macro.C_eq_F/(1 - n), macro.C_eq_F/n


def from_components(L, C1, C2, dcr_ohm=0.0, R_loss_ohm=None) -> ResonatorMacro:
    n = C1/(C1 + C2)
    C_eq = C1*C2/(C1 + C2)
    kw = dict(L_H=L, C_eq_F=C_eq, tap_n=n, dcr_ohm=dcr_ohm)
    if R_loss_ohm is not None:
        kw["R_loss_ohm"] = R_loss_ohm
    return ResonatorMacro(**kw)


# ------------------------------------------------------------ tolerance sweeps
def perturb_bank(bank, rng, sigma_L=0.10, sigma_C=0.05, correlated_caps=True):
    out = []
    for m in bank:
        L, C1, C2 = components(m)
        common = rng.normal(0, sigma_C) if correlated_caps else 0.0
        out.append(from_components(
            L*(1 + rng.normal(0, sigma_L)),
            C1*(1 + common + rng.normal(0, sigma_C)),
            C2*(1 + common + rng.normal(0, sigma_C)),
            dcr_ohm=m.dcr_ohm))
    return out


def bank_stats(bank, dt) -> dict:
    rf = spectral.RFBank(list(bank), dt)
    b = rf.enbw_hz()
    return dict(f0=rf.f0_hz.copy(), enbw=b.copy(), lam=rf.lam.copy(),
                zeta=np.array([m.zeta for m in bank]),
                enbw_cv=float(b.std()/b.mean()))


def tolerance_sweep(bank_fn, dt, sigmas, n_draws=64, seed=0, **kw) -> list:
    rng = np.random.default_rng(seed)
    nominal = bank_fn()
    s0 = bank_stats(nominal, dt)
    rows = []
    for sL, sC in sigmas:
        cv, ferr, enbw_err = [], [], []
        for _ in range(n_draws):
            b = perturb_bank(nominal, rng, sL, sC, **kw)
            s = bank_stats(b, dt)
            cv.append(s["enbw_cv"])
            ferr.append(np.abs(s["f0"]/s0["f0"] - 1))
            enbw_err.append(np.abs(s["enbw"]/s0["enbw"] - 1))
        rows.append(dict(sigma_L=sL, sigma_C=sC,
                         enbw_cv=float(np.mean(cv)),
                         f0_err_pct=float(100*np.mean(ferr)),
                         f0_err_p95_pct=float(100*np.percentile(ferr, 95)),
                         enbw_err_pct=float(100*np.mean(enbw_err)),
                         enbw_err_p95_pct=float(100*np.percentile(enbw_err, 95))))
    return rows
