"""Builds an oscillator bank and turns a signal into a feature matrix"""

from __future__ import annotations
import numpy as np
from ferronds.analog.macromodels import ResonatorMacro, IntegratorMacro, TAU_M_MS, Rails

def build_bank(n_resonators=32, band=(30.0, 200.0), zeta=0.15, L_H=1.0, rails=None):
    freqs = np.logspace(np.log10(band[0]), np.log10(band[1]), n_resonators)
    return [ResonatorMacro.for_target(f, zeta, L_H=L_H, rails=rails) for f in freqs]

def build_mixed_bank(n_resonators=32, band=(30.0, 200.0), zetas=(0.05, 0.15, 0.30),
                     L_H=1.0, rails=None):
    freqs = np.logspace(np.log10(band[0]), np.log10(band[1]),
                        int(np.ceil(n_resonators/len(zetas))))
    out = []
    for z in zetas:
        for f in freqs:
            out.append(ResonatorMacro.for_target(f, z, L_H=L_H, rails=rails))
    return out[:n_resonators] if len(out) > n_resonators else out

def default_taus(multipliers=(0.25, 1.0, 4.0)):
    return [TAU_M_MS*m for m in multipliers]

def feature_matrix(signal, bank, taus_ms, dt, normalize=True, rails=None):
    R = np.stack([m.response(signal, dt) for m in bank])
    if normalize:
        R = R/(R.std(axis=1, keepdims=True) + 1e-12)
    kw = {} if rails is None else dict(rails=rails)
    blocks = [np.stack([IntegratorMacro(tau_m_ms=tm, **kw).response(r, dt) for r in R])
              for tm in taus_ms]
    return np.concatenate(blocks, axis=0).T

def rf_feature_matrix(signal, bank, taus_ms, dt, quadrature=True, normalize=True,
                      rails=None):
    from ferronds.dynamics.spectral import RFBank
    Z = RFBank(list(bank), dt).response(signal)
    R = np.concatenate([Z.real, Z.imag], axis=0) if quadrature else Z.real
    if normalize:
        R = R/(R.std(axis=1, keepdims=True) + 1e-12)
    kw = {} if rails is None else dict(rails=rails)
    blocks = [np.stack([IntegratorMacro(tau_m_ms=tm, **kw).response(r, dt) for r in R])
              for tm in taus_ms]
    return np.concatenate(blocks, axis=0).T

def n_channels(bank, taus_ms, quadrature=False):
    return len(bank)*len(taus_ms)*(2 if quadrature else 1)
