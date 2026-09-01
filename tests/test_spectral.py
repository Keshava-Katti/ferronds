"""Checks the bank against an exponential-window STFT. The identity has to be exact."""

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from ferronds.dynamics import features
from ferronds.dynamics import spectral
from ferronds.dynamics.spectral import RFBank, rf_params, window_length_samples, equivalent_noise_bandwidth_hz

ok = lambda c, m: print(("  PASS  " if c else "  FAIL  ") + m)
rng = np.random.default_rng(0)
DT = 1/2000.
bank = features.build_bank(n_resonators=8, band=(30., 200.), zeta=0.15)
rf = RFBank(bank, DT)
x = rng.normal(size=3000)

print("Unrolled identity, Orchard eq. (5)\n" + "-"*70)
Z = rf.response(x)
worst = 0.0
for k in (0, 3, 7):
    for t in (50, 500, 2999):
        got, want = Z[k, t], rf.unrolled(x, k, t)
        worst = max(worst, abs(got - want)/(abs(want) + 1e-300))
ok(worst < 1e-10, f"z[t] vs unrolled sum: {worst:.2e} relative")

print("\nExponential-window STFT\n" + "-"*70)
S = spectral.exponential_window_stft(x, rf.f0_hz, 1/DT, rf.lam)
err = np.abs(Z - S).max()/np.abs(S).max()
ok(err < 1e-10, f"Bank vs windowed DFT: {err:.2e}")

from ferronds.data.signals import Timebase, KINDS, make_signal, signal_seed
_tb = Timebase()
_builders = (lambda n: spectral.build_constant_q_bank(n, (30., 200.), 0.15),
             lambda n: spectral.build_constant_bw_bank(n, (30., 200.), 12.0, 1/DT))
worst_own = 1.0
for _build in _builders:
    _rf = RFBank(_build(48), DT)
    for _kind in KINDS:
        _s = make_signal(_kind, _tb, np.random.default_rng(signal_seed(_kind, 0)))
        worst_own = min(worst_own, spectral.spectrogram_agreement(
            _rf.response(_s),
            spectral.exponential_window_stft(_s, _rf.f0_hz, 1/DT, _rf.lam))["correlation"])
ok(1 - worst_own < 1e-12,
   f"Both banks, {len(KINDS)} families: worst 1-r {1 - worst_own:.1e}")

print("\nPoles from circuit\n" + "-"*70)
m = bank[3]
lam, th = rf_params(m, DT)
w = 2*np.pi*m.f0_hz; K = 2.0/DT
den = np.array([K*K + 2*m.zeta*w*K + w*w, 2*w*w - 2*K*K, K*K - 2*m.zeta*w*K + w*w])
r = np.roots(den)
ok(abs(np.abs(r[0]) - lam) < 1e-12 and abs(abs(np.angle(r[0])) - th) < 1e-12,
   f"Poles vs denominator roots: |r| {np.abs(r[0]):.9f}")
f_pole = th/(2*np.pi*DT)
f_damped = m.f0_hz*np.sqrt(1 - m.zeta**2)
f_warp = 2*np.arctan(2*np.pi*f_damped*DT/2)/(2*np.pi*DT)
ok(abs(f_pole - f_warp)/f_warp < 2e-3,
   f"Pole {f_pole:.2f} Hz, warped {f_warp:.2f} Hz, component {m.f0_hz:.2f} Hz, "
   f"shift {100*(f_pole/m.f0_hz - 1):+.1f}%")
lam_ct = np.exp(-m.zeta*2*np.pi*m.f0_hz*DT)
ok(abs(lam - lam_ct)/lam_ct < 0.01,
   f"Pole radius {lam:.5f}, continuous-time {lam_ct:.5f}")

print("\nWindow length and bandwidth\n" + "-"*70)
for k in (0, 7):
    L = window_length_samples(rf.lam[k])
    B = equivalent_noise_bandwidth_hz(rf.lam[k], 1/DT)
    print(f"Channel {k}: f0 {rf.f0_hz[k]:.1f} Hz, lam {rf.lam[k]:.5f}, "
          f"window {L:.1f} samples, {1e3*L*DT:.1f} ms, ENBW {B:.2f} Hz, "
          f"Q {rf.f0_hz[k]/B:.1f}")
q = rf.q_factor()
ok(q.std()/q.mean() < 0.05,
   f"Constant-zeta Q: {q.mean():.1f} +/- {q.std():.2f}")
ok(rf.enbw_hz().max()/rf.enbw_hz().min() > 3,
   f"ENBW span: {rf.enbw_hz().min():.2f} to {rf.enbw_hz().max():.2f} Hz")

print("\nConstant-bandwidth bank\n" + "-"*70)
cb = spectral.build_constant_bw_bank(n_resonators=8, band=(30., 200.), enbw_hz=12.0, fs=2000.)
rfb = RFBank(cb, DT)
b = rfb.enbw_hz()
ok(b.std()/b.mean() < 1e-6,
   f"ENBW spread: {100*b.std()/b.mean():.2e}%, mean {b.mean():.4f} Hz")
ok(abs(b.mean() - 12.0)/12.0 < 1e-6, f"ENBW vs 12 Hz target: {b.mean():.4f} Hz")
qb = rfb.q_factor()
ok(qb.max()/qb.min() > 5, f"Q span: {qb.min():.1f} to {qb.max():.1f}")
taps = spectral.tap_fractions(cb)
ok(taps.min() > 2.5 and taps.max() < 97.5,
   f"Tap fractions: {taps.min():.1f}% to {taps.max():.1f}%, limit 2.5-97.5%")

print("\nComplex vs magnitude readout\n" + "-"*70)
t = np.arange(4000)*DT
chirp = np.sin(2*np.pi*(30 + (170/ (4000*DT))*t/2)*t)
big = features.build_bank(n_resonators=48, band=(30., 200.), zeta=0.15)
rfx = RFBank(big, DT)
Zc = rfx.response(chirp)
rec = rfx.reconstruct(Zc)
r_full = spectral.reconstruction_correlation(chirp, rec)
Zr = Zc.copy(); Zr.imag = 0.0
rec_r = rfx.reconstruct(np.abs(Zc).astype(complex))
r_mag = spectral.reconstruction_correlation(chirp, rec_r)
ok(r_full > 0.9, f"Complex readout r: {r_full:.4f}")
ok(r_mag < 0.5, f"Magnitude-only r: {r_mag:.4f}")
print()
