# FerroNDS

Source code for **"Neural dynamical systems on ferroelectric compute-in-memory for real-time forecasting"** (Katti, Selvakumar, Chaudhari, Jariwala; *Neuromorphic Computing and Engineering*, under revision).

FerroNDS is a neural dynamical system that pairs bandpass oscillator and leaky integrator primitives with a multi-bit ferrodiode (FeD) synaptic weighting circuit for real-time signal prediction on analog compute-in-memory hardware.

## Requirements

- Python 3.10+
- NumPy 1.23, SciPy 1.10

`ferronds/baselines` additionally needs PyTorch 2.7 and scikit-learn 1.2. `mamba-ssm` is optional; a pure-PyTorch state-space block is used when it is absent.

```bash
pip install numpy scipy                 # core
pip install torch scikit-learn          # baselines
```

## Repository structure

### `ferronds/analog` (devices and circuits)

| File | Description |
|---|---|
| `macromodels.py` | The five blocks FerroNDS is built from; supply rails, FeD weight bank over eight measured conductance states, bandpass oscillator, leaky integrator, and crossbar readout. Everything else imports from here. |
| `lc_tank.py` | Maps an oscillator macro onto physical `L` and `C` and back. Damping from a series or shunt resistor, integrator tap that yields quadrature, and component tolerance sweeps. |
| `power.py` | Per-component power, energy, and latency. Holds five-stage synaptic chain and its four mitigated variants (i.e., no log amplifier, current-mode column readout, 2 V supply, all levers together). |
| `variability.py` | Non-idealities as samplers and analytic sensitivities: programming and process spread in `A` and `G`, temperature, retention, shot and kTC noise, and how each channel reaches supply. |
| `transistors.py` | NMOS rectifier and differential pair written as circuits, with ideal square law alongside for comparison. |
| `nonlinearity.py` | Two nonlinearities the circuit already has, expanded into feature channels. |

### `ferronds/dynamics` (computation)

| File | Description |
|---|---|
| `spectral.py` | Oscillator bank as short-time Fourier transform (STFT). Pole parameters, equivalent noise bandwidth, constant-Q and constant-bandwidth constructors, and exponential-window reference transform that bank is equivalent to. |
| `features.py` | Builds bank and turns signal into a feature matrix. |
| `readout.py` | Multi-variate non-linear readout that quantizes, clips, and scales onto FeD crossbar. |
| `autoregressive.py` | Streaming front-end and closed-loop rollout used for chaotic function prediction. |
| `spectral_tasks.py` | Tasks whose target is property of the spectrum, i.e., fault power and frequency tracking. |

### `ferronds/data` (signals and corpora)

| File | Description |
|---|---|
| `signals.py` | Shared timebase and signal generators. Horizons are quoted in signal periods and resolved to integer samples here. |
| `mackey_glass.py` | Mackey-Glass series, both published NeuroBench instantiations and local RK4 regeneration from same parameters. |
| `keyword_spotting.py` | Google Speech Commands v2, standard 12-class split. |
| `keyword_spotting_frontend.py` | FerroNDS and log-mel front-ends at readout held to fixed width. |

### `ferronds/baselines` (comparison)

| File | Description |
|---|---|
| `registry.py` | Every model behind one interface; naive floors, FerroNDS, ridge regression, MLP, GRU, and Mamba. Includes state-space (S6) block in pure PyTorch that computes same function as `mamba-ssm` without needing `nvcc`. |
| `tasks.py` | Task registry; every task owns its own protocol, including purge gap between train and test. |
| `protocol_mackey_glass.py` | NeuroBench's chaotic function prediction protocol, worked in Lyapunov times. |
| `protocol_keyword_spotting.py` | Fixed-readout front-end comparison on Speech Commands. |
| `digital_power.py` | MACs, parameters, and ASIC energy for a digital baseline and inference rate at which analog stops winning. |
| `config.py` | Dataset location, device selection, and seed policy. |

### `ferronds/evaluation` and `tests`

| File | Description |
|---|---|
| `splitting.py` | Purged splitting and ridge penalty selection, identical for every model. Purge gap covers both the target overlap at horizon `H` and filter memory. |
| `test_spectral.py` | Checks bank against exponential-window STFT and that constant-bandwidth bank has requested ENBW exactly. |

`data/` and `results/` are where LTspice exports and published corpora are expected; see `ferronds/paths.py`.

## From claim to experiment

One of the paper's central claims is that the oscillator bank computes STFT exactly under the exponential window that the tank response applies. Corresponding code is in `spectral.py`:

```python
import numpy as np
from ferronds.dynamics.spectral import RFBank, build_constant_bw_bank, exponential_window_stft

fs = 2000.0
bank = RFBank(build_constant_bw_bank(n_resonators=8, band=(30., 200.), enbw_hz=12.0), 1/fs)
t = np.arange(2000)/fs
x = np.sin(2*np.pi*(30 + 170*t/t[-1])*t)          # 30-200 Hz chirp

Z = bank.response(x)                               # tank states
S = exponential_window_stft(x, bank.f0_hz, fs, bank.lam)
print(np.abs(Z - S).max())                         # 1.9e-13
```

The energy, power, and latency that abstract reports come out of `power.py` on evaluated fault-power design:

```python
from ferronds.analog import power

d = power.Design("fault power", n_res=16, n_channels=16, n_weights=16, n_integrators=16)
for f in (200, 10_000):
    e = power.energy_per_neuron_per_inference_j(d, f)
    p = power.power_w(d, f)["p_total_w"] + power.fed_power_w(d)
    print(f, "Hz:", round(e*1e9, 2), "nJ,", round(p*1e6, 1), "uW,",
          round(power.latency_s(f)*1e3, 4), "ms")
# 200 Hz:    89.53 nJ,  287.2 uW, 3.1831 ms
# 10000 Hz:  14.84 nJ, 2375.3 uW, 0.0637 ms
```

## Citation

```bibtex
@article{katti2026ferronds,
  title   = {Neural dynamical systems on ferroelectric compute-in-memory
             for real-time forecasting},
  author  = {Katti, Keshava and Selvakumar, Adithya and Chaudhari, Pratik
             and Jariwala, Deep},
  year    = {2026},
  note    = {Submitted}
}
```

## License

This repository accompanies an academic publication. Please cite the paper if you use this code.
