"""Per-component power, energy, and latency for FerroNDS instance"""

from __future__ import annotations
from dataclasses import dataclass, field


# ------------------------------------------------------------------ amplifiers
@dataclass(frozen=True)
class OpAmp:
    name: str
    iq_a: float
    vdd_v: float
    f_hz: float
    f_3db_hz: float

OPAMPS = {
    200:     OpAmp("LT6003 / LTC6003",  850e-9, 1.6,     200.0,     200.0),
    10000:   OpAmp("LTC2068 / LTC6259", 7.5e-6, 1.7,   10000.0,   10000.0),
    3000000: OpAmp("LTC6261",           245e-6, 1.8, 3000000.0, 3000000.0),
}


# ----------------------------------------------- synaptic chain configurations
@dataclass(frozen=True)
class SynapseRails:
    v_stage1: float = 8.0
    v_lv: float = 3.3
    has_stage1: bool = True
    has_sense: bool = True
    has_log: bool = True
    has_out: bool = True
    log_iq_factor: float = 1.0
    column_shared: bool = False

    def stage_volts(self) -> dict:
        return {"Input Scale/Bias": self.v_stage1 if self.has_stage1 else 0.0,
                "Sense Amp":        self.v_lv if self.has_sense else 0.0,
                "Log Amp":          self.v_lv*self.log_iq_factor if self.has_log else 0.0,
                "Output Scale/Bias": self.v_lv if self.has_out else 0.0}

    @property
    def volt_sum(self) -> float:
        return sum(self.stage_volts().values())

    @property
    def n_opamps(self) -> int:
        return sum(1 for k, v in self.stage_volts().items() if v > 0)

PUBLISHED = SynapseRails()


NO_LOGAMP = SynapseRails(has_log=False)

CURRENT_MODE = SynapseRails(has_log=False, column_shared=True)

TWO_VOLT = SynapseRails(v_stage1=2.0, v_lv=2.0, has_stage1=False,
                        has_out=False)
TWO_VOLT_NO_LOGAMP = SynapseRails(v_stage1=2.0, v_lv=2.0, has_stage1=False,
                                  has_out=False, has_log=False)
ALL_LEVERS = SynapseRails(v_stage1=2.0, v_lv=2.0, has_stage1=False, has_out=False,
                          has_log=False, column_shared=True)


# ------------------------------------------------------------------- network
@dataclass(frozen=True)
class Design:
    name: str
    n_res: int
    n_channels: int
    n_weights: int
    n_integrators: int = 0
    n_stages: int = 4
    n_out_columns: int = 1

    @property
    def n_devices(self) -> int:
        return 2*self.n_weights


def _stage_counts(d: Design, rails: SynapseRails) -> dict:
    shared = d.n_out_columns if rails.column_shared else d.n_weights
    return {"Input Scale/Bias": d.n_weights,
            "Sense Amp": shared,
            "Log Amp": shared,
            "Output Scale/Bias": shared}


# ------------------------------------------------------ power, energy, latency
def power_w(d: Design, f_hz: int, rails: SynapseRails = PUBLISHED) -> dict:
    op = OPAMPS[f_hz]
    volts, counts = rails.stage_volts(), _stage_counts(d, rails)
    p_neuron = d.n_res*op.iq_a*op.vdd_v
    p_integ = integrator_power_w(d, f_hz)
    p_synapse = sum(counts[k]*op.iq_a*v for k, v in volts.items())
    n_amps = d.n_res + sum(counts[k] for k, v in volts.items() if v > 0)
    return dict(opamp=op.name, f_hz=op.f_hz,
                p_neuron_w=p_neuron, p_integrator_w=p_integ, p_synapse_w=p_synapse,
                p_total_w=p_neuron + p_integ + p_synapse, n_opamps=n_amps)


def energy_per_neuron_per_inference_j(d: Design, f_hz: int,
                                      rails: SynapseRails = PUBLISHED) -> float:
    p = power_w(d, f_hz, rails)
    return p["p_total_w"]/(p["f_hz"]*d.n_res)


def latency_s(f_hz: int, n_stages: int = 4) -> float:
    return n_stages/(2*3.141592653589793*OPAMPS[f_hz].f_3db_hz)


def neurons_within_budget(d: Design, f_hz: int, budget_w: float = 1.0,
                          rails: SynapseRails = PUBLISHED) -> float:
    per_neuron = power_w(d, f_hz, rails)["p_total_w"]/d.n_res
    return budget_w/per_neuron


XYLO_SHAPED = Design("published reference (Xylo-A2 shape)", n_res=76,
                     n_channels=76, n_weights=1632)


# ------------------------------------------------------------ ferrodiode array

# Read current over operating window is 0.4 to 12 nA across selected
# levels, mean about 3 nA (Kim et al. 2024, Fig. 5(c)); read bias is
# midpoint of [6.45, 7.45] V window
FED_I_READ_A = 3.0e-9
FED_I_READ_RANGE_A = (0.4e-9, 12.0e-9)
FED_V_READ_V = 6.95


def fed_power_w(d: "Design", i_read_a: float = FED_I_READ_A,
                v_read_v: float = FED_V_READ_V) -> float:
    return d.n_devices*i_read_a*v_read_v


def breakdown_w(d: "Design", f_hz: int, rails: SynapseRails = PUBLISHED,
                i_read_a: float = FED_I_READ_A) -> dict:
    op = OPAMPS[f_hz]
    volts, counts = rails.stage_volts(), _stage_counts(d, rails)
    fed = fed_power_w(d, i_read_a=i_read_a)
    parts = {k: counts[k]*op.iq_a*v for k, v in volts.items()}
    parts["Input Scale/Bias"] += fed
    parts["Oscillator"] = d.n_res*op.iq_a*op.vdd_v
    parts["Integrator"] = integrator_power_w(d, f_hz)
    tot = sum(parts.values())
    parts.update(total_w=tot, fed_w=fed, fed_fraction=fed/tot,
                 opamp_fraction=(tot - fed)/tot)
    return parts


def energy_per_inference_j(d: "Design", f_hz: int,
                           rails: SynapseRails = PUBLISHED,
                           i_read_a: float = FED_I_READ_A) -> float:
    return breakdown_w(d, f_hz, rails, i_read_a)["total_w"]/OPAMPS[f_hz].f_hz


# ------------------------------------------------------------------ integrator

# From LTspice integrator; mean 0.83 uA and peak 3.9 uA into 500 kOhm
# gives 1.50 V versus measured 1.52 V membrane peak on lowest state;
# I_mean is averaged over three pulses in 100 ms, so duty cycle moves it
RLEAK_OHM = 500e3
C_MEMBRANE_F = 0.01e-6
INTEGRATOR_I_MEAN_A = 0.832e-6
INTEGRATOR_I_PEAK_A = 3.904e-6


def integrator_tau_s() -> float:
    return RLEAK_OHM*C_MEMBRANE_F


def integrator_power_w(d: "Design", f_hz: int, i_a: float | None = None) -> float:
    op = OPAMPS[f_hz]
    return d.n_integrators*(INTEGRATOR_I_MEAN_A if i_a is None else i_a)*op.vdd_v
