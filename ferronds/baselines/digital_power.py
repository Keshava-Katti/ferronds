"""MACs, parameters, and ASIC energy for a digital baseline"""

# 45 nm energies from tabulation of Horowitz's ISSCC 2014 data in
# arXiv:1602.04183, Table I
E_ADD16, E_MUL16 = 0.18e-12, 0.62e-12
E_RF64, E_SRAM4K, E_SRAM32K, E_DRAM = 0.23e-12, 8e-12, 11e-12, 640e-12
E_MAC16 = E_ADD16 + E_MUL16


def mlp_macs(n_in, hidden, n_out=1):
    dims = [n_in] + list(hidden) + [n_out]
    return sum(dims[i]*dims[i+1] for i in range(len(dims)-1))


def mlp_params(n_in, hidden, n_out=1):
    dims = [n_in] + list(hidden) + [n_out]
    return sum(dims[i]*dims[i+1] + dims[i+1] for i in range(len(dims)-1))


def asic_energy_j(n_in, hidden, n_out=1, mem="sram4k", weights_resident=False):
    macs = mlp_macs(n_in, hidden, n_out)
    par = mlp_params(n_in, hidden, n_out)
    e_mem = {"rf": E_RF64, "sram4k": E_SRAM4K, "sram32k": E_SRAM32K, "dram": E_DRAM}[
        "rf" if weights_resident else mem]
    return dict(macs=macs, params=par,
                e_arith_j=macs*E_MAC16, e_mem_j=par*e_mem,
                e_total_j=macs*E_MAC16 + par*e_mem)


MACS_ONEPOLE_COMPLEX = 4
MACS_DETECT = {"ReZ": 0, "ReZ2": 1, "absZ2": 2}
MACS_MEMBRANE = 2


def ferronds_digital(n_res, mode="ReZ2", membrane=True, mem="sram4k",
                     weights_resident=False):
    per_ch = MACS_ONEPOLE_COMPLEX + MACS_DETECT[mode] + (MACS_MEMBRANE if membrane else 0)
    macs = n_res*per_ch + n_res
    words = 2*n_res + 2*n_res + n_res
    e_mem = {"rf": E_RF64, "sram4k": E_SRAM4K, "sram32k": E_SRAM32K, "dram": E_DRAM}[
        "rf" if weights_resident else mem]
    return dict(macs=macs, words=words, macs_per_channel=per_ch,
                e_arith_j=macs*E_MAC16, e_mem_j=words*e_mem,
                e_total_j=macs*E_MAC16 + words*e_mem)


def crossover_hz(p_analog_w, e_digital_j):
    return p_analog_w/e_digital_j
