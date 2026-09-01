"""Non-linearity circuit already has; exponential synapse and MOS square law"""

from __future__ import annotations
import numpy as np


A_MIN, A_MAX = 0.897, 1.341
STAGE1_SLOPE = 0.3030303

# --------------------------------------------------------- exponential synapse
def exponential_synapse(z, k):
    e = np.exp(np.clip(k*z, -20, 20))
    return e - e.mean()

def expand_exponential(F, rates=None, drive=1.0, n_rates=3):
    if rates is None:
        rates = np.linspace(A_MIN, A_MAX, n_rates)*STAGE1_SLOPE*drive
    out = [exponential_synapse(F[:, j], k) for j in range(F.shape[1]) for k in rates]
    out = np.stack(out, 1)
    return out/(out.std(0, keepdims=True) + 1e-12)

# -------------------------------------------------------------- MOS square law
def mos_square_law(z, vt=0.3, beta=1.0):
    v = z - z.min() + vt*0.5
    return beta*np.where(v > vt, (v - vt)**2, 0.0)

def expand_mos(F, vts=(0.2, 0.4, 0.6)):
    out = [mos_square_law(F[:, j], vt=v) for j in range(F.shape[1]) for v in vts]
    out = np.stack(out, 1)
    out = out - out.mean(0, keepdims=True)
    return out/(out.std(0, keepdims=True) + 1e-12)
