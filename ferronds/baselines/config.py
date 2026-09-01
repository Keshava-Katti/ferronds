"""Dataset location, device selection, and seed policy every baseline shares"""

import os
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASETS = Path(os.environ.get("FERRONDS_ML_DATA", REPO_ROOT / "data" / "corpora"))
DTYPE = np.float64

SEEDS_PILOT = 10
SEEDS_MAX = 40
HELD_OUT_SEED0 = 20

# ------------------------------------------------------------------ device

def get_device(prefer: str | None = None):
    import torch

    want = prefer or os.environ.get("FERRONDS_ML_DEVICE")
    if want:
        return torch.device(want)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        # mamba-ssm's selective scan is CUDA kernel; MPS fallback is slow
        # sequential scan, so Mamba timing taken here is not comparable
        print("Device: MPS")
        print("Mamba: sequential fallback, timings invalid")
        return torch.device("mps")
    return torch.device("cpu")

# -------------------------------------------------------------------- seeds

def seed_everything(seed: int) -> None:
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
