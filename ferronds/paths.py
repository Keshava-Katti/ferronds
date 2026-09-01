"""Where the LTspice exports and published corpora live"""

from __future__ import annotations
import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("FERRONDS_DATA",
                                Path(__file__).resolve().parents[1] / "data"))

def states_dir() -> Path:      return DATA_ROOT / "States"
def neuron_circuit_dir() -> Path: return DATA_ROOT / "Neuron Circuit"
def damping_dir() -> Path:     return DATA_ROOT / "Damping Analysis"
def mackey_glass_dir() -> Path:
    return Path(os.environ.get("FERRONDS_MG_DATA", DATA_ROOT / "mackey_glass"))

def require(p: Path) -> Path:
    return p
