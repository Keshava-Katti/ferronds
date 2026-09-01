"""Task registry; every task owns its protocol"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import numpy as np

from ferronds.baselines import config
from ferronds.dynamics import spectral_tasks as S
from ferronds.evaluation.splitting import Split, make_split, purge_gap, xy
from ferronds.data.signals import Timebase, make_hard_signal, make_signal, signal_seed

PERIODS = {"freq_track": 800, "band_power": 400}
DEFAULT_HORIZONS = (0.0, 0.5, 2.0, 8.0)

WAVEFORM_PERIODS = 800

TRIVIAL_WAVEFORMS = ("noisy_sine", "noisy_square", "am_sine", "env_sine",
                     "composite")
NONTRIVIAL_WAVEFORMS = ("chirp", "wander_sine", "wander_am", "jitter_impulse")

WAVEFORM_HORIZONS = (0.5, 8.0)


@dataclass(frozen=True)
class TaskSpec:
    name: str
    horizon_periods: float
    family: str
    target_is_latent: bool
    metric: str = "mse"

    @property
    def key(self) -> str:
        return f"{self.name}|H={self.horizon_periods}"


@dataclass
class TaskData:
    spec: TaskSpec
    x: np.ndarray
    y: np.ndarray
    tb: Timebase
    split: Split
    purge: int
    seed: int

    def raw(self, which: str = "train"):
        sl = self.split.train if which == "train" else self.split.test
        xs = self.x[:sl.stop] if which == "train" else self.x[sl]
        _, ys = xy(np.asarray(self.x)[:, None], self.y, self.split, which)
        return xs, ys

    def features(self, F: np.ndarray, which: str = "train"):
        return xy(F, self.y, self.split, which)

    def lags(self, n_lags: int = 64, which: str = "train"):
        L = np.stack([np.concatenate([np.full(j, np.nan), self.x[:len(self.x)-j]])
                      for j in range(n_lags)], axis=1)
        X, y = xy(L, self.y, self.split, which)
        keep = np.isfinite(X).all(axis=1)
        return X[keep], y[keep]

    def sequence(self, window: int = 512, stride: int = 128, which: str = "train",
                 warmup: int = 64):
        X, y = self.lags(1, which)
        n = len(X)
        starts = np.arange(0, n - window + 1, stride)
        xs = np.stack([X[s:s+window, 0] for s in starts])[:, :, None]
        ys = np.stack([y[s:s+window] for s in starts])
        mask = np.ones_like(ys, dtype=bool)
        mask[:, :warmup] = False
        return xs, ys, mask

    @property
    def n_train(self) -> int:
        return self.split.train.stop

    @property
    def n_test(self) -> int:
        return self.split.test.stop - self.split.test.start


SPECTRAL = [TaskSpec(n, h, "spectral", target_is_latent=True)
            for n in ("freq_track", "band_power") for h in DEFAULT_HORIZONS]

WAVEFORM = [TaskSpec(n, h, "waveform", target_is_latent=False)
            for n in TRIVIAL_WAVEFORMS + NONTRIVIAL_WAVEFORMS
            for h in WAVEFORM_HORIZONS]

REGISTRY = {t.key: t for t in SPECTRAL + WAVEFORM}


def make(spec: TaskSpec | str, seed: int) -> TaskData:
    spec = REGISTRY[spec] if isinstance(spec, str) else spec

    if spec.family == "spectral":
        tb = replace(Timebase(), n_periods=PERIODS[spec.name],
                     horizon_periods=spec.horizon_periods)
        x, y = S.make_task(spec.name, tb, seed)
        purge = purge_gap(tb.horizon, [max(20.0, S.TAU_DETECTOR_MS)], tb.fs_hz)

    elif spec.family == "waveform":
        tb = replace(Timebase(), n_periods=WAVEFORM_PERIODS,
                     horizon_periods=spec.horizon_periods)
        rng = np.random.default_rng(signal_seed(spec.name, seed))
        hard = spec.name in ("wander_sine", "wander_am", "jitter_impulse",
                             "driven_resonance")
        x = (make_hard_signal(spec.name, tb, rng) if hard
             else make_signal(spec.name, tb, rng))
        y = x
        purge = purge_gap(tb.horizon, [max(20.0, S.TAU_DETECTOR_MS)], tb.fs_hz)

    return TaskData(spec=spec, x=np.asarray(x, float), y=np.asarray(y, float),
                    tb=tb, split=make_split(len(x), tb.horizon, purge),
                    purge=purge, seed=seed)


def mse(pred, true) -> float:
    p, t = np.asarray(pred, float).ravel(), np.asarray(true, float).ravel()
    n = min(len(p), len(t))
    return float(np.mean((p[:n] - t[:n])**2))


def skill_vs_mean(pred, true, train_target) -> float:
    base = mse(np.full_like(np.asarray(true, float), np.mean(train_target)), true)
    m = mse(pred, true)
    return float(base/m) if m > 0 else float("inf")


METRICS: dict[str, Callable] = {"mse": mse}
