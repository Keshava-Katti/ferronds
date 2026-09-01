"""Mackey-Glass under NeuroBench's chaotic function prediction protocol"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ferronds.baselines import config
from ferronds.data import mackey_glass as MG
from ferronds.evaluation.splitting import Split
from ferronds.data.signals import Timebase
from ferronds.baselines.tasks import TaskData, TaskSpec

N_INSTANTIATIONS = 30
OFFSET_LYAPUNOV = 0.5
TAU_PUBLISHED = 17
TARGET_TONE_HZ = 65.0
FED_N_RES = 32
FED_ZETA = 0.05

PUBLISHED = {
    "LSTM (NeuroBench)": {"score": 13.37, "params": 61001, "footprint_bytes": 4.90e5},
    "ESN (NeuroBench)": {"score": 14.79, "params": 35156, "footprint_bytes": 2.81e5},
}


def offsets(n=N_INSTANTIATIONS, spacing_lyap=OFFSET_LYAPUNOV):
    return [int(round(k*spacing_lyap*MG.POINTS_PER_LYAPUNOV)) for k in range(n)]


@dataclass(frozen=True)
class MGSpec(TaskSpec):
    tau: int = TAU_PUBLISHED
    offset: int = 0


def make_mg(tau: int = TAU_PUBLISHED, offset: int = 0,
            source: str = "published") -> TaskData:
    idx = next(i for i, (t, _, _) in enumerate(MG.NEUROBENCH_SERIES) if t == tau)
    x_full = np.asarray(MG.load_series([idx], source=source)[0]["x"], float)
    n_tr = int(round(MG.BENCH_TRAIN_LYAP*MG.POINTS_PER_LYAPUNOV))
    n_te = int(round(MG.BENCH_TEST_LYAP*MG.POINTS_PER_LYAPUNOV))
    x = x_full[offset:offset + n_tr + n_te]

    base = Timebase()
    fs_hz = TARGET_TONE_HZ*MG.dominant_period_samples(x)
    tb = replace(base, fs_hz=fs_hz, horizon_periods=base.f0_hz/fs_hz)
    assert tb.horizon == 1, f"horizon {tb.horizon}, need 1"

    spec = MGSpec(name=f"mackey_glass_tau{tau}", horizon_periods=tb.horizon_periods,
                  family="chaotic", target_is_latent=False, metric="smape",
                  tau=tau, offset=offset)
    split = Split(train=slice(0, n_tr), test=slice(n_tr, n_tr + n_te - 1),
                  horizon=1, purge=0)
    return TaskData(spec=spec, x=x, y=x, tb=tb, split=split, purge=0, seed=offset)


@dataclass(frozen=True)
class _DeviceCount:
    n: int

    def n_devices(self) -> int:
        return self.n


def free_run(step_fn, warmup, n_steps: int, clip=(-5.0, 5.0)) -> np.ndarray:
    hist = list(np.asarray(warmup, float))
    out = np.empty(n_steps)
    for k in range(n_steps):
        p = float(np.clip(float(step_fn(np.asarray(hist))), *clip))
        out[k] = p
        hist.append(p)
    return out


def _pad_left(v, n):
    return v if len(v) >= n else np.concatenate([np.full(n - len(v), v[0]), v])


def step_fn_for(model):
    import numpy as _np

    if hasattr(model, "sx") and hasattr(model, "n_lags"):
        L = model.n_lags

        def step(hist):
            v = _pad_left(_np.asarray(hist)[-L:][::-1], L)[None, :]
            return float(model.net.predict(model.sx.transform(v))[0]
                         * model.sg + model.mu)
        return step

    if hasattr(model, "w") and hasattr(model, "n_lags"):
        L = model.n_lags
        def step(hist):
            v = _pad_left(_np.asarray(hist)[-L:][::-1], L)
            return float(_np.concatenate([v, [1.0]]) @ model.w)
        return step

    if hasattr(model, "window") and hasattr(model, "xmu"):
        W = model.window

        def step(hist):
            import torch
            w = _pad_left(_np.asarray(hist)[-W:], W)
            dev = config.get_device()
            seq = torch.tensor(((w - model.xmu)/model.xsg)[None, :, None],
                               dtype=torch.float32, device=dev)
            model.net.eval()
            with torch.no_grad():
                out = model.net(seq).squeeze(-1).squeeze(0)[-1].item()
            return float(out*model.sg + model.mu)
        return step


def rollout_smape(model, td: TaskData) -> float:
    n_tr = td.split.train.stop
    truth = td.x[n_tr:]
    if model.family == "analog":
        from ferronds.dynamics import autoregressive as AR
        from ferronds.dynamics import features as F
        bank = F.build_bank(FED_N_RES, (td.tb.band_lo_hz, td.tb.band_hi_hz),
                            FED_ZETA)
        taus = F.default_taus()
        r = AR.run_closed_loop(td.x, split_frac=n_tr/len(td.x),
                               bank=bank, taus_ms=taus, dt=td.tb.dt)
        model.n_channels = len(bank)*len(taus)
        model.readout = _DeviceCount(2*model.n_channels)
        model._fitted = True
        return MG.smape(r["pred"][:len(truth)], truth[:len(r["pred"])])
    model.fit(td)
    pred = free_run(step_fn_for(model), td.x[:n_tr], len(truth))
    return MG.smape(pred, truth)


def one_step_smape(model, td: TaskData) -> float:
    _, yte = td.lags(1, "test")
    return MG.smape(model.predict(td, "test"), yte)


def persistence_one_step(td: TaskData) -> float:
    n_tr = td.split.train.stop
    truth = td.x[n_tr:]
    return MG.smape(td.x[n_tr - 1:n_tr - 1 + len(truth)], truth)


def mean_rollout(td: TaskData) -> float:
    n_tr = td.split.train.stop
    truth = td.x[n_tr:]
    return MG.smape(np.full(len(truth), td.x[:n_tr].mean()), truth)
