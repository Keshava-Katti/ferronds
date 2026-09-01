"""Model registry; one interface, so no model can be given more help than another"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ferronds.baselines import config
from ferronds.dynamics import spectral_tasks as S
from ferronds.analog.macromodels import FeDWeightBank, LinearReadout
from ferronds.evaluation.splitting import select_lambda, _fit_predict_factory
from ferronds.baselines.tasks import TaskData, mse

WB = FeDWeightBank(signed=True, ideal=True)


# ------------------------------------------------------------------- interface
class Model:
    name = "base"
    family = "baseline"

    def __init__(self, **kw):
        self.cfg = kw
        self._fitted = False

    def fit(self, td: TaskData) -> "Model":
        raise NotImplementedError

    def predict(self, td: TaskData, which: str = "test") -> np.ndarray:
        raise NotImplementedError

    @property
    def n_trained_params(self) -> int:
        raise NotImplementedError

    @property
    def n_devices(self) -> int:
        return 0

    def score(self, td: TaskData, which: str = "test") -> float:
        _, y = td.lags(1, which)
        return mse(self.predict(td, which), y)

    def describe(self) -> dict:
        return {"name": self.name, "family": self.family,
                "trained_params": self.n_trained_params,
                "devices": self.n_devices, "config": dict(self.cfg)}


# ---------------------------------------------------------------- naive floors
class MeanPredictor(Model):
    name, family = "predict the mean", "naive"

    def fit(self, td):
        _, y = td.lags(1, "train")
        self.mu = float(np.mean(y)); self._fitted = True; return self

    def predict(self, td, which="test"):
        X, _ = td.lags(1, which)
        return np.full(len(X), self.mu)

    @property
    def n_trained_params(self): return 0


class BestFixedDelay(Model):
    name, family = "best fixed delay", "naive"

    def __init__(self, max_delay: int = 64, **kw):
        super().__init__(max_delay=max_delay, **kw)
        self.max_delay = max_delay

    def fit(self, td):
        X, y = td.lags(self.max_delay, "train")
        errs = [mse(X[:, d], y) for d in range(self.max_delay)]
        self.delay = int(np.argmin(errs)); self._fitted = True; return self

    def predict(self, td, which="test"):
        X, _ = td.lags(self.max_delay, which)
        return X[:, self.delay]

    @property
    def n_trained_params(self): return 1


# -------------------------------------------------------------------- FerroNDS
@dataclass
class FerroNDSConfig:
    mode: str
    n_res: int
    zeta: float = 0.05


FROZEN = {
    "freq_track": FerroNDSConfig(mode="absZ2", n_res=32),
    "band_power": FerroNDSConfig(mode="ReZ2", n_res=16),
}
FROZEN_DEFAULT = FerroNDSConfig(mode="absZ2", n_res=16)

FROZEN_WAVEFORM = FerroNDSConfig(mode="BP", n_res=32)


class FerroNDS(Model):
    name, family = "FerroNDS", "analog"

    def __init__(self, cfg: FerroNDSConfig | None = None, **kw):
        super().__init__(**kw)
        self.cfg_obj = cfg

    def _resolve(self, td) -> FerroNDSConfig:
        if self.cfg_obj is not None:
            return self.cfg_obj
        if td.spec.name in FROZEN:
            return FROZEN[td.spec.name]
        return (FROZEN_WAVEFORM if td.spec.family == "waveform"
                else FROZEN_DEFAULT)

    def _features(self, td):
        c = self._resolve(td)
        return S.front_end(td.x, td.tb, c.mode, c.n_res, c.zeta)

    def fit(self, td):
        c = self._resolve(td)
        F = self._features(td)
        Xtr, ytr = td.features(F, "train")
        self.mu, self.sg = float(ytr.mean()), float(ytr.std())
        z = (ytr - self.mu)/self.sg
        lam, _ = select_lambda(Xtr, z, td.purge, _fit_predict_factory(WB, True))
        self.readout = LinearReadout(bank=WB).fit_ridge(Xtr, z, lam=lam)
        self.lam, self.n_channels = lam, c.n_res
        self._fitted = True
        return self

    def predict(self, td, which="test"):
        X, _ = td.features(self._features(td), which)
        return self.readout.predict(X, quantize=True)*self.sg + self.mu

    @property
    def n_trained_params(self) -> int:
        return int(self.n_channels)

    @property
    def n_devices(self) -> int:
        return int(self.readout.n_devices())


# ------------------------------------------------------ linear and feedforward
class RidgeLags(Model):
    name, family = "ridge on lags", "linear"
    ALPHAS = (1e-6, 1e-4, 1e-2, 1e0, 1e2, 1e4)

    def __init__(self, n_lags: int = 64, alpha: float | None = None, **kw):
        super().__init__(n_lags=n_lags, alpha=alpha, **kw)
        self.n_lags, self.alpha = n_lags, alpha

    @staticmethod
    def _solve(A, y, a):
        return np.linalg.solve(A.T@A + a*np.eye(A.shape[1]), A.T@y)

    def fit(self, td):
        from ferronds.evaluation.splitting import purged_blocked_folds
        X, y = td.lags(self.n_lags, "train")
        A = np.hstack([X, np.ones((len(X), 1))])
        if self.alpha is None:
            best = (np.inf, self.ALPHAS[0])
            for a in self.ALPHAS:
                errs = [mse(A[va]@self._solve(A[tr], y[tr], a), y[va])
                        for tr, va in purged_blocked_folds(len(A), td.purge)]
                if errs and np.mean(errs) < best[0]:
                    best = (float(np.mean(errs)), a)
            self.alpha_used = best[1]
        else:
            self.alpha_used = self.alpha
        self.w = self._solve(A, y, self.alpha_used)
        self._fitted = True; return self

    def predict(self, td, which="test"):
        X, _ = td.lags(self.n_lags, which)
        return np.hstack([X, np.ones((len(X), 1))])@self.w

    @property
    def n_trained_params(self): return self.n_lags + 1


class SklearnMLP(Model):
    name, family = "MLP", "digital"

    def __init__(self, hidden=(64, 64), n_lags: int = 64, alpha: float = 1e-4,
                 max_iter: int = 800, n_iter_no_change: int = 60, **kw):
        super().__init__(hidden=hidden, n_lags=n_lags, alpha=alpha, **kw)
        self.hidden, self.n_lags, self.alpha, self.max_iter = \
            tuple(hidden), n_lags, alpha, max_iter
        self.n_iter_no_change = n_iter_no_change

    def fit(self, td):
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler
        X, y = td.lags(self.n_lags, "train")
        self.sx = StandardScaler().fit(X)
        self.mu, self.sg = float(y.mean()), float(y.std())
        self.net = MLPRegressor(hidden_layer_sizes=self.hidden, alpha=self.alpha,
                                max_iter=self.max_iter, early_stopping=True,
                                n_iter_no_change=self.n_iter_no_change,
                                validation_fraction=0.2, learning_rate_init=3e-3,
                                random_state=td.seed)
        self.net.fit(self.sx.transform(X), (y - self.mu)/self.sg)
        self._fitted = True; return self

    def predict(self, td, which="test"):
        X, _ = td.lags(self.n_lags, which)
        return self.net.predict(self.sx.transform(X))*self.sg + self.mu

    @property
    def n_trained_params(self) -> int:
        sizes = (self.n_lags,) + self.hidden + (1,)
        return sum(a*b + b for a, b in zip(sizes[:-1], sizes[1:]))


# --------------------------------------------------- recurrent and state-space
class TorchSequenceModel(Model):
    family = "digital"

    def __init__(self, d_model: int = 32, n_layers: int = 1, window: int = 512,
                 stride: int = 64, warmup: int = 64, epochs: int = 200,
                 lr: float = 3e-3, batch: int = 32, patience: int = 20,
                 clip: float | None = None, **kw):
        super().__init__(d_model=d_model, n_layers=n_layers, window=window,
                         stride=stride, epochs=epochs, lr=lr, clip=clip, **kw)
        self.d_model, self.n_layers = d_model, n_layers
        self.window, self.stride, self.warmup = window, stride, warmup
        self.epochs, self.lr, self.batch, self.patience = epochs, lr, batch, patience
        self.clip = clip
        self.net = None

    def _build(self, device):
        raise NotImplementedError

    def fit(self, td):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        dev = config.get_device()
        config.seed_everything(td.seed)
        xs, ys, mk = td.sequence(self.window, self.stride, "train", self.warmup)
        self.mu, self.sg = float(ys[mk].mean()), float(ys[mk].std())
        self.xmu, self.xsg = float(xs.mean()), float(xs.std() + 1e-12)

        X = torch.tensor((xs - self.xmu)/self.xsg, dtype=torch.float32)
        Y = torch.tensor((ys - self.mu)/self.sg, dtype=torch.float32)
        M = torch.tensor(mk)
        self.net = self._build(dev).to(dev)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)

        n_val = max(1, int(0.2*len(X)))
        dl = DataLoader(TensorDataset(X[:-n_val], Y[:-n_val], M[:-n_val]),
                        batch_size=self.batch, shuffle=True)
        Xv, Yv, Mv = X[-n_val:].to(dev), Y[-n_val:].to(dev), M[-n_val:].to(dev)

        best, bad, best_state = float("inf"), 0, None
        for _ in range(self.epochs):
            self.net.train()
            for xb, yb, mb in dl:
                xb, yb, mb = xb.to(dev), yb.to(dev), mb.to(dev)
                loss = (((self.net(xb).squeeze(-1) - yb)**2)*mb).sum()/mb.sum()
                opt.zero_grad(); loss.backward()
                if self.clip:
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.clip)
                opt.step()
            self.net.eval()
            with torch.no_grad():
                v = float((((self.net(Xv).squeeze(-1) - Yv)**2)*Mv).sum()/Mv.sum())
            if v < best - 1e-9:
                best, bad = v, 0
                best_state = {k: t.detach().clone() for k, t in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self._fitted = True
        return self

    def predict(self, td, which="test"):
        import torch
        dev = config.get_device()
        X, _ = td.lags(1, which)
        seq = torch.tensor(((X[:, 0] - self.xmu)/self.xsg)[None, :, None],
                           dtype=torch.float32, device=dev)
        self.net.eval()
        with torch.no_grad():
            out = self.net(seq).squeeze(-1).squeeze(0).cpu().numpy()
        return out*self.sg + self.mu

    @property
    def n_trained_params(self) -> int:
        return int(sum(p.numel() for p in self.net.parameters() if p.requires_grad))


class GRU(TorchSequenceModel):
    name = "GRU"

    def _build(self, device):
        import torch.nn as nn

        class _Net(nn.Module):
            def __init__(self, d, n):
                super().__init__()
                self.rnn = nn.GRU(1, d, n, batch_first=True)
                self.head = nn.Linear(d, 1)

            def forward(self, x):
                h, _ = self.rnn(x)
                return self.head(h)

        return _Net(self.d_model, self.n_layers)


# ---------------------------------------------- Mamba S6 block
def _reference_mamba_block(d_model, d_state=16, d_conv=4, expand=2):
    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class S6(nn.Module):
        def __init__(self):
            super().__init__()
            self.d_inner = expand*d_model
            self.dt_rank = max(1, math.ceil(d_model/16))
            self.d_state = d_state
            self.in_proj = nn.Linear(d_model, 2*self.d_inner, bias=False)
            self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv,
                                    groups=self.d_inner, padding=d_conv - 1)
            self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2*d_state, bias=False)
            self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
            A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
            self.A_log = nn.Parameter(torch.log(A))
            self.D = nn.Parameter(torch.ones(self.d_inner))
            self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        def forward(self, u):
            B, L, _ = u.shape
            xz = self.in_proj(u)
            x, z = xz.chunk(2, dim=-1)
            x = self.conv1d(x.transpose(1, 2))[:, :, :L].transpose(1, 2)
            x = F.silu(x)

            dbc = self.x_proj(x)
            dt, Bm, Cm = torch.split(dbc, [self.dt_rank, self.d_state, self.d_state], -1)
            dt = F.softplus(self.dt_proj(dt))
            A = -torch.exp(self.A_log)

            dA = torch.exp(dt.unsqueeze(-1)*A)
            dBx = dt.unsqueeze(-1)*Bm.unsqueeze(2)*x.unsqueeze(-1)

            h = torch.zeros(B, self.d_inner, self.d_state, device=u.device, dtype=u.dtype)
            ys = []
            for t in range(L):
                h = dA[:, t]*h + dBx[:, t]
                ys.append((h*Cm[:, t].unsqueeze(1)).sum(-1))
            y = torch.stack(ys, dim=1) + x*self.D
            return self.out_proj(y*F.silu(z))

    return S6()


def _shim_transformers_generation() -> str:
    try:
        import transformers.generation as g
    except Exception as exc:
        return f"not applied ({type(exc).__name__})"
    target = getattr(g, "GenerateDecoderOnlyOutput", None)
    if target is None:
        target = type("_MambaShimDecoderOnlyOutput", (), {})
    added = [nm for nm in ("GreedySearchDecoderOnlyOutput",
                           "SampleDecoderOnlyOutput") if not hasattr(g, nm)]
    for nm in added:
        setattr(g, nm, target)
    return (f"shimmed {', '.join(added)}" if added else "shim not needed")


def _probe_mamba_block(block, d_model, d_state, device, attempts) -> tuple:
    import inspect
    import torch

    base = dict(d_model=d_model, d_state=d_state)
    trials = [(base, "fused fast path")]
    if "use_fast_path" in inspect.signature(block.__init__).parameters:
        trials.append(({**base, "use_fast_path": False},
                       "use_fast_path=False, fused selective scan"))
    for kw, mode in trials:
        try:
            b = block(**kw).to(device)
            with torch.no_grad():
                b(torch.randn(2, 16, d_model, device=device))
            return kw, mode
        except Exception as exc:
            attempts.append(f"forward with {mode}: {type(exc).__name__}: "
                            f"{str(exc).splitlines()[0][:140]}")
    return None, None


class Mamba(TorchSequenceModel):
    name = "Mamba"

    def __init__(self, impl: str = "auto", d_state: int = 16, **kw):
        super().__init__(impl=impl, d_state=d_state, **kw)
        self.impl, self.d_state = impl, d_state
        self.impl_used = None

    def _block_factory(self, device):
        want_cuda = self.impl in ("auto", "cuda") and device.type == "cuda"
        if want_cuda:
            shim = _shim_transformers_generation()
            attempts, block = [], None
            for path, attr in (("mamba_ssm", "Mamba"),
                               ("mamba_ssm.modules.mamba_simple", "Mamba"),
                               ("mamba_ssm.modules.mamba2", "Mamba2")):
                try:
                    mod = __import__(path, fromlist=[attr])
                    block = getattr(mod, attr)
                    self.impl_used = f"mamba-ssm ({path}.{attr}; {shim}"
                    break
                except Exception as exc:
                    attempts.append(f"{path}.{attr}: {type(exc).__name__}: {exc}")
            if block is not None:
                kw, mode = _probe_mamba_block(block, self.d_model, self.d_state,
                                              device, attempts)
                if kw is not None:
                    self.impl_used = f"{self.impl_used}; {mode})"
                    return lambda: block(**kw)
            detail = "\n      ".join([f"transformers shim: {shim}"] + attempts)
        self.impl_used = "reference S6 (pure PyTorch)"
        return lambda: _reference_mamba_block(self.d_model, self.d_state)

    def _build(self, device):
        import torch.nn as nn
        make_block = self._block_factory(device)

        class _Net(nn.Module):
            def __init__(self, d, n):
                super().__init__()
                self.inp = nn.Linear(1, d)
                self.blocks = nn.ModuleList([make_block() for _ in range(n)])
                self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(n)])
                self.norm = nn.LayerNorm(d)
                self.head = nn.Linear(d, 1)

            def forward(self, x):
                h = self.inp(x)
                for blk, nrm in zip(self.blocks, self.norms):
                    h = h + blk(nrm(h))
                return self.head(self.norm(h))

        return _Net(self.d_model, self.n_layers)

    def describe(self):
        d = super().describe()
        d["impl_used"] = self.impl_used
        return d


# --------------------------------------------------- model families
def default_registry(include_mamba: bool = True, mamba_impl: str = "auto") -> list:
    models = [
        MeanPredictor(), BestFixedDelay(max_delay=64),
        RidgeLags(n_lags=64), FerroNDS(),
        SklearnMLP(hidden=(4,)), SklearnMLP(hidden=(8,)),
        SklearnMLP(hidden=(16,)), SklearnMLP(hidden=(32,)),
        SklearnMLP(hidden=(64, 64)),
        *[GRU(d_model=d, stride=16, lr=3e-3, epochs=800, patience=60)
          for d in (1, 2, 4, 8, 16, 32)],
    ]
    if include_mamba:
        models += [Mamba(d_model=d, stride=16, lr=3e-3, epochs=800, patience=60,
                         impl=mamba_impl)
                   for d in (2, 4, 8, 16, 32)]
    return models
