"""The Mackey-Glass series, published and regenerated"""

from __future__ import annotations
import numpy as np
from pathlib import Path

BETA, GAMMA, N_EXP = 0.2, 0.1, 10
POINTS_PER_LYAPUNOV = 75
LYAPUNOV_TIMES = 50
BENCH_TRAIN_LYAP = 10.0
BENCH_TEST_LYAP = 10.0

NEUROBENCH_SERIES = [
    (17, 197, 0.7206597), (18, 138, 0.7744313), (19, 315, 0.7783468),
    (20, 131, 0.9225991), (21, 191, 0.9479431), (22, 119, 0.5455960),
    (23, 106, 0.8622247), (24,  97, 0.3259660), (25,  98, 0.8297825),
    (26, 104, 1.0033490), (27, 112, 0.6491406), (28, 119, 1.0957495),
    (29, 131, 0.9256179), (30, 139, 0.2713639),
]

HF_REPO = "https://huggingface.co/datasets/NeuroBench/mackey_glass/resolve/main"

def mackey_glass(tau, lyapunov_time, x0, n_points=None, oversample=20):
    n_points = n_points or LYAPUNOV_TIMES*POINTS_PER_LYAPUNOV
    dt_out = lyapunov_time/POINTS_PER_LYAPUNOV
    dt = dt_out/oversample
    n_fine = n_points*oversample
    d = int(round(tau/dt))
    x = np.empty(n_fine + 1)
    x[:d+1] = x0
    f = lambda xd, xc: BETA*xd/(1.0 + xd**N_EXP) - GAMMA*xc
    for k in range(d, n_fine):
        xd = x[k-d]
        xd_h = x[k-d] if k-d+oversample//2 >= len(x) else 0.5*(x[k-d] + x[min(k-d+1, k)])
        k1 = f(xd, x[k])
        k2 = f(xd_h, x[k] + 0.5*dt*k1)
        k3 = f(xd_h, x[k] + 0.5*dt*k2)
        k4 = f(x[min(k-d+1, k)], x[k] + dt*k3)
        x[k+1] = x[k] + dt*(k1 + 2*k2 + 2*k3 + k4)/6.0
    return x[:n_fine:oversample].copy(), dt_out

def fetch_published(dest=None, force=False):
    import tarfile, io, urllib.request
    from ferronds.paths import mackey_glass_dir
    dest = Path(dest) if dest else mackey_glass_dir()
    dest.mkdir(parents=True, exist_ok=True)
    if not force and (dest / "mg_17.npy").exists():
        return dest
    with urllib.request.urlopen(f"{HF_REPO}/data.tar.gz", timeout=120) as r:
        blob = r.read()
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as t:
        for m in t.getmembers():
            if m.isfile() and m.name.endswith(".npy"):
                m.name = Path(m.name).name
                t.extract(m, dest)
    for extra in ("mackey_glass_parameters.csv", "generator.py"):
        try:
            with urllib.request.urlopen(f"{HF_REPO}/{extra}", timeout=60) as r:
                (dest / extra).write_bytes(r.read())
        except Exception:
            pass
    return dest


def load_series(indices=None, source="published"):
    from ferronds.paths import mackey_glass_dir
    idx = range(len(NEUROBENCH_SERIES)) if indices is None else indices
    out = []
    for i in idx:
        tau, lyap, x0 = NEUROBENCH_SERIES[i]
        dt_out = lyap/POINTS_PER_LYAPUNOV
        if source == "published":
            p = mackey_glass_dir() / f"mg_{tau}.npy"
            raw = np.asarray(np.load(p), float).squeeze()
        elif source == "regenerated":
            raw, dt_out = mackey_glass(tau, lyap, x0)
        out.append(dict(index=i, tau=tau, lyapunov_time=lyap, x0=x0, source=source,
                        x=raw, dt_mg=dt_out, n=len(raw)))
    return out

def neurobench_window(x, train_lyap=BENCH_TRAIN_LYAP, test_lyap=BENCH_TEST_LYAP):
    n_tr = int(round(train_lyap*POINTS_PER_LYAPUNOV))
    n_te = int(round(test_lyap*POINTS_PER_LYAPUNOV))
    if len(x) < n_tr + n_te:
        raise ValueError(f"series {len(x)} points, need {n_tr + n_te}")
    return np.asarray(x, float)[:n_tr + n_te], n_tr


def dominant_period_samples(x):
    z = x - x.mean()
    sp = np.abs(np.fft.rfft(z*np.hanning(len(z))))
    fr = np.fft.rfftfreq(len(z), 1.0)
    k = int(np.argmax(sp[1:])) + 1
    return 1.0/fr[k]

def smape(pred, true):
    p, t = np.asarray(pred, float), np.asarray(true, float)
    return float(200.0*np.mean(np.abs(p - t)/(np.abs(p) + np.abs(t) + 1e-12)))


if __name__ == "__main__":
    import sys
    if "--fetch" in sys.argv:
        d = fetch_published(force="--force" in sys.argv)
        print(f"NeuroBench arrays: {d}")
        for r in load_series():
            print(f"tau {r['tau']}: lyap {r['lyapunov_time']}, n {r['n']}, "
                  f"range {r['x'].min():.3f} to {r['x'].max():.3f}")
