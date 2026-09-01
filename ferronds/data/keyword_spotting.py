"""Google Speech Commands v2, the standard 12-class split"""

from __future__ import annotations
import os, glob, wave, numpy as np

ROOT = os.environ.get("GSC_ROOT", os.path.expanduser("~/gsc"))
CORE = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
LABELS = CORE + ["_unknown_", "_silence_"]
FS = 16000


def _read_full(p):
    with wave.open(p) as w:
        return np.frombuffer(w.readframes(w.getnframes()),
                             dtype=np.int16).astype(np.float32)/32768.0


def _read(p):
    a = _read_full(p)
    if len(a) < FS:
        a = np.pad(a, (0, FS - len(a)))
    return a[:FS]


def _split_sets():
    val = set(l.strip() for l in open(f"{ROOT}/validation_list.txt"))
    tst = set(l.strip() for l in open(f"{ROOT}/testing_list.txt"))
    return val, tst


def load(split="train", n_per_class=None, unknown_frac=0.1, seed=0):
    val, tst = _split_sets()
    rng = np.random.default_rng(seed)
    words = sorted(d for d in os.listdir(ROOT)
                   if os.path.isdir(f"{ROOT}/{d}") and not d.startswith("_"))
    X, y = [], []
    for w in words:
        rel = [f"{w}/{os.path.basename(p)}" for p in sorted(glob.glob(f"{ROOT}/{w}/*.wav"))]
        keep = [r for r in rel if (r in val if split == "val" else
                                   r in tst if split == "test" else
                                   (r not in val and r not in tst))]
        is_core = w in CORE
        lab = LABELS.index(w) if is_core else LABELS.index("_unknown_")
        if not is_core:
            keep = list(rng.permutation(keep)[:max(1, int(unknown_frac*len(keep)))])
        if n_per_class:
            keep = list(rng.permutation(keep)[:n_per_class if is_core
                                              else max(1, n_per_class//len(words))])
        for r in keep:
            X.append(_read(f"{ROOT}/{r}")); y.append(lab)
    bg = [_read_full(p) for p in glob.glob(f"{ROOT}/_background_noise_/*.wav")] \
        if os.path.isdir(f"{ROOT}/_background_noise_") else []
    bg = [b if len(b) >= FS else np.pad(b, (0, FS - len(b))) for b in bg]
    n_sil = n_per_class or int(np.mean(np.bincount(y)))
    for _ in range(n_sil):
        if not bg: break
        b = bg[rng.integers(len(bg))]
        i = rng.integers(0, len(b) - FS + 1)
        X.append(b[i:i+FS]*rng.uniform(0.0, 1.0)); y.append(LABELS.index("_silence_"))
    return np.stack(X), np.array(y)
