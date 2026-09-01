"""Google Speech Commands v2 under fixed-readout front-end comparison"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field

import numpy as np

from ferronds.baselines import config
from ferronds.data import keyword_spotting as KD
from ferronds.data.keyword_spotting_frontend import SPEECH_BAND_HZ, KWSFrontEnd, MelReference

N_CHANNELS = 32
ZETA = 0.05
TAU_MS = 8.0
HOP_MS = 10.0
FS_HZ = 16000.0
UNKNOWN_FRAC = 0.1
FRONT_ENDS = ("ferronds", "mel")
N_CLASSES = len(KD.LABELS)
CACHE = config.DATASETS/"kws"
N_PROC = int(os.environ.get("KWS_NPROC", "1"))


def front_end(name: str):
    if name == "ferronds":
        return KWSFrontEnd(n_channels=N_CHANNELS, band_hz=SPEECH_BAND_HZ,
                           zeta=ZETA, tau_ms=TAU_MS, fs_hz=FS_HZ, hop_ms=HOP_MS)
    if name == "mel":
        return MelReference(n_channels=N_CHANNELS, band_hz=SPEECH_BAND_HZ,
                            fs_hz=FS_HZ, hop=int(round(HOP_MS*1e-3*FS_HZ)))


def pool(F: np.ndarray) -> np.ndarray:
    F = np.asarray(F)
    return np.concatenate([F.mean(0), F.std(0)])


def majority_class_accuracy(y) -> float:
    y = np.asarray(y).ravel()
    return float(np.bincount(y).max()/len(y)) if len(y) else float("nan")


@dataclass
class KWSData:
    Xtr: np.ndarray
    ytr: np.ndarray
    Xte: np.ndarray
    yte: np.ndarray
    front_end_name: str
    Ftr: np.ndarray | None = None
    Fte: np.ndarray | None = None
    labels: tuple = tuple(KD.LABELS)
    meta: dict = field(default_factory=dict)

    @property
    def n_features(self) -> int:
        return int(self.Xtr.shape[1])

    @property
    def n_classes(self) -> int:
        return len(self.labels)

    def with_frames(self) -> tuple:
        return self.Ftr, self.Fte


_WORK: dict = {"audio": None, "fe": None, "frames": False}


def _feat_one(i):
    F = _WORK["fe"](_WORK["audio"][i])
    return pool(F), (np.asarray(F, np.float32) if _WORK["frames"] else None)


def extract(audio, name: str, want_frames: bool = False, n_proc: int = N_PROC,
            label: str = "") -> tuple:
    _WORK.update(audio=audio, fe=front_end(name), frames=bool(want_frames))
    t0 = time.perf_counter()
    if n_proc > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(n_proc) as p:
            out = list(p.imap(_feat_one, range(len(audio)), chunksize=32))
    else:
        out = [_feat_one(i) for i in range(len(audio))]
    X = np.stack([o[0] for o in out])
    F = np.stack([o[1] for o in out]) if want_frames else None
    if label:
        print(f"{label}: {len(audio)} clips, {time.perf_counter() - t0:.0f} s",
              flush=True)
    _WORK.update(audio=None, fe=None)
    return X, F


def make_from_audio(name: str, Xtr_audio, ytr, Xte_audio, yte, seed: int = 0,
                    with_frames: bool = False, n_proc: int = N_PROC,
                    labels=tuple(KD.LABELS)) -> KWSData:
    Xtr, Ftr = extract(Xtr_audio, name, with_frames, n_proc, f"{name} train")
    Xte, Fte = extract(Xte_audio, name, with_frames, n_proc, f"{name} test")
    return KWSData(Xtr=Xtr, ytr=np.asarray(ytr), Xte=Xte, yte=np.asarray(yte),
                   front_end_name=name, Ftr=Ftr, Fte=Fte, labels=tuple(labels),
                   meta={"n_channels": N_CHANNELS, "zeta": ZETA,
                         "tau_ms": TAU_MS, "hop_ms": HOP_MS, "seed": seed,
                         "band_hz": list(SPEECH_BAND_HZ), "source": "in-memory"})


def _tag(name, split, seed, n_per_class) -> str:
    key = json.dumps({"fe": name, "split": split, "seed": seed,
                      "n_per_class": n_per_class, "n_channels": N_CHANNELS,
                      "zeta": ZETA, "tau_ms": TAU_MS, "hop_ms": HOP_MS,
                      "band_hz": list(SPEECH_BAND_HZ),
                      "unknown_frac": UNKNOWN_FRAC}, sort_keys=True)
    return f"{name}_{split}_s{seed}_{hashlib.md5(key.encode()).hexdigest()[:10]}"


def _load_split(name, split, seed, n_per_class, with_frames, n_proc, use_cache):
    tag = _tag(name, split, seed, n_per_class)
    fx, fy, ff = (CACHE/f"{tag}_X.npy"), (CACHE/f"{tag}_y.npy"), (CACHE/f"{tag}_F.npy")
    if use_cache and fx.is_file() and fy.is_file() and (ff.is_file() or not with_frames):
        X, y = np.load(fx), np.load(fy)
        F = np.load(ff, mmap_mode="r") if with_frames else None
        print(f"{name} {split}: {len(y)} clips, cached", flush=True)
        return X, y, F
    audio, y = KD.load(split, n_per_class=n_per_class, seed=seed,
                       unknown_frac=UNKNOWN_FRAC)
    X, F = extract(audio, name, with_frames, n_proc, f"{name} {split}")
    if use_cache:
        CACHE.mkdir(parents=True, exist_ok=True)
        np.save(fx, X); np.save(fy, y)
        if with_frames:
            np.save(ff, F)
    return X, y, F


def make_kws(name: str, seed: int = 0, n_per_class: int | None = None,
             with_frames: bool = False, n_proc: int = N_PROC,
             use_cache: bool = True) -> KWSData:
    Xtr, ytr, Ftr = _load_split(name, "train", seed, n_per_class, with_frames,
                                n_proc, use_cache)
    Xte, yte, Fte = _load_split(name, "test", seed, n_per_class, with_frames,
                                n_proc, use_cache)
    return KWSData(Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, front_end_name=name,
                   Ftr=Ftr, Fte=Fte,
                   meta={"n_channels": N_CHANNELS, "zeta": ZETA, "tau_ms": TAU_MS,
                         "hop_ms": HOP_MS, "band_hz": list(SPEECH_BAND_HZ),
                         "unknown_frac": UNKNOWN_FRAC, "seed": seed,
                         "n_per_class": n_per_class, "source": KD.ROOT})


def make_both(seed: int = 0, n_per_class: int | None = None,
              with_frames: bool = False, n_proc: int = N_PROC,
              use_cache: bool = True) -> dict:
    out = {n: make_kws(n, seed=seed, n_per_class=n_per_class,
                       with_frames=with_frames, n_proc=n_proc,
                       use_cache=use_cache) for n in FRONT_ENDS}
    a, b = (out[n] for n in FRONT_ENDS)
    assert np.array_equal(a.ytr, b.ytr) and np.array_equal(a.yte, b.yte), \
        "front ends: clip sets differ"
    return out
