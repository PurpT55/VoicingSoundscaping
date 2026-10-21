"""Audio file I/O and format helpers. Internal format: float32, shape (n,) mono or (n, 2) stereo."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def load(path: str | Path, sr: int | None = None, mono: bool = False) -> tuple[np.ndarray, int]:
    path = Path(path)
    try:
        x, file_sr = sf.read(str(path), dtype="float32", always_2d=False)
    except (sf.LibsndfileError, RuntimeError):
        x, file_sr = _ffmpeg_decode(path)
    if mono:
        x = to_mono(x)
    if sr and file_sr != sr:
        x = resample(x, file_sr, sr)
        file_sr = sr
    return x.astype(np.float32, copy=False), file_sr


def _ffmpeg_decode(path: Path) -> tuple[np.ndarray, int]:
    if not shutil.which("ffmpeg"):
        raise RuntimeError(f"cannot decode {path} (install ffmpeg)")
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "decoded.wav"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(path), "-c:a", "pcm_f32le", str(out)], check=True)
        x, sr = sf.read(str(out), dtype="float32", always_2d=False)
    return x, sr


def save(path: str | Path, x: np.ndarray, sr: int, subtype: str | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = path.suffix.lower().lstrip(".")
    if fmt in ("wav", "flac"):
        sf.write(str(path), np.asarray(x, dtype=np.float32), sr, subtype=subtype or "PCM_24")
    else:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "tmp.wav"
            sf.write(str(tmp), np.asarray(x, dtype=np.float32), sr, subtype="PCM_24")
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(tmp), str(path)], check=True)
    return path


def resample(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to or len(x) == 0:
        return x
    g = gcd(int(sr_from), int(sr_to))
    return resample_poly(x, sr_to // g, sr_from // g, axis=0).astype(np.float32)


def to_mono(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=1).astype(np.float32) if x.ndim == 2 else x


def to_stereo(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2:
        return x if x.shape[1] == 2 else np.repeat(x[:, :1], 2, axis=1)
    return np.stack([x, x], axis=1)


def silence(seconds: float, sr: int, stereo: bool = False) -> np.ndarray:
    n = max(0, int(round(seconds * sr)))
    return np.zeros((n, 2) if stereo else n, dtype=np.float32)


def duration(path: str | Path) -> float:
    info = sf.info(str(path))
    return info.frames / float(info.samplerate)
