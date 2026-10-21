"""Loudness and level measurement: ITU-R BS.1770 LUFS, RMS, sample/true peak, noise floor."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyloudnorm

from .dsp import EPS, _intersample_peaks, db_to_gain, frame_rms_db


def lufs(x: np.ndarray, sr: int) -> float:
    """Integrated loudness. Clips shorter than one 400 ms gating block fall back to RMS."""
    if len(x) < int(0.42 * sr):
        return rms_db(x) - 0.7  # rough K-weighting offset for speech
    meter = pyloudnorm.Meter(sr)
    with np.errstate(divide="ignore"):
        val = meter.integrated_loudness(np.asarray(x, dtype=np.float64))
    return float(val) if np.isfinite(val) else -120.0


def rms_db(x: np.ndarray) -> float:
    if len(x) == 0:
        return -120.0
    return float(20 * np.log10(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)) + EPS))


def peak_db(x: np.ndarray) -> float:
    if len(x) == 0:
        return -120.0
    return float(20 * np.log10(np.max(np.abs(x)) + EPS))


def true_peak_db(x: np.ndarray, sr: int) -> float:
    if len(x) == 0:
        return -120.0
    return float(20 * np.log10(np.max(_intersample_peaks(x, 4)) + EPS))


def noise_floor_db(x: np.ndarray, sr: int, win: float = 0.05, quantile: float = 0.05,
                   ignore_digital_silence: bool = True) -> float:
    """Level of the quietest regions (ACX measures the floor in pauses/room tone)."""
    db = frame_rms_db(x, sr, win)
    if ignore_digital_silence:
        db = db[db > -110]
    if len(db) == 0:
        return -120.0
    return float(np.quantile(db, quantile))


def normalize_lufs(x: np.ndarray, sr: int, target: float, max_gain_db: float = 30.0) -> np.ndarray:
    cur = lufs(x, sr)
    if cur <= -100:
        return x
    g = np.clip(target - cur, -max_gain_db, max_gain_db)
    return (x * db_to_gain(g)).astype(np.float32)


def normalize_rms(x: np.ndarray, target_db: float) -> np.ndarray:
    cur = rms_db(x)
    if cur <= -100:
        return x
    return (x * db_to_gain(target_db - cur)).astype(np.float32)


@dataclass
class LevelReport:
    lufs: float
    rms_db: float
    peak_db: float
    true_peak_db: float
    noise_floor_db: float
    duration: float

    def as_dict(self) -> dict:
        return {k: round(v, 2) for k, v in self.__dict__.items()}


def measure(x: np.ndarray, sr: int) -> LevelReport:
    return LevelReport(
        lufs=lufs(x, sr), rms_db=rms_db(x), peak_db=peak_db(x), true_peak_db=true_peak_db(x, sr),
        noise_floor_db=noise_floor_db(x, sr), duration=len(x) / float(sr),
    )
