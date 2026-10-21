"""Mastering presets: final loudness, peak control and delivery-standard compliance."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import dsp
from .loudness import lufs, measure, normalize_lufs, normalize_rms, rms_db


@dataclass(frozen=True)
class MasterPreset:
    name: str
    description: str
    measure: str          # "rms" (ACX) or "lufs"
    target: float         # dBFS RMS or LUFS
    ceiling_db: float     # limiter ceiling (true-peak-ish)
    glue: tuple[float, float] | None = (-22.0, 1.6)  # bus compressor (threshold, ratio)
    tilt_db: float = 0.0  # >0 brighter, <0 darker (high shelf @ 6 kHz)
    sample_rate: int = 44100


PRESETS: dict[str, MasterPreset] = {
    "acx": MasterPreset("acx", "Audible/ACX: RMS -20 dBFS, peaks <= -3.5 dBFS, 44.1 kHz", "rms", -20.0, -3.5),
    "audiobook": MasterPreset("audiobook", "Generic audiobook stores (Apple/Google/Spotify): -18 LUFS, -2 dBTP",
                              "lufs", -18.0, -2.0),
    "podcast": MasterPreset("podcast", "Podcast/streaming speech: -16 LUFS, -1 dBTP", "lufs", -16.0, -1.0),
    "cinematic": MasterPreset("cinematic", "Immersive drama, wide dynamics (EBU R128): -23 LUFS, -1 dBTP",
                              "lufs", -23.0, -1.0, glue=(-26.0, 1.3), sample_rate=48000),
    "bedtime": MasterPreset("bedtime", "Soft bedtime listening: -22 LUFS, darker top, gentle dynamics",
                            "lufs", -22.0, -3.0, glue=(-28.0, 2.0), tilt_db=-2.0),
}


def master(x: np.ndarray, sr: int, preset: str | MasterPreset = "acx") -> tuple[np.ndarray, dict]:
    p = PRESETS[preset] if isinstance(preset, str) else preset
    y = np.asarray(x, dtype=np.float32)
    y = dsp.highpass(y, sr, 28.0, order=2)
    if p.tilt_db:
        y = dsp.shelf(y, sr, 6000.0, p.tilt_db, high=True)
    if p.glue:
        y = dsp.compress(y, sr, threshold_db=p.glue[0], ratio=p.glue[1], attack=0.03, release=0.4, knee_db=8.0)
    # loudness then peak control; iterate because limiting lowers RMS slightly
    for _ in range(3):
        y = normalize_rms(y, p.target) if p.measure == "rms" else normalize_lufs(y, sr, p.target)
        y = dsp.limiter(y, sr, ceiling_db=p.ceiling_db)
        cur = rms_db(y) if p.measure == "rms" else lufs(y, sr)
        if abs(cur - p.target) < 0.3:
            break
    report = measure(y, sr).as_dict()
    report["preset"] = p.name
    return y, report
