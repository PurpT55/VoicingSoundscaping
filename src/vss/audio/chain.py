"""Voice processing chains and acoustic spaces.

* chains clean and polish each synthesized take (HPF, mud cut, presence, de-ess, compression)
* spaces place dialogue inside a scene (room, hall, cave, forest, phone, inner voice ...)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import dsp
from .io import to_stereo

CHAINS: dict[str, dict[str, Any]] = {
    "raw": {},
    "dialog": {
        "hpf": 80, "eq": [("peak", 280, -1.5, 1.0), ("peak", 3500, 1.5, 0.9), ("high", 10000, 1.0)],
        "deess": -30, "comp": (-24, 2.5, 0.006, 0.09),
    },
    "narrator": {
        "hpf": 65, "eq": [("low", 160, 1.0), ("peak", 300, -1.0, 1.0), ("peak", 3200, 1.0, 0.9), ("high", 11000, 1.0)],
        "deess": -31, "comp": (-25, 2.2, 0.008, 0.12),
    },
    "child": {
        "hpf": 110, "eq": [("peak", 2500, 1.0, 1.0), ("high", 9000, 0.5)], "deess": -28, "comp": (-24, 2.5, 0.005, 0.08),
    },
    "deep": {
        "hpf": 50, "eq": [("low", 120, 1.5), ("peak", 350, -1.5, 1.2), ("peak", 3000, 1.0, 1.0)],
        "deess": -31, "comp": (-25, 2.5, 0.008, 0.12),
    },
}

SPACES: dict[str, dict[str, Any]] = {
    "dry": {},
    "booth": {"room": 0.08, "damp": 0.7, "wet": 0.04, "width": 0.4},
    "bedroom": {"room": 0.2, "damp": 0.8, "wet": 0.07, "predelay": 0.004},
    "small_room": {"room": 0.25, "damp": 0.6, "wet": 0.1, "predelay": 0.005},
    "living_room": {"room": 0.35, "damp": 0.55, "wet": 0.12, "predelay": 0.008},
    "kitchen": {"room": 0.3, "damp": 0.3, "wet": 0.13, "predelay": 0.006},
    "tavern": {"room": 0.5, "damp": 0.5, "wet": 0.14, "predelay": 0.012},
    "stairwell": {"room": 0.6, "damp": 0.2, "wet": 0.22, "predelay": 0.01},
    "hall": {"room": 0.72, "damp": 0.4, "wet": 0.2, "predelay": 0.02},
    "cathedral": {"room": 0.95, "damp": 0.25, "wet": 0.3, "predelay": 0.035},
    "cave": {"room": 0.88, "damp": 0.15, "wet": 0.28, "predelay": 0.025, "lowpass": 7000},
    "forest": {"room": 0.45, "damp": 0.85, "wet": 0.07, "predelay": 0.015},
    "outdoor_open": {"room": 0.3, "damp": 0.9, "wet": 0.03},
    "car": {"room": 0.1, "damp": 0.9, "wet": 0.06, "lowpass": 9000},
    "phone": {"band": (300, 3400), "drive": 0.2},
    "radio": {"band": (180, 5500), "drive": 0.08},
    "underwater": {"lowpass": 800, "room": 0.6, "damp": 0.6, "wet": 0.25},
    "dream": {"room": 0.92, "damp": 0.3, "wet": 0.3, "predelay": 0.04, "chorus": True},
    "inner_voice": {"room": 0.5, "damp": 0.5, "wet": 0.16, "predelay": 0.012, "highpass": 150, "chorus": True},
}


def apply_chain(x: np.ndarray, sr: int, name: str = "dialog", custom: dict | None = None) -> np.ndarray:
    spec = custom if custom is not None else CHAINS.get(name, CHAINS["dialog"])
    y = np.asarray(x, dtype=np.float32)
    if not spec or len(y) == 0:
        return y
    if spec.get("hpf"):
        y = dsp.highpass(y, sr, spec["hpf"], order=2)
    for band in spec.get("eq", []):
        if band[0] == "peak":
            y = dsp.peaking(y, sr, band[1], band[2], band[3])
        elif band[0] in ("low", "high"):
            y = dsp.shelf(y, sr, band[1], band[2], high=band[0] == "high")
    if spec.get("deess") is not None:
        y = dsp.deess(y, sr, threshold_db=spec["deess"])
    if spec.get("comp"):
        thr, ratio, att, rel = spec["comp"]
        y = dsp.compress(y, sr, threshold_db=thr, ratio=ratio, attack=att, release=rel)
    return y


def space_tail(name: str, custom: dict | None = None) -> float:
    spec = custom if custom is not None else SPACES.get(name, {})
    if not spec or "room" not in spec and "ir" not in spec:
        return 0.0
    return 0.3 + 2.6 * float(spec.get("room", 0.5)) ** 1.5


def apply_space(mono: np.ndarray, sr: int, name: str = "dry", custom: dict | None = None,
                pan: float = 0.0) -> np.ndarray:
    """Return a stereo clip placed in ``name`` space (with its reverb tail appended)."""
    spec = custom if custom is not None else SPACES.get(name, {})
    x = np.asarray(mono, dtype=np.float32)
    if spec.get("band"):
        lo, hi = spec["band"]
        x = dsp.bandpass(x, sr, lo, hi, order=4)
        if spec.get("drive"):
            d = 1.0 + 8.0 * spec["drive"]
            x = (np.tanh(x * d) / np.tanh(d)).astype(np.float32)
    if spec.get("highpass"):
        x = dsp.highpass(x, sr, spec["highpass"])
    if spec.get("lowpass"):
        x = dsp.lowpass(x, sr, spec["lowpass"], order=4)
    st = dsp.pan(x, pan) if pan else to_stereo(x)
    tail = space_tail(name, custom)
    if tail <= 0:
        return st
    import pedalboard

    st = np.concatenate([st, np.zeros((int(tail * sr), 2), dtype=np.float32)])
    boards = []
    if spec.get("ir"):
        boards.append(pedalboard.Convolution(str(spec["ir"]), mix=float(spec.get("wet", 0.2))))
    else:
        if spec.get("predelay"):
            pre = pedalboard.Pedalboard([pedalboard.Delay(delay_seconds=float(spec["predelay"]), feedback=0.0, mix=1.0)])
        else:
            pre = None
        wet_board = pedalboard.Pedalboard(([pedalboard.Chorus(rate_hz=0.4, depth=0.15, mix=0.35)] if spec.get("chorus") else [])
                                          + [pedalboard.Reverb(room_size=float(spec["room"]), damping=float(spec.get("damp", 0.5)),
                                                               wet_level=1.0, dry_level=0.0, width=float(spec.get("width", 1.0)))])
        src = st.T.copy()
        if pre is not None:
            src = pre(src, sr)
        wet = wet_board(src, sr).T
        y = st + wet * float(spec.get("wet", 0.15))
        return y.astype(np.float32)
    return pedalboard.Pedalboard(boards)(st.T.copy(), sr).T.astype(np.float32)
