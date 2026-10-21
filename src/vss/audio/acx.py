"""ACX / Audible submission checks (the de-facto audiobook delivery standard).

* each file: one chapter, <= 120 min
* RMS between -23 and -18 dBFS
* peaks <= -3 dBFS
* noise floor <= -60 dBFS RMS
* 0.5-1 s of room tone at the head, 1-5 s at the tail
* 44.1 kHz, MP3 192 kbps+ CBR (handled by the exporter)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .dsp import frame_rms_db
from .loudness import measure


@dataclass
class Check:
    name: str
    ok: bool
    value: str
    expected: str

    def __post_init__(self):
        self.ok = bool(self.ok)


@dataclass
class AcxReport:
    checks: list[Check] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)

    def format(self) -> str:
        rows = [f"{'PASS' if c.ok else 'FAIL'}  {c.name:<16} {c.value:<14} (expected {c.expected})" for c in self.checks]
        return "\n".join(rows + [f"ACX: {'compliant' if self.ok else 'NOT compliant'}"])

    def as_dict(self) -> dict:
        return {"ok": self.ok, "checks": [c.__dict__ for c in self.checks]}


def _edge_silence(x: np.ndarray, sr: int, head: bool, threshold_db: float = -50.0) -> float:
    db = frame_rms_db(x, sr, 0.01)
    if len(db) == 0:
        return 0.0
    idx = np.where(db > threshold_db)[0]
    if len(idx) == 0:
        return len(x) / sr
    return (idx[0] if head else len(db) - 1 - idx[-1]) * 0.01


def check(x: np.ndarray, sr: int) -> AcxReport:
    m = measure(x, sr)
    head = _edge_silence(x, sr, True)
    tail = _edge_silence(x, sr, False)
    r = AcxReport()
    r.checks.append(Check("rms", -23.0 <= m.rms_db <= -18.0, f"{m.rms_db:.1f} dBFS", "-23..-18 dBFS"))
    r.checks.append(Check("peak", m.true_peak_db <= -3.0, f"{m.true_peak_db:.1f} dBFS", "<= -3 dBFS"))
    r.checks.append(Check("noise_floor", m.noise_floor_db <= -60.0, f"{m.noise_floor_db:.1f} dBFS", "<= -60 dBFS"))
    r.checks.append(Check("head_room_tone", 0.5 <= head <= 1.0 + 0.05, f"{head:.2f} s", "0.5-1.0 s"))
    r.checks.append(Check("tail_room_tone", 1.0 <= tail <= 5.0, f"{tail:.2f} s", "1-5 s"))
    r.checks.append(Check("length", m.duration <= 120 * 60, f"{m.duration / 60:.1f} min", "<= 120 min"))
    r.checks.append(Check("sample_rate", sr == 44100, f"{sr} Hz", "44100 Hz"))
    return r
