"""Emotional-range coverage: how much of the taxonomy each backend can really perform.

Levels per emotion:

* ``native`` - a direct control targets it (tag, instruction, vector dimension, edit type)
* ``bank``   - native *if* an emotion reference clip exists for it (IndexTTS emo prompt,
               reference-style cloners); see ``vss bank``
* ``approx`` - reachable only through a neighbour emotion and/or prosody tweaks
* ``none``   - not controllable

The project goal is >= 80-90 % of the taxonomy at ``native``/``bank`` level through the best
backend or a routed combination of backends.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import taxonomy as tx
from .mapping import step_is_exact

BASIC_FAMILIES = {"joy", "sadness", "anger", "fear", "surprise", "disgust", "calm"}
LEVEL_SCORE = {"native": 1.0, "bank": 1.0, "approx": 0.5, "none": 0.0}
LEVEL_ORDER = {"native": 3, "bank": 2, "approx": 1, "none": 0}


def emotion_level(e: tx.Emotion, caps, *, bank: bool = False) -> str:
    """Coverage level of one taxonomy entry for a backend's ``Capabilities`` (best mechanism wins)."""
    mech = set(caps.emotion)
    levels = ["none"]
    extreme_ok = not e.hard or caps.strong_extremes
    if "freeform_tags" in mech or "instruct" in mech:
        levels.append("native" if extreme_ok else "approx")
    if "step_edit" in mech:
        levels.append("native" if step_is_exact(e) else ("approx" if e.step else "none"))
    if "vector8" in mech:
        levels.append("native" if (e.kind == "emotion" and e.family in BASIC_FAMILIES and not e.hard) else "approx")
    if "exaggeration" in mech or "reference_style" in mech:
        levels.append("approx")
    if bank and caps.emotion_reference:
        levels.append("bank")
    return max(levels, key=LEVEL_ORDER.__getitem__)


def nonverbal_level(nv: tx.Nonverbal, caps, *, bank: bool = False) -> str:
    if caps.supports_nonverbal(nv.id):
        return "native"
    return "bank" if bank else "none"


@dataclass
class CoverageReport:
    backend: str
    levels: dict[str, str]
    nonverbals: dict[str, str]

    @property
    def native_pct(self) -> float:
        return 100.0 * sum(1 for v in self.levels.values() if v in ("native", "bank")) / len(self.levels)

    @property
    def weighted_pct(self) -> float:
        return 100.0 * sum(LEVEL_SCORE[v] for v in self.levels.values()) / len(self.levels)

    @property
    def nonverbal_pct(self) -> float:
        if not self.nonverbals:
            return 0.0
        return 100.0 * sum(1 for v in self.nonverbals.values() if v in ("native", "bank")) / len(self.nonverbals)


def report(name: str, caps, *, bank: bool = False) -> CoverageReport:
    return CoverageReport(
        backend=name,
        levels={e.id: emotion_level(e, caps, bank=bank) for e in tx.EMOTIONS},
        nonverbals={n.id: nonverbal_level(n, caps, bank=bank) for n in tx.NONVERBALS},
    )


def combine(name: str, reports: list[CoverageReport]) -> CoverageReport:
    """Best level per emotion across several backends (emotion routing)."""
    best = {}
    for eid in reports[0].levels:
        best[eid] = max((r.levels[eid] for r in reports), key=LEVEL_ORDER.__getitem__)
    best_nv = {}
    for nid in reports[0].nonverbals:
        best_nv[nid] = max((r.nonverbals[nid] for r in reports), key=LEVEL_ORDER.__getitem__)
    return CoverageReport(backend=name, levels=best, nonverbals=best_nv)


def format_table(reports: list[CoverageReport], show_rows: bool = False) -> str:
    lines = []
    w = max(len(r.backend) for r in reports)
    lines.append(f"{'backend'.ljust(w)}  native+bank  weighted  nonverbals")
    for r in reports:
        lines.append(f"{r.backend.ljust(w)}  {r.native_pct:10.1f}%  {r.weighted_pct:7.1f}%  {r.nonverbal_pct:9.1f}%")
    if show_rows:
        lines.append("")
        header = "emotion".ljust(16) + "".join(r.backend[:10].ljust(11) for r in reports)
        lines.append(header)
        for e in tx.EMOTIONS:
            lines.append(e.id.ljust(16) + "".join(r.levels[e.id].ljust(11) for r in reports))
    return "\n".join(lines)
