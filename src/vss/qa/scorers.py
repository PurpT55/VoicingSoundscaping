"""Automatic take scoring (best-of-N selection).

* ``signal``   - always on: duration plausibility, clipping, long internal silences, level
* ``asr``      - faster-whisper transcription -> word error rate (catches hallucinated / missing words)
* ``emotion``  - emotion2vec+ (FunASR) probability of the intended coarse emotion class
* ``speaker``  - optional speaker similarity hook (embedding cosine vs the voice reference)

Scores are 0..1 (higher is better); details are stored in the take cache and the QA report.
"""

from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass, field

import numpy as np

from ..audio import dsp, io
from ..audio.loudness import peak_db, rms_db
from ..emotion.mapping import Blend, e2v_target


@dataclass
class TakeScore:
    score: float
    parts: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"score": round(self.score, 3), "parts": {k: round(v, 3) for k, v in self.parts.items()}, "notes": self.notes}


def expected_seconds(text: str, rate: float = 1.0, cps: float = 14.5) -> float:
    """Typical narration runs ~14-16 characters per second."""
    return max(0.4, len(text) / (cps * max(0.5, rate)))


class SignalScorer:
    name = "signal"

    def score(self, x: np.ndarray, sr: int, text: str, blend: Blend, rate: float = 1.0, **_) -> tuple[float, list[str]]:
        notes = []
        if len(x) == 0 or rms_db(x) < -60:
            return 0.0, ["empty or silent take"]
        dur = len(x) / sr
        exp = expected_seconds(text, rate)
        ratio = dur / exp
        # short exclamations, screams, sobs and laughter legitimately run long
        sigma = 0.35 + (0.25 if len(text) < 25 else 0.0) + (0.25 if any(p.emotion.hard for p in blend.parts) else 0.0)
        s_dur = float(np.exp(-((np.log(ratio)) ** 2) / (2 * sigma ** 2)))  # 1.0 at ratio 1
        limit = 1.8 * (1 + sigma - 0.35)
        if ratio > limit:
            notes.append(f"take is {ratio:.1f}x longer than expected (babble/hallucination?)")
        if ratio < 0.5:
            notes.append(f"take is {ratio:.1f}x shorter than expected (truncated?)")
        clip = float(np.mean(np.abs(x) > 0.995))
        s_clip = 1.0 if clip < 1e-4 else max(0.0, 1.0 - clip * 200)
        if clip >= 1e-3:
            notes.append(f"clipping {clip * 100:.2f}% of samples")
        gaps = dsp.internal_silences(x, sr, min_len=1.3)
        s_gap = 1.0 if not gaps else max(0.2, 1.0 - 0.25 * len(gaps))
        if gaps:
            notes.append(f"{len(gaps)} long internal silence(s)")
        crest = peak_db(x) - rms_db(x)
        s_crest = 1.0 if crest < 30 else 0.7
        return float(0.45 * s_dur + 0.2 * s_clip + 0.25 * s_gap + 0.1 * s_crest), notes


def _words(s: str) -> list[str]:
    s = s.lower().replace("’", "'")
    return re.findall(r"[a-z0-9À-ɏ']+", s)


def wer(ref: str, hyp: str) -> float:
    r, h = _words(ref), _words(hyp)
    if not r:
        return 0.0 if not h else 1.0
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return d[len(h)] / len(r)


class AsrScorer:
    name = "asr"

    def __init__(self, model: str = "small.en", device: str = "auto", compute_type: str = "default", max_wer: float = 0.15):
        from faster_whisper import WhisperModel

        self.model = WhisperModel(model, device=device, compute_type=compute_type)
        self.max_wer = max_wer

    def transcribe(self, x: np.ndarray, sr: int, language: str | None = None) -> str:
        x16 = io.resample(x, sr, 16000)
        segments, _ = self.model.transcribe(x16.astype(np.float32), language=language, beam_size=1, vad_filter=False)
        return " ".join(s.text for s in segments).strip()

    def score(self, x: np.ndarray, sr: int, text: str, language: str = "en", **_) -> tuple[float, list[str]]:
        hyp = self.transcribe(x, sr, language[:2] if language else None)
        e = wer(text, hyp)
        notes = [f"WER {e:.2f}: heard '{hyp[:120]}'"] if e > self.max_wer else []
        return float(max(0.0, 1.0 - e / max(1e-6, 2 * self.max_wer))), notes


class EmotionScorer:
    """emotion2vec+ via FunASR. Coarse (8 classes) - a tie-breaker, never the only judge."""

    name = "emotion"
    LABELS = ("angry", "disgusted", "fearful", "happy", "neutral", "other", "sad", "surprised", "unknown")

    def __init__(self, model: str = "iic/emotion2vec_plus_large"):
        from funasr import AutoModel

        self.model = AutoModel(model=model, disable_update=True)

    def probs(self, x: np.ndarray, sr: int) -> dict[str, float]:
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            io.save(f.name, io.resample(x, sr, 16000), 16000, subtype="PCM_16")
            res = self.model.generate(f.name, granularity="utterance", extract_embedding=False)
        r = res[0]
        out = {}
        for label, sc in zip(r["labels"], r["scores"]):
            name = label.split("/")[-1].strip().lower()
            out[name] = float(sc)
        return out

    def score(self, x: np.ndarray, sr: int, text: str, blend: Blend, **_) -> tuple[float, list[str]]:
        target = e2v_target(blend)
        p = self.probs(x, sr)
        val = p.get(target, 0.0)
        top = max(p, key=p.get) if p else "?"
        notes = [] if top == target else [f"emotion model hears '{top}' (wanted '{target}')"]
        return float(min(1.0, val * 1.6)), notes


def build_scorers(qa_cfg: dict) -> list:
    scorers: list = [SignalScorer()]
    if qa_cfg.get("asr") == "faster-whisper":
        try:
            scorers.append(AsrScorer(qa_cfg.get("asr_model", "small.en"), max_wer=float(qa_cfg.get("max_wer", 0.15))))
        except ImportError:
            pass
    if qa_cfg.get("emotion_model") == "emotion2vec":
        try:
            scorers.append(EmotionScorer(qa_cfg.get("emotion2vec_model", "iic/emotion2vec_plus_large")))
        except ImportError:
            pass
    return scorers


WEIGHTS = {"signal": 0.35, "asr": 0.45, "emotion": 0.2, "speaker": 0.2}


def score_take(scorers: list, x: np.ndarray, sr: int, text: str, blend: Blend, rate: float, language: str) -> TakeScore:
    parts, notes = {}, []
    for s in scorers:
        val, n = s.score(x, sr, text=text, blend=blend, rate=rate, language=language)
        parts[s.name] = val
        notes.extend(n)
    total_w = sum(WEIGHTS.get(k, 0.2) for k in parts)
    score = sum(parts[k] * WEIGHTS.get(k, 0.2) for k in parts) / total_w if total_w else 0.0
    if parts.get("signal", 1.0) < 0.15:
        score = min(score, parts["signal"])
    return TakeScore(score, parts, notes)
