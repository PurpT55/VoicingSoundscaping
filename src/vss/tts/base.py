"""Backend contract for speech synthesis.

A backend turns one :class:`SynthesisRequest` (a short, single-voice unit of speech, usually
one sentence or a few) into mono float32 audio. The planner decides how a script line is cut
into requests based on each backend's :class:`Capabilities`:

* backends with ``inline_emotion`` get mixed-emotion pieces in one request (rendered as inline
  tags), others get one request per emotion segment;
* nonverbals the backend supports natively are passed inline; the others are rendered from a
  nonverbal bank or dropped (with a lint warning).
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..emotion.mapping import Blend, Prosody, prosody

# Emotion control mechanisms a backend can expose.
VECTOR8 = "vector8"            # IndexTTS-style 8-dim emotion vector
INSTRUCT = "instruct"          # natural-language instruction (Qwen3-TTS, OpenAI-style servers)
FREEFORM = "freeform_tags"     # free-form inline tags (Fish S2-Pro / Drama 3, Higgs v3)
STEP_EDIT = "step_edit"        # Step-Audio-EditX emotion/style editing
EXAGGERATION = "exaggeration"  # scalar expressiveness (Chatterbox)
REFERENCE_STYLE = "reference_style"  # emotion carried by the reference clip (cloners)
PROSODY = "prosody"            # only rate/pitch/volume (espeak, Kokoro speed)


@dataclass(frozen=True)
class Capabilities:
    emotion: frozenset[str] = frozenset()
    nonverbals: frozenset[str] | str = frozenset()  # "freeform" = any nonverbal as inline tag
    inline_emotion: bool = False
    emotion_reference: bool = False  # accepts a separate emotion reference clip (emotion bank)
    # "any": the clip may come from any speaker (timbre/emotion disentangled, e.g. IndexTTS);
    # "voice": the clip must be the same voice (cloners that copy the clip's style AND timbre)
    emotion_reference_scope: str = "any"
    cloning: bool = False
    speed: bool = False
    languages: tuple[str, ...] = ("en",)
    max_chars: int = 300
    sample_rate: int = 24000
    strong_extremes: bool = False    # convincingly does screams, sobs, laughing speech
    deterministic_seed: bool = True

    def supports_nonverbal(self, nv_id: str) -> bool:
        return self.nonverbals == "freeform" or nv_id in self.nonverbals


@dataclass
class VoiceSpec:
    name: str
    reference: Path | None = None
    reference_text: str = ""
    description: str = ""
    language: str = "en"
    options: dict[str, Any] = field(default_factory=dict)  # backend-specific options for this voice

    def fingerprint(self) -> dict:
        ref_hash = file_hash(self.reference) if self.reference and Path(self.reference).exists() else None
        return {
            "name": self.name, "ref": ref_hash, "ref_text": self.reference_text,
            "desc": self.description, "lang": self.language, "opts": self.options,
        }


@dataclass(frozen=True)
class Piece:
    """One element of a request: ``speech`` text with a blend, or an inline ``nonverbal``."""
    kind: str  # "speech" | "nonverbal"
    text: str = ""
    blend: Blend = Blend()
    nonverbal: str = ""


@dataclass
class SynthesisRequest:
    pieces: list[Piece]
    voice: VoiceSpec
    blend: Blend                      # dominant blend of the request
    language: str = "en"
    seed: int = 0
    emotion_reference: Path | None = None   # emotion-bank clip, if the backend uses one
    emotion_reference_text: str = ""
    params: dict[str, Any] = field(default_factory=dict)  # per-line overrides (rate=, temperature= ...)

    @property
    def text(self) -> str:
        return " ".join(p.text for p in self.pieces if p.kind == "speech").strip()

    @property
    def prosody(self) -> Prosody:
        p = prosody(self.blend)
        if "rate" in self.params:
            p = Prosody(rate=p.rate * float(self.params["rate"]), pitch=p.pitch, level=p.level)
        if "pitch" in self.params:
            p = Prosody(rate=p.rate, pitch=p.pitch + float(self.params["pitch"]), level=p.level)
        return p

    def fingerprint(self) -> dict:
        return {
            "pieces": [(p.kind, p.text, p.blend.key(), p.nonverbal) for p in self.pieces],
            "voice": self.voice.fingerprint(), "blend": self.blend.key(), "lang": self.language,
            "seed": self.seed,
            "emo_ref": file_hash(self.emotion_reference) if self.emotion_reference else None,
            "params": self.params,
        }


@dataclass
class Audio:
    samples: np.ndarray  # float32, mono
    sr: int
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return len(self.samples) / float(self.sr) if self.sr else 0.0


class BackendUnavailable(RuntimeError):
    pass


class TTSBackend(ABC):
    """Base class; subclasses set ``name`` and implement :meth:`synthesize`."""

    name: str = "base"
    description: str = ""
    license: str = ""

    def __init__(self, **options: Any) -> None:
        self.options = options

    # -- introspection ------------------------------------------------------------------
    def capabilities(self) -> Capabilities:
        return Capabilities()

    def available(self) -> tuple[bool, str]:
        """Return (ok, human-readable reason). Must be cheap and must not load models."""
        return True, "ok"

    def fingerprint(self) -> dict:
        """What identifies this backend's output, for the render cache."""
        keep = {k: v for k, v in self.options.items() if k not in ("device", "url", "timeout", "workers")}
        return {"backend": self.name, "options": keep}

    # -- synthesis ----------------------------------------------------------------------
    @abstractmethod
    def synthesize(self, req: SynthesisRequest) -> Audio:
        ...

    def render_text(self, req: SynthesisRequest) -> str:
        """The exact text sent to the model (inline tags included). Used for logs/QA."""
        return req.text

    def close(self) -> None:
        pass


def file_hash(path: Path | str | None, chunk: int = 1 << 20) -> str | None:
    if not path:
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()[:16]


def stable_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=_json_default).encode()).hexdigest()


def _json_default(o: Any):
    if isinstance(o, Path):
        return str(o)
    if hasattr(o, "__dataclass_fields__"):
        return asdict(o)
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    return str(o)
