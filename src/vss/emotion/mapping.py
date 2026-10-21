"""Turn a backend-agnostic emotion blend into backend-specific controls.

A *blend* is what a script line asks for: one or more emotions with intensities, an optional
vocal delivery (whisper, shout ...) and an optional free-text acting direction::

    MIRA (whisper, fear 0.6, "afraid to wake something"): Is anyone up there?

Each helper here renders that blend for one family of controls. Backends pick the helpers
that match their capabilities (see ``vss.tts.base.Capabilities``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import taxonomy as tx
from .taxonomy import Emotion

DEFAULT_INTENSITY = 0.6


@dataclass(frozen=True)
class EmotionRef:
    emotion: Emotion
    intensity: float = DEFAULT_INTENSITY

    def __str__(self) -> str:
        return f"{self.emotion.id} {self.intensity:.2f}"


@dataclass(frozen=True)
class Blend:
    parts: tuple[EmotionRef, ...] = ()
    direction: str = ""

    # -- construction -------------------------------------------------------------------
    @classmethod
    def parse(cls, spec: str, direction: str = "") -> "Blend":
        """Parse ``"whisper, fear 0.6"`` (commas or '+' separate parts)."""
        parts: list[EmotionRef] = []
        for item in re.split(r"[,+]", spec or ""):
            item = item.strip()
            if not item:
                continue
            m = re.fullmatch(r"([A-Za-z_\- ]+?)\s*(\d*\.?\d+)?", item)
            if not m:
                raise ValueError(f"cannot parse emotion '{item}'")
            name, inten = m.group(1), m.group(2)
            parts.append(EmotionRef(tx.get(name), clamp01(float(inten)) if inten else DEFAULT_INTENSITY))
        return cls(tuple(parts), direction.strip())

    def merged(self, other: "Blend") -> "Blend":
        """Combine two blends; parts of ``other`` override same-id parts of ``self``.

        A new non-delivery emotion in ``other`` replaces the base emotions (a line that says
        ``(angry)`` should not stay ``content`` underneath), but deliveries persist unless
        ``other`` brings its own delivery.
        """
        if not other.parts:
            return Blend(self.parts, other.direction or self.direction)
        other_has_emotion = any(not p.emotion.is_delivery for p in other.parts)
        other_has_delivery = any(p.emotion.is_delivery for p in other.parts)
        kept = [
            p for p in self.parts
            if (p.emotion.is_delivery and not other_has_delivery)
            or (not p.emotion.is_delivery and not other_has_emotion)
        ]
        return Blend(tuple(kept) + other.parts, other.direction or self.direction)

    # -- accessors ----------------------------------------------------------------------
    @property
    def emotions(self) -> tuple[EmotionRef, ...]:
        return tuple(p for p in self.parts if not p.emotion.is_delivery)

    @property
    def deliveries(self) -> tuple[EmotionRef, ...]:
        return tuple(p for p in self.parts if p.emotion.is_delivery)

    @property
    def primary(self) -> EmotionRef | None:
        emos = self.emotions
        if emos:
            return emos[0]
        return self.parts[0] if self.parts else None

    @property
    def is_neutral(self) -> bool:
        return not self.parts or all(p.emotion.id == "neutral" for p in self.parts)

    def key(self) -> str:
        body = "+".join(f"{p.emotion.id}:{p.intensity:.2f}" for p in self.parts)
        return f"{body}|{self.direction}" if self.direction else body

    def __str__(self) -> str:
        s = ", ".join(str(p) for p in self.parts) or "neutral"
        return f"{s} \"{self.direction}\"" if self.direction else s


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def intensity_word(x: float) -> str:
    if x < 0.3:
        return "subtle"
    if x < 0.5:
        return "mild"
    if x < 0.75:
        return "moderate"
    if x < 0.9:
        return "strong"
    return "extreme"


def _adverb(x: float) -> str:
    if x < 0.3:
        return "slightly "
    if x < 0.75:
        return ""
    if x < 0.9:
        return "very "
    return "extremely "


# ---------------------------------------------------------------------------- IndexTTS
def index_vector(blend: Blend, cap: float = 0.8) -> list[float] | None:
    """IndexTTS-2/2.5 ``emo_vector`` (happy, angry, sad, afraid, disgusted, melancholic, surprised, calm).

    Taxonomy vectors describe the emotion at intensity 0.8; we scale linearly and cap the
    vector sum (large sums trade away speaker similarity on IndexTTS).
    """
    if blend.is_neutral:
        return None
    v = [0.0] * 8
    for p in blend.parts:
        for i, x in enumerate(p.emotion.vec):
            v[i] += x * p.intensity / 0.8
    total = sum(v)
    if total <= 0:
        return None
    if total > cap:
        v = [x * cap / total for x in v]
    return [round(min(1.0, x), 4) for x in v]


def index_alpha(blend: Blend) -> float:
    """``emo_alpha`` for an emotion *reference clip* (bank) - stronger lines get more of it."""
    p = blend.primary
    if p is None:
        return 0.0
    peak = max(x.intensity for x in blend.parts)
    return round(0.45 + 0.55 * peak, 3)


# -------------------------------------------------------------------- instruct / prompt
def instruct_text(blend: Blend, extra: str = "") -> str:
    """Natural-language direction for instruct models (Qwen3-TTS, OpenAI-style servers)."""
    if blend.is_neutral and not blend.direction and not extra:
        return ""
    chunks: list[str] = []
    for p in blend.emotions:
        if p.emotion.id == "neutral":
            continue
        chunks.append(f"{p.emotion.describe.rstrip('.')} ({intensity_word(p.intensity)} intensity).")
    for p in blend.deliveries:
        chunks.append(f"Delivery: {p.emotion.describe.rstrip('.')}.")
    if blend.direction:
        chunks.append(f"Direction: {blend.direction.rstrip('.')}.")
    if extra:
        chunks.append(extra)
    return " ".join(chunks)


# ------------------------------------------------------------------ free-form inline tags
def freeform_tags(blend: Blend, fmt: str = "[{}]", include_direction: bool = True) -> str:
    """Inline tags for Fish S2-Pro / Drama 3 style models, e.g. ``[whispering] [very afraid]``."""
    tags: list[str] = []
    for p in blend.deliveries:
        tags.append(_delivery_tag(p))
    for p in blend.emotions:
        if p.emotion.id == "neutral":
            continue
        tags.append(_adverb(p.intensity) + p.emotion.tag)
    if include_direction and blend.direction:
        tags.append(blend.direction.strip().rstrip("."))
    return " ".join(fmt.format(t) for t in tags)


def _delivery_tag(p: EmotionRef) -> str:
    named = {
        "whisper": "whispering", "murmur": "murmuring", "soft": "softly", "breathy": "breathy",
        "mumbling": "mumbling", "projected": "projecting", "shout": "shouting", "scream": "screaming",
        "roar": "roaring", "laughing": "laughing while speaking", "sing_song": "sing-song",
        "sobbing": "sobbing", "trembling": "trembling voice",
    }
    t = named.get(p.emotion.id, p.emotion.tag)
    if p.intensity < 0.3 and p.emotion.id in ("whisper", "shout", "scream"):
        t = {"whisper": "near-whisper", "shout": "raised voice", "scream": "yelping"}[p.emotion.id]
    return t


# ---------------------------------------------------------------------- Step-Audio-EditX
@dataclass(frozen=True)
class StepEdit:
    edit_type: str  # emotion | style | paralinguistic
    edit_info: str
    iterations: int = 1


def step_edits(blend: Blend) -> list[StepEdit]:
    edits: list[StepEdit] = []
    for p in blend.parts:
        if not p.emotion.step:
            continue
        kind, value = p.emotion.step
        it = 1 if p.intensity < 0.5 else 2 if p.intensity < 0.8 else 3
        if value == "whisper":
            it = max(it, 2)
        edits.append(StepEdit(kind, value, it))
    return edits


def step_is_exact(e: Emotion) -> bool:
    if not e.step:
        return False
    value = e.step[1]
    names = {e.id, *e.aliases}
    return value in names or e.id[:5] == value[:5]


# ---------------------------------------------------------------------------- Chatterbox
def chatterbox_params(blend: Blend) -> dict[str, float]:
    """Map arousal/intensity onto Chatterbox ``exaggeration`` and ``cfg_weight``.

    Resemble's guidance: dramatic speech wants exaggeration ~0.7+ with cfg ~0.3.
    """
    if blend.is_neutral:
        return {"exaggeration": 0.5, "cfg_weight": 0.5}
    drive = 0.0
    for p in blend.parts:
        drive = max(drive, (p.emotion.arousal + 1) / 2 * p.intensity + (0.15 if p.emotion.hard else 0.0))
    exaggeration = round(min(1.3, max(0.25, 0.3 + drive * 0.9)), 3)
    cfg = round(max(0.2, 0.5 - max(0.0, exaggeration - 0.5) * 0.45), 3)
    return {"exaggeration": exaggeration, "cfg_weight": cfg}


# -------------------------------------------------------------------------------- prosody
@dataclass(frozen=True)
class Prosody:
    rate: float = 1.0       # speaking-rate multiplier (>1 faster)
    pitch: float = 0.0      # semitones
    level: float = 0.0      # dB relative to the dialog target (mixer)
    extras: dict = field(default_factory=dict)


def prosody(blend: Blend) -> Prosody:
    rate, pitch, level = 1.0, 0.0, 0.0
    for p in blend.parts:
        s = min(1.6, p.intensity / DEFAULT_INTENSITY)
        rate *= 1.0 + (p.emotion.rate - 1.0) * s
        pitch += p.emotion.pitch * s
        level += p.emotion.level if p.emotion.is_delivery else p.emotion.level * s
    return Prosody(rate=round(max(0.6, min(1.5, rate)), 3), pitch=round(pitch, 2), level=round(level, 2))


def e2v_target(blend: Blend) -> str:
    """Expected emotion2vec+ class for automatic QA (coarse, 8 classes)."""
    p = blend.primary
    return p.emotion.e2v if p else "neutral"
