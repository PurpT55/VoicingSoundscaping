"""Data model of a parsed story script (``.vss`` files)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Union

from ..emotion.mapping import Blend


@dataclass(frozen=True)
class Source:
    file: str
    line: int

    def __str__(self) -> str:
        return f"{self.file}:{self.line}"


# ----------------------------------------------------------------------- inline elements
@dataclass
class Text:
    text: str          # spoken text (respellings applied, markup stripped)
    display: str = ""  # original text as written (for captions/transcripts)


@dataclass
class EmotionShift:
    blend: Blend
    reset: bool = False


@dataclass
class NonverbalEvent:
    name: str  # canonical nonverbal id


@dataclass
class PauseEvent:
    seconds: float


@dataclass
class SfxEvent:
    name: str
    params: dict[str, Any] = field(default_factory=dict)


InlineItem = Union[Text, EmotionShift, NonverbalEvent, PauseEvent, SfxEvent]


# --------------------------------------------------------------------------- top level
@dataclass
class Line:
    speaker: str
    blend: Blend
    items: list[InlineItem]
    params: dict[str, Any] = field(default_factory=dict)
    source: Source | None = None
    narration: bool = False  # True when no speaker tag was written (prose paragraph)

    @property
    def display_text(self) -> str:
        return " ".join(i.display or i.text for i in self.items if isinstance(i, Text)).strip()

    @property
    def spoken_text(self) -> str:
        return " ".join(i.text for i in self.items if isinstance(i, Text)).strip()


@dataclass
class Directive:
    name: str                      # scene | ambience | music | sfx | pause | space
    args: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    source: Source | None = None


@dataclass
class Scene:
    title: str = ""
    source: Source | None = None


Event = Union[Line, Directive, Scene]


@dataclass
class Chapter:
    number: int
    title: str
    events: list[Event] = field(default_factory=list)
    source: Source | None = None

    @property
    def lines(self) -> list[Line]:
        return [e for e in self.events if isinstance(e, Line)]

    @property
    def slug(self) -> str:
        import re

        s = re.sub(r"[^a-z0-9]+", "_", self.title.lower()).strip("_")
        return f"{self.number:02d}_{s or 'chapter'}"[:60]


@dataclass
class Diagnostic:
    level: str  # error | warning | info
    message: str
    source: Source | None = None

    def __str__(self) -> str:
        where = f"{self.source}: " if self.source else ""
        return f"{where}{self.level}: {self.message}"


@dataclass
class Script:
    chapters: list[Chapter] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    diagnostics: list[Diagnostic] = field(default_factory=list)

    @property
    def errors(self) -> list[Diagnostic]:
        return [d for d in self.diagnostics if d.level == "error"]

    def speakers(self) -> set[str]:
        return {ln.speaker for ch in self.chapters for ln in ch.lines}
