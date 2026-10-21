"""Planner: script chapter -> ordered steps (speech units, pauses, cues) ready to render.

Decisions made here:
* which backend speaks each unit (line param > cast > voice > emotion routing > default)
* how text is normalised and chunked (per-backend max length)
* whether emotion shifts / nonverbals stay inline (backend supports it) or become separate
  units / bank clips
* the gaps between units (sentence, paragraph, speaker change, scene)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..config import Project
from ..emotion.mapping import Blend
from ..script import model as m
from ..script.normalize import ChunkPolicy, chunk, normalize
from ..tts.base import Capabilities, Piece


@dataclass
class Unit:
    uid: str
    speaker: str
    voice: str
    backend: str
    pieces: list[Piece]
    blend: Blend
    params: dict[str, Any]
    display: str
    gap_before: float
    source: m.Source | None = None
    pan: float = 0.0
    narrator: bool = False
    line_id: int = 0          # units of the same script line share it (captions)
    line_text: str = ""       # the line as written (captions show this, not the normalised text)

    @property
    def text(self) -> str:
        return " ".join(p.text for p in self.pieces if p.kind == "speech").strip()


@dataclass
class Pause:
    seconds: float


@dataclass
class NonverbalClip:
    speaker: str
    voice: str
    nonverbal: str
    gap_before: float
    source: m.Source | None = None


@dataclass
class Cue:
    """A sound-design directive carried to the timeline (scene, ambience, music, sfx, space)."""
    kind: str  # scene | ambience | music | sfx | space
    args: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    title: str = ""
    source: m.Source | None = None


Step = Unit | Pause | NonverbalClip | Cue


@dataclass
class Plan:
    chapter: m.Chapter
    steps: list[Step]
    warnings: list[str] = field(default_factory=list)

    @property
    def units(self) -> list[Unit]:
        return [s for s in self.steps if isinstance(s, Unit)]


def route_backend(blend: Blend, routing: dict[str, str], default: str) -> str:
    if not routing:
        return default
    for p in blend.parts:
        if p.emotion.id in routing:
            return routing[p.emotion.id]
    for p in blend.parts:
        if p.emotion.hard and "hard" in routing:
            return routing["hard"]
        for key in (f"kind:{p.emotion.kind}", f"family:{p.emotion.family}"):
            if key in routing:
                return routing[key]
    return default


def spoken_title(title: str) -> str:
    t = re.sub(r"\s*[:—–-]\s*", ". ", title.strip())
    return t if t.endswith((".", "!", "?")) else t + "."


class Planner:
    def __init__(self, project: Project, caps_for: Callable[[str], Capabilities], draft_backend: str | None = None):
        self.p = project
        self.caps_for = caps_for
        self.draft = draft_backend
        r = project.cfg["render"]
        self.pauses = r["pauses"]
        self.chunk_cfg = r["chunk"]
        self.routing = dict(project.cfg["backends"].get("routing") or {})
        self.default_backend = project.cfg["backends"]["default"]
        self.nv_fallback = r.get("nonverbal_fallback", "bank")

    # -- helpers ------------------------------------------------------------------------
    def backend_for(self, line: m.Line, blend: Blend) -> str:
        if self.draft:
            return self.draft
        if line.params.get("backend"):
            return str(line.params["backend"])
        member = self.p.member(line.speaker)
        voice = self.p.voices[member.voice]
        base = member.backend or voice.backend or self.default_backend
        return route_backend(blend, self.routing, base)

    def policy(self, caps: Capabilities) -> ChunkPolicy:
        mx = min(int(self.chunk_cfg.get("max_chars", 300)), caps.max_chars)
        return ChunkPolicy(target_chars=min(int(self.chunk_cfg.get("target_chars", 220)), mx), max_chars=mx)

    def norm(self, text: str, lang: str) -> str:
        return normalize(text, lang, self.p.cfg.get("lexicon") or {})

    # -- main ---------------------------------------------------------------------------
    def plan(self, chapter: m.Chapter) -> Plan:
        steps: list[Step] = []
        warnings: list[str] = []
        prev_speaker: str | None = None
        counter = 0
        line_no = 0
        pending_gap = 0.0

        if self.p.cfg["render"].get("announce_chapters", True) and chapter.title:
            narrator = self.p.default_speaker
            member = self.p.member(narrator)
            backend = self.draft or member.backend or self.p.voices[member.voice].backend or self.default_backend
            blend = member.default
            text = self.norm(spoken_title(chapter.title), self.p.voices[member.voice].language)
            steps.append(Unit(f"{chapter.number:02d}-title", narrator, member.voice, backend,
                              [Piece("speech", text, blend)], blend, {}, chapter.title, 0.0, chapter.source,
                              member.pan, True, line_id=0, line_text=chapter.title))
            steps.append(Pause(float(self.pauses.get("after_title", 1.2))))
            prev_speaker = narrator

        for ev in chapter.events:
            if isinstance(ev, m.Scene):
                steps.append(Cue("scene", title=ev.title, source=ev.source))
                pending_gap = max(pending_gap, float(self.pauses.get("scene", 2.0)))
                prev_speaker = None
                continue
            if isinstance(ev, m.Directive):
                if ev.name == "pause":
                    secs = float(ev.args[0]) if ev.args else float(ev.params.get("seconds", 1.0))
                    steps.append(Pause(secs))
                elif ev.name in ("scene", "ambience", "music", "sfx", "space"):
                    steps.append(Cue(ev.name, list(ev.args), dict(ev.params), source=ev.source))
                continue
            # ---- a line of speech
            line: m.Line = ev
            line_no += 1
            member = self.p.member(line.speaker)
            voice = self.p.voices[member.voice]
            base_blend = member.default.merged(line.blend)
            if line.speaker == prev_speaker:
                gap = float(self.pauses["paragraph"])
            else:
                gap = float(self.pauses["speaker_change"]) if prev_speaker else 0.0
            gap = max(gap, pending_gap)
            pending_gap = 0.0
            prev_speaker = line.speaker
            params = {**member.params, **line.params}
            params.pop("backend", None)

            first = True
            for run in self._runs(self._segments(line, base_blend), line):
                if isinstance(run, m.PauseEvent):
                    steps.append(Pause(run.seconds))
                    continue
                if isinstance(run, m.SfxEvent):
                    steps.append(Cue("sfx", [run.name], dict(run.params), source=line.source))
                    continue
                backend, caps, segs = run
                pieces: list = []
                for blend, items in segs:
                    if pieces and not caps.inline_emotion:
                        pieces.append(None)
                    pieces.extend(self._pieces(items, blend, caps, voice.language, line, warnings))
                for chunk_pieces in self._chunk(pieces, caps):
                    counter += 1
                    if len(chunk_pieces) == 1 and isinstance(chunk_pieces[0], _Nonverbal):
                        steps.append(NonverbalClip(line.speaker, member.voice, chunk_pieces[0].name,
                                                   gap if first else 0.12, line.source))
                        first = False
                        continue
                    speech = [p for p in chunk_pieces if p.kind == "speech" and p.text]
                    if not speech:
                        continue
                    steps.append(Unit(
                        uid=f"{chapter.number:02d}-{counter:04d}", speaker=line.speaker, voice=member.voice,
                        backend=backend, pieces=chunk_pieces, blend=speech[0].blend, params=params,
                        display=" ".join(p.text for p in speech),
                        gap_before=gap if first else float(self.pauses["sentence"]), source=line.source,
                        pan=float(params.get("pan", member.pan)),
                        narrator=line.narration or line.speaker == self.p.default_speaker,
                        line_id=line_no, line_text=line.display_text,
                    ))
                    first = False
        return Plan(chapter, steps, warnings)

    def _segments(self, line: m.Line, base: Blend) -> list:
        """Split a line at emotion shifts, pauses and sfx: [(blend, items) | PauseEvent | SfxEvent].

        Inline shifts are relative to the line's base blend (``[sad]`` then ``[relief]`` each
        modify the line's own emotion), ``[reset]`` returns to it.
        """
        out: list = []
        cur_blend, cur = base, []
        for it in line.items:
            if isinstance(it, m.EmotionShift):
                if cur:
                    out.append((cur_blend, cur))
                    cur = []
                cur_blend = base if it.reset else base.merged(it.blend)
            elif isinstance(it, (m.PauseEvent, m.SfxEvent)):
                if cur:
                    out.append((cur_blend, cur))
                    cur = []
                out.append(it)
            else:
                cur.append(it)
        if cur:
            out.append((cur_blend, cur))
        return out

    def _runs(self, segments: list, line: m.Line) -> list:
        """Group consecutive emotion segments that go to the same backend into one run."""
        runs: list = []
        for seg in segments:
            if not isinstance(seg, tuple):
                runs.append(seg)
                continue
            backend = self.backend_for(line, seg[0])
            if runs and isinstance(runs[-1], tuple) and runs[-1][0] == backend:
                runs[-1][2].append(seg)
            else:
                runs.append((backend, self.caps_for(backend), [seg]))
        return runs

    def _pieces(self, items: list, blend: Blend, caps: Capabilities, lang: str, line: m.Line,
                warnings: list[str]) -> list:
        """Turn items into Pieces; ``None`` marks a forced unit split, ``_Nonverbal`` a bank clip."""
        pieces: list = []
        for it in items:
            if isinstance(it, m.Text):
                pieces.append(Piece("speech", self.norm(it.text, lang), blend))
            elif isinstance(it, m.NonverbalEvent):
                if caps.supports_nonverbal(it.name):
                    pieces.append(Piece("nonverbal", nonverbal=it.name, blend=blend))
                elif self.nv_fallback == "bank":
                    pieces.append(None)
                    pieces.append(_Nonverbal(it.name))
                    pieces.append(None)
                else:
                    warnings.append(f"{line.source}: <{it.name}> dropped (backend cannot render it; no bank)")
        return pieces

    def _chunk(self, pieces: list, caps: Capabilities) -> list[list[Piece]]:
        """Group pieces into requests no longer than the backend allows."""
        policy = self.policy(caps)
        groups: list[list[Piece]] = []
        buf: list[Piece] = []
        size = 0

        def flush():
            nonlocal buf, size
            if buf:
                groups.append(buf)
            buf, size = [], 0

        for p in pieces:
            if p is None:
                flush()
                continue
            if isinstance(p, _Nonverbal):
                flush()
                groups.append([p])  # standalone bank clip marker
                continue
            if p.kind == "nonverbal":
                buf.append(p)
                continue
            for sentence in chunk(p.text, policy):
                if buf and size + len(sentence) + 1 > policy.target_chars:
                    flush()
                buf.append(Piece("speech", sentence, p.blend))
                size += len(sentence) + 1
        flush()
        return groups


@dataclass(frozen=True)
class _Nonverbal:
    name: str
    kind: str = "bank_nonverbal"


def make_plan(project: Project, chapter: m.Chapter, caps_for: Callable[[str], Capabilities],
              draft_backend: str | None = None) -> Plan:
    return Planner(project, caps_for, draft_backend).plan(chapter)


def bank_lookup(directory: Path | None, key: str, variant: int = 0) -> tuple[Path, str] | None:
    """Find ``<dir>/<key>.wav`` (or numbered variants ``<key>_1.wav``) and its transcript."""
    if not directory or not Path(directory).exists():
        return None
    d = Path(directory)
    hits = sorted(p for ext in ("wav", "flac", "mp3", "ogg")
                  for p in d.glob(f"{key}*.{ext}") if re.fullmatch(rf"{re.escape(key)}(_\d+)?", p.stem))
    if not hits:
        return None
    path = hits[variant % len(hits)]
    txt = path.with_suffix(".txt")
    return path, (txt.read_text(encoding="utf-8").strip() if txt.exists() else "")


def emotion_bank_clip(project: Project, voice_name: str, blend: Blend, caps: Capabilities,
                      variant: int = 0) -> tuple[Path, str] | None:
    """Pick the emotion reference clip for a blend: delivery first (hardest to fake), then primary."""
    if not caps.emotion_reference or blend.is_neutral:
        return None
    voice = project.voices.get(voice_name)
    dirs = [voice.emotion_bank if voice else None]
    if caps.emotion_reference_scope == "any":
        dirs.append(project.bank_dir("emotions"))
    keys = [p.emotion.id for p in blend.deliveries] + [p.emotion.id for p in blend.emotions if p.emotion.id != "neutral"]
    for key in keys:
        for d in dirs:
            hit = bank_lookup(d, key, variant)
            if hit:
                return hit
    return None


def nonverbal_bank_clip(project: Project, voice_name: str, nv: str, variant: int = 0) -> tuple[Path, str] | None:
    voice = project.voices.get(voice_name)
    for d in (voice.nonverbal_bank if voice else None, project.bank_dir("nonverbals")):
        hit = bank_lookup(d, nv, variant)
        if hit:
            return hit
    return None


__all__ = ["Plan", "Planner", "Unit", "Pause", "NonverbalClip", "Cue", "make_plan", "route_backend",
           "emotion_bank_clip", "nonverbal_bank_clip", "bank_lookup", "spoken_title"]
