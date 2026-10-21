"""Render pipeline: script -> plan -> takes (cached, QA-selected) -> timeline -> mix -> master -> export."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from ..audio import dsp, io
from ..audio.acx import check as acx_check
from ..audio.chain import apply_chain
from ..audio.loudness import normalize_lufs
from ..audio.master import PRESETS, master
from ..config import Project
from ..emotion.mapping import prosody
from ..mix.timeline import layout, mixdown
from ..qa.scorers import SignalScorer, build_scorers, score_take
from ..script import model as m
from ..script.parser import parse_files
from ..sound import procedural
from ..sound.resolve import SoundResolver
from ..tts import registry
from ..tts.base import Audio, BackendUnavailable, SynthesisRequest, TTSBackend, VoiceSpec, stable_hash
from .cache import TakeCache
from .export import build_m4b, srt, write_chapter
from .plan import NonverbalClip, Plan, Unit, emotion_bank_clip, make_plan, nonverbal_bank_clip

SAFE_FALLBACKS = ("espeak", "placeholder")


@dataclass
class UnitResult:
    uid: str
    speaker: str
    backend: str
    text: str
    emotion: str
    render_text: str = ""
    take: int = 0
    score: float = 0.0
    notes: list[str] = field(default_factory=list)
    cached: bool = False
    seconds: float = 0.0
    flagged: bool = False
    source: str = ""
    emotion_reference: str = ""


@dataclass
class ChapterResult:
    number: int
    title: str
    slug: str
    files: dict[str, str] = field(default_factory=dict)
    duration: float = 0.0
    acx: dict = field(default_factory=dict)
    levels: dict = field(default_factory=dict)
    units: list[UnitResult] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    seconds_to_render: float = 0.0


class Renderer:
    def __init__(self, project: Project, *, draft: bool = False, takes: int | None = None, use_cache: bool = True,
                 log: Callable[[str], None] = print, preset: str | None = None, qa: bool = True):
        self.p = project
        cfg = project.cfg
        self.preset = preset or cfg["master"]["preset"]
        self.sr = int(cfg["render"]["sample_rate"])
        if self.preset in PRESETS and PRESETS[self.preset].sample_rate != self.sr and self.preset == "acx":
            self.sr = 44100
        self.draft_backend = cfg["backends"]["draft"] if draft else None
        self.takes = int(takes or cfg["render"]["takes"])
        self.max_takes = max(self.takes, int(cfg["render"]["max_takes"]))
        self.log = log
        self.cache = TakeCache(project.cache_dir, use_cache)
        self.resolver = SoundResolver(project, self.sr)
        self.scorers = build_scorers(cfg["qa"]) if qa and not draft else [SignalScorer()]
        self._backends: dict[str, TTSBackend] = {}
        self.fallback_notes: dict[str, str] = {}

    # -- backends -----------------------------------------------------------------------
    def backend(self, name: str) -> TTSBackend:
        if name in self._backends:
            return self._backends[name]
        chain = [name] + list(self.p.cfg["backends"].get("fallback", []) or [])
        if self.p.cfg["backends"].get("draft"):
            chain.append(self.p.cfg["backends"]["draft"])
        chain += list(SAFE_FALLBACKS)
        reasons = []
        for cand in dict.fromkeys(chain):
            try:
                b = registry.create(cand, None, self.p)
            except (KeyError, TypeError) as exc:
                reasons.append(f"{cand}: {exc}")
                continue
            ok, why = b.available()
            if ok:
                if cand != name:
                    note = f"backend '{name}' unavailable ({'; '.join(reasons) or why}) -> using '{cand}'"
                    self.fallback_notes[name] = note
                    self.log(f"  ! {note}")
                self._backends[name] = b
                return b
            reasons.append(f"{cand}: {why}")
        raise BackendUnavailable(f"no usable backend for '{name}': {'; '.join(reasons)}")

    def caps_for(self, name: str):
        return self.backend(name).capabilities()

    def close(self) -> None:
        for b in {id(b): b for b in self._backends.values()}.values():
            b.close()

    # -- voices -------------------------------------------------------------------------
    def voice_spec(self, voice_name: str) -> VoiceSpec:
        v = self.p.voices[voice_name]
        return VoiceSpec(v.name, v.reference, v.reference_text, v.description, v.language, v.options)

    # -- units --------------------------------------------------------------------------
    def render_unit(self, unit: Unit) -> tuple[np.ndarray, UnitResult]:
        backend = self.backend(unit.backend)
        caps = backend.capabilities()
        voice = self.p.voices[unit.voice]
        bank = emotion_bank_clip(self.p, unit.voice, unit.blend, caps)
        base_seed = int(self.p.cfg["render"]["seed"])
        res = UnitResult(unit.uid, unit.speaker, backend.name, unit.text, str(unit.blend),
                         source=str(unit.source or ""), emotion_reference=str(bank[0]) if bank else "")
        best: tuple[float, Audio, int, list[str], bool] | None = None
        rate = prosody(unit.blend).rate * float(unit.params.get("rate", 1.0))
        takes_wanted = int(unit.params.get("takes", self.takes))
        min_score = float(self.p.cfg["qa"]["min_score"])
        for take in range(max(takes_wanted, 1) if self.draft_backend else self.max_takes):
            seed = int(stable_hash([base_seed, unit.voice, unit.text, unit.blend.key(), take])[:8], 16) % (2 ** 31)
            if "seed" in unit.params:
                seed = int(unit.params["seed"]) + take
            req = SynthesisRequest(
                pieces=list(unit.pieces), voice=self.voice_spec(unit.voice), blend=unit.blend, language=voice.language,
                seed=seed, emotion_reference=bank[0] if bank else None, emotion_reference_text=bank[1] if bank else "",
                params={k: v for k, v in unit.params.items() if not k.startswith("_") and k not in ("takes", "pan", "seed")},
            )
            key = TakeCache.key(backend.fingerprint(), req.fingerprint(), take)
            hit = self.cache.get(key)
            cached = hit is not None
            if hit:
                audio, meta = hit
            else:
                audio = backend.synthesize(req)
                meta = {"render_text": backend.render_text(req), "uid": unit.uid}
            x = audio.samples.astype(np.float32)
            sc = score_take(self.scorers, x, audio.sr, unit.text, unit.blend, rate, voice.language)
            if not cached:
                meta["qa"] = sc.as_dict()
                self.cache.put(key, audio, meta)
            res.render_text = meta.get("render_text", unit.text)
            if best is None or sc.score > best[0]:
                best = (sc.score, audio, take, sc.notes, cached)
            if take + 1 >= takes_wanted and best[0] >= min_score:
                break
        assert best is not None
        score, audio, take, notes, cached = best
        res.take, res.score, res.notes, res.cached = take, round(score, 3), notes, cached
        res.flagged = score < min_score or bool(notes)
        y = self.process_voice(audio, unit.voice, unit.blend, extra_gain=float(unit.params.get("gain", 0.0)))
        res.seconds = round(len(y) / self.sr, 2)
        return y, res

    def process_voice(self, audio: Audio, voice_name: str, blend, extra_gain: float = 0.0) -> np.ndarray:
        v = self.p.voices[voice_name]
        x = io.resample(audio.samples.astype(np.float32), audio.sr, self.sr)
        x = dsp.trim_silence(x, self.sr, pad=0.035)
        x = apply_chain(x, self.sr, v.chain)
        target = float(self.p.cfg["mix"]["dialog_lufs"]) + prosody(blend).level + v.gain + extra_gain
        x = normalize_lufs(x, self.sr, target)
        return dsp.declick(x, self.sr)

    def nonverbal_audio(self, step: NonverbalClip) -> np.ndarray | None:
        hit = nonverbal_bank_clip(self.p, step.voice, step.nonverbal)
        if not hit:
            return None
        x, sr = io.load(hit[0], mono=True)
        from ..emotion.mapping import Blend

        return self.process_voice(Audio(x, sr), step.voice, Blend(), extra_gain=-3.0)

    def breath(self, unit: Unit) -> np.ndarray | None:
        mode = self.p.cfg["render"].get("breaths", "off")
        if mode == "off" or any(p.emotion.id in ("whisper", "murmur") for p in unit.blend.deliveries):
            return None
        hit = nonverbal_bank_clip(self.p, unit.voice, "breath", variant=hash(unit.uid) % 7)
        if hit:
            x, sr = io.load(hit[0], mono=True)
            x = io.resample(x, sr, self.sr)
        elif mode == "procedural":
            x = procedural.render("breath", 0.3 + (hash(unit.uid) % 5) * 0.03, self.sr, seed=unit.uid)[:, 0]
        else:
            return None
        return normalize_lufs(x, self.sr, float(self.p.cfg["mix"]["dialog_lufs"]) - 22.0)

    # -- chapters -----------------------------------------------------------------------
    def plan(self, chapter: m.Chapter) -> Plan:
        return make_plan(self.p, chapter, self.caps_for, self.draft_backend)

    def render_chapter(self, chapter: m.Chapter, out_dir: Path) -> ChapterResult:
        t0 = time.time()
        plan = self.plan(chapter)
        result = ChapterResult(chapter.number, chapter.title, chapter.slug, warnings=list(plan.warnings))
        clips: dict[str, np.ndarray] = {}
        units = plan.units
        self.log(f"chapter {chapter.number}: {chapter.title} - {len(units)} units")
        for i, u in enumerate(units, 1):
            y, ur = self.render_unit(u)
            clips[u.uid] = y
            result.units.append(ur)
            mark = "cached" if ur.cached else f"take {ur.take + 1}"
            flag = "  <-- CHECK" if ur.flagged else ""
            self.log(f"  [{i:>3}/{len(units)}] {u.speaker:<10} {ur.backend:<11} {mark:<7} score {ur.score:.2f} "
                     f"| {u.text[:60]}{flag}")
        tl = layout(plan, clips, self.sr, self.p.cfg, self.resolver, breath_fn=self.breath,
                    nonverbal_audio=self.nonverbal_audio)
        beds = self.beds_mode()
        exp = self.p.cfg["export"]
        mix, sources, stems = mixdown(tl, self.resolver, self.p.cfg, beds=beds, stems=bool(exp.get("stems")))
        result.sources = sources
        mastered, levels = master(mix, self.sr, self.preset)
        mastered = self.add_room_tone(mastered, chapter.number)
        result.levels = levels
        result.acx = acx_check(mastered, self.sr).as_dict()
        result.acx["applies"] = self.preset == "acx"
        if beds != "full":
            result.warnings.append(f"beds={beds}: ambience/music layers were left out of this master")
        margin = tl.speech_to_bed_db
        levels["speech_to_bed_db"] = margin
        lo, hi = self.p.cfg["mix"].get("speech_to_bed_db", [14.0, 26.0])
        if margin is not None and margin < lo:
            result.warnings.append(f"beds sit only {margin} dB under speech (< {lo}): they may mask words; "
                                   "lower mix.ambience_db / mix.music_db or layer gains")
        elif margin is not None and margin > hi:
            result.warnings.append(f"beds sit {margin} dB under speech (> {hi}): nearly inaudible on small speakers; "
                                   "raise mix.ambience_db / mix.music_db or layer gains")
        files = write_chapter(mastered, self.sr, out_dir, chapter.slug, list(exp["formats"]) + ["wav"],
                              exp.get("mp3_bitrate", "192k"), title=chapter.title, album=self.p.title,
                              artist=self.p.cfg.get("author", ""), track=chapter.number)
        if stems:
            for name, bus in stems.items():
                files[f"stem_{name}"] = io.save(out_dir / "stems" / f"{chapter.slug}_{name}.wav", bus, self.sr)
        if exp.get("captions", True):
            cap = out_dir / f"{chapter.slug}.srt"
            cap.write_text(srt(tl.captions, narrator=self.p.default_speaker), encoding="utf-8")
            files["srt"] = cap
        result.files = {k: str(v) for k, v in files.items()}
        result.duration = round(len(mastered) / self.sr, 2)
        result.seconds_to_render = round(time.time() - t0, 1)
        return result

    def beds_mode(self) -> str:
        mode = str(self.p.cfg["mix"].get("beds", "auto"))
        if mode == "auto":
            # ACX's -60 dB noise-floor rule cannot be met under continuous beds
            return "off" if self.preset == "acx" else "full"
        return mode

    def add_room_tone(self, x: np.ndarray, seed: int) -> np.ndarray:
        level = float(self.p.cfg["mix"].get("room_tone_db", -66.0))
        if level <= -100:
            return x
        tone = procedural.render("room_tone", min(60.0, len(x) / self.sr), self.sr, seed=seed)
        from ..sound.resolve import loop_to

        tone = loop_to(tone, len(x), self.sr)
        tone = tone / (np.sqrt(np.mean(tone ** 2)) + 1e-9) * dsp.db_to_gain(level)
        return (x + tone).astype(np.float32)

    # -- book ---------------------------------------------------------------------------
    def render(self, chapters: list[int] | None = None, out_dir: Path | None = None, force: bool = False) -> dict:
        from .lint import lint

        out_dir = Path(out_dir or self.p.out_dir)
        script = parse_files(self.p.script_files(), self.p.default_speaker)
        issues = lint(self.p, script, self)
        errors = [d for d in issues if d.level == "error"]
        for d in issues:
            if d.level != "info":
                self.log(f"  {d}")
        if errors and not force:
            raise SystemExit(f"{len(errors)} script error(s); fix them or use --force")
        selected = [c for c in script.chapters if not chapters or c.number in chapters]
        results = []
        try:
            for ch in selected:
                results.append(self.render_chapter(ch, out_dir))
        finally:
            self.close()
        book = {"title": self.p.title, "preset": self.preset, "draft": bool(self.draft_backend),
                "chapters": [asdict(r) for r in results], "backend_fallbacks": self.fallback_notes,
                "cache": {"hits": self.cache.hits, "misses": self.cache.misses}}
        if "m4b" in self.p.cfg["export"]["formats"] and results and not chapters:
            m4b = out_dir / f"{_slug(self.p.title)}.m4b"
            build_m4b([(r.title, Path(r.files["wav"])) for r in results], m4b, self.p.title,
                      self.p.cfg.get("author", ""), self.p.cfg.get("narrator_credit", ""),
                      self.p.resolve(self.p.cfg.get("cover")), self.p.cfg["export"].get("m4b_bitrate", "96k"))
            book["m4b"] = str(m4b)
        (out_dir / "render_report.json").write_text(json.dumps(book, indent=1, default=str))
        (out_dir / "QA_SHEET.md").write_text(qa_sheet(book), encoding="utf-8")
        return book


def _slug(s: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_") or "book"


def qa_sheet(book: dict) -> str:
    """Human listening checklist: flagged takes first, then per-chapter delivery checks."""
    lines = [f"# QA sheet - {book['title']}", "", f"preset: {book['preset']}  draft: {book['draft']}", ""]
    if book.get("backend_fallbacks"):
        lines += ["## Backend fallbacks", ""] + [f"- {v}" for v in book["backend_fallbacks"].values()] + [""]
    for ch in book["chapters"]:
        acx = ch["acx"]
        label = "ACX" if acx.get("applies") else "ACX (informational, preset is not acx)"
        lines += [f"## Chapter {ch['number']}: {ch['title']}  ({ch['duration'] / 60:.1f} min)", "",
                  f"{label}: {'PASS' if acx.get('ok') else 'FAIL'} - " + ", ".join(
                      f"{c['name']} {c['value']}{'' if c['ok'] else ' (!)'}" for c in acx.get("checks", [])), ""]
        flagged = [u for u in ch["units"] if u["flagged"]]
        lines.append(f"Flagged takes: {len(flagged)} / {len(ch['units'])}")
        for u in flagged:
            lines.append(f"- [ ] `{u['uid']}` {u['speaker']} ({u['emotion']}) score {u['score']}: {u['text'][:90]}")
            for n in u["notes"]:
                lines.append(f"      - {n}")
        for w in ch.get("warnings", []):
            lines.append(f"- note: {w}")
        missing = [s for s in ch["sources"] if s.startswith("MISSING")]
        if missing:
            lines += ["", "Missing sounds:"] + [f"- {s}" for s in missing]
        lines += ["", "Listening checklist:", "- [ ] pronunciations (names, places)", "- [ ] emotion lands on key lines",
                  "- [ ] music/ambience never masks words (car + earbuds)", "- [ ] no clicks, cut-offs or doubled words", ""]
    return "\n".join(lines)
