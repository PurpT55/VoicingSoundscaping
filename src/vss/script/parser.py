"""Parser for the ``.vss`` story-script format (see docs/SCRIPT_FORMAT.md).

Quick reference::

    ---                                   optional YAML front matter
    title: The Lantern Keeper
    ---
    # Chapter One: The Light             chapter heading
    ## On the cliff                      scene heading (resets ambience and space)
    @scene ambience="sea_waves:-6, wind_soft:-12" space=outdoor_open
    @music start wonder prompt="gentle celtic harp, wonder" gain=-20 fade=4
    @sfx gull_cry gain=-10 wait=false
    @pause 1.5
    @narrator NARRATOR                   default speaker for untagged prose

    Untagged prose is read by the narrator.
    MIRA (whisper, fear 0.6, "afraid to wake something", rate=0.95): Is anyone up there?
    TOM (dry_humor): Only me. <chuckle> [warm] Come up, the kettle's on.
    // comment
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Any

from ..emotion import taxonomy as tx
from ..emotion.mapping import Blend, EmotionRef, clamp01, DEFAULT_INTENSITY
from .model import (
    Chapter, Diagnostic, Directive, EmotionShift, Line, NonverbalEvent, PauseEvent, Scene,
    Script, SfxEvent, Source, Text,
)

DIRECTIVES = {"scene", "ambience", "music", "sfx", "pause", "space", "narrator"}
PAUSES = {"beat": 0.3, "pause": 0.6, "long_pause": 1.5, "longpause": 1.5}
_SPEAKER = re.compile(r"^([A-Z][A-Z0-9_'.\- ]{0,40}?)\s*(?=[(:])")
_INLINE = re.compile(
    r"(?P<esc>\\[\[\]<>{}*\\])"
    r"|\[(?P<emo>[^\]]*)\]"
    r"|<(?P<evt>[^>]*)>"
    r"|\{(?P<written>[^}|]*)\|(?P<spoken>[^}]*)\}"
    r"|\*(?P<emph>[^*\n]+)\*"
)


def parse_value(v: str) -> Any:
    low = v.lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    try:
        if re.fullmatch(r"[+-]?\d+", v):
            return int(v)
        if re.fullmatch(r"[+-]?(\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?", v):
            return float(v)
    except ValueError:
        pass
    return v


def split_top(s: str, sep: str = ",") -> list[str]:
    """Split on ``sep`` outside double quotes."""
    out, buf, q = [], [], False
    for ch in s:
        if ch == '"':
            q = not q
        if ch == sep and not q:
            out.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    out.append("".join(buf))
    return [x.strip() for x in out if x.strip()]


class _Parser:
    def __init__(self, filename: str, chapter_offset: int, default_speaker: str):
        self.filename = filename
        self.diags: list[Diagnostic] = []
        self.chapters: list[Chapter] = []
        self.chapter_offset = chapter_offset
        self.default_speaker = default_speaker
        self.meta: dict[str, Any] = {}

    # -- diagnostics --------------------------------------------------------------------
    def _src(self, n: int) -> Source:
        return Source(self.filename, n)

    def err(self, n: int, msg: str) -> None:
        self.diags.append(Diagnostic("error", msg, self._src(n)))

    def warn(self, n: int, msg: str) -> None:
        self.diags.append(Diagnostic("warning", msg, self._src(n)))

    # -- structure ----------------------------------------------------------------------
    def chapter(self, n: int) -> Chapter:
        if not self.chapters:
            title = str(self.meta.get("chapter") or self.meta.get("title") or Path(self.filename).stem)
            self.chapters.append(Chapter(self.chapter_offset + 1, title, source=self._src(n)))
        return self.chapters[-1]

    def parse(self, text: str) -> None:
        lines = text.splitlines()
        i = 0
        if lines and lines[0].strip() == "---":
            end = next((k for k in range(1, len(lines)) if lines[k].strip() == "---"), None)
            if end is None:
                self.err(1, "front matter opened with '---' but never closed")
            else:
                import yaml

                try:
                    self.meta = yaml.safe_load("\n".join(lines[1:end])) or {}
                except yaml.YAMLError as exc:
                    self.err(1, f"front matter is not valid YAML: {exc}")
                i = end + 1
        current: Line | None = None
        for n in range(i, len(lines)):
            raw = lines[n]
            lineno = n + 1
            s = raw.strip()
            if not s:
                current = None
                continue
            if s.startswith("//"):
                continue
            if s.startswith("# "):
                current = None
                self.chapters.append(Chapter(self.chapter_offset + len(self.chapters) + 1, s[2:].strip(),
                                             source=self._src(lineno)))
                continue
            if s.startswith("## "):
                current = None
                self.chapter(lineno).events.append(Scene(s[3:].strip(), self._src(lineno)))
                continue
            if s.startswith("@"):
                current = None
                d = self.directive(s[1:], lineno)
                if d:
                    if d.name == "narrator":
                        if d.args:
                            self.default_speaker = d.args[0].upper()
                    self.chapter(lineno).events.append(d)
                continue
            m = _SPEAKER.match(s)
            if m and not s.startswith(("(", "[")):
                parsed = self.speaker_line(s, m, lineno)
                if parsed:
                    current = parsed
                    self.chapter(lineno).events.append(parsed)
                    continue
            if current is not None:
                current.items.extend(self.inline(s, lineno, current.blend))
                continue
            current = Line(self.default_speaker, Blend(), [], source=self._src(lineno), narration=True)
            current.items = self.inline(s, lineno, current.blend)
            self.chapter(lineno).events.append(current)

    def directive(self, body: str, n: int) -> Directive | None:
        try:
            tokens = shlex.split(body, posix=True)
        except ValueError as exc:
            self.err(n, f"cannot parse directive: {exc}")
            return None
        if not tokens:
            self.err(n, "empty directive")
            return None
        name = tokens[0].lower()
        if name not in DIRECTIVES:
            self.err(n, f"unknown directive '@{name}' (known: {', '.join(sorted(DIRECTIVES))})")
            return None
        args, params = [], {}
        for t in tokens[1:]:
            if "=" in t and not t.startswith("="):
                k, v = t.split("=", 1)
                params[k.strip().lower()] = parse_value(v)
            else:
                args.append(t)
        return Directive(name, args, params, self._src(n))

    def speaker_line(self, s: str, m: re.Match, n: int) -> Line | None:
        speaker = m.group(1).strip()
        rest = s[m.end():]
        blend, params = Blend(), {}
        if rest.startswith("("):
            close = _matching_paren(rest)
            if close < 0:
                self.err(n, "unclosed '(' in speaker directions")
                return None
            blend, params = self.directions(rest[1:close], n)
            rest = rest[close + 1:].lstrip()
        if not rest.startswith(":"):
            return None
        line = Line(speaker.upper(), blend, [], params, self._src(n))
        line.items = self.inline(rest[1:].strip(), n, blend)
        return line

    def directions(self, body: str, n: int) -> tuple[Blend, dict[str, Any]]:
        parts: list[EmotionRef] = []
        params: dict[str, Any] = {}
        direction = ""
        for item in split_top(body):
            if item.startswith('"') and item.endswith('"') and len(item) >= 2:
                direction = item[1:-1].strip()
                continue
            if "=" in item:
                k, v = item.split("=", 1)
                params[k.strip().lower()] = parse_value(v.strip().strip('"'))
                continue
            ref = self.emotion_ref(item, n)
            if ref:
                parts.append(ref)
        return Blend(tuple(parts), direction), params

    def emotion_ref(self, item: str, n: int) -> EmotionRef | None:
        m = re.fullmatch(r"([A-Za-z_\- ]+?)\s*(\d*\.?\d+)?", item.strip())
        if not m:
            self.err(n, f"cannot read emotion '{item}' (expected 'name' or 'name 0.7')")
            return None
        name, inten = m.group(1), m.group(2)
        e = tx.find(name)
        if e is None:
            hint = tx.suggest(name)
            self.err(n, f"unknown emotion '{name}'" + (f" (did you mean: {', '.join(hint)}?)" if hint else ""))
            return None
        value = float(inten) if inten else DEFAULT_INTENSITY
        if value > 1.0:
            self.warn(n, f"intensity {value} for '{name}' clamped to 1.0")
        return EmotionRef(e, clamp01(value))

    def inline(self, s: str, n: int, base: Blend) -> list:
        items: list = []
        spoken, shown = [], []

        def flush():
            if "".join(spoken).strip():
                items.append(Text(_squash("".join(spoken)), _squash("".join(shown))))
            spoken.clear()
            shown.clear()

        pos = 0
        for m in _INLINE.finditer(s):
            spoken.append(s[pos:m.start()])
            shown.append(s[pos:m.start()])
            pos = m.end()
            if m.group("esc"):
                spoken.append(m.group("esc")[1])
                shown.append(m.group("esc")[1])
            elif m.group("emo") is not None:
                flush()
                body = m.group("emo").strip()
                if body.lower() in ("reset", "/", ""):
                    items.append(EmotionShift(base, reset=True))
                else:
                    blend, params = self.directions(body, n)
                    if params:
                        self.warn(n, f"parameters inside [..] are ignored: {params}")
                    items.append(EmotionShift(blend))
            elif m.group("evt") is not None:
                flush()
                evt = self.event(m.group("evt").strip(), n)
                if evt:
                    items.append(evt)
            elif m.group("written") is not None:
                shown.append(m.group("written"))
                spoken.append(m.group("spoken"))
            elif m.group("emph") is not None:
                shown.append(m.group("emph"))
                spoken.append(m.group("emph"))
        spoken.append(s[pos:])
        shown.append(s[pos:])
        flush()
        return items

    def event(self, body: str, n: int):
        try:
            tokens = shlex.split(body)
        except ValueError:
            tokens = body.split()
        if not tokens:
            self.err(n, "empty <> event")
            return None
        head = tokens[0].lower().replace("-", "_")
        if head == "long" and len(tokens) > 1 and tokens[1].lower() == "pause":
            return PauseEvent(PAUSES["long_pause"])
        if head in PAUSES:
            if len(tokens) > 1:
                try:
                    return PauseEvent(float(tokens[1].rstrip("s")))
                except ValueError:
                    self.err(n, f"bad pause length '{tokens[1]}'")
                    return None
            return PauseEvent(PAUSES[head])
        if head == "sfx":
            if len(tokens) < 2:
                self.err(n, "<sfx> needs a name, e.g. <sfx door_creak>")
                return None
            params = {}
            for t in tokens[2:]:
                if "=" in t:
                    k, v = t.split("=", 1)
                    params[k.lower()] = parse_value(v)
            return SfxEvent(tokens[1], params)
        nv = tx.find_nonverbal(head)
        if nv is None:
            hint = tx.suggest(head)
            self.err(n, f"unknown event <{body}>" + (f" (did you mean: {', '.join(hint)}?)" if hint else ""))
            return None
        return NonverbalEvent(nv.id)


def _matching_paren(s: str) -> int:
    depth, q = 0, False
    for i, ch in enumerate(s):
        if ch == '"':
            q = not q
        elif not q and ch == "(":
            depth += 1
        elif not q and ch == ")":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _squash(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def parse_text(text: str, filename: str = "<text>", *, chapter_offset: int = 0,
               default_speaker: str = "NARRATOR") -> Script:
    p = _Parser(filename, chapter_offset, default_speaker)
    p.parse(text)
    return Script(p.chapters, p.meta, p.diags)


def parse_files(paths: list[Path], default_speaker: str = "NARRATOR") -> Script:
    script = Script()
    for path in paths:
        part = parse_text(Path(path).read_text(encoding="utf-8"), str(path),
                          chapter_offset=len(script.chapters), default_speaker=default_speaker)
        script.chapters.extend(part.chapters)
        script.diagnostics.extend(part.diagnostics)
        for k, v in part.meta.items():
            script.meta.setdefault(k, v)
    return script
