"""Local sound library: index folders of SFX / ambience / music files and find them by name or tags.

Point it at your own collections (Sonniss GDC bundles, BBC Sound Effects under their licence,
freesound CC0 downloads, your own field recordings ...). Files are matched by normalised stem
(``door_creak_02.wav`` -> ``door_creak``) and by path/filename tokens, and numbered variants are
rotated so repeated effects don't sound copy-pasted.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from pathlib import Path

AUDIO_EXT = {".wav", ".flac", ".ogg", ".mp3", ".aif", ".aiff", ".m4a", ".opus"}


def _norm(s: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")
    return re.sub(r"_?\d+$", "", s)  # drop variant numbers


def _tokens(s: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", s.lower()) if t and not t.isdigit()}


@dataclass
class Entry:
    path: Path
    key: str
    tokens: set[str] = field(default_factory=set)


class AssetLibrary:
    def __init__(self, dirs: list[Path]):
        self.dirs = [Path(d) for d in dirs if d]
        self.entries: list[Entry] = []
        for d in self.dirs:
            if not d.exists():
                continue
            for p in sorted(d.rglob("*")):
                if p.suffix.lower() in AUDIO_EXT and p.is_file():
                    rel = p.relative_to(d)
                    toks = _tokens(str(rel.with_suffix("")))
                    self.entries.append(Entry(p, _norm(p.stem), toks))
        self.by_key: dict[str, list[Entry]] = {}
        for e in self.entries:
            self.by_key.setdefault(e.key, []).append(e)

    def __len__(self) -> int:
        return len(self.entries)

    def find_all(self, name: str) -> list[Path]:
        key = _norm(name)
        if key in self.by_key:
            return [e.path for e in self.by_key[key]]
        want = _tokens(name)
        if want:
            scored = []
            for e in self.entries:
                if want <= e.tokens:
                    scored.append((len(e.tokens - want), str(e.path), e.path))
            if scored:
                scored.sort()
                best = scored[0][0]
                return [p for s, _, p in scored if s == best]
        close = difflib.get_close_matches(key, list(self.by_key), n=1, cutoff=0.82)
        if close:
            return [e.path for e in self.by_key[close[0]]]
        return []

    def find(self, name: str, variant: int = 0) -> Path | None:
        hits = self.find_all(name)
        return hits[variant % len(hits)] if hits else None
