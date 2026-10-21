"""Resolve a named sound (sfx / ambience / music) to audio.

Resolution order per kind (configurable under ``assets.generators`` in project.yaml):

1. project / user libraries (``assets/*_dirs``) - real recordings always win
2. configured generators in order, e.g. ``[acestep, procedural]`` for music or
   ``[moss_sfx, procedural]`` for effects; generated audio is cached by content hash
3. ``procedural`` - the built-in synthesiser (always available)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..audio import dsp, io
from ..audio.loudness import normalize_lufs
from ..tts.base import stable_hash
from . import procedural
from .generators import GeneratorError, make_generator
from .library import AssetLibrary

MOOD_WORDS = {
    "wonder": "wonder", "magic": "wonder", "magical": "wonder", "dream": "wonder", "awe": "wonder", "stars": "wonder",
    "warm": "warm", "happy": "warm", "cosy": "warm", "cozy": "warm", "home": "warm", "gentle": "warm", "joy": "warm",
    "sad": "melancholy", "melancholy": "melancholy", "grief": "melancholy", "loss": "melancholy", "rain": "melancholy",
    "mystery": "mystery", "mysterious": "mystery", "secret": "mystery", "night": "mystery",
    "tension": "tension", "scary": "tension", "danger": "tension", "fear": "tension", "dark": "tension", "suspense": "tension",
    "lullaby": "lullaby", "sleep": "lullaby", "bedtime": "lullaby", "music_box": "lullaby",
    "adventure": "adventure", "journey": "adventure", "quest": "adventure",
    "triumph": "triumph", "victory": "triumph", "epic": "triumph", "heroic": "triumph",
}


def mood_from_text(text: str) -> str:
    for word in re.findall(r"[a-z_]+", text.lower()):
        if word in MOOD_WORDS:
            return MOOD_WORDS[word]
    return "warm"


@dataclass
class Sound:
    audio: np.ndarray  # stereo float32, ~-20 LUFS
    sr: int
    source: str        # "library:<path>" | "generator:<name>" | "procedural:<name>"


def loop_to(x: np.ndarray, n: int, sr: int, xfade: float = 2.0) -> np.ndarray:
    """Extend a (stereo) clip to ``n`` samples by looping with equal-power crossfades."""
    if len(x) >= n:
        return x[:n]
    k = min(int(xfade * sr), len(x) // 3)
    if k < 16:
        reps = -(-n // max(1, len(x)))
        return np.concatenate([x] * reps)[:n]
    t = np.linspace(0, np.pi / 2, k, dtype=np.float32)[:, None]
    fin, fout = np.sin(t), np.cos(t)
    out = x.copy()
    while len(out) < n:
        head = out[:-k]
        joint = out[-k:] * fout + x[:k] * fin
        out = np.concatenate([head, joint, x[k:]])
    return out[:n]


class SoundResolver:
    KINDS = ("sfx", "ambience", "music")

    def __init__(self, project, sr: int, cache_dir: Path | None = None):
        self.project = project
        self.sr = sr
        self.cache_dir = Path(cache_dir or project.cache_dir) / "sound"
        a = project.cfg["assets"]
        res = project.resolve
        sfx_dirs = [res(d) for d in a.get("sfx_dirs", [])]
        self.libs = {
            "sfx": AssetLibrary(sfx_dirs),
            "ambience": AssetLibrary([res(d) for d in a.get("ambience_dirs", [])] + sfx_dirs),
            "music": AssetLibrary([res(d) for d in a.get("music_dirs", [])]),
        }
        self.order = {k: list(a["generators"].get(k, ["procedural"])) for k in self.KINDS}
        self._gens: dict[str, object] = {}
        self._checked: dict[str, tuple[bool, str]] = {}
        self.log: list[str] = []

    # -- introspection (lint) -----------------------------------------------------------
    def describe(self, kind: str, name: str, prompt: str | None = None) -> str | None:
        if self.libs[kind].find(name):
            return f"library:{self.libs[kind].find(name)}"
        for g in self.order[kind]:
            if g == "procedural":
                if procedural.lookup(name) or kind == "music":
                    return f"procedural:{name}"
                continue
            if prompt or kind in ("music", "ambience", "sfx"):
                return f"generator:{g}"
        return None

    # -- resolution ---------------------------------------------------------------------
    def get(self, kind: str, name: str, seconds: float | None = None, prompt: str | None = None,
            seed: int | str = 0, variant: int = 0, **params) -> Sound:
        lib_path = self.libs[kind].find(name, variant)
        if lib_path:
            x, _ = io.load(lib_path, sr=self.sr)
            x = io.to_stereo(x)
            if kind != "sfx":
                x = normalize_lufs(x, self.sr, -20.0)
                if seconds:
                    x = loop_to(x, int(seconds * self.sr), self.sr)
            else:
                x = normalize_lufs(x, self.sr, -20.0)
            return Sound(x, self.sr, f"library:{lib_path}")
        errors = []
        for g in self.order[kind]:
            if g == "procedural":
                snd = self._procedural(kind, name, seconds, prompt, seed, params)
                if snd is not None:
                    return snd
                continue
            try:
                return self._generated(g, kind, name, seconds, prompt, seed, params)
            except (GeneratorError, OSError, KeyError) as exc:
                errors.append(f"{g}: {exc}")
                self.log.append(f"generator {g} failed for {kind} '{name}': {exc}")
        raise LookupError(f"cannot resolve {kind} '{name}'" + (f" ({'; '.join(errors)})" if errors else ""))

    def _procedural(self, kind, name, seconds, prompt, seed, params) -> Sound | None:
        proc = procedural.lookup(name)
        if proc is None and kind == "music":
            mood = mood_from_text(f"{name} {prompt or ''}")
            proc = procedural.lookup(f"pad_{mood}")
        if proc is None:
            return None
        secs = seconds if (seconds and proc.kind != "oneshot") else float(params.pop("duration", proc.default_seconds))
        allowed = {k: v for k, v in params.items() if k in ("intensity", "strength", "calm", "size", "crickets", "birds",
                                                             "people", "bpm", "distance", "pitch", "count", "notes", "mood")}
        x = proc.fn(float(secs), self.sr, procedural.rng_for(proc.name, seed, sorted(allowed.items())), **allowed)
        return Sound(x, self.sr, f"procedural:{proc.name}")

    def _generator(self, name: str):
        if name not in self._gens:
            cfg = dict(self.project.generator_options(name))
            if not cfg and name == "acestep":
                cfg = {"type": "acestep"}
            cfg.setdefault("type", name if name in ("acestep", "command", "http") else cfg.get("type", name))
            self._gens[name] = make_generator(name, cfg)
        return self._gens[name]

    def _generated(self, gname, kind, name, seconds, prompt, seed, params) -> Sound:
        gen = self._generator(gname)
        if kind not in getattr(gen, "kinds", self.KINDS):
            raise GeneratorError(f"{gname} does not produce {kind}")
        if gname not in self._checked:
            ok, why = gen.available()
            self._checked[gname] = (ok, why)
            if not ok:
                self.log.append(f"generator {gname} unavailable: {why}")
        if not self._checked[gname][0]:
            raise GeneratorError(self._checked[gname][1])
        text = prompt or name.replace("_", " ")
        gen_secs = float(params.get("duration") or seconds or 8.0)
        if kind == "music":
            gen_secs = min(gen_secs, float(params.get("max_generate", 240.0)))
        elif kind == "ambience":
            gen_secs = min(gen_secs, float(params.get("max_generate", 30.0)))
        key = stable_hash({"g": gname, "k": kind, "t": text, "s": round(gen_secs, 1), "seed": str(seed),
                           "p": {k: v for k, v in params.items() if k not in ("gain", "fade", "offset", "wait", "pan")}})
        out = self.cache_dir / kind / f"{key[:24]}.wav"
        if not out.exists():
            gen.generate(text, gen_secs, out, seed=int(key[:8], 16), **params)
        x, _ = io.load(out, sr=self.sr)
        x = normalize_lufs(io.to_stereo(x), self.sr, -20.0)
        if kind != "sfx" and seconds:
            x = loop_to(dsp.fade(x, self.sr, 0.05, 0.05), int(seconds * self.sr), self.sr)
        return Sound(x, self.sr, f"generator:{gname}")
