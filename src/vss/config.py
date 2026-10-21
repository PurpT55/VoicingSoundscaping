"""Project configuration (``project.yaml``): cast, voices, backends, render, mix and master settings."""

from __future__ import annotations

import copy
import glob
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .emotion.mapping import Blend

DEFAULTS: dict[str, Any] = {
    "title": "Untitled",
    "author": "",
    "narrator_credit": "",
    "language": "en",
    "cover": None,
    "script": ["script/*.vss"],
    "default_speaker": "NARRATOR",
    "backends": {
        "default": "espeak",
        "draft": "espeak",
        "routing": {},
        "config": {},
    },
    "voices": {},
    "cast": {},
    "banks": {"emotions": None, "nonverbals": None},
    "lexicon": {},
    "render": {
        "sample_rate": 44100,
        "takes": 1,
        "max_takes": 4,
        "seed": 1234,
        "cache_dir": ".vss_cache",
        "out_dir": "out",
        "chunk": {"target_chars": 220, "max_chars": 300},
        "pauses": {
            "sentence": 0.35, "paragraph": 0.8, "speaker_change": 0.5, "scene": 2.0,
            "chapter_head": 0.75, "chapter_tail": 3.0, "after_title": 1.2,
        },
        "breaths": "off",           # off | bank | procedural
        "narrator_in_space": False,  # narrator stays dry ("outside" the scene) by default
        "announce_chapters": True,
        "nonverbal_fallback": "bank",  # bank | drop
    },
    "qa": {
        "asr": "none",              # none | faster-whisper
        "asr_model": "small.en",
        "max_wer": 0.15,
        "emotion_model": "none",    # none | emotion2vec
        "speaker_model": "none",
        "min_score": 0.45,
    },
    "mix": {
        "dialog_lufs": -20.0,
        # bus levels relative to the dialogue target; beds end up ~16-22 dB under speech
        "ambience_db": -12.0,
        "music_db": -11.0,
        "sfx_db": -6.0,
        "duck": {"depth_db": 6.0, "attack": 0.15, "release": 0.9, "threshold_db": -42.0, "music_extra_db": 3.0},
        "speech_to_bed_db": [14.0, 26.0],  # QA window: below -> beds mask words, above -> beds inaudible
        "room_tone_db": -66.0,
        "beds": "auto",             # auto (off for the acx preset) | full | off | music | ambience
    },
    "master": {"preset": "acx"},
    "assets": {
        "sfx_dirs": ["assets/sfx"],
        "music_dirs": ["assets/music"],
        "ambience_dirs": ["assets/ambience"],
        "generators": {
            "sfx": ["procedural"],
            "ambience": ["procedural"],
            "music": ["procedural"],
            "config": {},
        },
    },
    "export": {
        "formats": ["wav", "mp3"],  # wav | flac | mp3 | m4b
        "mp3_bitrate": "192k",
        "m4b_bitrate": "96k",
        "captions": True,            # .srt per chapter (read-along)
        "stems": False,              # also write dialog/ambience/music/sfx stems (for a DAW)
    },
}


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


@dataclass
class Voice:
    name: str
    description: str = ""
    reference: Path | None = None
    reference_text: str = ""
    language: str = "en"
    backend: str | None = None
    emotion_bank: Path | None = None
    nonverbal_bank: Path | None = None
    options: dict[str, dict[str, Any]] = field(default_factory=dict)
    chain: str = "dialog"
    gain: float = 0.0


@dataclass
class CastMember:
    speaker: str
    voice: str
    default: Blend = Blend()
    pan: float = 0.0
    backend: str | None = None
    params: dict[str, Any] = field(default_factory=dict)


class Project:
    def __init__(self, root: Path, cfg: dict[str, Any], path: Path | None = None):
        self.root = Path(root)
        self.cfg = cfg
        self.path = path
        self.warnings: list[str] = []
        self.voices = {name: self._voice(name, v or {}) for name, v in cfg["voices"].items()}
        self.cast = {spk.upper(): self._cast(spk.upper(), c or {}) for spk, c in cfg["cast"].items()}

    # -- loading ------------------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path) -> "Project":
        path = Path(path)
        if path.is_dir():
            path = path / "project.yaml"
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        unknown = set(data) - set(DEFAULTS)
        proj = cls(path.parent, deep_merge(DEFAULTS, data), path)
        for k in sorted(unknown):
            proj.warnings.append(f"unknown top-level key '{k}' in {path.name}")
        return proj

    @classmethod
    def from_dict(cls, data: dict[str, Any], root: str | Path = ".") -> "Project":
        return cls(Path(root), deep_merge(DEFAULTS, data))

    def resolve(self, p: str | Path | None) -> Path | None:
        if p in (None, ""):
            return None
        p = Path(os.path.expanduser(str(p)))
        return p if p.is_absolute() else (self.root / p)

    def _voice(self, name: str, v: dict) -> Voice:
        return Voice(
            name=name,
            description=v.get("description", ""),
            reference=self.resolve(v.get("reference")),
            reference_text=v.get("reference_text", ""),
            language=v.get("language", self.cfg["language"]),
            backend=v.get("backend"),
            emotion_bank=self.resolve(v.get("emotion_bank")),
            nonverbal_bank=self.resolve(v.get("nonverbal_bank")),
            options=v.get("options", {}) or {},
            chain=v.get("chain", "dialog"),
            gain=float(v.get("gain", 0.0)),
        )

    def _cast(self, speaker: str, c: dict) -> CastMember:
        voice = c.get("voice", speaker.lower())
        if voice not in self.voices:
            self.warnings.append(f"cast '{speaker}' uses voice '{voice}' which is not defined under voices:")
            self.voices[voice] = Voice(name=voice, language=self.cfg["language"])
        try:
            default = Blend.parse(str(c.get("default", "") or ""))
        except (KeyError, ValueError) as exc:
            self.warnings.append(f"cast '{speaker}': bad default emotion: {exc}")
            default = Blend()
        params = {k: v for k, v in c.items() if k not in ("voice", "default", "pan", "backend")}
        return CastMember(speaker, voice, default, float(c.get("pan", 0.0)), c.get("backend"), params)

    # -- accessors ----------------------------------------------------------------------
    @property
    def title(self) -> str:
        return self.cfg["title"]

    @property
    def language(self) -> str:
        return self.cfg["language"]

    @property
    def default_speaker(self) -> str:
        return str(self.cfg["default_speaker"]).upper()

    def script_files(self) -> list[Path]:
        pats = self.cfg["script"]
        pats = [pats] if isinstance(pats, str) else pats
        files: list[Path] = []
        for pat in pats:
            matches = sorted(glob.glob(str(self.resolve(pat))))
            files.extend(Path(m) for m in matches if Path(m) not in files)
        return files

    def member(self, speaker: str) -> CastMember:
        speaker = speaker.upper()
        if speaker not in self.cast:
            voice = speaker.lower()
            if voice not in self.voices:
                self.voices[voice] = Voice(name=voice, language=self.language)
            self.cast[speaker] = CastMember(speaker, voice)
        return self.cast[speaker]

    def voice_of(self, speaker: str) -> Voice:
        return self.voices[self.member(speaker).voice]

    def backend_options(self, name: str) -> dict[str, Any]:
        opts = dict(self.cfg["backends"]["config"].get(name, {}) or {})
        for key in ("model_dir", "model", "voices", "checkpoint", "repo_dir", "python"):
            if key in opts and isinstance(opts[key], str) and not opts[key].startswith(("http:", "https:")):
                opts[key] = str(self.resolve(opts[key]))
        return opts

    def generator_options(self, name: str) -> dict[str, Any]:
        return dict(self.cfg["assets"]["generators"]["config"].get(name, {}) or {})

    @property
    def cache_dir(self) -> Path:
        return self.resolve(self.cfg["render"]["cache_dir"])

    @property
    def out_dir(self) -> Path:
        return self.resolve(self.cfg["render"]["out_dir"])

    def bank_dir(self, kind: str) -> Path | None:
        return self.resolve(self.cfg["banks"].get(kind))
