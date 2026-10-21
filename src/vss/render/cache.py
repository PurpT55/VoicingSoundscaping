"""Content-addressed take cache: re-renders only regenerate what changed.

Key = hash(backend fingerprint + request fingerprint + take index). Takes are stored as 24-bit
FLAC with a JSON sidecar (QA scores, render text, backend args).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..audio import io
from ..tts.base import Audio, stable_hash


class TakeCache:
    def __init__(self, root: Path, enabled: bool = True):
        self.root = Path(root) / "takes"
        self.enabled = enabled
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(backend_fp: dict, request_fp: dict, take: int) -> str:
        return stable_hash({"b": backend_fp, "r": request_fp, "t": take})[:32]

    def _paths(self, key: str) -> tuple[Path, Path]:
        d = self.root / key[:2]
        return d / f"{key}.flac", d / f"{key}.json"

    def get(self, key: str) -> tuple[Audio, dict] | None:
        if not self.enabled:
            return None
        audio_p, meta_p = self._paths(key)
        if not audio_p.exists():
            self.misses += 1
            return None
        x, sr = io.load(audio_p, mono=True)
        meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
        self.hits += 1
        return Audio(x, sr, meta.get("audio_meta", {})), meta

    def put(self, key: str, audio: Audio, meta: dict) -> None:
        if not self.enabled:
            return
        audio_p, meta_p = self._paths(key)
        audio_p.parent.mkdir(parents=True, exist_ok=True)
        io.save(audio_p, np.asarray(audio.samples, dtype=np.float32), audio.sr, subtype="PCM_24")
        meta = dict(meta)
        meta["audio_meta"] = _jsonable(audio.meta)
        meta_p.write_text(json.dumps(meta, indent=1, default=str))

    def update_meta(self, key: str, extra: dict) -> None:
        _, meta_p = self._paths(key)
        if meta_p.exists():
            meta = json.loads(meta_p.read_text())
            meta.update(extra)
            meta_p.write_text(json.dumps(meta, indent=1, default=str))


def _jsonable(d: dict) -> dict:
    try:
        json.dumps(d, default=str)
        return d
    except (TypeError, ValueError):
        return {k: str(v) for k, v in d.items()}
