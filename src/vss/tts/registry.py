"""Backend registry. Classes are imported lazily so optional dependencies stay optional."""

from __future__ import annotations

import importlib
from typing import Any

from .base import TTSBackend

BACKENDS: dict[str, str] = {
    "placeholder": "vss.tts.simple:PlaceholderBackend",
    "espeak": "vss.tts.simple:EspeakBackend",
    "kokoro": "vss.tts.simple:KokoroBackend",
    "indextts": "vss.tts.indextts:IndexTTSBackend",
    "fish": "vss.tts.fish:FishBackend",
    "openai_speech": "vss.tts.openai_speech:OpenAISpeechBackend",
    "qwen3": "vss.tts.qwen3:Qwen3Backend",
    "chatterbox": "vss.tts.chatterbox:ChatterboxBackend",
    "step_edit": "vss.tts.step_edit:StepEditBackend",
}


def backend_class(kind: str) -> type[TTSBackend]:
    if kind not in BACKENDS:
        raise KeyError(f"unknown backend '{kind}' (known: {', '.join(BACKENDS)})")
    mod, cls = BACKENDS[kind].split(":")
    return getattr(importlib.import_module(mod), cls)


def create(name: str, options: dict[str, Any] | None = None, project=None) -> TTSBackend:
    """Create backend ``name``. A configured name may alias a kind: ``higgs: {type: openai_speech, ...}``."""
    opts = dict(options if options is not None else (project.backend_options(name) if project else {}))
    kind = opts.pop("type", name)
    if kind == "step_edit":
        base_name = opts.get("base", "indextts")
        base = create(base_name, None, project) if project else create(base_name, {})
        b = backend_class(kind)(base_backend=base, **opts)
    else:
        b = backend_class(kind)(**opts)
    b.name = name if name != kind else b.name
    return b
