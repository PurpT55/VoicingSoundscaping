"""Chatterbox backend (Resemble AI, MIT): original / turbo / nano / multilingual (V3).

* original + multilingual: ``exaggeration`` and ``cfg_weight`` from the emotion's arousal/intensity
* turbo / nano: native paralinguistic tags ([laugh], [chuckle], [cough], [sigh] ...)
* emotion banks must be the SAME voice (Chatterbox copies the clip's timbre and style)
"""

from __future__ import annotations

import os
import tempfile

from ..audio import io
from ..emotion import taxonomy as tx
from ..emotion.mapping import chatterbox_params
from .base import EXAGGERATION, REFERENCE_STYLE, Audio, Capabilities, SynthesisRequest, TTSBackend
from .worker import Engine

TURBO_TAGS = frozenset(n.id for n in tx.NONVERBALS if n.chatterbox)


class ChatterboxBackend(TTSBackend):
    name = "chatterbox"
    description = "Chatterbox (Resemble AI): exaggeration control, Turbo paralinguistic tags"
    license = "MIT (outputs watermarked with PerTh)"

    def __init__(self, **options):
        super().__init__(**options)
        self._engine: Engine | None = None

    @property
    def model(self) -> str:
        return str(self.options.get("model", "original"))

    def capabilities(self) -> Capabilities:
        turbo = self.model in ("turbo", "nano")
        mech = {REFERENCE_STYLE} | (set() if turbo else {EXAGGERATION})
        langs = ("*",) if self.model == "multilingual" else ("en",)
        return Capabilities(emotion=frozenset(mech), nonverbals=TURBO_TAGS if turbo else frozenset(),
                            emotion_reference=True, emotion_reference_scope="voice", cloning=True,
                            languages=langs, max_chars=300, sample_rate=24000)

    def available(self) -> tuple[bool, str]:
        if self.options.get("python"):
            ok = os.path.exists(os.path.expanduser(self.options["python"]))
            return ok, "worker python " + ("ok" if ok else "not found")
        try:
            __import__("chatterbox")
            return True, "chatterbox importable"
        except ImportError:
            return False, "pip install chatterbox-tts (or set backends.config.chatterbox.python)"

    def engine(self) -> Engine:
        if self._engine is None:
            init = {"model": self.model}
            for k in ("device", "t3_model"):
                if k in self.options:
                    init[k] = self.options[k]
            py = os.path.expanduser(self.options["python"]) if self.options.get("python") else None
            self._engine = Engine("chatterbox", init, python=py)
        return self._engine

    def render_text(self, req: SynthesisRequest) -> str:
        out = []
        for p in req.pieces:
            if p.kind == "nonverbal":
                nv = tx.get_nonverbal(p.nonverbal)
                if nv.chatterbox and self.model in ("turbo", "nano"):
                    out.append(nv.chatterbox)
            else:
                out.append(p.text)
        return " ".join(out).strip()

    def build_args(self, req: SynthesisRequest, out: str) -> dict:
        args = {"text": self.render_text(req), "output_path": out, "seed": req.seed}
        ref = req.emotion_reference or req.voice.reference
        if ref:
            args["audio_prompt_path"] = str(ref)
        if self.model in ("original", "multilingual"):
            args.update(chatterbox_params(req.blend))
            for k in ("exaggeration", "cfg_weight"):
                if k in req.params:
                    args[k] = float(req.params[k])
        if self.model == "multilingual":
            args["language_id"] = req.language[:2]
        if "temperature" in req.params:
            args["temperature"] = float(req.params["temperature"])
        return args

    def synthesize(self, req: SynthesisRequest) -> Audio:
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "out.wav")
            args = self.build_args(req, out)
            self.engine().call("synth", args)
            x, sr = io.load(out, mono=True)
        return Audio(x, sr, {"backend": self.name, "args": {k: v for k, v in args.items() if k != "output_path"}})

    def close(self) -> None:
        if self._engine:
            self._engine.close()
            self._engine = None
