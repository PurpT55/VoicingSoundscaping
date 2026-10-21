"""Qwen3-TTS backend (Apache-2.0, 10 languages).

Modes:
* ``custom_voice`` - 9 premium speakers + natural-language ``instruct`` (emotion, style, prosody)
* ``voice_design`` - voice from a text description (voice.description) + emotion instruct
* ``clone``        - 3-second cloning from voice.reference (no instruct: emotion follows the text)
"""

from __future__ import annotations

import os
import tempfile

from ..audio import io
from ..emotion.mapping import instruct_text
from .base import INSTRUCT, REFERENCE_STYLE, Audio, BackendUnavailable, Capabilities, SynthesisRequest, TTSBackend
from .worker import Engine

MODELS = {
    "custom_voice": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "voice_design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "clone": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
}


class Qwen3Backend(TTSBackend):
    name = "qwen3"
    description = "Qwen3-TTS: instruct-controlled emotion, voice design, cloning"
    license = "Apache-2.0"

    def __init__(self, **options):
        super().__init__(**options)
        self._engine: Engine | None = None

    @property
    def mode(self) -> str:
        return str(self.options.get("mode", "custom_voice"))

    def capabilities(self) -> Capabilities:
        mech = {REFERENCE_STYLE} if self.mode == "clone" else {INSTRUCT}
        return Capabilities(emotion=frozenset(mech), cloning=self.mode == "clone",
                            emotion_reference=self.mode == "clone", emotion_reference_scope="voice",
                            languages=("zh", "en", "ja", "ko", "de", "fr", "ru", "pt", "es", "it"),
                            max_chars=300, sample_rate=24000)

    def available(self) -> tuple[bool, str]:
        if self.options.get("python"):
            ok = os.path.exists(os.path.expanduser(self.options["python"]))
            return ok, "worker python " + ("ok" if ok else "not found")
        try:
            __import__("qwen_tts")
            return True, "qwen-tts importable"
        except ImportError:
            return False, "pip install qwen-tts (or set backends.config.qwen3.python)"

    def engine(self) -> Engine:
        if self._engine is None:
            o = self.options
            init = {"model": o.get("model", MODELS.get(self.mode, MODELS["custom_voice"]))}
            for k in ("device", "dtype", "attn_implementation"):
                if k in o:
                    init[k] = o[k]
            py = os.path.expanduser(o["python"]) if o.get("python") else None
            self._engine = Engine("qwen3", init, python=py)
        return self._engine

    def build_args(self, req: SynthesisRequest, out: str) -> dict:
        vo = req.voice.options.get("qwen3", {})
        args = {"mode": self.mode, "text": req.text, "output_path": out, "language": req.language, "seed": req.seed}
        pace = ""
        rate = req.prosody.rate
        if rate < 0.93:
            pace = "Speak slowly."
        elif rate > 1.07:
            pace = "Speak quickly."
        if self.mode == "custom_voice":
            args["speaker"] = vo.get("speaker", self.options.get("speaker", "Ryan"))
            args["instruct"] = instruct_text(req.blend, pace)
        elif self.mode == "voice_design":
            desc = vo.get("description") or req.voice.description
            if not desc:
                raise BackendUnavailable(f"voice '{req.voice.name}' needs a description for qwen3 voice_design")
            emo = instruct_text(req.blend, pace)
            args["instruct"] = f"{desc}. {emo}".strip() if emo else desc
        else:
            ref = str(req.emotion_reference or req.voice.reference or "")
            if not ref:
                raise BackendUnavailable(f"voice '{req.voice.name}' needs a reference clip for qwen3 clone")
            args["ref_audio"] = ref
            args["ref_text"] = req.emotion_reference_text if req.emotion_reference else req.voice.reference_text
        if self.options.get("gen_kwargs"):
            args["gen_kwargs"] = self.options["gen_kwargs"]
        return args

    def synthesize(self, req: SynthesisRequest) -> Audio:
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "out.wav")
            args = self.build_args(req, out)
            self.engine().call("synth", args)
            x, sr = io.load(out, mono=True)
        return Audio(x, sr, {"backend": self.name, "instruct": args.get("instruct", "")})

    def close(self) -> None:
        if self._engine:
            self._engine.close()
            self._engine = None
