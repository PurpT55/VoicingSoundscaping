"""IndexTTS-2 / 2.5 backend: zero-shot cloning with timbre/emotion disentanglement.

Emotion is driven, in priority order, by
1. an emotion-bank clip (``emo_audio_prompt``) - from ANY speaker, IndexTTS keeps the timbre;
2. the 8-dim emotion vector derived from the taxonomy (default);
3. optional ``emotion_mode: text`` -> IndexTTS' own text-to-emotion (Qwen) on our acting notes.
"""

from __future__ import annotations

import os
import tempfile

from ..audio import io
from ..emotion.mapping import index_alpha, index_vector, instruct_text
from .base import VECTOR8, Audio, BackendUnavailable, Capabilities, SynthesisRequest, TTSBackend
from .worker import Engine


class IndexTTSBackend(TTSBackend):
    name = "indextts"
    description = "IndexTTS-2.5 (bilibili): cloning + disentangled emotion vectors/reference clips"
    license = "bilibili IndexTTS licence (commercial use: ask bilibili)"

    def __init__(self, **options):
        super().__init__(**options)
        self._engine: Engine | None = None

    @property
    def version(self) -> str:
        return str(self.options.get("version", "2.5"))

    def capabilities(self) -> Capabilities:
        langs = ("en", "zh", "ja", "es", "ar") if self.version.startswith("2.5") else ("en", "zh")
        return Capabilities(emotion=frozenset({VECTOR8}), emotion_reference=True, emotion_reference_scope="any",
                            cloning=True, speed=self.version.startswith("2.5"), languages=langs, max_chars=260,
                            sample_rate=22050)

    def available(self) -> tuple[bool, str]:
        md = self.options.get("model_dir")
        if not md or not os.path.exists(os.path.join(os.path.expanduser(md), "config.yaml")):
            return False, f"model_dir with config.yaml not found ({md}); see docs/SETUP.md#indextts"
        if self.options.get("python") and not os.path.exists(os.path.expanduser(self.options["python"])):
            return False, f"python not found: {self.options['python']}"
        if not self.options.get("python"):
            try:
                __import__("indextts")
            except ImportError:
                return False, "indextts not importable here; set backends.config.indextts.python to its venv"
        return True, f"IndexTTS-{self.version} at {md}"

    def engine(self) -> Engine:
        if self._engine is None:
            o = self.options
            init = {"model_dir": os.path.expanduser(o["model_dir"]), "version": self.version}
            for k in ("use_bf16", "use_fp16", "use_cuda_kernel", "use_deepspeed", "device"):
                if k in o:
                    init[k] = o[k]
            if o.get("emotion_mode") == "text":
                init["use_qwen_emo"] = True
            py = os.path.expanduser(o["python"]) if o.get("python") else None
            cwd = os.path.expanduser(o["repo_dir"]) if o.get("repo_dir") else None
            self._engine = Engine("indextts", init, python=py, cwd=cwd)
        return self._engine

    def build_args(self, req: SynthesisRequest, out_path: str) -> dict:
        if not req.voice.reference:
            raise BackendUnavailable(f"IndexTTS needs a reference clip for voice '{req.voice.name}'")
        args = {"text": req.text, "spk_audio_prompt": str(req.voice.reference), "output_path": out_path,
                "lang": req.language, "seed": req.seed}
        mode = self.options.get("emotion_mode", "vector")
        if req.emotion_reference:
            args["emo_audio_prompt"] = str(req.emotion_reference)
            args["emo_alpha"] = float(req.params.get("emo_alpha", index_alpha(req.blend)))
        elif mode == "text" and not req.blend.is_neutral:
            args["use_emo_text"] = True
            args["emo_text"] = instruct_text(req.blend)
            args["emo_alpha"] = float(req.params.get("emo_alpha", 0.6))
        else:
            vec = index_vector(req.blend, cap=float(self.options.get("vector_cap", 0.8)))
            if vec:
                args["emo_vector"] = vec
        rate = req.prosody.rate
        if self.version.startswith("2.5") and abs(rate - 1.0) > 0.02:
            args["duration_factor"] = round(1.0 / rate, 3)
        if self.options.get("use_random"):
            args["use_random"] = True
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
