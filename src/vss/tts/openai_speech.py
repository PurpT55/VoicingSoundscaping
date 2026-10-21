"""Generic backend for LOCAL OpenAI-compatible speech servers (``POST {url}/audio/speech``).

In 2026 most open models are served this way on your own GPU: SGLang-Omni (Higgs Audio v3,
Fish S2-Pro/Drama 3, MOSS-TTS), vLLM-Omni (Qwen3-TTS, MOSS-TTS family, IndexTTS-2.5, Fish),
Kokoro-FastAPI, etc. Point ``url`` at localhost - nothing leaves your machine.

Options::

    url: http://127.0.0.1:8000/v1
    model: bosonai/higgs-audio-v3-tts-4b
    emotion_mode: inline_tags        # inline_tags | instructions | none
    tag_format: "[{}]"               # how inline tags are written for this model
    nonverbal_tags: true             # send nonverbals as inline tags
    speed: false                     # server honours the "speed" field
    ref_audio_field: ref_audio       # optional: per-request cloning fields (base64 wav)
    ref_text_field: ref_text
    extra: {language: English}       # merged into every payload
"""

from __future__ import annotations

import base64
import io as _io
import json
import urllib.error
import urllib.request

from ..emotion import taxonomy as tx
from ..emotion.mapping import freeform_tags, instruct_text
from .base import FREEFORM, INSTRUCT, PROSODY, Audio, BackendUnavailable, Capabilities, SynthesisRequest, TTSBackend


class OpenAISpeechBackend(TTSBackend):
    name = "openai_speech"
    description = "any local OpenAI-compatible /audio/speech server (SGLang-Omni, vLLM-Omni ...)"
    license = "depends on the served model"

    @property
    def url(self) -> str:
        return str(self.options.get("url", "http://127.0.0.1:8000/v1")).rstrip("/")

    @property
    def mode(self) -> str:
        return str(self.options.get("emotion_mode", "instructions"))

    def capabilities(self) -> Capabilities:
        mech = {"inline_tags": FREEFORM, "instructions": INSTRUCT}.get(self.mode, PROSODY)
        return Capabilities(
            emotion=frozenset({mech}),
            nonverbals="freeform" if (self.mode == "inline_tags" and self.options.get("nonverbal_tags", True)) else frozenset(),
            inline_emotion=self.mode == "inline_tags", cloning=bool(self.options.get("ref_audio_field")),
            speed=bool(self.options.get("speed", False)), languages=("*",),
            max_chars=int(self.options.get("max_chars", 400)), sample_rate=int(self.options.get("sample_rate", 24000)),
            strong_extremes=bool(self.options.get("strong_extremes", False)),
        )

    def available(self) -> tuple[bool, str]:
        try:
            urllib.request.urlopen(self.url + "/models", timeout=2.0).read()
            return True, f"speech server at {self.url}"
        except urllib.error.HTTPError:
            return True, f"speech server at {self.url} (no /models endpoint)"
        except Exception as exc:  # noqa: BLE001
            return False, f"speech server not reachable at {self.url} ({exc.__class__.__name__})"

    def render_text(self, req: SynthesisRequest) -> str:
        if self.mode != "inline_tags":
            return req.text
        fmt = self.options.get("tag_format", "[{}]")
        out, last = [], None
        for p in req.pieces:
            if p.kind == "nonverbal":
                if self.options.get("nonverbal_tags", True):
                    out.append(fmt.format(tx.get_nonverbal(p.nonverbal).describe))
                continue
            if p.blend.key() != last:
                out.append(freeform_tags(p.blend, fmt=fmt))
                last = p.blend.key()
            out.append(p.text)
        return " ".join(s for s in out if s).strip()

    def payload(self, req: SynthesisRequest) -> dict:
        o = self.options
        vo = req.voice.options.get(self.name, {}) or req.voice.options.get(o.get("profile", ""), {})
        body = {
            "model": o.get("model", "default"), "input": self.render_text(req),
            "voice": vo.get("voice", o.get("voice", req.voice.name)), "response_format": "wav",
        }
        if self.mode == "instructions":
            instr = instruct_text(req.blend)
            if instr:
                body["instructions"] = instr
        if o.get("speed"):
            body["speed"] = round(req.prosody.rate, 3)
        if o.get("ref_audio_field") and req.voice.reference and not vo.get("voice"):
            with open(req.voice.reference, "rb") as f:
                body[o["ref_audio_field"]] = base64.b64encode(f.read()).decode()
            if o.get("ref_text_field"):
                body[o["ref_text_field"]] = req.voice.reference_text
        if o.get("seed_field"):
            body[o["seed_field"]] = int(req.seed)
        body.update(o.get("extra", {}) or {})
        body.update(vo.get("extra", {}) or {})
        return body

    def synthesize(self, req: SynthesisRequest) -> Audio:
        headers = {"Content-Type": "application/json"}
        if self.options.get("api_key"):
            headers["Authorization"] = f"Bearer {self.options['api_key']}"
        r = urllib.request.Request(self.url + "/audio/speech", data=json.dumps(self.payload(req)).encode(),
                                   headers=headers, method="POST")
        try:
            with urllib.request.urlopen(r, timeout=float(self.options.get("timeout", 600))) as resp:
                data = resp.read()
        except urllib.error.URLError as exc:
            raise BackendUnavailable(f"speech server error: {exc}") from exc
        import soundfile as sf

        x, sr = sf.read(_io.BytesIO(data), dtype="float32", always_2d=False)
        if x.ndim == 2:
            x = x.mean(axis=1)
        return Audio(x, sr, {"backend": self.name, "text": self.render_text(req)})
