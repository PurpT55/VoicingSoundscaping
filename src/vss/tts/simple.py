"""Always-available backends: ``placeholder`` (timing tones), ``espeak`` (formant synth) and
``kokoro`` (fast neural CPU voice). Good for proofing timing, drafts and tests - not for emotion."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from ..audio import dsp, io
from .base import PROSODY, Audio, BackendUnavailable, Capabilities, SynthesisRequest, TTSBackend, stable_hash


class PlaceholderBackend(TTSBackend):
    """Deterministic speech-shaped tones (no dependencies). Each syllable is a short formant
    tone; pitch, speed and breathiness follow the emotion prosody. Used for timing drafts and CI."""

    name = "placeholder"
    description = "speech-shaped timing tones (no model)"
    license = "n/a"

    def capabilities(self) -> Capabilities:
        return Capabilities(emotion=frozenset({PROSODY}), speed=True, languages=("*",), max_chars=400,
                            sample_rate=24000)

    def synthesize(self, req: SynthesisRequest) -> Audio:
        sr = 24000
        pr = req.prosody
        seed = int(stable_hash([req.voice.name, req.text, req.seed])[:8], 16)
        rng = np.random.default_rng(seed)
        base = 110 + int(stable_hash(req.voice.name)[:4], 16) % 120
        f0 = base * 2 ** (pr.pitch / 12)
        whisper = any(p.emotion.id in ("whisper", "murmur") for p in req.blend.deliveries)
        parts = []
        for piece in req.pieces:
            if piece.kind == "nonverbal":
                k = int(0.5 * sr)
                parts.append(dsp.bandpass(rng.standard_normal(k).astype(np.float32), sr, 300, 3000) * np.hanning(k) * 0.1)
                continue
            for word in piece.text.split():
                syl = max(1, sum(ch in "aeiouy" for ch in word.lower()))
                for _ in range(syl):
                    dur = rng.uniform(0.11, 0.18) / pr.rate
                    k = int(dur * sr)
                    t = np.arange(k) / sr
                    f = f0 * (1 + rng.uniform(-0.08, 0.08)) * (1 - 0.1 * t / dur)
                    if whisper:
                        src = rng.standard_normal(k).astype(np.float32) * 0.3
                    else:
                        ph = 2 * np.pi * np.cumsum(f) / sr
                        src = sum(np.sin(h * ph) / h for h in range(1, 8)).astype(np.float32)
                    src = dsp.bandpass(src, sr, 250, 3500)
                    parts.append(src * np.hanning(k).astype(np.float32) * 0.3)
                parts.append(np.zeros(int(0.05 * sr / pr.rate), np.float32))
            if piece.text.rstrip()[-1:] in ".!?":
                parts.append(np.zeros(int(0.15 * sr), np.float32))
        x = np.concatenate(parts) if parts else np.zeros(int(0.2 * sr), np.float32)
        return Audio(x.astype(np.float32), sr, {"backend": self.name})


class EspeakBackend(TTSBackend):
    """eSpeak NG formant synthesiser: robotic but intelligible, instant, 100+ languages.

    Voice options: ``voice`` (e.g. en-gb, en-us, fr, en-gb+f3), ``pitch`` (0-99), ``wpm``.
    Whisper deliveries use espeak's ``+whisper`` variant.
    """

    name = "espeak"
    description = "eSpeak NG formant synth (drafts/proofing)"
    license = "GPL-3.0 (binary, called as a program)"

    def capabilities(self) -> Capabilities:
        return Capabilities(emotion=frozenset({PROSODY}), speed=True, languages=("*",), max_chars=500,
                            sample_rate=22050)

    def available(self) -> tuple[bool, str]:
        exe = shutil.which(self.options.get("binary", "espeak-ng")) or shutil.which("espeak")
        return (exe is not None), (exe or "espeak-ng not installed (apt install espeak-ng)")

    def synthesize(self, req: SynthesisRequest) -> Audio:
        exe = shutil.which(self.options.get("binary", "espeak-ng")) or shutil.which("espeak")
        if not exe:
            raise BackendUnavailable("espeak-ng not installed")
        vo = req.voice.options.get("espeak", {})
        voice = vo.get("voice") or self.options.get("voice") or _espeak_lang(req.language)
        if any(p.emotion.id in ("whisper", "murmur") for p in req.blend.deliveries) and "+" not in voice:
            voice += "+whisper"
        pr = req.prosody
        wpm = int(float(vo.get("wpm", self.options.get("wpm", 160))) * pr.rate)
        pitch = int(max(0, min(99, float(vo.get("pitch", 50)) + pr.pitch * 4)))
        amp = int(max(20, min(200, 100 + pr.level * 4)))
        text = " ".join(p.text for p in req.pieces if p.kind == "speech")
        proc = subprocess.run([exe, "-v", voice, "-s", str(wpm), "-p", str(pitch), "-a", str(amp), "--stdout", text],
                              capture_output=True, check=False)
        if proc.returncode != 0 or not proc.stdout:
            raise RuntimeError(f"espeak failed: {proc.stderr.decode(errors='ignore')[:300]}")
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            f.write(proc.stdout)
            f.flush()
            x, sr = io.load(f.name, mono=True)
        return Audio(x, sr, {"backend": self.name, "voice": voice})


def _espeak_lang(lang: str) -> str:
    return {"en": "en-gb"}.get(lang[:2], lang[:2] or "en")


class KokoroBackend(TTSBackend):
    """Kokoro-82M (Apache-2.0): natural, very fast, CPU-friendly; no emotion control (speed only).

    ``impl: onnx`` (default) uses kokoro-onnx with ``model``/``voices`` files
    (github.com/thewh1teagle/kokoro-onnx releases); ``impl: torch`` uses the ``kokoro`` package.
    Voice options: ``voice`` (af_heart, af_bella, bf_emma, am_michael, bm_george ...), ``lang``.
    """

    name = "kokoro"
    description = "Kokoro-82M fast neural voice (drafts, quick bedtime reads)"
    license = "Apache-2.0"

    def __init__(self, **options):
        super().__init__(**options)
        self._model = None

    def capabilities(self) -> Capabilities:
        return Capabilities(emotion=frozenset({PROSODY}), speed=True,
                            languages=("en", "es", "fr", "it", "pt", "ja", "zh", "hi"), max_chars=350, sample_rate=24000)

    def available(self) -> tuple[bool, str]:
        impl = self.options.get("impl", "onnx")
        try:
            __import__("kokoro_onnx" if impl == "onnx" else "kokoro")
        except ImportError:
            return False, f"pip install {'kokoro-onnx' if impl == 'onnx' else 'kokoro'}"
        if impl == "onnx":
            for key in ("model", "voices"):
                p = self.options.get(key)
                if not p or not Path(p).exists():
                    return False, f"kokoro '{key}' file missing ({p}); see docs/SETUP.md"
        return True, "ok"

    def _load(self):
        if self._model is None:
            if self.options.get("impl", "onnx") == "onnx":
                from kokoro_onnx import Kokoro

                self._model = Kokoro(str(self.options["model"]), str(self.options["voices"]))
            else:
                from kokoro import KPipeline

                self._model = KPipeline(lang_code=self.options.get("lang_code", "a"))
        return self._model

    def synthesize(self, req: SynthesisRequest) -> Audio:
        m = self._load()
        vo = req.voice.options.get("kokoro", {})
        voice = vo.get("voice", self.options.get("voice", "af_heart"))
        speed = float(vo.get("speed", 1.0)) * req.prosody.rate
        text = " ".join(p.text for p in req.pieces if p.kind == "speech")
        if self.options.get("impl", "onnx") == "onnx":
            lang = vo.get("lang", self.options.get("lang", "en-gb" if voice.startswith("b") else "en-us"))
            samples, sr = m.create(text, voice=voice, speed=max(0.5, min(2.0, speed)), lang=lang)
            x = np.asarray(samples, dtype=np.float32)
        else:
            chunks = [np.asarray(a, dtype=np.float32) for _, _, a in m(text, voice=voice, speed=speed)]
            x, sr = np.concatenate(chunks), 24000
        return Audio(x, int(sr), {"backend": self.name, "voice": voice})
