"""Qwen3-TTS engine (Alibaba Qwen, Apache-2.0): CustomVoice / VoiceDesign / Base (clone).

Setup::  python -m venv ~/venvs/qwen3tts && ~/venvs/qwen3tts/bin/pip install -U qwen-tts soundfile
         (optional) pip install -U flash-attn --no-build-isolation
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _runtime import seed_everything, serve  # noqa: E402

LANG_NAMES = {"zh": "Chinese", "en": "English", "ja": "Japanese", "ko": "Korean", "de": "German", "fr": "French",
              "ru": "Russian", "pt": "Portuguese", "es": "Spanish", "it": "Italian"}


class Engine:
    def __init__(self, model="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", device="cuda:0", dtype="bfloat16",
                 attn_implementation=None):
        import torch
        from qwen_tts import Qwen3TTSModel

        kw = {"device_map": device, "dtype": getattr(torch, dtype)}
        if attn_implementation:
            kw["attn_implementation"] = attn_implementation
        self.model = Qwen3TTSModel.from_pretrained(model, **kw)
        self.prompts = {}

    def synth(self, mode, text, output_path, language="en", speaker=None, instruct=None, ref_audio=None,
              ref_text=None, seed=None, gen_kwargs=None):
        seed_everything(seed)
        lang = LANG_NAMES.get((language or "auto")[:2].lower(), "Auto")
        gk = dict(gen_kwargs or {})
        if mode == "custom_voice":
            kw = {"text": text, "language": lang, "speaker": speaker}
            if instruct:
                kw["instruct"] = instruct
            wavs, sr = self.model.generate_custom_voice(**kw, **gk)
        elif mode == "voice_design":
            wavs, sr = self.model.generate_voice_design(text=text, language=lang, instruct=instruct or "", **gk)
        elif mode == "clone":
            key = (ref_audio, ref_text)
            if key not in self.prompts:
                self.prompts[key] = self.model.create_voice_clone_prompt(
                    ref_audio=ref_audio, ref_text=ref_text or None, x_vector_only_mode=not ref_text)
            wavs, sr = self.model.generate_voice_clone(text=text, language=lang, voice_clone_prompt=self.prompts[key], **gk)
        else:
            raise ValueError(f"unknown qwen3 mode {mode}")
        import soundfile as sf

        sf.write(output_path, wavs[0], sr)
        return {"path": output_path, "sr": sr}


if __name__ == "__main__":
    serve(Engine)
