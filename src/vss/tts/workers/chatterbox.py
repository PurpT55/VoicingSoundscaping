"""Chatterbox engine (Resemble AI, MIT): original (exaggeration/CFG), Turbo/Nano (paralinguistic
tags), Multilingual V3. Outputs carry Resemble's imperceptible PerTh watermark.

Setup::  python -m venv ~/venvs/chatterbox && ~/venvs/chatterbox/bin/pip install chatterbox-tts soundfile
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _runtime import seed_everything, serve  # noqa: E402


class Engine:
    def __init__(self, model="original", device="cuda", t3_model="v3"):
        self.kind = model
        if model in ("turbo", "nano"):
            from chatterbox.tts_turbo import ChatterboxTurboTTS

            kw = {"device": device}
            if model == "nano":
                kw["nano"] = True
            self.m = ChatterboxTurboTTS.from_pretrained(**kw)
        elif model == "multilingual":
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS

            self.m = ChatterboxMultilingualTTS.from_pretrained(device=device, t3_model=t3_model)
        else:
            from chatterbox.tts import ChatterboxTTS

            self.m = ChatterboxTTS.from_pretrained(device=device)

    def synth(self, text, output_path, audio_prompt_path=None, exaggeration=None, cfg_weight=None,
              language_id=None, temperature=None, seed=None):
        seed_everything(seed)
        kw = {}
        if audio_prompt_path:
            kw["audio_prompt_path"] = audio_prompt_path
        if self.kind in ("original", "multilingual"):
            if exaggeration is not None:
                kw["exaggeration"] = float(exaggeration)
            if cfg_weight is not None:
                kw["cfg_weight"] = float(cfg_weight)
        if self.kind == "multilingual" and language_id:
            kw["language_id"] = language_id
        if temperature is not None:
            kw["temperature"] = float(temperature)
        wav = self.m.generate(text, **kw)
        import soundfile as sf

        data = wav.squeeze(0).detach().cpu().numpy() if hasattr(wav, "detach") else wav
        sf.write(output_path, data, self.m.sr)
        return {"path": output_path, "sr": int(self.m.sr)}


if __name__ == "__main__":
    serve(Engine)
