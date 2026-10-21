"""IndexTTS-2 / IndexTTS-2.5 engine (bilibili). Runs inside the index-tts repo's own venv.

Setup (on the GPU machine)::

    git clone https://github.com/index-tts/index-tts.git && cd index-tts && uv sync --all-extras
    hf download IndexTeam/IndexTTS-2.5 --local-dir=checkpoints

project.yaml::

    backends:
      config:
        indextts: {repo_dir: ~/index-tts, python: ~/index-tts/.venv/bin/python, model_dir: ~/index-tts/checkpoints, version: "2.5"}
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _runtime import seed_everything, serve  # noqa: E402

LANGS = {"en": "EN", "zh": "ZH", "ja": "JA", "es": "ES", "ar": "AR"}


class Engine:
    def __init__(self, model_dir, version="2.5", use_bf16=True, use_fp16=False, use_cuda_kernel=False,
                 use_deepspeed=False, use_qwen_emo=False, device=None):
        self.version = str(version)
        cfg = os.path.join(model_dir, "config.yaml")
        if self.version.startswith("2.5"):
            from indextts.infer_v2_5 import IndexTTS2

            kw = {"cfg_path": cfg, "model_dir": model_dir, "use_bf16": use_bf16}
            if use_qwen_emo:
                kw["use_qwen_emo"] = True
        else:
            from indextts.infer_v2 import IndexTTS2

            kw = {"cfg_path": cfg, "model_dir": model_dir, "use_fp16": use_fp16,
                  "use_cuda_kernel": use_cuda_kernel, "use_deepspeed": use_deepspeed}
        if device:
            kw["device"] = device
        self.tts = IndexTTS2(**kw)

    def synth(self, text, spk_audio_prompt, output_path, lang=None, emo_audio_prompt=None, emo_alpha=None,
              emo_vector=None, emo_text=None, use_emo_text=False, duration_factor=None, seed=None,
              use_random=False, extra=None):
        seed_everything(seed)
        kw = {"spk_audio_prompt": spk_audio_prompt, "text": text, "output_path": output_path, "verbose": False,
              "use_random": bool(use_random)}
        if self.version.startswith("2.5") and lang:
            kw["lang"] = LANGS.get(lang.lower()[:2], lang.upper())
        if emo_audio_prompt:
            kw["emo_audio_prompt"] = emo_audio_prompt
        if emo_alpha is not None:
            kw["emo_alpha"] = float(emo_alpha)
        if emo_vector:
            kw["emo_vector"] = [float(v) for v in emo_vector]
        if use_emo_text:
            kw["use_emo_text"] = True
            if emo_text:
                kw["emo_text"] = emo_text
        if duration_factor and self.version.startswith("2.5"):
            kw["duration_factor"] = max(0.5, min(2.0, float(duration_factor)))
        kw.update(extra or {})
        self.tts.infer(**kw)
        import soundfile as sf

        return {"path": output_path, "sr": sf.info(output_path).samplerate}


if __name__ == "__main__":
    serve(Engine)
