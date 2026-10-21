"""Step-Audio-EditX engine (StepFun, Apache-2.0): iterative emotion / style / paralinguistic editing.

Runs inside the Step-Audio-EditX repo (cwd = repo dir) and its venv::

    git clone https://github.com/stepfun-ai/Step-Audio-EditX && cd Step-Audio-EditX && pip install -r requirements.txt
    hf download stepfun-ai/Step-Audio-EditX --local-dir ~/models/Step-Audio-EditX
    hf download stepfun-ai/Step-Audio-Tokenizer --local-dir ~/models/Step-Audio-Tokenizer
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _runtime import seed_everything, serve  # noqa: E402


class Engine:
    def __init__(self, model_path, tokenizer_path=None, model_source="local", extra=None):
        os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TRITON_ATTN")
        from tokenizer import StepAudioTokenizer
        from tts import StepAudioTTS

        tok = StepAudioTokenizer(tokenizer_path, model_source=model_source)
        self.m = StepAudioTTS(model_path, tok, model_source=model_source, **(extra or {}))

    def edit(self, prompt_wav, prompt_text, edit_type, edit_info, output_path, iterations=1, target_text="", seed=None):
        import torchaudio

        seed_everything(seed)
        cur, sr = prompt_wav, None
        for _ in range(max(1, int(iterations))):
            audio, sr = self.m.edit(prompt_wav_path=cur, prompt_text=prompt_text, target_text=target_text,
                                    edit_type=edit_type, edit_info=edit_info)
            torchaudio.save(output_path, audio.cpu(), sr)
            cur = output_path
        return {"path": output_path, "sr": int(sr)}

    def clone(self, prompt_wav, prompt_text, target_text, output_path, seed=None):
        import torchaudio

        seed_everything(seed)
        audio, sr = self.m.clone(prompt_wav_path=prompt_wav, prompt_text=prompt_text, target_text=target_text)
        torchaudio.save(output_path, audio.cpu(), sr)
        return {"path": output_path, "sr": int(sr)}


if __name__ == "__main__":
    serve(Engine)
