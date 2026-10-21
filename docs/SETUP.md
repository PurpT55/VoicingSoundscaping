# Setup

## 1. VSS itself (any OS, Python >= 3.10)

```bash
git clone <this repo> && cd VoicingSoundscaping
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"          # numpy, scipy, soundfile, pyloudnorm, pedalboard, pyyaml, num2words, msgpack
sudo apt install ffmpeg espeak-ng  # mp3/m4b export, instant draft voice (macOS: brew install ffmpeg espeak-ng)
vss doctor
vss render examples/lantern_keeper --draft
```

## 2. Pick a voice stack for your hardware

| hardware | recommended stack | coverage |
|---|---|---|
| CPU only / laptop | Kokoro (drafts), Chatterbox-Nano, MOSS-TTS-Nano via `openai_speech` | drafts; emotion only via banks |
| 8-12 GB GPU | **IndexTTS-2.5** (bf16) + shared emotion bank; Qwen3-TTS 0.6B/1.7B; ACE-Step 1.5 2B turbo | 100 % with a full bank |
| 16-24 GB GPU | + **Fish S2-Pro** (24 GB) for `routing: hard`; Step-Audio-EditX AWQ polish; ACE-Step XL with offload | 100 % |
| 48 GB+ (or 2x24 GB) | **Fish Drama 3** for the hardest lines or everything; Higgs v3 via SGLang-Omni | 100 %, strongest extremes |

Every heavy model lives in **its own venv**. VSS runs it as a worker subprocess (JSON over stdin)
or talks to its local HTTP server, so dependency conflicts can't break VSS.

## 3. Backends

### IndexTTS-2.5 (`indextts`)
```bash
git clone https://github.com/index-tts/index-tts.git ~/index-tts && cd ~/index-tts
pip install -U uv && uv sync --all-extras
uv tool install huggingface-hub && hf download IndexTeam/IndexTTS-2.5 --local-dir=checkpoints
```
```yaml
backends:
  default: indextts
  config:
    indextts: {repo_dir: ~/index-tts, python: ~/index-tts/.venv/bin/python, model_dir: ~/index-tts/checkpoints, version: "2.5"}
    # optional: use_bf16: true, emotion_mode: vector|text, vector_cap: 0.8, device: cuda:0
```
Each voice needs `reference:` (5-15 s clean clip) and `reference_text:`.

### Fish Audio S2-Pro / Drama 3 (`fish`)
```bash
git clone https://github.com/fishaudio/fish-speech ~/fish-speech && cd ~/fish-speech   # follow speech.fish.audio/install
hf download fishaudio/s2-pro --local-dir checkpoints/s2-pro     # or fishaudio/drama-3 (>= 48 GB, NVIDIA)
python tools/api_server.py --listen 127.0.0.1:8080 --llama-checkpoint-path checkpoints/s2-pro \
  --decoder-checkpoint-path checkpoints/s2-pro/codec.pth --decoder-config-name modded_dac_vq
# Drama 3: python tools/api_server.py --listen 127.0.0.1:8080 --llama-checkpoint-path checkpoints/drama-3 --device cuda:0
```
```yaml
    fish: {url: "http://127.0.0.1:8080", model: s2-pro}   # model: drama-3 enables strong extremes + single_chunk
```
Voices use `reference:`/`reference_text:` (sent with each request) or a saved server voice:
`options: {fish: {reference_id: narrator}}`.

### Qwen3-TTS (`qwen3`)
```bash
python -m venv ~/venvs/qwen3tts && ~/venvs/qwen3tts/bin/pip install -U qwen-tts soundfile
```
```yaml
    qwen3: {python: ~/venvs/qwen3tts/bin/python, mode: custom_voice, device: "cuda:0"}   # mode: voice_design | clone
```
Voice options: `qwen3: {speaker: Ryan}` (custom_voice); `description:` drives voice_design.

### Step-Audio-EditX polish (`step_edit`)
```bash
git clone https://github.com/stepfun-ai/Step-Audio-EditX ~/Step-Audio-EditX && cd ~/Step-Audio-EditX
python -m venv .venv && .venv/bin/pip install -r requirements.txt
hf download stepfun-ai/Step-Audio-EditX --local-dir ~/models/Step-Audio-EditX
hf download stepfun-ai/Step-Audio-Tokenizer --local-dir ~/models/Step-Audio-Tokenizer
```
```yaml
  default: step_edit
  config:
    step_edit: {base: indextts, repo_dir: ~/Step-Audio-EditX, python: ~/Step-Audio-EditX/.venv/bin/python,
                model_path: ~/models/Step-Audio-EditX, tokenizer_path: ~/models/Step-Audio-Tokenizer,
                max_edits: 2, max_iterations: 3, paralinguistic: true}
```

### Chatterbox (`chatterbox`)
```bash
python -m venv ~/venvs/chatterbox && ~/venvs/chatterbox/bin/pip install chatterbox-tts soundfile
```
```yaml
    chatterbox: {python: ~/venvs/chatterbox/bin/python, model: original}   # turbo | nano | multilingual
```

### Any OpenAI-compatible local server (`openai_speech`): Higgs v3, MOSS-TTS, Qwen3 via vLLM-Omni ...
```bash
sgl-omni serve --model-path bosonai/higgs-audio-v3-tts-4b --port 8000      # example: Higgs Audio v3
```
```yaml
backends:
  default: higgs
  config:
    higgs: {type: openai_speech, url: "http://127.0.0.1:8000/v1", model: bosonai/higgs-audio-v3-tts-4b,
            emotion_mode: inline_tags, tag_format: "[{}]", strong_extremes: false}
```
Several named profiles can share a type (`type: openai_speech`). Check each server's docs for its
voice and cloning fields (`ref_audio_field`, `ref_text_field`, `extra:`).

### Kokoro (`kokoro`): CPU drafts
```bash
pip install kokoro-onnx
mkdir -p ~/models/kokoro && cd ~/models/kokoro
curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
```
```yaml
    kokoro: {model: ~/models/kokoro/kokoro-v1.0.onnx, voices: ~/models/kokoro/voices-v1.0.bin}
```
Voices: af_heart, af_bella, af_sky, bf_emma, bf_isabella, am_michael, am_puck, bm_george, bm_lewis ...

## 4. Music and sound effects

### ACE-Step 1.5 (`acestep`)
```bash
git clone https://github.com/ACE-Step/ACE-Step-1.5.git ~/ACE-Step-1.5 && cd ~/ACE-Step-1.5 && uv sync
uv run acestep-api            # http://127.0.0.1:8001
```
```yaml
assets:
  generators:
    music: [acestep, procedural]
    config:
      acestep: {type: acestep, url: "http://127.0.0.1:8001", steps: 8, thinking: true}   # model: acestep-v15-xl-turbo
```
Cues are instrumental (`lyrics: [Instrumental]`) unless a cue passes `lyrics=`. Write prompts like
a film composer's brief: instruments, mood, tempo, "no vocals".

### MOSS-SoundEffect v2 / Stable Audio Open / AudioX (`command` or `http`)
```yaml
assets:
  generators:
    sfx: [moss_sfx, procedural]
    ambience: [moss_sfx, procedural]
    config:
      moss_sfx: {type: command, kinds: [sfx, ambience], max_seconds: 30,
                 cmd: "~/MOSS-TTS/.venv/bin/python ~/MOSS-TTS/moss_soundeffect_v2/infer.py --text {prompt} --duration {seconds} --seed {seed} --output {out}"}
```
Adapt `cmd` to the tool's actual CLI. Generated files are cached by prompt, length and seed.

## 5. Automatic QA (optional, recommended for long books)

```bash
pip install "vss[qa]"          # faster-whisper: word error rate per take
pip install "vss[qa-emotion]"  # emotion2vec+ via FunASR: emotion check per take
```
```yaml
qa: {asr: faster-whisper, asr_model: small.en, max_wer: 0.12, emotion_model: emotion2vec, min_score: 0.5}
render: {takes: 2, max_takes: 5}   # extra takes only when the best score is below min_score
```

## 6. Local LLM for manuscript annotation (optional)

Any OpenAI-compatible chat server: Ollama (`ollama serve`, `ollama pull qwen3:14b`), llama.cpp
`llama-server`, LM Studio, vLLM.
```bash
vss annotate manuscript.txt --llm http://127.0.0.1:11434/v1 --model qwen3:14b --out script/01.vss
```
