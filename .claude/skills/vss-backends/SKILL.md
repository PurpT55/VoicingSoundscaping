---
name: vss-backends
description: Install, configure, run and troubleshoot VSS synthesis backends and generators locally - IndexTTS-2.5, Fish Audio S2-Pro/Drama 3, Qwen3-TTS, Step-Audio-EditX, Chatterbox, Kokoro, espeak, any OpenAI-compatible local speech server (Higgs v3, MOSS via SGLang-Omni/vLLM-Omni), ACE-Step 1.5, command/http SFX generators; hardware tiers, licences, fallbacks and vss doctor. Use when setting up a machine, adding or switching a model, or when a backend fails or falls back.
---

# Backends and generators

Full recipes: `docs/SETUP.md`. Landscape and licences: `docs/RESEARCH.md`.

## Principles
* **Local only.** Every backend runs on the user's machine (in-process, as a worker in its own venv,
  or as a localhost HTTP server). No cloud TTS or music APIs: don't add one.
* **One venv per heavy model**, referenced by `python:` (worker) or by its server's `url:`.
* Check everything with `vss doctor <project>`: availability, reasons, coverage table, banks,
  references.

## Picking a stack (see the SETUP.md table)
* 8-12 GB: IndexTTS-2.5 + shared emotion bank (+ Qwen3-TTS for voice design), ACE-Step 1.5 turbo.
* 24 GB: + Fish S2-Pro for `routing: {hard: fish}`, Step-Audio-EditX polish.
* 48 GB: Fish Drama 3 for extremes, or as the main voice.
* CPU: Kokoro drafts, Chatterbox-Nano, MOSS-TTS-Nano via `openai_speech`.

## Configuration patterns
```yaml
backends:
  default: indextts
  fallback: [fish, kokoro]
  draft: kokoro
  routing: {hard: fish, sobbing: fish}
  config:
    indextts: {repo_dir: ~/index-tts, python: ~/index-tts/.venv/bin/python, model_dir: ~/index-tts/checkpoints, version: "2.5"}
    fish: {url: "http://127.0.0.1:8080", model: drama-3}
    higgs: {type: openai_speech, url: "http://127.0.0.1:8000/v1", model: bosonai/higgs-audio-v3-tts-4b, emotion_mode: inline_tags}
```

## Troubleshooting
| symptom | check |
|---|---|
| "unavailable -> using espeak" in logs/QA sheet | `vss doctor`: wrong path, server not running, missing package |
| worker exits immediately | run the worker's python by hand: `~/index-tts/.venv/bin/python -c "import indextts"`; check `repo_dir` (cwd) |
| CUDA OOM | bf16/fp16, smaller model, one model per GPU, close other servers; ACE-Step `--offload` |
| Fish 4xx/5xx | `curl http://127.0.0.1:8080/v1/health`; reference audio must be WAV with a transcript |
| robotic/garbled takes | reference clip too noisy or too short; lower emotion strength; more takes |
| very slow | IndexTTS `use_cuda_kernel`, vLLM/SGLang serving recipes; drafts with Kokoro |

## Adding a model
Most new models expose `/v1/audio/speech`: add an `openai_speech` profile (no code). Otherwise
follow `vss-dev` ("New TTS backend"). Record the model, its licence and its VRAM in
docs/RESEARCH.md and 0_Index.txt.

## Licences (summary; verify before commercial use)
Apache/MIT: Qwen3-TTS, Step-Audio-EditX, Chatterbox (watermarked), Kokoro, ACE-Step 1.5.
Fish Audio Community Licence (S2-Pro, Drama 3): free non-commercial; commercial under $500k
revenue; audiobook production allowed; no hosted TTS service. IndexTTS: bilibili licence
(commercial: contact bilibili). Higgs v3: research/non-commercial (creator grant with credit).
