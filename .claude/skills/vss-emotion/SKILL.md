---
name: vss-emotion
description: The VSS emotion system - 96-entry taxonomy (emotions, attitudes, states, deliveries) + 23 nonverbals, intensities and blends, how each backend realises them (IndexTTS vectors/reference clips, Fish free-form tags, Qwen3 instruct, Step-Audio-EditX edits, Chatterbox exaggeration), coverage tracking toward the 80-90% goal, routing, and fixing flat or wrong emotion. Use when an emotion doesn't land, when adding/changing emotions, or when choosing backends for emotional range.
---

# Emotion system

Code: `src/vss/emotion/{taxonomy,mapping,coverage}.py`. Docs: `docs/EMOTIONS.md`.

## Mental model
`Blend` = emotions + deliveries + acting note -> each backend renders it with its mechanism:
* IndexTTS: bank clip (`emo_audio_prompt`, `emo_alpha` = 0.45 + 0.55*intensity), else an 8-dim
  vector scaled by intensity/0.8 with the sum capped (`vector_cap`, default 0.8). Optional
  `emotion_mode: text` uses IndexTTS's own Qwen emotion reader on our acting notes.
* Fish: `[whispering] [very afraid] [acting note]` inline. Drama 3 handles the extremes.
* Qwen3 / OpenAI-style servers: instruct sentences with intensity words.
* Step-Audio-EditX: 1-3 edit iterations (intensity < 0.5 -> 1, < 0.8 -> 2, else 3; whisper >= 2).
* Chatterbox: `exaggeration` 0.3-1.3 and `cfg_weight` from arousal x intensity.
* Prosody hints (rate/pitch/level) feed speed controls, DSP fallbacks and the mixer.

## Measuring range
`vss emotions --backend indextts fish qwen3 --rows` -> per-emotion matrix + BEST-OF-ALL row;
`vss emotions --bank` assumes full banks. The goal is >= 80-90 % native+bank for the backends a
project actually routes to (see the 0_Index.txt "COVERAGE" line). Coverage = controllability;
**always confirm by listening** and with emotion2vec scores in QA.

## Fixing emotion that doesn't land
1. Is the right backend speaking? `vss plan` shows backend + exact model text per line.
2. Is it only `approx` there? `vss lint -v` lists approximated emotions per backend.
   -> add a bank clip for it, or `routing: {<emotion>: fish}`, or wrap with `step_edit`.
3. Intensity: 0.6 is moderate; strong moments want 0.8; deliveries ignore intensity mostly.
4. Text matters: emotion fights the words. Adding an acting note (`"through gritted teeth"`) helps
   tag and instruct models; IndexTTS benefits from a matching bank clip.
5. Takes: `takes=4` on a key line; with QA models installed, emotion2vec picks the best.
6. Too much (wobbly timbre, artefacts): lower intensity or `vector_cap`, or use a bank clip.

## Changing the taxonomy
Add/edit `E(...)` entries in `taxonomy.py`: unique id/aliases, all 8 vector dims 0..1, an
adjective-first `describe`, a short `tag`, the nearest valid Step mapping, prosody hints, a
reference `line` that *means* the emotion, and `hard=True` for extremes that instruct models
under-deliver. Then run `pytest tests/test_emotion.py`, regenerate docs
(`vss emotions --markdown`), and log the change in 0_Index.txt.
