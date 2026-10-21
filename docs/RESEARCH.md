# Research: open, local audio models (state of October 2026)

The landscape VSS builds on. Everything runs on your own hardware: no ElevenLabs, no Suno, no
cloud APIs. Licences change, so re-check the model card before any commercial release.

Method: model READMEs fetched from GitHub on 2026-10-21, plus leaderboards (Artificial Analysis
open-weights TTS, TTS Arena). Benchmark numbers come from the vendors unless stated otherwise.

## Speech (TTS)

| model | size / VRAM | emotion control | cloning | licence | role in VSS |
|---|---|---|---|---|---|
| **Fish Audio Drama 3** (Oct 2026) | 35B MoE (3B active), >= 48 GB FP8, NVIDIA only | free-form inline tags, multi-turn "director" revisions, infilling; strong crying/whisper/laughter | 10-30 s ref | Fish Audio Community Licence: free non-commercial, commercial < $500k revenue, audiobook production explicitly OK | top-tier performer for the hardest lines (`routing: hard`) |
| **Fish Audio S2-Pro** (Mar 2026) | 4B + 400M, >= 24 GB (GPU or CPU) | 15,000+ free-form tags (`[whisper]`, `[excited]`), multi-speaker | yes | same licence | main expressive backend on 24 GB cards |
| **IndexTTS-2.5** (bilibili, Aug 2026) | 0.8B, ~8-12 GB, RTF ~0.2 on 4090 | 8-dim emotion vector, emotion reference clip from ANY speaker (timbre/emotion disentangled), emotion from text, `duration_factor` | 5 s ref | bilibili licence (commercial: ask) | default narrator/cast voice; emotion banks unlock 100 % coverage |
| **Qwen3-TTS** (Jan 2026) | 0.6B / 1.7B | natural-language `instruct`; VoiceDesign from a description; 9 premium speakers | 3 s (Base, no instruct) | Apache-2.0 | instruct emotions, voice design for new characters |
| **Step-Audio-EditX** (StepFun) | 3B, 6-8 GB AWQ | iterative edits: 14 emotions, 30+ styles (whisper, shout, roar, murmur ...), paralinguistics ([sigh], [laugh], [breath] ...) | yes | Apache-2.0 | refinement pass over any backend (`step_edit`) |
| **Higgs Audio v3** (Boson AI, Jun 2026) | 4B | inline emotion/style/prosody, 100+ languages | yes | research/non-commercial (creator grant with credit) | via SGLang-Omni -> `openai_speech` backend |
| **MOSS-TTS v1.5 / TTSD / VoiceGenerator** (OpenMOSS, 2026) | 1.7-8B (llama.cpp: 8 GB) | long-form, dialogue, voice design; 48 kHz stereo | yes | see repo | via vLLM-Omni/SGLang-Omni -> `openai_speech` |
| **Chatterbox** (Resemble AI): Multilingual V3, Turbo 350M, Nano 110M | 0.1-0.5B, Nano runs on CPU | `exaggeration`/`cfg`; Turbo/Nano native [laugh] [chuckle] [cough] ... | yes | MIT, PerTh watermark | light GPUs, CPU drafts with tags |
| **VibeVoice** (Microsoft) | 1.5B / Realtime 0.5B | context-driven, up to 90 min, 4 speakers | yes | MIT (TTS code withdrawn upstream in 2025; weights remain) | long-form dialogue experiments |
| **Kokoro-82M** | 82M, CPU real-time | none (speed only) | no (presets) | Apache-2.0 | instant drafts, proofing, bedtime quickies |
| **VoxCPM2, OmniVoice, CosyVoice3, FireRedTTS-2** | 0.5-2B | varied | yes | varied | candidates (see the IndexTTS-2.5 CV3-Eval tables) |

The open-weights TTS leaderboard (Artificial Analysis, Oct 2026) ranks Fish Audio Drama 3 Preview
first, with Breeze TTS 2 and Fish S2-Pro close behind; Higgs v3 is about 10th. Rankings differ
between arenas, so trust your own listening tests.

## Music

| model | VRAM | notes | licence | in VSS |
|---|---|---|---|---|
| **ACE-Step 1.5** (Feb 2026) / **1.5 XL** (Apr 2026, 4B DiT) | < 4 GB (2B turbo) / 12-20 GB (XL) | full songs or instrumentals 10-600 s, BPM/key control, LoRA from a few songs, REST API | MIT | `acestep` generator (instrumental underscore by default) |
| **HeartMuLa** (Jan 2026) | - | Llama-3.2 based music family; benchmarks below ACE-Step 1.5 | see repo | via `command` generator |
| YuE, DiffRhythm, SongBloom, MusicGen | - | older options | varied | via `command` generator |

## Sound effects and ambience

| model | notes | in VSS |
|---|---|---|
| **MOSS-SoundEffect v2.0** (May 2026) | DiT + flow matching, 48 kHz, up to 30 s, bilingual prompts | `command` / `http` generator |
| **Stable Audio Open** | up to 47 s samples/SFX | `command` generator |
| **AudioX-Turbo**, TangoFlux, MMAudio | anything-to-audio / fast text-to-audio | `command` generator |
| **VSS procedural engine** (built in) | 41 deterministic generators: rain, wind, sea, fire, night, forest, stream, crowd, tavern, cave, thunder, bells, chimes, magic sparkle, knock, page turn, owl, breath, 8 mood pads | always available, no model |
| real libraries | Sonniss GDC bundles (royalty-free), BBC Sound Effects (RemArc licence), freesound CC0 | `assets/*_dirs`, highest priority |

## Quality control

| tool | use |
|---|---|
| **faster-whisper** | ASR -> word error rate per take (catches hallucinated or skipped words) |
| **emotion2vec+ large** (FunASR) | 9-class speech emotion recognition, a tie-breaker between takes |
| pyloudnorm (BS.1770), own true-peak limiter | loudness and peak compliance (ACX, -18 LUFS stores, -16 podcast, R128) |
| UTMOSv2, Audiobox Aesthetics, speaker-embedding similarity | candidate scorers (roadmap) |

## Serving pattern (2026)

Most new models ship with **SGLang-Omni** or **vLLM-Omni** recipes and expose an OpenAI-compatible
`/v1/audio/speech` endpoint on localhost. VSS's `openai_speech` backend talks to any of them, so a
new model usually needs config, not code.

## Sources

* Fish Speech README and inference/install docs (github.com/fishaudio/fish-speech), Fish Audio Community Licence (2026-10-13)
* IndexTTS README (github.com/index-tts/index-tts), IndexTTS-2.5 tech report arXiv:2601.03888
* Qwen3-TTS README (github.com/QwenLM/Qwen3-TTS)
* Step-Audio-EditX README and tts_infer.py (github.com/stepfun-ai/Step-Audio-EditX)
* Higgs Audio README (github.com/boson-ai/higgs-audio), model card bosonai/higgs-audio-v3-tts-4b
* Chatterbox README (github.com/resemble-ai/chatterbox)
* VibeVoice README (github.com/microsoft/VibeVoice)
* ACE-Step 1.5 README and API docs (github.com/ace-step/ACE-Step-1.5), arXiv:2602.00744
* MOSS-TTS family README (github.com/OpenMOSS/MOSS-TTS)
* HeartMuLa arXiv:2601.10547; Artificial Analysis open-weights TTS leaderboard
* emotion2vec arXiv:2312.15185, model card emotion2vec/emotion2vec_plus_large
