# Architecture

```
 manuscript.txt ──(vss annotate / skill vss-director)──▶ script/*.vss + project.yaml
                                                              │
 script/parser.py ── Script(chapters, lines, directives, diagnostics)
        │                                   render/lint.py (speakers, sounds, coverage, refs)
        ▼
 render/plan.py ── Plan(steps): Unit | Pause | NonverbalClip | Cue
        │   backend choice (line > cast > voice > routing > default), normalisation + chunking,
        │   inline emotion/nonverbal kept or split by backend Capabilities
        ▼
 render/pipeline.py  for each Unit:
        │   emotion bank lookup ─▶ SynthesisRequest ─▶ TTSBackend.synthesize (worker / HTTP / in-proc)
        │   take cache (content hash) · best-of-N with qa/scorers (signal, ASR WER, emotion2vec)
        │   voice post: resample ▸ trim ▸ chain (HPF/EQ/de-ess/comp) ▸ LUFS + emotion level offset
        ▼
 mix/timeline.py  layout(): dialogue clips, scenes, ambience layers, music cues, sfx, breaths, captions
                  mixdown(): spaces (reverb/phone/...) ▸ duck curve ▸ beds ▸ sfx ▸ speech-to-bed metric
        ▼
 audio/master.py  preset (acx | audiobook | podcast | cinematic | bedtime) ▸ room tone ▸ ACX check
        ▼
 render/export.py  WAV/FLAC/MP3 per chapter · SRT read-along · M4B with chapters + cover · stems
                   render_report.json · QA_SHEET.md (flagged takes, ACX, mix notes, listening checklist)
```

## Packages

| path | responsibility |
|---|---|
| `vss/emotion/` | taxonomy (96 entries + 23 nonverbals), `Blend`, backend mappings, coverage model |
| `vss/script/` | `.vss` data model, parser (diagnostics with suggestions), text normalisation, chunking |
| `vss/config.py` | project.yaml defaults + deep merge, `Project`, `Voice`, `CastMember` |
| `vss/tts/` | backend contract (`base.py`), registry, adapters, worker runtime (`worker.py`, `workers/*.py`) |
| `vss/sound/` | procedural generators, asset library, generative clients (ACE-Step, command, http), resolver |
| `vss/audio/` | io, DSP (filters, envelope, compressor, de-esser, look-ahead limiter, time-stretch), loudness, ACX, chains/spaces, mastering |
| `vss/mix/` | timeline layout + mixdown + ducking |
| `vss/qa/` | take scorers and selection |
| `vss/render/` | planner, take cache, lint, pipeline, export |
| `vss/voices/` | emotion banks (coverage, bootstrapping) |
| `vss/llm/` | local-LLM manuscript annotation + auto-repair |
| `vss/cli.py` | `vss` command |

## Key decisions

1. **Backend-agnostic emotion vocabulary.** Scripts never contain model-specific syntax; each adapter
   renders the same `Blend` into vectors, tags, instructions or edits. Switching models never means
   rewriting a book.
2. **One venv per model.** SOTA repos pin incompatible torch stacks. Workers (`tts/workers/*.py`,
   stdlib-only runtime) run inside each model's venv and speak JSON lines; HTTP servers
   (Fish, SGLang/vLLM-Omni, ACE-Step) are first-class.
3. **Capabilities drive planning.** `Capabilities` (mechanisms, inline emotion, nonverbals, max chars,
   emotion-reference scope, strong extremes) decide chunking, inline vs split, bank usage, routing,
   and coverage reporting.
4. **Emotion banks over labels.** A reference clip carries more emotion than a label. IndexTTS's
   timbre/emotion disentanglement lets one shared bank serve every voice.
5. **Deterministic, cached renders.** Seeds come from content hashes; takes are cached by backend
   fingerprint plus request fingerprint, so editing one line re-renders one line.
6. **Measured mixing.** Bus levels relative to the dialogue target, sidechain ducking, and a
   reported speech-to-bed margin (QA window 14-26 dB) instead of guesswork.
7. **Two kinds of master.** ACX compliance (beds off, RMS -20, peaks <= -3.5, floor < -60) and immersive
   presets for family, streaming and cinematic listening.
8. **Graceful degradation.** Missing backends fall back (configured list -> draft -> espeak -> placeholder),
   and missing sounds fall back to procedural. Everything is reported, nothing fails silently.

## Extending

* **New TTS backend**: subclass `TTSBackend` (capabilities, available, synthesize, render_text),
  register in `tts/registry.py`, add a worker if it needs its own venv, add tests with a mock or fake
  engine (`tests/test_backends.py`), document it in docs/SETUP.md, and update 0_Index.txt.
* **New emotion**: add an `E(...)` entry in `emotion/taxonomy.py` (vector, describe, tag, step, line).
  Tests enforce completeness and alias uniqueness. Regenerate docs/EMOTIONS.md.
* **New procedural sound**: write `fn(seconds, sr, rng, **params) -> stereo` ending in `finish()`,
  then `_reg(...)`. Tests check every registry entry for length, level and determinism.
* **New generator** (music/SFX model): usually just a `command`/`http` config entry; otherwise
  add a client class in `sound/generators.py` and a branch in `make_generator`.
