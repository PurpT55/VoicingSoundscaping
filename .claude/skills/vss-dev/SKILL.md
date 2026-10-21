---
name: vss-dev
description: Developer guide to the VSS codebase - architecture, module map, conventions, tests, and step-by-step recipes for adding a TTS backend, emotion, procedural sound, generator, master preset or CLI command. Use when changing code in src/vss, writing tests, debugging the render pipeline, or reviewing changes.
---

# Developing VSS

Read `docs/ARCHITECTURE.md` first (data flow + key decisions).

## Setup and checks
```bash
uv venv .venv && uv pip install --python .venv/bin/python -e ".[dev]"   # or python -m venv + pip
.venv/bin/python -m pytest -q          # full suite (~1 min, no GPU, no network)
.venv/bin/vss render examples/lantern_keeper --draft   # end-to-end smoke (espeak/kokoro)
```
System tools: `ffmpeg` (mp3/m4b; tests skip m4b without it) and `espeak-ng` (draft voice).

## Conventions
* Audio inside VSS: float32 numpy, mono `(n,)` or stereo `(n, 2)`; sample rates are explicit.
* DSP must be vectorised (whole chapters are 50-150M samples); Python loops only at frame rate.
* Heavy/optional imports go inside functions; `vss` must import with only the core deps.
* Model code runs in workers (`src/vss/tts/workers/*.py`): stdlib + model libs only, no `vss`
  imports, and must work in-process too (`Engine(worker, init)` without `python=`).
* Everything user-visible is reported (fallbacks, missing sounds, approximations); nothing fails silently.
* Determinism: seeds come from content hashes (`stable_hash`); caches are keyed on fingerprints.
* Match the surrounding style: dataclasses, type hints, short docstrings that say *why*.

## Recipes
**New TTS backend**
1. `src/vss/tts/<name>.py`: subclass `TTSBackend`; implement `capabilities()` truthfully (it
   drives planning and coverage), `available()` (cheap, no model load), `synthesize()`,
   `render_text()`. Use `emotion.mapping` helpers for emotion.
2. Needs its own venv? Add `workers/<name>.py` with `Engine` + `serve(Engine)` and call it via `tts.worker.Engine`.
3. Register in `tts/registry.py::BACKENDS`.
4. Tests in `tests/test_backends.py`: payload/args against a mock HTTP server or fake engine.
5. Docs: SETUP.md recipe, RESEARCH.md row, `vss emotions --markdown` refresh, 0_Index.txt.

**New emotion / nonverbal**: `emotion/taxonomy.py` (see the `vss-emotion` skill) and tests.
**New procedural sound**: `sound/procedural.py` (see the `vss-soundscape` skill).
**New generator type**: client class in `sound/generators.py` + branch in `make_generator`.
**New master preset**: `audio/master.py::PRESETS` + the test in `tests/test_audio.py` covers it automatically.
**New CLI command**: `cli.py` (`cmd_*` + parser entry) + a smoke assertion in `tests/test_pipeline.py::test_cli_smoke`.

## Debugging a render
* `vss plan <project> --chapter N`: steps, backends, gaps, exact model text.
* `render_report.json`: per-unit scores/notes; `.vss_cache/takes/*/<key>.json` holds per-take metadata.
* `export.stems: true`: listen to dialog/ambience/music/sfx separately.
* `--no-cache` to rule out stale takes; delete `.vss_cache/sound` to regenerate sounds.

## Before committing
`pytest -q` green, example draft render works, docs and 0_Index.txt updated (STATUS, items,
SESSION LOG). Commit messages reference workstream IDs.
