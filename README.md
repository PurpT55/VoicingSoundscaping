# VoicingSoundscaping

**Voice, music and soundscapes for audiobooks and stories, made on your own machine.**

VSS turns a story into a finished, mastered audiobook. It casts expressive voices with a wide
emotional range, places them in real-sounding spaces, adds ambience, music and effects, and
delivers ACX-compliant chapters, an M4B with chapter markers, and read-along captions. It uses the
best open models of 2026 locally (IndexTTS-2.5, Fish Audio S2-Pro / Drama 3, Qwen3-TTS,
Step-Audio-EditX, Chatterbox, Kokoro, ACE-Step 1.5 ...). No cloud APIs.

```
MIRA (whisper, afraid 0.7): Grandpa? Grandpa, the light. The ships can't see us.
TOM (calm 0.7, authoritative 0.4): Steady now. Take my hand.
PIP (peeved 0.8, projected): Do you mind? Some of us are trying to sleep!
MIRA (surprised 0.8): A seagull? [laughing] A talking seagull!
```

## What's inside
* **A 96-entry emotion vocabulary**: emotions, attitudes (dry humor, cheeky, sarcastic ...),
  states (crying, sleepy, out of breath ...) and deliveries (whisper, shout, scream ...), plus 23
  nonverbals (`<laugh> <sigh> <sob> <gasp> <yawn> ...`). Each backend gets it in its own terms:
  emotion vectors, reference clips, free-form tags, instructions or edit passes. Coverage is
  tracked per backend; 100 % is reachable with Fish Drama 3, or with IndexTTS-2.5 plus an emotion
  bank (`vss emotions`).
* **9 voice backends** behind one interface. Heavy models run in their own venvs (JSON-line
  workers) or as local servers, and anything OpenAI-compatible (`/v1/audio/speech`) plugs in by
  config. Emotion routing sends the hardest lines to the most expressive model.
* **Soundscapes**: a built-in procedural engine (41 sounds: rain, wind, sea, fire, night,
  forest, crowd, cave, thunder, bells, magic, breaths, 8 mood pads), your own libraries, and
  ACE-Step 1.5 / MOSS-SoundEffect / Stable Audio via generators. 19 acoustic spaces, sidechain
  ducking, and a measured speech-to-background margin.
* **Professional finishing**: per-voice chains (EQ, de-ess, compression), loudness by emotion
  (whispers stay small), five master presets (ACX, audiobook, podcast, cinematic, bedtime), a
  look-ahead limiter, room tone, and an ACX checker.
* **Quality control**: best-of-N takes, scored automatically (signal checks, plus faster-whisper
  WER and emotion2vec when installed), a content-addressed cache (edit one line, re-render one
  line), and a `QA_SHEET.md` listening checklist.
* **Writing tools**: a human-friendly `.vss` script format, lint with suggestions, `vss plan` to see
  exactly what each model will receive, and `vss annotate` to draft scripts with a local LLM.

## Quick start
```bash
pip install -e ".[dev]"            # Python 3.10+, plus ffmpeg and espeak-ng from your OS
vss render examples/lantern_keeper --draft      # instant draft with espeak/Kokoro voices
vss doctor examples/lantern_keeper              # what's installed, what each backend can perform
```
Then set up a production voice for your hardware (`docs/SETUP.md`), add reference clips, and
render without `--draft`.

## Docs
* [docs/SETUP.md](docs/SETUP.md): install every backend, hardware tiers
* [docs/SCRIPT_FORMAT.md](docs/SCRIPT_FORMAT.md): writing `.vss` scripts
* [docs/EMOTIONS.md](docs/EMOTIONS.md): the emotion system and coverage tables
* [docs/PROJECT_CONFIG.md](docs/PROJECT_CONFIG.md): every `project.yaml` key
* [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): how it works, how to extend it
* [docs/RESEARCH.md](docs/RESEARCH.md): the open-model landscape (Oct 2026) and licences
* [0_Index.txt](0_Index.txt): goals, progress and next steps (start here when contributing)

## Responsible use
Clone only voices you have permission to use. Disclose AI narration when you publish. Respect each
model's licence (summarised in docs/RESEARCH.md).
