# VoicingSoundscaping (vss)

Local audiobook and storytelling engine: expressive voices, soundscapes, music, mastering.

**Start every session by reading `0_Index.txt`** (goals, status, workstreams, next actions), and
update it before you finish (protocol: `.claude/skills/vss-index/SKILL.md`).

Skills in `.claude/skills/`:
* `vss-director`: story to script
* `vss-voices`: casting, references, banks
* `vss-emotion`: the emotion system
* `vss-soundscape`: ambience, music, effects
* `vss-mastering-qa`: QA and delivery
* `vss-backends`: model setup
* `vss-dev`: codebase

Ground rules
* Local only: no cloud TTS/music APIs (no ElevenLabs, no Suno). Models run on the user's machine.
* Scripts stay backend-agnostic; emotion syntax is defined in `src/vss/emotion/taxonomy.py`.
* Never claim a model integration works without running it; mark `(gpu)` items in the index.
* Clone only voices with consent; personal recordings stay out of git.

Checks: `.venv/bin/python -m pytest -q` (no GPU or network needed) and
`.venv/bin/vss render examples/lantern_keeper --draft`.
