# project.yaml reference

Every key is optional; defaults live in `src/vss/config.py` (`DEFAULTS`). Relative paths resolve
against the project folder; `~` expands.

```yaml
title: The Lantern Keeper
author: ...
narrator_credit: ...           # m4b "composer" tag
language: en                   # default for voices
cover: cover.jpg               # embedded in the m4b
script: [script/*.vss]         # globs, sorted
default_speaker: NARRATOR      # reads untagged prose

backends:
  default: indextts            # production voice for everyone
  fallback: [fish, kokoro]     # tried in order when a backend is unavailable (then draft, espeak, placeholder)
  draft: kokoro                # used by --draft
  routing:                     # per-emotion overrides: emotion id | hard | kind:<kind> | family:<family>
    hard: fish
  config:                      # options per backend name; `type:` makes named profiles
    indextts: {repo_dir: ..., python: ..., model_dir: ..., version: "2.5"}
    higgs: {type: openai_speech, url: "http://127.0.0.1:8000/v1", model: ..., emotion_mode: inline_tags}

voices:
  narrator:
    description: Warm, mature storyteller ...   # used by voice-design backends and as documentation
    reference: voices/narrator/reference.wav    # cloning backends
    reference_text: "exact words of the clip"
    language: en
    backend: null                                # per-voice backend override
    emotion_bank: voices/narrator/emotions       # per-voice emotion clips
    nonverbal_bank: voices/narrator/nonverbals   # laugh/sigh/breath clips
    chain: narrator                              # raw | dialog | narrator | child | deep
    gain: 0                                      # dB trim
    options:                                     # per-backend voice settings
      kokoro: {voice: bf_emma}
      qwen3: {speaker: Serena}
      fish: {reference_id: narrator}
      espeak: {voice: en-gb+f3, pitch: 45, wpm: 150}

cast:
  NARRATOR: {voice: narrator, default: storytelling 0.5}
  PIP: {voice: pip, default: cheeky 0.5, pan: 0.25, backend: fish, rate: 1.05}   # extra keys become line params

banks: {emotions: banks/emotions, nonverbals: banks/nonverbals}
lexicon: {Gullhaven: GULL-hay-ven}   # whole-word, case-sensitive respellings

render:
  sample_rate: 44100
  takes: 1                     # takes per unit
  max_takes: 4                 # keep trying while the best score < qa.min_score
  seed: 1234
  cache_dir: .vss_cache
  out_dir: out
  chunk: {target_chars: 220, max_chars: 300}   # also capped by each backend
  pauses: {sentence: 0.35, paragraph: 0.8, speaker_change: 0.5, scene: 2.0,
           chapter_head: 0.75, chapter_tail: 3.0, after_title: 1.2}
  breaths: off                 # off | bank | procedural
  narrator_in_space: false
  announce_chapters: true
  nonverbal_fallback: bank     # bank | drop

qa: {asr: none, asr_model: small.en, max_wer: 0.15, emotion_model: none, min_score: 0.45}

mix:
  dialog_lufs: -20             # per-take target before mastering (+ emotion level offset)
  ambience_db: -12             # bus levels relative to dialogue
  music_db: -11
  sfx_db: -6
  duck: {depth_db: 6, attack: 0.15, release: 0.9, threshold_db: -42, music_extra_db: 3}
  speech_to_bed_db: [14, 26]   # QA window for the measured margin
  room_tone_db: -66
  beds: auto                   # auto (off for acx) | full | off | music | ambience

master: {preset: acx}          # acx | audiobook | podcast | cinematic | bedtime

assets:
  sfx_dirs: [assets/sfx]
  music_dirs: [assets/music]
  ambience_dirs: [assets/ambience]
  generators:
    sfx: [procedural]
    ambience: [procedural]
    music: [acestep, procedural]
    config:
      acestep: {type: acestep, url: "http://127.0.0.1:8001"}

export:
  formats: [wav, mp3, m4b]     # wav | flac | mp3 | m4b
  mp3_bitrate: 192k
  m4b_bitrate: 96k
  captions: true               # SRT per chapter
  stems: false                 # dialog/ambience/music/sfx stems for a DAW
```

## Master presets

| preset | measure | target | ceiling | notes |
|---|---|---|---|---|
| acx | RMS | -20 dBFS | -3.5 dBFS | Audible/ACX; beds off by default (noise floor rule) |
| audiobook | LUFS | -18 | -2 dBTP | Apple/Google/Spotify-style stores, immersive |
| podcast | LUFS | -16 | -1 dBTP | streaming speech |
| cinematic | LUFS | -23 | -1 dBTP | EBU R128, wide dynamics (pair with render.sample_rate: 48000) |
| bedtime | LUFS | -22 | -3 dBTP | softer, darker top end, gentle dynamics |
