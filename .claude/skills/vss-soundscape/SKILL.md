---
name: vss-soundscape
description: Sound design for VSS stories - ambience beds, acoustic spaces, music cues (ACE-Step 1.5 prompts or procedural mood pads), sound effects (libraries, MOSS-SoundEffect/Stable Audio via generators, procedural), levels, ducking and the speech-to-bed check. Use when designing a scene's sound, adding music or effects, tuning mix levels, adding a sound library or generator, or creating new procedural sounds.
---

# Soundscapes, music and effects

Code: `src/vss/sound/` (procedural, library, generators, resolve), `src/vss/mix/timeline.py`,
`src/vss/audio/chain.py` (spaces).

## Resolution order
library files (`assets/sfx|ambience|music`, matched by stem/tokens, numbered variants rotate)
-> configured generators (`assets.generators.<kind>`) -> procedural (`vss sounds`). Missing sounds
show as `MISSING` in the QA sheet. Audition any procedural sound with `vss sound rain --seconds 20`.

## Scene recipes
| scene | ambience | space | music |
|---|---|---|---|
| cosy indoors, storm outside | `rain:-2, wind:-8, fire:-12` | kitchen / living_room | warm, sparse |
| night outdoors | `night:-6, wind_soft:-12` | outdoor_open / forest | mystery or wonder pad |
| seaside | `sea_waves:-3, wind_soft:-10` | outdoor_open | wonder |
| forest by day | `forest_day:-4, stream:-10` | forest | adventure |
| tavern / market | `tavern:-6` / `crowd:-6` | tavern | none (the crowd is the music) |
| cave / dungeon | `cave:-4` | cave | tension, very low |
| dream / memory | `wind_soft:-14` | dream | lullaby / wonder |
| thoughts, telepathy | (none) | inner_voice | - |

## Levels (all relative to the dialogue target)
Defaults: ambience bus -12 dB, music -11 dB, sfx -6 dB, ducking 6 dB (+3 dB more on music),
attack 0.15 s, release 0.9 s. Per-layer gains in the spec (`rain:-2`). The render report states the
**speech-to-bed margin**; the healthy window is 14-26 dB (QA_SHEET warns outside it). Kids'
bedtime: aim near 22 dB; immersive drama: ~16 dB.

## Music with ACE-Step 1.5
Prompt like a film composer: instrumentation + mood + tempo + texture + "instrumental".
* `"solo felt piano and soft strings, tender, nostalgic, slow 70 bpm, warm room, instrumental"`
* `"low tense strings, sparse taiko pulse, suspense, dark, slow build, no vocals"`
* `"celtic harp and flute, playful, light, 100 bpm, bright, instrumental"`
Use `duration=` for stingers and `fade=`/`fade_out=` for transitions; `bpm=` and `key_scale=` pass
through. Cached by prompt+length+seed. Without ACE-Step, cue names and prompts pick a procedural pad
(wonder/warm/melancholy/mystery/tension/lullaby/adventure/triumph).

## Effects
* Prefer real recordings for hero sounds (door, footsteps, animals); procedural is great for
  weather, fire, water, magic, bells, thunder, knocks, page turns, breaths.
* `wait=true` holds the story until the effect finishes (a knock before "Who's there?").
* Generative SFX: configure `command`/`http` generators (docs/SETUP.md §4). Write prompts
  concretely: `"old wooden door creaks open slowly, small room, close mic"`.

## Adding a procedural generator
`fn(seconds, sr, rng, **params)` building stereo from the helpers (`pink`, `brown`, `lfo`,
`varying_bandpass`, `scatter`, `_place_events`, `overlay`), ending with `finish(x, sr)`
(-20 LUFS, fades, limiter). Register with `_reg(name, kind, fn, description, default_seconds, aliases)`.
Tests check every registry entry automatically. Keep it vectorised: avoid per-sample Python loops,
and never `np.convolve` with long kernels (use fftconvolve or uniform_filter1d).
