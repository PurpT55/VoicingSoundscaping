# The `.vss` script format

Plain text you can write by hand, have Claude write (skill `vss-director`), or draft with a local
LLM (`vss annotate`). Prose stays prose; you mark only what the performance needs.

```
---                                        optional YAML front matter (title, chapter ...)
title: The Lantern Keeper
---
# Chapter One: The Night the Light Went Out     chapter (one file may hold several)
## The keeper's kitchen                          scene: closes previous ambience, resets space
@scene ambience="rain:-2, wind:-8, fire:-12" space=kitchen
@music start home prompt="gentle acoustic guitar, cosy" gain=-2 fade=5
// comments start with two slashes

Untagged prose is read by the narrator. Consecutive lines
join into one paragraph; a blank line ends it.

MIRA (bored 0.7): There is nothing to do. <sigh>
TOM (dry_humor 0.6, "barely looking up", rate=0.95): Three hours, love.
MIRA (peeved 0.5): I wasn't counting. [cheeky 0.5] I was up to four thousand and twelve.
```

## Speaker lines

`SPEAKER (directions): text`. The speaker is UPPERCASE (letters, digits, spaces, `_ ' . -`) and must
appear under `cast:` in project.yaml (otherwise an automatic voice is used and lint warns).

Directions are comma-separated, in any order:

| item | example | meaning |
|---|---|---|
| emotion | `afraid`, `afraid 0.8` | taxonomy id or alias, intensity 0..1 (default 0.6) |
| delivery | `whisper`, `shout`, `scream` | vocal mode, combines with emotions |
| acting note | `"barely holding back tears"` | free text: instruct/tag backends use it verbatim |
| parameter | `rate=0.9`, `pitch=1`, `gain=-2`, `pan=-0.3`, `backend=fish`, `takes=4`, `seed=7` | per-line overrides |

Run `vss emotions --list` for every emotion with its aliases (`cry` -> crying, `deadpan` -> dry_humor,
`peaky` -> peppy ...). Unknown names are errors, with suggestions.

## Inline markup

| markup | meaning |
|---|---|
| `[sad 0.7]` | switch emotion for the rest of the line (relative to the line's own) |
| `[whisper, afraid]` | blends work inline too |
| `[reset]` | back to the line's emotion |
| `<laugh> <sigh> <sob> <gasp> <yawn> <sniff> <cough> <groan> <breath> <hmm> ...` | nonverbal sounds (23, see EMOTIONS.md) |
| `<pause>` `<pause 1.2>` `<beat>` `<long pause>` | silence (0.6 s, 1.2 s, 0.3 s, 1.5 s) |
| `<sfx door_creak gain=-6>` | effect anchored at this point of the line |
| `{Aoife\|EE-fa}` | written form for captions, spoken form for the voice |
| `*word*` | emphasis (asterisks are stripped) |
| `\[` `\<` `\{` `\*` | literal characters |

Nonverbals stay inline when the backend can perform them (Fish: all; Chatterbox Turbo: laugh, chuckle,
cough, sigh, gasp, sniff, groan, shush, clear_throat; Step-Audio-EditX: sigh, laugh, breath ...).
Otherwise VSS splices a clip from the nonverbal bank, or drops the sound with a lint note.

## Directives

| directive | parameters |
|---|---|
| `@scene` | `ambience="name:gain, name:gain"`, `space=<space>`, `fade=` |
| `@ambience add rain:-6` / `remove rain` / `stop` / `set a, b` | `fade=` (seconds) |
| `@space cave` | change the acoustic space from here on |
| `@music start <cue> prompt="..."` / `@music stop` | `gain=` (dB), `fade=` (in), `fade_out=`, `duration=`, `offset=`, `bpm=`, `key_scale=` |
| `@sfx <name>` | `gain=`, `offset=` (s), `wait=true` (hold the story until it ends), `pan=`, `prompt=`, `duration=`, plus generator params (`distance=` for thunder, `count=` for knock, `pitch=` for bell) |
| `@pause 1.5` | silence |
| `@narrator NAME` | default speaker for untagged prose (first-person chapters) |

**Spaces**: dry, booth, bedroom, small_room, living_room, kitchen, tavern, stairwell, hall,
cathedral, cave, forest, outdoor_open, car, phone, radio, underwater, dream, inner_voice. The
narrator stays dry unless `render.narrator_in_space: true`, so it reads as the storyteller, outside
the scene.

**Sound names** resolve, in order, to your libraries (`assets/sfx|ambience|music`, matched by
file name and tags), then the configured generators (ACE-Step, MOSS-SoundEffect ...), then the
procedural engine (`vss sounds`). Music falls back to a procedural mood pad chosen from the cue
name or prompt (wonder, warm, melancholy, mystery, tension, lullaby, adventure, triumph).

## Pacing defaults (project.yaml `render.pauses`)

sentence 0.35 s, paragraph 0.8 s, speaker change 0.5 s, scene 2.0 s, chapter head 0.75 s
(room tone), chapter tail 3.0 s, after the announced chapter title 1.2 s.

## Tips from the director's chair

* Narration carries the book: give the narrator a `default:` (e.g. `storytelling 0.5`) and colour
  only the turning points.
* Contrast sells emotion: a whisper lands harder after a normal line; save `extreme` (0.9+) for the
  one moment that needs it.
* Put the reaction sound before the words (`<gasp> What was that?`), not after.
* Keep each line to one breath of thought; long speeches split naturally at sentences.
* Read `vss plan` before rendering: it shows exactly what each model will receive.
