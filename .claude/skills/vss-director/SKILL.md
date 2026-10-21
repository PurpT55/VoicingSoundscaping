---
name: vss-director
description: Turn a story or manuscript into a performance-ready VSS script (.vss) - casting speakers, choosing emotions/intensities/deliveries/nonverbals, pacing, scenes, ambience, music and SFX cues - then lint and plan it. Use when adapting any text into an audiobook or audio drama, when annotating dialogue with emotions, or when reviewing/improving an existing script's performance direction.
---

# Directing a story for VSS

You are the audiobook director: faithful to the author's words, generous with feeling, sparing with
effects. Format reference: `docs/SCRIPT_FORMAT.md`. Emotion reference: `vss emotions --list` or
`docs/EMOTIONS.md`.

## Workflow
1. **Read the whole text first.** Note POV (third person -> NARRATOR; first person -> `@narrator MIRA`),
   characters, locations, time of day, weather, and the 3-5 emotional peaks of each chapter.
2. **Cast** in project.yaml: one `cast:` entry per speaker, each with a `voice:` and a `default:`
   temperament (TOM `dry_humor 0.3`, PIP `cheeky 0.5`). Write a vivid `description:` for each voice
   (age, texture, accent, energy): voice-design backends use it.
3. **Mark dialogue.** Quoted speech becomes `SPEAKER (...): text`; keep dialogue tags ("she
   whispered") in the narration. Never drop or invent words: the author's text is sacred.
   Exception: pure tags like `he said` between two halves of one quote may stay in the narrator
   line between two speaker lines.
4. **Direct each line.** Ask: what does the character *want* in this line, and what are they
   *hiding*? Pick the emotion for the want, a delivery for the circumstance (whisper in the dark),
   and an acting note in quotes for the subtext (`"trying to sound brave"`).
5. **Shape arcs.** Use inline shifts for turns inside a line: `I'm fine. [hurt 0.5] I'm fine.`
6. **Sound design**, per scene: `## Scene` + `@scene ambience="..." space=...`, music only at
   structural moments (opening, turn, resolution), SFX only when the text names the sound or the
   moment needs a punctuation.
7. **Validate**: `vss lint <project> -v` (zero errors), then `vss plan <project> --chapter N`
   to read what each model receives. Fix anything that reads wrong.
8. **Draft render** `vss render <project> --draft --chapter N`, read `QA_SHEET.md`, adjust.

## Emotion choice cheatsheet
| moment | try |
|---|---|
| bedtime narration | `storytelling 0.5`, `tender 0.4`, `soft` |
| comic understatement | `dry_humor 0.6`; never `amused` (too warm) |
| playful kid | `cheeky`, `giddy`, `peppy`, `<giggle>` |
| irritation ladder | `peeved` -> `irritated` -> `frustrated` -> `angry` -> `furious` |
| fear ladder | `worried` -> `anxious` -> `afraid` -> `terrified`/`panicked`, add `trembling` or `whisper` |
| sadness ladder | `disappointed` -> `sad` -> `crying` -> `sobbing`; quiet grief = `melancholic`/`wistful` |
| low energy | `tired`, `exhausted`, `sleepy` (+`<yawn>`), `bored` (+`<sigh>`) |
| warmth | `warm`, `tender`, `comforting`, `grateful`, `content` |
| awe and magic | `awe 0.8` + `soft`, `<gasp>` before, sparkle SFX after |
| villains (family-safe) | `menacing` (quiet is scarier), `smug`, `arrogant`, `contemptuous` |

Intensity: 0.3 colour, 0.6 clear, 0.8 strong, 0.9+ rare peak. Contrast beats volume: put a whisper
after a normal line, and a scream after silence (`@pause 1.0`).

## Nonverbals and pacing
* Reactions go *before* the words: `<gasp> The bridge is gone?`
* At most one nonverbal per line, except for comic effect.
* `<beat>` for comic timing, `<pause 1.2>` for weight, `@pause 2` between beats of a scene.

## Sound design rules of thumb
* One bed per scene (two or three layers at most): `rain:-2, fire:-12`.
* Music needs a reason: under narration it should be slow and instrumental; stop it before
  dialogue-heavy passages, or let ducking handle it.
* Name SFX simply (`door_knock`, `thunder distance=0.2`); check `vss sounds` for procedural ones.
* Space follows the camera: `space=stairwell`, `phone` for calls, `inner_voice` for thoughts.

## Deliverable checklist
- [ ] every word of the source present, in order
- [ ] every speaker is in `cast:` with a voice and a default
- [ ] `vss lint` shows 0 errors; notes reviewed
- [ ] each chapter has 3+ deliberate emotional peaks, not a flat level
- [ ] scenes have ambience and space; music cues have prompts
