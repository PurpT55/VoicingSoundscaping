---
name: vss-voices
description: Casting and voice work for VSS - designing character voices, recording or generating reference clips, building emotion banks and nonverbal banks, voice consistency, and consent/ethics for cloning. Use when adding a character, choosing or recording a reference voice, filling emotion banks (vss bank), or when a voice sounds wrong, inconsistent or flat.
---

# Voices, references and banks

## Casting a voice
1. Write the `description:` like a casting call: age, gender, texture (gravelly, breathy, bright),
   accent, pace, energy, one personality hook. Example: *"Gravelly seventy-year-old lighthouse
   keeper, slow Scottish lilt, dry wit, kind underneath."*
2. Get a **reference clip** (cloning backends: IndexTTS, Fish, Chatterbox, Qwen3-clone):
   * record a real person: the warmest option for family books; or
   * design one: Qwen3-TTS VoiceDesign (`mode: voice_design`), MOSS-VoiceGenerator, Fish voice
     design. Render the voice's `reference_text`, keep the best take, and freeze it as the
     reference so the voice stays consistent across the book.
3. Reference clip spec: 8-15 s (IndexTTS works from ~5 s; Fish prefers 10-30 s), one speaker,
   neutral-to-warm delivery, no music/reverb/noise, mono 24-48 kHz, exact transcript in
   `reference_text`. Trim breaths at the edges; keep natural ones inside.
4. Per-backend fallbacks for drafts in `options:` (`kokoro: {voice: ...}`, `espeak: {...}`).

## Emotion banks (the biggest lever for authenticity)
* Shared bank `banks/emotions/<emotion>.wav` + `.txt`: used by IndexTTS as the emotion prompt,
  and may be **any** speaker. Per-voice bank `voices/<v>/emotions/`: required for
  Chatterbox/Qwen3-clone (they copy timbre too).
* Fill order: the hard set first (`crying sobbing scream laughing whisper shout trembling
  out_of_breath sleepy dry_humor`), then the book's frequent emotions (`vss lint -v` lists what
  is only approximated).
* Generate: `vss bank build <project> --backend fish --emotions crying,scream,whisper` (the most
  expressive backend you have), or `--voice tom` for a per-voice bank.
* Record: ask family or friends to act the taxonomy line for each emotion (`vss emotions --list`);
  one take per emotion, 3-8 s, intensity "strong". Numbered variants (`joy_2.wav`) rotate.
* Check: `vss bank list <project> -v`, then `vss doctor <project>`.

## Nonverbal bank
`banks/nonverbals/<id>.wav` (laugh, sigh, sob, gasp, yawn, breath ...) for backends that can't
produce them. Per-voice `nonverbal_bank:` sounds best (the same person's laugh). Breath clips enable
`render.breaths: bank`.

## Consistency problems
* Voice drifts between lines: use a longer, cleaner reference; lower IndexTTS `vector_cap`
  (0.6); prefer bank clips over strong vectors; fix the `seed=` for a line.
* Emotion too weak: raise the intensity, add a bank clip, route to a stronger backend
  (`routing:`), or wrap with `step_edit`.
* Accent bleed in multilingual books: give each language its own reference.

## Ethics (non-negotiable)
Clone only voices you have permission to use: your own, consenting family and friends, or
designed voices. Never imitate real public figures or other narrators. Keep personal recordings
out of git (`voices/**/*.wav` is ignored). Chatterbox output is watermarked (PerTh); disclose AI
narration when you publish.
