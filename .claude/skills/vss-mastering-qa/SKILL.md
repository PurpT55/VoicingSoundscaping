---
name: vss-mastering-qa
description: Quality control and delivery for VSS audiobooks - master presets (ACX/Audible, audiobook stores, podcast, cinematic, bedtime), ACX compliance, reading QA_SHEET.md and render_report.json, fixing flagged takes (pickups), listening checklist, exports (MP3, M4B chapters + cover, SRT captions, stems). Use before delivering or sharing a book, when ACX checks fail, when takes are flagged, or when choosing export formats.
---

# Mastering, QA and delivery

## Choose the master
| goal | preset | notes |
|---|---|---|
| Audible / ACX upload | `acx` | beds off automatically (noise floor < -60 dB is impossible under beds); SFX stay |
| family, friends, own site, Apple/Google/Spotify | `audiobook` | -18 LUFS, full soundscape |
| podcast feeds | `podcast` | -16 LUFS |
| headphones / immersive drama | `cinematic` | -23 LUFS, wide dynamics; set `render.sample_rate: 48000` |
| children's bedtime | `bedtime` | -22 LUFS, darker, gentle |
`vss render <p> --preset acx --out out/acx` and `--preset audiobook --out out/family` reuse the same
cached takes, so a second master is cheap.

## Read the reports
`out/QA_SHEET.md`:
* **Flagged takes**, each with a reason: too long (hallucination/babble), too short
  (truncation), clipping, long silences, high WER (with ASR), wrong emotion (with emotion2vec).
* **ACX** line: rms, peak, noise floor, head/tail room tone, length, sample rate.
* **notes**: speech-to-bed margin warnings, beds disabled, backend fallbacks.
`out/render_report.json` has everything machine-readable (per-unit backend, take, score, notes,
render text, emotion reference used).

## Fixing a flagged line (pickup)
1. Listen at the caption time (`.srt` has the timestamps).
2. Fix the cause: wording (lexicon/respelling `{word|spoken}`), emotion (intensity, bank,
   routing), or the model (try `backend=fish` on that line).
3. Force new takes: `takes=4` or `seed=11` on that line. Only that unit re-renders (cache).
4. Re-render the chapter: `vss render <p> --chapter 3`.

## Listening checklist (always, on two systems: earbuds and a speaker)
- [ ] names and places pronounced right (add to `lexicon:`)
- [ ] the 3-5 emotional peaks per chapter land; nothing is flat for minutes
- [ ] no doubled, skipped or invented words (enable ASR QA for long books)
- [ ] beds never mask words; music swells only in pauses
- [ ] no clicks at edits, no clipped shouts, no reverb on the narrator (unless intended)
- [ ] chapter starts and ends breathe (room tone), titles announced consistently

## Delivery files
`<slug>.wav` (24-bit master), `.mp3` (192k CBR, 44.1 kHz), `.flac` (optional), `.srt` (read-along
captions; speaker names shown for dialogue), `<Title>.m4b` (all chapters, chapter markers,
metadata, cover via `cover:`), stems when `export.stems: true`. Before uploading, check each file
with `vss acx out/<file>.mp3`.
