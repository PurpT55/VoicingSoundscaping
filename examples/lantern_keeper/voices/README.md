# Voices

Each voice folder holds the reference clip a cloning backend copies the timbre from:

```
voices/narrator/reference.wav   + reference_text in project.yaml
voices/mira/reference.wav
voices/tom/reference.wav
voices/pip/reference.wav
voices/<voice>/emotions/<emotion>.wav  (optional per-voice emotion bank)
```

Reference clips that clone well:

* 8-15 seconds of one person speaking naturally (IndexTTS works from ~5 s, Fish S2 prefers 10-30 s)
* clean: no music, no reverb, no other voices; a quiet room and a phone held 20 cm away is fine
* the *neutral* version of the voice; emotions come from the script (and the emotion bank)
* 24-48 kHz WAV, mono; write the exact words into `reference_text`

Only clone voices you have permission to use: your own, family and friends who agreed, or
voices you designed (Qwen3-TTS VoiceDesign, MOSS-VoiceGenerator, Fish voice design). Audio files
in `voices/` are git-ignored so personal recordings never get pushed by accident.

No clips yet? The draft backends (Kokoro, espeak) use the preset voices in `options:`.
