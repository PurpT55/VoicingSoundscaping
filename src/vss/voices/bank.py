"""Emotion banks: one short reference clip per emotion (``<bank>/<emotion_id>.wav`` + ``.txt``).

Why: reference audio carries emotion far more convincingly than labels. IndexTTS-2/2.5 takes an
emotion clip from ANY speaker and keeps the target timbre, so a single shared bank lifts every
voice in the project to (near) full emotional range. Cloners like Chatterbox/Qwen3-clone copy the
clip's timbre too, so they need a per-voice bank.

Ways to fill a bank (best first):
1. record real people acting each line (family voices make the warmest audiobooks)
2. generate with the most expressive backend you have (Fish Drama 3 / S2-Pro, Qwen3 instruct,
   Step-Audio-EditX edits) - ``vss bank build``
3. curate clips from licensed datasets/recordings you own
"""

from __future__ import annotations

from pathlib import Path

from ..audio import dsp, io
from ..audio.loudness import normalize_lufs
from ..emotion import taxonomy as tx
from ..emotion.mapping import Blend, EmotionRef
from ..tts.base import Piece, SynthesisRequest, VoiceSpec


def coverage(bank_dir: Path | None) -> dict[str, bool]:
    have = set()
    if bank_dir and Path(bank_dir).exists():
        have = {p.stem.rsplit("_", 1)[0] if p.stem.rsplit("_", 1)[-1].isdigit() else p.stem
                for p in Path(bank_dir).iterdir() if p.suffix.lower() in (".wav", ".flac", ".mp3", ".ogg")}
    return {e.id: e.id in have for e in tx.EMOTIONS}


def build(backend, voice: VoiceSpec, out_dir: Path, emotions: list[str] | None = None, intensity: float = 0.8,
          overwrite: bool = False, sr_out: int = 24000, log=print, seed: int = 7) -> list[Path]:
    """Synthesize the taxonomy reference line for each emotion with ``backend`` into ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    made: list[Path] = []
    ids = emotions or [e.id for e in tx.EMOTIONS if e.id != "neutral"]
    for eid in ids:
        e = tx.get(eid)
        wav = out_dir / f"{e.id}.wav"
        if wav.exists() and not overwrite:
            continue
        blend = Blend((EmotionRef(e, intensity),))
        nv = [Piece("nonverbal", nonverbal=n, blend=blend) for n in e.nonverbals[:1]
              if backend.capabilities().supports_nonverbal(n)]
        req = SynthesisRequest(pieces=nv + [Piece("speech", e.line, blend)], voice=voice, blend=blend,
                               language=voice.language, seed=seed)
        audio = backend.synthesize(req)
        x = io.resample(audio.samples, audio.sr, sr_out)
        x = dsp.trim_silence(x, sr_out, pad=0.05)
        x = normalize_lufs(x, sr_out, -20.0)
        io.save(wav, x, sr_out, subtype="PCM_16")
        wav.with_suffix(".txt").write_text(e.line + "\n", encoding="utf-8")
        made.append(wav)
        log(f"  {e.id:<14} {len(x) / sr_out:5.1f}s  {e.line[:60]}")
    return made
