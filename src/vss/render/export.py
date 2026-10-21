"""Delivery: chapter files (WAV/FLAC/MP3), read-along captions (SRT), and a full-book M4B with
chapter markers, metadata and cover art (via ffmpeg)."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from ..audio import io
from ..mix.timeline import Caption


def _ts(t: float) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def srt(captions: list[Caption], show_speaker: bool = True, narrator: str = "NARRATOR") -> str:
    blocks = []
    for i, c in enumerate(captions, 1):
        who = f"{c.speaker.title()}: " if show_speaker and c.speaker != narrator else ""
        blocks.append(f"{i}\n{_ts(c.start)} --> {_ts(c.end)}\n{who}{c.text}\n")
    return "\n".join(blocks)


def ffmpeg_ok() -> bool:
    return shutil.which("ffmpeg") is not None


def write_chapter(x: np.ndarray, sr: int, out_dir: Path, slug: str, formats: list[str], mp3_bitrate: str = "192k",
                  title: str = "", album: str = "", artist: str = "", track: int | None = None) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    wav = out_dir / f"{slug}.wav"
    io.save(wav, x, sr, subtype="PCM_24")
    paths["wav"] = wav
    if "flac" in formats:
        paths["flac"] = io.save(out_dir / f"{slug}.flac", x, sr, subtype="PCM_24")
    if "mp3" in formats and ffmpeg_ok():
        mp3 = out_dir / f"{slug}.mp3"
        cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(wav), "-codec:a", "libmp3lame", "-b:a", mp3_bitrate,
               "-ar", "44100", "-write_xing", "0"]
        for k, v in (("title", title), ("album", album), ("artist", artist), ("genre", "Audiobook")):
            if v:
                cmd += ["-metadata", f"{k}={v}"]
        if track:
            cmd += ["-metadata", f"track={track}"]
        subprocess.run(cmd + [str(mp3)], check=True)
        paths["mp3"] = mp3
    if "wav" not in formats:
        paths.pop("wav")
        wav_keep = any(f in formats for f in ("m4b",))
        if not wav_keep:
            wav.unlink(missing_ok=True)
    return paths


def build_m4b(chapters: list[tuple[str, Path]], out: Path, title: str, author: str = "", narrator: str = "",
              cover: Path | None = None, bitrate: str = "96k", comment: str = "") -> Path:
    """Concatenate chapter audio into one AAC .m4b with chapter markers."""
    if not ffmpeg_ok():
        raise RuntimeError("ffmpeg is required for m4b export")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        lst = td / "list.txt"
        lst.write_text("".join(f"file '{p.resolve().as_posix()}'\n" for _, p in chapters))
        meta = [";FFMETADATA1", f"title={_esc(title)}", f"album={_esc(title)}", f"artist={_esc(author)}",
                f"composer={_esc(narrator)}", "genre=Audiobook"]
        if comment:
            meta.append(f"comment={_esc(comment)}")
        t = 0
        for name, p in chapters:
            d = int(round(io.duration(p) * 1000))
            meta += ["[CHAPTER]", "TIMEBASE=1/1000", f"START={t}", f"END={t + d}", f"title={_esc(name)}"]
            t += d
        mfile = td / "meta.txt"
        mfile.write_text("\n".join(meta) + "\n")
        cmd = ["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-i", str(mfile)]
        if cover and Path(cover).exists():
            cmd += ["-i", str(cover), "-map", "0:a", "-map", "2:v", "-c:v", "copy", "-disposition:v:0", "attached_pic"]
        else:
            cmd += ["-map", "0:a"]
        cmd += ["-map_metadata", "1", "-map_chapters", "1", "-c:a", "aac", "-b:a", bitrate, "-movflags", "+faststart",
                str(out.with_suffix(".m4a"))]
        subprocess.run(cmd, check=True)
        shutil.move(str(out.with_suffix(".m4a")), str(out))
    return out


def _esc(s: str) -> str:
    return str(s).replace("\\", "\\\\").replace("=", "\\=").replace(";", "\\;").replace("#", "\\#").replace("\n", " ")
