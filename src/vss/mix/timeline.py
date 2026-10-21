"""Chapter timeline: lay out dialogue, ambience beds, music cues and effects, then mix.

Layout rules
* chapter head/tail are room tone only (ACX), content starts at ``chapter_head``
* a ``## scene`` heading closes the previous scene's ambience (crossfade) and resets the space
* ambience and music are ducked under dialogue (sidechain envelope with attack/release)
* effects are placed at the cursor (``@sfx`` / ``<sfx>``) and can hold the cursor (``wait=true``)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..audio import dsp
from ..audio.chain import apply_space
from ..audio.io import to_stereo
from ..render.plan import Cue, NonverbalClip, Pause, Plan, Unit


@dataclass
class DialogClip:
    start: float
    mono: np.ndarray
    speaker: str
    uid: str
    space: str
    pan: float
    line_id: int = 0
    text: str = ""
    kind: str = "speech"  # speech | nonverbal | breath

    def duration(self, sr: int) -> float:
        return len(self.mono) / sr


@dataclass
class Layer:
    kind: str  # ambience | music
    name: str
    start: float
    end: float | None = None
    gain_db: float = 0.0
    fade_in: float = 2.0
    fade_out: float = 2.5
    prompt: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    seed: str = ""


@dataclass
class SfxClip:
    start: float
    name: str
    gain_db: float = 0.0
    pan: float = 0.0
    params: dict[str, Any] = field(default_factory=dict)
    seed: str = ""


@dataclass
class Caption:
    start: float
    end: float
    speaker: str
    text: str


@dataclass
class Timeline:
    sr: int
    dialog: list[DialogClip] = field(default_factory=list)
    layers: list[Layer] = field(default_factory=list)
    sfx: list[SfxClip] = field(default_factory=list)
    captions: list[Caption] = field(default_factory=list)
    content_end: float = 0.0
    duration: float = 0.0
    sources: list[str] = field(default_factory=list)
    speech_to_bed_db: float | None = None


def parse_layers(spec: str) -> list[tuple[str, float]]:
    """``"sea_waves:-6, wind_soft:-12"`` -> [("sea_waves", -6.0), ("wind_soft", -12.0)]."""
    out = []
    for part in str(spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        name, _, gain = part.partition(":")
        out.append((name.strip(), float(gain) if gain.strip() else 0.0))
    return out


def layout(plan: Plan, clips: dict[str, np.ndarray], sr: int, cfg: dict, resolver, breath_fn=None,
           nonverbal_audio=None) -> Timeline:
    """Place everything on a timeline. ``clips`` maps unit uid -> processed mono audio."""
    r = cfg["render"]
    pauses = r["pauses"]
    tl = Timeline(sr)
    t = float(pauses.get("chapter_head", 0.8))
    space = "dry"
    open_amb: dict[str, Layer] = {}
    music: Layer | None = None
    narrator_in_space = bool(r.get("narrator_in_space", False))
    ch = plan.chapter.number
    sfx_count = 0
    line_span: dict[int, list] = {}

    def close_amb(at: float, fade: float = 2.5):
        for layer in open_amb.values():
            layer.end = at
            layer.fade_out = fade
        open_amb.clear()

    def open_layers(spec: str, at: float, fade: float = 2.0):
        for name, gain in parse_layers(spec):
            if name in open_amb:
                open_amb[name].gain_db = gain
                continue
            layer = Layer("ambience", name, at, gain_db=gain, fade_in=fade, seed=f"{ch}-{name}-{at:.1f}")
            open_amb[name] = layer
            tl.layers.append(layer)

    for step in plan.steps:
        if isinstance(step, Pause):
            t += step.seconds
        elif isinstance(step, Cue):
            k, a, p = step.kind, step.args, step.params
            if k == "scene":
                if step.title or "ambience" in p:
                    close_amb(t, float(p.get("fade", 2.5)))
                if step.title:
                    space = "dry"
                if "ambience" in p:
                    open_layers(p["ambience"], t, float(p.get("fade", 2.0)))
                if "space" in p:
                    space = str(p["space"])
            elif k == "space":
                space = a[0] if a else str(p.get("name", "dry"))
            elif k == "ambience":
                action = a[0].lower() if a else "set"
                rest = ", ".join(a[1:]) if a and action in ("add", "remove", "stop", "set") else ", ".join(a)
                fade = float(p.get("fade", 2.0))
                if action == "stop":
                    close_amb(t, fade)
                elif action == "remove":
                    for name, _ in parse_layers(rest):
                        if name in open_amb:
                            open_amb[name].end, open_amb[name].fade_out = t, fade
                            del open_amb[name]
                elif action == "add":
                    open_layers(rest, t, fade)
                else:
                    close_amb(t, fade)
                    open_layers(rest, t, fade)
            elif k == "music":
                action = a[0].lower() if a else "start"
                if action in ("stop", "end"):
                    if music:
                        music.end, music.fade_out = t, float(p.get("fade", 5.0))
                        music = None
                    continue
                name = a[1] if action == "start" and len(a) > 1 else (a[0] if a and action != "start" else p.get("name", "underscore"))
                fade = float(p.get("fade", 4.0))
                if music:
                    music.end, music.fade_out = t, fade
                music = Layer("music", str(name), t + float(p.get("offset", 0.0)), gain_db=float(p.get("gain", 0.0)),
                              fade_in=fade, prompt=p.get("prompt"), seed=f"{ch}-{name}-{t:.1f}",
                              params={kk: vv for kk, vv in p.items() if kk not in ("gain", "fade", "offset", "prompt")})
                if p.get("duration"):
                    music.end = music.start + float(p["duration"])
                    music.fade_out = float(p.get("fade_out", fade))
                    tl.layers.append(music)
                    music = None
                else:
                    tl.layers.append(music)
            elif k == "sfx":
                if not a:
                    continue
                sfx_count += 1
                at = t + float(p.get("offset", 0.0))
                clip = SfxClip(at, a[0], float(p.get("gain", 0.0)), float(p.get("pan", 0.0)),
                               {kk: vv for kk, vv in p.items() if kk not in ("gain", "offset", "wait", "pan")},
                               seed=f"{ch}-{a[0]}-{sfx_count}")
                tl.sfx.append(clip)
                if p.get("wait"):
                    snd = resolver.get("sfx", clip.name, prompt=clip.params.get("prompt"), seed=clip.seed,
                                       variant=sfx_count, **{kk: vv for kk, vv in clip.params.items() if kk != "prompt"})
                    clip.params["_resolved"] = snd
                    t = at + len(snd.audio) / sr + 0.15
        elif isinstance(step, (Unit, NonverbalClip)):
            t += step.gap_before
            if isinstance(step, Unit):
                mono = clips.get(step.uid)
                if mono is None or len(mono) == 0:
                    continue
                if breath_fn is not None and step.gap_before >= 0.3:
                    br = breath_fn(step)
                    if br is not None and len(br):
                        bstart = max(0.0, t - len(br) / sr - 0.05)
                        tl.dialog.append(DialogClip(bstart, br, step.speaker, step.uid + "-breath",
                                                    space if (narrator_in_space or not step.narrator) else "dry",
                                                    step.pan, kind="breath"))
                clip_space = space if (narrator_in_space or not step.narrator) else "dry"
                tl.dialog.append(DialogClip(t, mono, step.speaker, step.uid, clip_space, step.pan, step.line_id, step.display))
                end = t + len(mono) / sr
                span = line_span.setdefault(step.line_id, [t, end, step.speaker, step.line_text or step.display])
                span[1] = end
                t = end
            else:
                mono = nonverbal_audio(step) if nonverbal_audio else None
                if mono is None or len(mono) == 0:
                    continue
                tl.dialog.append(DialogClip(t, mono, step.speaker, f"nv-{step.nonverbal}", space, 0.0, kind="nonverbal"))
                t += len(mono) / sr
    tl.content_end = t
    tail_fade = 1.5
    for layer in tl.layers:
        if layer.end is None:
            layer.end = t + 0.3
            layer.fade_out = max(layer.fade_out, tail_fade)
    tl.duration = t + float(pauses.get("chapter_tail", 3.0))
    if tl.layers:  # let musical fade-outs finish, then keep a clean room-tone tail
        last = max(layer.end + layer.fade_out for layer in tl.layers)
        tl.duration = max(tl.duration, last + 1.5)
    tl.captions = [Caption(s, e, spk, txt) for _, (s, e, spk, txt) in sorted(line_span.items())]
    return tl


# ----------------------------------------------------------------------------- mixing
def duck_curve(dialog: np.ndarray, sr: int, depth_db: float, attack: float, release: float,
               threshold_db: float) -> np.ndarray:
    """Per-sample gain (linear) that dips ``depth_db`` while dialogue is active."""
    env = dsp.envelope(dialog, sr, attack=0.01, release=0.15, frame=0.01)
    env_db = 20 * np.log10(env + 1e-9)
    active = np.clip((env_db - threshold_db) / 10.0, 0.0, 1.0)
    # smooth the activity itself with musical attack/release
    act = dsp.envelope(active.astype(np.float32), sr, attack=attack, release=release, frame=0.01)
    return (10 ** (-depth_db * np.clip(act, 0, 1) / 20)).astype(np.float32)


def mixdown(tl: Timeline, resolver, cfg: dict, beds: str = "full",
            stems: bool = False) -> tuple[np.ndarray, list[str], dict[str, np.ndarray] | None]:
    """Mix the timeline. ``beds``: full | off | music | ambience (which layer kinds play).
    With ``stems`` the per-bus signals (dialog, ambience, music, sfx) are returned too."""
    sr = tl.sr
    mix_cfg = cfg["mix"]
    n = int(np.ceil(tl.duration * sr)) + 1
    out = np.zeros((n, 2), dtype=np.float32)
    sources: list[str] = []
    buses = {k: np.zeros((n, 2), dtype=np.float32) for k in ("ambience", "music", "sfx")} if stems else None

    # 1. dialogue bus (spaces + pan)
    for c in tl.dialog:
        st = apply_space(c.mono, sr, c.space, pan=c.pan) if (c.space != "dry" or c.pan) else to_stereo(c.mono)
        i = int(round(c.start * sr))
        e = min(n, i + len(st))
        if e > i:
            out[i:e] += st[: e - i]

    duck = mix_cfg["duck"]
    curve = duck_curve(out, sr, float(duck["depth_db"]), float(duck["attack"]), float(duck["release"]),
                       float(duck["threshold_db"]))
    music_extra = float(duck.get("music_extra_db", 2.0))
    curve_music = curve ** (1 + music_extra / max(1.0, float(duck["depth_db"])))

    dialog_stem = out.copy() if stems else None
    hop = int(0.4 * sr)
    frames = n // hop
    def _frame_energy(sig: np.ndarray, at: int) -> np.ndarray:
        e = np.zeros(frames)
        if frames == 0 or len(sig) == 0:
            return e
        full = np.zeros(frames * hop, dtype=np.float32)
        end = min(len(full), at + len(sig))
        if end > at:
            full[at:end] = (sig[: end - at] ** 2).mean(axis=1) if sig.ndim == 2 else sig[: end - at] ** 2
        return full.reshape(frames, hop).mean(axis=1)
    dialog_e = _frame_energy(out, 0)
    bed_e = np.zeros(frames)
    # 2. ambience + music layers
    for layer in tl.layers:
        if beds != "full" and beds != layer.kind:
            sources.append(f"{layer.kind} '{layer.name}' skipped (beds={beds})")
            continue
        start, end = max(0.0, layer.start), min(tl.duration, layer.end if layer.end is not None else tl.duration)
        length = end - start + layer.fade_out
        if length <= 0.05:
            continue
        try:
            snd = resolver.get(layer.kind, layer.name, seconds=length, prompt=layer.prompt, seed=layer.seed, **layer.params)
        except LookupError as exc:
            sources.append(f"MISSING {layer.kind} '{layer.name}': {exc}")
            continue
        sources.append(f"{layer.kind} '{layer.name}' <- {snd.source}")
        x = snd.audio[: int(length * sr)]
        fo = min(layer.fade_out, len(x) / sr / 2)
        x = dsp.fade(x, sr, min(layer.fade_in, len(x) / sr / 2), fo)
        bus = float(mix_cfg["ambience_db"] if layer.kind == "ambience" else mix_cfg["music_db"])
        g = dsp.db_to_gain(bus + layer.gain_db)
        i = int(round(start * sr))
        e = min(n, i + len(x))
        if e <= i:
            continue
        c = (curve if layer.kind == "ambience" else curve_music)[i:e, None]
        y = x[: e - i] * g * c
        out[i:e] += y
        bed_e += _frame_energy(y, i)
        if buses is not None:
            buses[layer.kind][i:e] += y

    # 3. effects (not ducked)
    for s in tl.sfx:
        snd = s.params.get("_resolved")
        if snd is None:
            try:
                snd = resolver.get("sfx", s.name, prompt=s.params.get("prompt"), seed=s.seed,
                                   **{k: v for k, v in s.params.items() if k not in ("prompt", "_resolved")})
            except LookupError as exc:
                sources.append(f"MISSING sfx '{s.name}': {exc}")
                continue
        sources.append(f"sfx '{s.name}' <- {snd.source}")
        x = snd.audio
        if s.pan:
            x = x * np.array([1 - max(0.0, s.pan), 1 + min(0.0, s.pan)], dtype=np.float32)
        g = dsp.db_to_gain(float(mix_cfg["sfx_db"]) + s.gain_db)
        i = int(round(s.start * sr))
        e = min(n, i + len(x))
        if e > i:
            out[i:e] += x[: e - i] * g
            if buses is not None:
                buses["sfx"][i:e] += x[: e - i] * g
    if buses is not None:
        buses["dialog"] = dialog_stem
    tl.speech_to_bed_db = speech_to_bed(dialog_e, bed_e)
    return out, sorted(set(sources)), buses


def speech_to_bed(dialog_e: np.ndarray, bed_e: np.ndarray) -> float | None:
    """Median level difference (dB) between dialogue and beds while someone is speaking."""
    d = 10 * np.log10(dialog_e + 1e-12)
    b = 10 * np.log10(bed_e + 1e-12)
    speaking = (d > -45) & (b > -100)
    if speaking.sum() < 3:
        return None
    return round(float(np.median(d[speaking]) - np.median(b[speaking])), 1)
