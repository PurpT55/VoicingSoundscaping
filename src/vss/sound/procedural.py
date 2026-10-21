"""Procedural soundscapes: ambience beds, one-shot effects and mood pads, synthesised from scratch.

No models, no samples, fully deterministic (seeded) and any length. Quality goal: convincing
*backgrounds* under narration (rain, wind, sea, fire, night, forest, stream, crowd ...) and
tasteful musical beds. For foreground hero sounds prefer a real library or a generative model.

Every generator returns stereo float32 normalised to about -20 LUFS (beds/music) so the mixer
can place them with simple relative gains.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, fftconvolve, lfilter, sosfilt

from ..audio import dsp
from ..audio.loudness import normalize_lufs

Generator = Callable[..., np.ndarray]


@dataclass(frozen=True)
class Proc:
    name: str
    kind: str  # bed | oneshot | music
    fn: Generator
    description: str
    default_seconds: float
    aliases: tuple[str, ...] = ()


def rng_for(*keys) -> np.random.Generator:
    h = hashlib.sha256("|".join(map(str, keys)).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


# ------------------------------------------------------------------------------ helpers
def _n(seconds: float, sr: int) -> int:
    return max(1, int(round(seconds * sr)))


def pink(n: int, rng: np.random.Generator) -> np.ndarray:
    """Pink noise via Paul Kellet's economy filter."""
    b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
    a = [1, -2.494956002, 2.017265875, -0.522189400]
    y = lfilter(b, a, rng.standard_normal(n + 2000))[2000:]
    return (y / (np.std(y) + 1e-9)).astype(np.float32)


def brown(n: int, rng: np.random.Generator, sr: int = 44100) -> np.ndarray:
    y = lfilter([1.0], [1.0, -0.997], rng.standard_normal(n + 4000))[4000:]
    y = sosfilt(butter(2, 20 / (sr / 2), "highpass", output="sos"), y)
    return (y / (np.std(y) + 1e-9)).astype(np.float32)


def lfo(n: int, sr: int, rate_hz: float, rng: np.random.Generator, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    """Smooth random control signal in [lo, hi] changing about ``rate_hz`` times per second."""
    pts = max(4, int(n / sr * rate_hz) + 4)
    knots = rng.uniform(0, 1, pts)
    pos = np.arange(n, dtype=np.float64) * (pts - 1) / max(1, n - 1)
    i = np.minimum(pos.astype(np.int64), pts - 2)
    frac = pos - i
    w = (1 - np.cos(np.pi * frac)) / 2  # cosine interpolation: smooth, no overshoot
    y = knots[i] * (1 - w) + knots[i + 1] * w
    return (lo + (hi - lo) * y).astype(np.float32)


def varying_bandpass(x: np.ndarray, sr: int, center: np.ndarray, q: float = 1.0, block: int = 2048) -> np.ndarray:
    """Band-pass with a slowly moving centre frequency (block-wise coefficients, carried state)."""
    out = np.empty_like(x)
    zi = np.zeros((1, 2))
    nyq = sr / 2
    for s in range(0, len(x), block):
        c = float(np.clip(center[min(s + block // 2, len(center) - 1)], 40, nyq * 0.9))
        # quantise to 1/24 octave so coefficient design is cached
        step = round(24 * np.log2(c / 40.0))
        out[s:s + block], zi = sosfilt(_band_sos(step, round(q * 4), sr), x[s:s + block], zi=zi)
    return out


@lru_cache(maxsize=4096)
def _band_sos(step: int, q4: int, sr: int):
    nyq = sr / 2
    c = 40.0 * 2 ** (step / 24)
    q = max(0.25, q4 / 4)
    lo, hi = c / (1 + 0.5 / q), min(c * (1 + 0.5 / q), nyq * 0.95)
    return butter(1, [lo / nyq, hi / nyq], btype="band", output="sos")


def scatter(n: int, rate: float, sr: int, rng: np.random.Generator, amp_sigma: float = 0.6) -> np.ndarray:
    """Poisson impulse train with log-normal amplitudes."""
    count = rng.poisson(rate * n / sr)
    imp = np.zeros(n, dtype=np.float32)
    if count:
        pos = rng.integers(0, n, count)
        np.add.at(imp, pos, rng.lognormal(0, amp_sigma, count).astype(np.float32))
    return imp


def decay_kernel(sr: int, ms: float, rng: np.random.Generator, lo: float, hi: float) -> np.ndarray:
    k = _n(ms / 1000 * 5, sr)
    t = np.arange(k) / sr
    noise = rng.standard_normal(k)
    noise = dsp.bandpass(noise.astype(np.float32), sr, lo, hi)
    return (noise * np.exp(-t / (ms / 1000))).astype(np.float32)


def overlay(dst: np.ndarray, src: np.ndarray, at: int, gain: float = 1.0) -> None:
    if at >= len(dst) or at + len(src) <= 0:
        return
    s0 = max(0, -at)
    e = min(len(src), len(dst) - at)
    dst[at + s0: at + e] += src[s0:e] * gain


def stereo(l: np.ndarray, r: np.ndarray) -> np.ndarray:
    return np.stack([l, r], axis=1).astype(np.float32)


def finish(x: np.ndarray, sr: int, target_lufs: float = -20.0, fade: float = 0.5) -> np.ndarray:
    x = np.nan_to_num(x).astype(np.float32)
    if x.ndim == 1:
        x = stereo(x, x)
    x = dsp.fade(x, sr, min(fade, len(x) / sr / 4), min(fade, len(x) / sr / 4))
    y = normalize_lufs(x, sr, target_lufs)
    return dsp.limiter(y, sr, -1.0, oversample=1)


def _place_events(n: int, sr: int, rng, rate: float, make: Callable[[], np.ndarray], spread: float = 0.8):
    """Overlap-add stereo events made by ``make()`` at Poisson times with random pan."""
    L = np.zeros(n, np.float32)
    R = np.zeros(n, np.float32)
    count = rng.poisson(rate * n / sr)
    for at in np.sort(rng.integers(0, n, count)):
        ev = make()
        p = rng.uniform(-spread, spread)
        gl, gr = np.cos((p + 1) * np.pi / 4), np.sin((p + 1) * np.pi / 4)
        overlay(L, ev, int(at), gl)
        overlay(R, ev, int(at), gr)
    return L, R


# ---------------------------------------------------------------------------- beds
def room_tone(seconds: float, sr: int, rng, **_) -> np.ndarray:
    n = _n(seconds, sr)
    l = dsp.lowpass(pink(n, rng), sr, 3500)
    r = dsp.lowpass(pink(n, rng), sr, 3500)
    hvac = dsp.lowpass(brown(n, rng, sr), sr, 180) * 0.6
    return finish(stereo(l * 0.7 + hvac, r * 0.7 + hvac), sr)


def rain(seconds: float, sr: int, rng, intensity: float = 0.5, **_) -> np.ndarray:
    n = _n(seconds, sr)
    swell = lfo(n, sr, 0.08, rng, 0.75, 1.0)
    bed = [dsp.bandpass(pink(n, rng), sr, 500, 9000) * swell for _ in range(2)]
    drops = []
    for _ in range(2):
        layer = np.zeros(n, np.float32)
        for lo, hi, ms in ((1800, 6000, 3), (900, 3000, 6), (3000, 9000, 1.5)):
            imp = scatter(n, 60 + 900 * intensity, sr, rng, 0.8)
            layer += fftconvolve(imp, decay_kernel(sr, ms, rng, lo, hi), mode="full")[:n].astype(np.float32)
        drops.append(layer / (np.std(layer) + 1e-9))
    rumble = dsp.lowpass(brown(n, rng, sr), sr, 300) * (0.2 + 0.5 * intensity)
    mix_bed = 0.55 + 0.4 * intensity
    return finish(stereo(bed[0] * mix_bed + drops[0] * 0.35 + rumble, bed[1] * mix_bed + drops[1] * 0.35 + rumble), sr)


def wind(seconds: float, sr: int, rng, strength: float = 0.5, **_) -> np.ndarray:
    n = _n(seconds, sr)
    chans = []
    gust = lfo(n, sr, 0.12 + 0.2 * strength, rng, 0.25, 1.0)
    for _ in range(2):
        src = brown(n, rng, sr) * 0.6 + pink(n, rng) * 0.4
        center = lfo(n, sr, 0.15, rng, 250 + 300 * strength, 900 + 1400 * strength)
        whistle = varying_bandpass(src, sr, center, q=2.5 + 3 * strength)
        body = dsp.lowpass(src, sr, 500 + 600 * strength)
        g = gust * lfo(n, sr, 0.3, rng, 0.8, 1.0)
        chans.append((body * 0.6 + whistle * 1.4) * g)
    return finish(stereo(*chans), sr)


def sea_waves(seconds: float, sr: int, rng, calm: float = 0.5, **_) -> np.ndarray:
    n = _n(seconds, sr)
    env = np.zeros(n, np.float32)
    foam = np.zeros(n, np.float32)
    t = 0.0
    while t < seconds + 10:
        period = rng.uniform(6.0, 11.0) * (0.8 + 0.4 * calm)
        rise, fall = rng.uniform(1.2, 2.2), rng.uniform(3.0, 5.5)
        peak = rng.uniform(0.6, 1.0)
        k = _n(rise + fall, sr)
        tt = np.arange(k) / sr
        shape = np.where(tt < rise, (tt / rise) ** 2, np.exp(-(tt - rise) / (fall / 3)))
        at = int(t * sr) - _n(rng.uniform(0, 1), sr)
        overlay(env, (shape * peak).astype(np.float32), at)
        crest = np.exp(-((tt - rise - 0.3) ** 2) / 0.6)
        overlay(foam, (crest * peak).astype(np.float32), at)
        t += period
    env = np.clip(env, 0, 1.2)
    out = []
    for d in (0, int(0.21 * sr)):
        e = np.roll(env, d)
        f = np.roll(foam, d)
        src = pink(n, rng)
        # brighter as the wave breaks: blend lowpassed and full-band noise by envelope
        dark = dsp.lowpass(src, sr, 350)
        mid = dsp.bandpass(src, sr, 300, 2500)
        hiss = dsp.highpass(pink(n, rng), sr, 2500)
        out.append(dark * (0.35 + 0.65 * e) + mid * e * 0.8 + hiss * f * 0.45)
    return finish(stereo(*out), sr)


def stream(seconds: float, sr: int, rng, **_) -> np.ndarray:
    n = _n(seconds, sr)
    bed = [dsp.bandpass(pink(n, rng), sr, 250, 3500) * lfo(n, sr, 0.5, rng, 0.7, 1.0) for _ in range(2)]

    def bubble():
        dur = rng.uniform(0.015, 0.06)
        k = _n(dur, sr)
        tt = np.arange(k) / sr
        f0 = rng.uniform(350, 1600)
        f = f0 * (1 + tt / dur * rng.uniform(0.4, 1.4))
        ph = 2 * np.pi * np.cumsum(f) / sr
        return (np.sin(ph) * np.exp(-tt / (dur / 3)) * rng.uniform(0.2, 1.0)).astype(np.float32)

    L, R = _place_events(n, sr, rng, 55, bubble)
    return finish(stereo(bed[0] * 0.8 + L * 1.6, bed[1] * 0.8 + R * 1.6), sr)


def fire(seconds: float, sr: int, rng, size: float = 0.5, **_) -> np.ndarray:
    n = _n(seconds, sr)
    roar = dsp.lowpass(brown(n, rng, sr), sr, 300 + 300 * size) * lfo(n, sr, 0.7, rng, 0.6, 1.0)
    hiss = dsp.bandpass(pink(n, rng), sr, 1500, 7000) * lfo(n, sr, 2.0, rng, 0.1, 0.35)
    chans = []
    for _ in range(2):
        crack = np.zeros(n, np.float32)
        for rate, ms, lo, hi, g in ((14, 1.2, 1500, 8000, 1.0), (4, 3.5, 600, 3000, 1.4), (0.6, 8, 200, 1500, 2.2)):
            imp = scatter(n, rate * (0.6 + size), sr, rng, 1.0)
            crack += fftconvolve(imp, decay_kernel(sr, ms, rng, lo, hi), mode="full")[:n].astype(np.float32) * g
        crack /= np.std(crack) + 1e-9
        chans.append(roar * 0.9 + hiss * 0.6 + crack * 0.5)
    return finish(stereo(*chans), sr)


def _cricket(sr: int, rng, seconds: float) -> np.ndarray:
    n = _n(seconds, sr)
    f = rng.uniform(3800, 5200)
    carrier = np.sin(2 * np.pi * f * np.arange(n) / sr)
    pulse_rate = rng.uniform(22, 34)
    pulses = rng.integers(2, 5)
    period = rng.uniform(0.45, 1.1)
    gate = np.zeros(n, np.float32)
    pl = _n(0.6 / pulse_rate, sr)
    win = np.hanning(pl)
    t = rng.uniform(0, period)
    while t < seconds:
        for k in range(pulses):
            overlay(gate, win.astype(np.float32), int((t + k / pulse_rate) * sr))
        t += period * rng.uniform(0.9, 1.1)
    return (carrier * gate).astype(np.float32)


def night(seconds: float, sr: int, rng, crickets: int = 5, **_) -> np.ndarray:
    n = _n(seconds, sr)
    L = np.zeros(n, np.float32)
    R = np.zeros(n, np.float32)
    for _ in range(crickets):
        c = _cricket(sr, rng, seconds)
        c = dsp.lowpass(c, sr, rng.uniform(5000, 9000)) * rng.uniform(0.2, 1.0)
        p = rng.uniform(-0.9, 0.9)
        L += c * np.cos((p + 1) * np.pi / 4)
        R += c * np.sin((p + 1) * np.pi / 4)
    breeze = wind(seconds, sr, rng, strength=0.15) * 0.25
    return finish(stereo(L, R) * 0.6 + breeze, sr)


def _bird_phrase(sr: int, rng) -> np.ndarray:
    syllables = rng.integers(2, 8)
    base = rng.uniform(2200, 5500)
    parts = []
    for _ in range(syllables):
        dur = rng.uniform(0.03, 0.14)
        k = _n(dur, sr)
        tt = np.linspace(0, 1, k)
        sweep = base * (1 + rng.uniform(-0.35, 0.35) * np.sin(np.pi * tt * rng.uniform(0.5, 2.0)) + rng.uniform(-0.2, 0.3) * tt)
        trill = 1 + 0.04 * np.sin(2 * np.pi * rng.uniform(30, 90) * tt * dur)
        ph = 2 * np.pi * np.cumsum(sweep * trill) / sr
        amp = np.sin(np.pi * tt) ** 1.5
        syl = np.sin(ph) * amp + 0.15 * np.sin(2 * ph) * amp
        parts.append(syl.astype(np.float32))
        parts.append(np.zeros(_n(rng.uniform(0.02, 0.09), sr), np.float32))
    return np.concatenate(parts) * rng.uniform(0.15, 1.0)


def forest_day(seconds: float, sr: int, rng, birds: float = 0.5, **_) -> np.ndarray:
    n = _n(seconds, sr)
    leaves = wind(seconds, sr, rng, strength=0.3)
    leaves = np.stack([dsp.highpass(leaves[:, c], sr, 900) for c in range(2)], axis=1) * 0.5
    L, R = _place_events(n, sr, rng, 0.4 + 1.2 * birds, lambda: dsp.lowpass(_bird_phrase(sr, rng), sr, 9000), spread=0.95)
    b = stereo(L, R)
    import pedalboard

    b = pedalboard.Pedalboard([pedalboard.Reverb(room_size=0.5, damping=0.7, wet_level=0.25, dry_level=0.8)])(b.T.copy(), sr).T
    return finish(leaves + b * 0.9, sr)


def city(seconds: float, sr: int, rng, **_) -> np.ndarray:
    n = _n(seconds, sr)
    rumble = dsp.lowpass(brown(n, rng, sr), sr, 220) * lfo(n, sr, 0.1, rng, 0.6, 1.0)
    L = rumble.copy()
    R = rumble.copy()

    def car():
        dur = rng.uniform(3.0, 7.0)
        k = _n(dur, sr)
        tt = np.linspace(-1, 1, k)
        env = np.exp(-(tt ** 2) * 5)
        src = pink(k, rng)
        return (dsp.bandpass(src, sr, 120, 2500) * env).astype(np.float32)

    cl, cr = _place_events(n, sr, rng, 0.25, car, spread=1.0)
    hum = dsp.bandpass(pink(n, rng), sr, 400, 3000) * 0.15
    return finish(stereo(L + cl * 0.8 + hum, R + cr * 0.8 + hum), sr)


def crowd(seconds: float, sr: int, rng, people: int = 14, **_) -> np.ndarray:
    """Babble texture: band-limited noise 'syllables' with formant colour and speech rhythm."""
    n = _n(seconds, sr)
    L = np.zeros(n, np.float32)
    R = np.zeros(n, np.float32)
    for _ in range(people):
        src = pink(n, rng)
        f1 = lfo(n, sr, rng.uniform(3, 6), rng, 350, 900)
        voice = varying_bandpass(src, sr, f1, q=2.0, block=512) + 0.5 * dsp.bandpass(src, sr, 1200, 2800)
        syll = lfo(n, sr, rng.uniform(4, 7), rng, 0, 1) ** 2
        talk = (lfo(n, sr, 0.25, rng, 0, 1) > 0.35).astype(np.float32)
        talk = uniform_filter1d(talk, size=_n(0.15, sr))
        v = voice * syll * talk * rng.uniform(0.4, 1.0)
        p = rng.uniform(-0.9, 0.9)
        L += v * np.cos((p + 1) * np.pi / 4)
        R += v * np.sin((p + 1) * np.pi / 4)
    out = stereo(dsp.lowpass(L, sr, 4000), dsp.lowpass(R, sr, 4000))
    import pedalboard

    out = pedalboard.Pedalboard([pedalboard.Reverb(room_size=0.6, damping=0.5, wet_level=0.35, dry_level=0.7)])(out.T.copy(), sr).T
    return finish(out, sr)


def tavern(seconds: float, sr: int, rng, **kw) -> np.ndarray:
    n = _n(seconds, sr)
    base = crowd(seconds, sr, rng, people=10)
    clinks = np.zeros(n, np.float32)
    count = rng.poisson(0.35 * seconds)
    for at in rng.integers(0, n, count):
        overlay(clinks, glass_clink(0.6, sr, rng)[:, 0], int(at), rng.uniform(0.2, 0.6))
    fireplace = fire(seconds, sr, rng, size=0.3) * 0.25
    return finish(base + stereo(clinks, clinks) * 0.5 + fireplace, sr)


def underwater(seconds: float, sr: int, rng, **_) -> np.ndarray:
    n = _n(seconds, sr)
    body = [dsp.lowpass(brown(n, rng, sr), sr, 500) * lfo(n, sr, 0.2, rng, 0.6, 1.0) for _ in range(2)]

    def bub():
        k = _n(rng.uniform(0.02, 0.08), sr)
        tt = np.arange(k) / sr
        f = rng.uniform(250, 900) * (1 + tt * 12)
        return (np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-tt * 40)).astype(np.float32)

    L, R = _place_events(n, sr, rng, 6, bub)
    return finish(stereo(body[0] + L * 0.6, body[1] + R * 0.6), sr)


def cave(seconds: float, sr: int, rng, **_) -> np.ndarray:
    n = _n(seconds, sr)
    tone = room_tone(seconds, sr, rng) * 0.6
    drips = np.zeros(n, np.float32)
    for at in rng.integers(0, n, rng.poisson(0.5 * seconds)):
        overlay(drips, water_drip(0.4, sr, rng)[:, 0], int(at), rng.uniform(0.3, 1.0))
    import pedalboard

    d = pedalboard.Pedalboard([pedalboard.Reverb(room_size=0.92, damping=0.2, wet_level=0.6, dry_level=0.5)])(
        stereo(drips, drips).T.copy(), sr).T
    return finish(tone + d, sr)


def clock_ticking(seconds: float, sr: int, rng, bpm: float = 60, **_) -> np.ndarray:
    n = _n(seconds, sr)
    x = np.zeros(n, np.float32)
    step = 60.0 / bpm
    t, k = 0.1, 0
    while t < seconds:
        overlay(x, clock_tick(0.1, sr, rng, tock=bool(k % 2))[:, 0], int(t * sr))
        t += step
        k += 1
    return finish(stereo(x, x) + room_tone(seconds, sr, rng) * 0.05, sr)


def heartbeat(seconds: float, sr: int, rng, bpm: float = 70, **_) -> np.ndarray:
    n = _n(seconds, sr)
    x = np.zeros(n, np.float32)
    t = 0.1
    while t < seconds:
        for off, g in ((0.0, 1.0), (0.28, 0.7)):
            k = _n(0.12, sr)
            tt = np.arange(k) / sr
            thump = np.sin(2 * np.pi * (55 - 120 * tt) * tt) * np.exp(-tt * 35) * g
            overlay(x, thump.astype(np.float32), int((t + off) * sr))
        t += 60.0 / bpm
    return finish(stereo(x, x), sr)


# ----------------------------------------------------------------------------- one-shots
def thunder(seconds: float, sr: int, rng, distance: float = 0.5, **_) -> np.ndarray:
    n = _n(seconds, sr)
    t = np.arange(n) / sr
    env = np.exp(-t / (1.2 + 2.5 * distance)) * lfo(n, sr, 3.0, rng, 0.4, 1.0)
    attack = np.clip(t / (0.02 + 0.4 * distance), 0, 1)
    rumble = [dsp.lowpass(brown(n, rng, sr), sr, 250 + 500 * (1 - distance)) * env * attack for _ in range(2)]
    crack = np.zeros(n, np.float32)
    if distance < 0.6:
        k = _n(0.4, sr)
        crack[:k] = dsp.bandpass(pink(k, rng), sr, 800, 7000) * np.exp(-np.arange(k) / sr / 0.06) * (1 - distance) * 3
    return finish(stereo(rumble[0] + crack, rumble[1] + crack), sr, -18.0, fade=0.01)


def _partials(seconds: float, sr: int, f0: float, ratios, decays, amps, rng) -> np.ndarray:
    n = _n(seconds, sr)
    t = np.arange(n) / sr
    x = np.zeros(n)
    for r, d, a in zip(ratios, decays, amps):
        x += a * np.sin(2 * np.pi * f0 * r * t + rng.uniform(0, 2 * np.pi)) * np.exp(-t / d)
    x *= np.clip(t / 0.002, 0, 1)
    return x.astype(np.float32)


def bell(seconds: float, sr: int, rng, pitch: float = 440.0, **_) -> np.ndarray:
    ratios = (0.5, 1.0, 1.183, 1.506, 2.0, 2.514, 2.662, 3.011, 4.166)
    decays = tuple(seconds * d for d in (0.9, 0.7, 0.55, 0.45, 0.35, 0.25, 0.22, 0.18, 0.12))
    amps = (0.6, 1.0, 0.7, 0.5, 0.45, 0.3, 0.25, 0.2, 0.12)
    x = _partials(seconds, sr, pitch, ratios, decays, amps, rng)
    return finish(stereo(x, x), sr, -20.0, fade=0.005)


def chime(seconds: float, sr: int, rng, notes: int = 5, **_) -> np.ndarray:
    n = _n(seconds, sr)
    L = np.zeros(n, np.float32)
    R = np.zeros(n, np.float32)
    scale = (523.25, 587.33, 659.25, 783.99, 880.0, 1046.5, 1174.66)
    for i in range(notes):
        f = scale[rng.integers(0, len(scale))]
        tone = _partials(min(2.5, seconds), sr, f, (1, 2.76, 5.4, 8.93), (1.2, 0.5, 0.25, 0.12), (1, 0.4, 0.2, 0.1), rng)
        at = int(rng.uniform(0, max(0.01, seconds - 2.5)) * sr) if i else 0
        p = rng.uniform(-0.6, 0.6)
        overlay(L, tone, at, np.cos((p + 1) * np.pi / 4))
        overlay(R, tone, at, np.sin((p + 1) * np.pi / 4))
    return finish(stereo(L, R), sr, -20.0, fade=0.005)


def sparkle(seconds: float, sr: int, rng, **_) -> np.ndarray:
    """Magic shimmer: a rising cloud of tiny high grains with a glassy tail."""
    n = _n(seconds, sr)
    L = np.zeros(n, np.float32)
    R = np.zeros(n, np.float32)
    grains = int(40 * seconds)
    for i in range(grains):
        pos = i / max(1, grains)
        f = 1800 + 4500 * pos * rng.uniform(0.7, 1.2)
        k = _n(rng.uniform(0.05, 0.25), sr)
        tt = np.arange(k) / sr
        g = (np.sin(2 * np.pi * f * tt) * np.exp(-tt * rng.uniform(10, 30))).astype(np.float32)
        at = int(pos * (n - k)) + int(rng.normal(0, 0.03) * sr)
        p = rng.uniform(-0.9, 0.9)
        overlay(L, g, at, np.cos((p + 1) * np.pi / 4) * (0.3 + pos))
        overlay(R, g, at, np.sin((p + 1) * np.pi / 4) * (0.3 + pos))
    import pedalboard

    y = pedalboard.Pedalboard([pedalboard.Reverb(room_size=0.85, damping=0.2, wet_level=0.6, dry_level=0.6)])(stereo(L, R).T.copy(), sr).T
    return finish(y, sr, -20.0, fade=0.02)


def whoosh(seconds: float, sr: int, rng, **_) -> np.ndarray:
    n = _n(seconds, sr)
    t = np.linspace(0, 1, n)
    env = np.sin(np.pi * t) ** 2
    center = 300 + 3500 * np.sin(np.pi * t) ** 1.5
    chans = [varying_bandpass(pink(n, rng), sr, center, q=1.5, block=256) * env for _ in range(2)]
    panned = stereo(chans[0] * (1 - t * 0.6), chans[1] * (0.4 + t * 0.6))
    return finish(panned, sr, -20.0, fade=0.01)


def knock(seconds: float, sr: int, rng, count: int = 3, **_) -> np.ndarray:
    n = _n(seconds, sr)
    x = np.zeros(n, np.float32)
    for i in range(count):
        k = _n(0.15, sr)
        tt = np.arange(k) / sr
        body = sum(np.sin(2 * np.pi * f * tt) * np.exp(-tt * d) for f, d in ((140, 35), (260, 50), (520, 90)))
        click = dsp.bandpass(rng.standard_normal(k).astype(np.float32), sr, 1500, 5000) * np.exp(-tt * 300)
        overlay(x, (body * 0.8 + click * 0.5).astype(np.float32), int((0.05 + i * rng.uniform(0.18, 0.24)) * sr),
                rng.uniform(0.8, 1.0))
    return finish(stereo(x, x), sr, -20.0, fade=0.003)


def water_drip(seconds: float, sr: int, rng, **_) -> np.ndarray:
    n = _n(seconds, sr)
    k = _n(0.05, sr)
    tt = np.arange(k) / sr
    f = rng.uniform(900, 1600) * (1 + 18 * tt)
    d = (np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-tt * 70)).astype(np.float32)
    x = np.zeros(n, np.float32)
    overlay(x, d, int(0.01 * sr))
    return finish(stereo(x, x), sr, -20.0, fade=0.002)


def clock_tick(seconds: float, sr: int, rng, tock: bool = False, **_) -> np.ndarray:
    n = _n(seconds, sr)
    tt = np.arange(n) / sr
    f = 1800 if tock else 2400
    x = (np.sin(2 * np.pi * f * tt) * np.exp(-tt * 260) + 0.4 * rng.standard_normal(n) * np.exp(-tt * 600)).astype(np.float32)
    return finish(stereo(x, x), sr, -24.0, fade=0.001)


def glass_clink(seconds: float, sr: int, rng, **_) -> np.ndarray:
    x = _partials(seconds, sr, rng.uniform(1800, 2600), (1, 2.32, 4.25, 6.63), (0.25, 0.12, 0.06, 0.04), (1, 0.5, 0.3, 0.2), rng)
    return finish(stereo(x, x), sr, -22.0, fade=0.002)


def page_turn(seconds: float, sr: int, rng, **_) -> np.ndarray:
    n = _n(seconds, sr)
    t = np.linspace(0, 1, n)
    env = np.sin(np.pi * np.clip(t * 1.2, 0, 1)) ** 2
    crinkle = scatter(n, 900, sr, rng, 1.2)
    crinkle = fftconvolve(crinkle, decay_kernel(sr, 0.8, rng, 2000, 9000), mode="full")[:n]
    swish = dsp.bandpass(pink(n, rng), sr, 1500, 8000) * 0.5
    x = (crinkle / (np.std(crinkle) + 1e-9) * 0.5 + swish) * env
    return finish(stereo(x, np.roll(x, 40)), sr, -24.0, fade=0.01)


def owl(seconds: float, sr: int, rng, **_) -> np.ndarray:
    n = _n(seconds, sr)
    x = np.zeros(n, np.float32)
    f0 = rng.uniform(330, 420)
    t = 0.05
    for dur, gap in ((0.35, 0.25), (0.18, 0.12), (0.18, 0.12), (0.55, 0.0)):
        k = _n(dur, sr)
        tt = np.linspace(0, 1, k)
        f = f0 * (1 + 0.05 * np.sin(np.pi * tt)) * (1 + 0.01 * np.sin(2 * np.pi * 6 * tt * dur))
        ph = 2 * np.pi * np.cumsum(f) / sr
        hoot = (np.sin(ph) + 0.25 * np.sin(2 * ph)) * np.sin(np.pi * tt) ** 1.2
        overlay(x, hoot.astype(np.float32), int(t * sr))
        t += dur + gap
    import pedalboard

    y = pedalboard.Pedalboard([pedalboard.Reverb(room_size=0.6, damping=0.6, wet_level=0.35, dry_level=0.8)])(stereo(x, x).T.copy(), sr).T
    return finish(y, sr, -22.0, fade=0.01)


def riser(seconds: float, sr: int, rng, **_) -> np.ndarray:
    n = _n(seconds, sr)
    t = np.linspace(0, 1, n)
    center = 200 * (30 ** t)
    noise = [varying_bandpass(pink(n, rng), sr, center, q=2.0, block=256) * t ** 2 for _ in range(2)]
    tone = np.sin(2 * np.pi * np.cumsum(110 * (4 ** t)) / sr) * t ** 3 * 0.3
    return finish(stereo(noise[0] + tone, noise[1] + tone), sr, -20.0, fade=0.01)


def singing_bowl(seconds: float, sr: int, rng, pitch: float = 220.0, **_) -> np.ndarray:
    n = _n(seconds, sr)
    t = np.arange(n) / sr
    x = np.zeros(n)
    for r, a, d in ((1.0, 1.0, 0.9), (2.71, 0.5, 0.6), (5.13, 0.25, 0.4), (8.21, 0.1, 0.25)):
        beat = 1 + 0.25 * np.sin(2 * np.pi * rng.uniform(0.5, 2.0) * t)
        x += a * beat * np.sin(2 * np.pi * pitch * r * t) * np.exp(-t / (seconds * d))
    x *= np.clip(t / 0.01, 0, 1)
    return finish(stereo(x, np.roll(x, 90)), sr, -20.0, fade=0.01)


def breath(seconds: float, sr: int, rng, depth: float = 0.5, **_) -> np.ndarray:
    """A soft inhale before a phrase (used when no breath bank exists)."""
    n = _n(seconds, sr)
    t = np.linspace(0, 1, n)
    env = np.sin(np.pi * np.clip(t * 1.1, 0, 1)) ** 1.6 * (0.6 + 0.4 * t)
    src = pink(n, rng)
    formant = dsp.bandpass(src, sr, 500, 2400) * 0.7 + dsp.bandpass(src, sr, 2400, 6000) * 0.3 * depth
    x = formant * env
    return finish(stereo(x, x), sr, -20.0, fade=0.02)


# ---------------------------------------------------------------------------- mood pads
MOODS = {
    # root midi note, chord progression as scale-degree triads/extended voicings (semitones from root)
    "wonder": (57, [(0, 7, 12, 16, 23), (5, 12, 16, 21, 26), (2, 9, 14, 17, 24), (7, 14, 19, 23, 26)], "major"),
    "warm": (55, [(0, 7, 12, 16), (5, 12, 17, 21), (9, 16, 21, 24), (7, 14, 19, 23)], "major"),
    "melancholy": (57, [(0, 7, 12, 15), (8, 15, 20, 24), (3, 10, 15, 19), (10, 17, 22, 26)], "minor"),
    "mystery": (50, [(0, 7, 14, 15), (1, 8, 13, 17), (0, 7, 12, 15, 26), (-2, 5, 10, 14)], "minor"),
    "tension": (45, [(0, 6, 12, 13), (0, 7, 13, 18), (-1, 6, 11, 12), (0, 6, 12, 15)], "dim"),
    "lullaby": (60, [(0, 7, 12, 16), (5, 9, 12, 17), (7, 11, 14, 19), (0, 4, 7, 12)], "major"),
    "adventure": (52, [(0, 7, 12, 16, 19), (-4, 3, 8, 12), (-2, 5, 10, 14), (5, 12, 17, 21)], "major"),
    "triumph": (55, [(0, 7, 12, 16, 19), (5, 12, 17, 21), (-3, 4, 9, 12, 16), (7, 14, 19, 23)], "major"),
}


def _midi(m: float) -> float:
    return 440.0 * 2 ** ((m - 69) / 12)


def pad(seconds: float, sr: int, rng, mood: str = "wonder", chord_seconds: float = 8.0,
        arpeggio: bool | None = None, **_) -> np.ndarray:
    root, prog, _kind = MOODS.get(mood, MOODS["wonder"])
    n = _n(seconds, sr)
    L = np.zeros(n, np.float32)
    R = np.zeros(n, np.float32)
    seg = _n(chord_seconds, sr)
    xf = _n(min(2.5, chord_seconds / 3), sr)
    i = 0
    for start in range(0, n, seg):
        chord = prog[i % len(prog)]
        i += 1
        k = min(seg + xf, n - start)
        if k <= 0:
            break
        t = np.arange(k) / sr
        env = np.minimum(1.0, t / (xf / sr)) * np.minimum(1.0, (k / sr - t) / (xf / sr) + 1e-9)
        env = np.clip(env, 0, 1) ** 1.5
        for v, semi in enumerate(chord):
            f = _midi(root + semi)
            for side, (buf, det) in enumerate(((L, -0.07), (R, 0.07))):
                ph0 = rng.uniform(0, 2 * np.pi)
                vib = 1 + 0.0015 * np.sin(2 * np.pi * (0.13 + 0.05 * v) * t + ph0)
                ff = f * (2 ** (det / 12)) * vib
                ph = 2 * np.pi * np.cumsum(ff) / sr + ph0
                voice = np.sin(ph) + 0.35 * np.sin(2 * ph) + 0.12 * np.sin(3 * ph) + 0.05 * np.sin(4 * ph)
                overlay(buf, (voice * env / (1 + v * 0.35)).astype(np.float32), start, 0.25)
    if arpeggio is None:
        arpeggio = mood in ("lullaby", "wonder")
    if arpeggio:
        step = 60.0 / (72 if mood == "lullaby" else 90) / 2
        t0 = 0.0
        j = 0
        while t0 < seconds - 1.0:
            chord = prog[int(t0 / chord_seconds) % len(prog)]
            note = root + 12 + chord[j % len(chord)] + (12 if (j // len(chord)) % 2 else 0)
            tone = _partials(1.6, sr, _midi(note), (1, 2.0, 3.0, 4.2), (0.8, 0.4, 0.2, 0.1), (1, 0.25, 0.08, 0.05), rng)
            p = np.sin(j * 0.7) * 0.4
            overlay(L, tone, int(t0 * sr), 0.18 * np.cos((p + 1) * np.pi / 4))
            overlay(R, tone, int(t0 * sr), 0.18 * np.sin((p + 1) * np.pi / 4))
            t0 += step
            j += 1
    y = stereo(dsp.lowpass(L, sr, 5000), dsp.lowpass(R, sr, 5000))
    import pedalboard

    y = pedalboard.Pedalboard([
        pedalboard.Chorus(rate_hz=0.25, depth=0.2, mix=0.3),
        pedalboard.Reverb(room_size=0.9, damping=0.4, wet_level=0.45, dry_level=0.65, width=1.0),
    ])(y.T.copy(), sr).T
    return finish(y, sr, -20.0, fade=2.0)


# ------------------------------------------------------------------------------- registry
REGISTRY: dict[str, Proc] = {}


def _reg(name, kind, fn, desc, secs, aliases=()):
    REGISTRY[name] = Proc(name, kind, fn, desc, secs, tuple(aliases))


_reg("room_tone", "bed", room_tone, "quiet room tone (ACX padding, silence that breathes)", 30, ("roomtone", "silence"))
_reg("rain", "bed", rain, "rain bed with droplets (intensity=0..1)", 60, ("rainfall", "storm_rain"))
_reg("rain_light", "bed", lambda s, sr, r, **k: rain(s, sr, r, intensity=0.2), "light drizzle", 60, ("drizzle",))
_reg("rain_heavy", "bed", lambda s, sr, r, **k: rain(s, sr, r, intensity=0.9), "heavy downpour", 60, ("downpour",))
_reg("wind", "bed", wind, "gusty wind (strength=0..1)", 60, ("breeze",))
_reg("wind_soft", "bed", lambda s, sr, r, **k: wind(s, sr, r, strength=0.2), "soft breeze", 60, ("soft_wind",))
_reg("wind_storm", "bed", lambda s, sr, r, **k: wind(s, sr, r, strength=0.95), "howling storm wind", 60, ("gale", "blizzard"))
_reg("sea_waves", "bed", sea_waves, "waves breaking on a shore", 60, ("ocean", "sea", "waves", "beach", "shore"))
_reg("stream", "bed", stream, "babbling brook", 60, ("brook", "creek", "river"))
_reg("fire", "bed", fire, "crackling fire (size=0..1)", 60, ("fireplace", "campfire", "hearth"))
_reg("night", "bed", night, "night crickets and a faint breeze", 60, ("crickets", "night_crickets", "summer_night"))
_reg("forest_day", "bed", forest_day, "daytime forest: leaves and birdsong", 60, ("forest", "birds", "birdsong", "meadow", "garden"))
_reg("city", "bed", city, "distant city traffic", 60, ("traffic", "street", "town"))
_reg("crowd", "bed", crowd, "indistinct crowd murmur", 60, ("murmur", "babble", "market", "crowd_murmur"))
_reg("tavern", "bed", tavern, "busy tavern: murmur, clinks, hearth", 60, ("inn", "pub"))
_reg("underwater", "bed", underwater, "submerged rumble and bubbles", 60, ("under_water",))
_reg("cave", "bed", cave, "cavern with distant drips", 60, ("cavern", "dungeon"))
_reg("clock_ticking", "bed", clock_ticking, "steady clock tick-tock (bpm=60)", 30, ("ticking", "clock"))
_reg("heartbeat", "bed", heartbeat, "heartbeat (bpm=70)", 20, ("heart",))
_reg("thunder", "oneshot", thunder, "thunder rumble (distance=0 close .. 1 far)", 7, ("thunderclap",))
_reg("bell", "oneshot", bell, "struck bell (pitch=Hz)", 5, ("church_bell", "ding"))
_reg("chime", "oneshot", chime, "wind/door chimes", 4, ("chimes", "wind_chimes", "door_chime"))
_reg("sparkle", "oneshot", sparkle, "magic shimmer", 3, ("magic", "shimmer", "fairy_dust", "spell"))
_reg("whoosh", "oneshot", whoosh, "air whoosh / swish", 1.2, ("swish", "swoosh"))
_reg("knock", "oneshot", knock, "knocks on a wooden door (count=3)", 1.2, ("door_knock", "knocking"))
_reg("water_drip", "oneshot", water_drip, "single water drip", 0.4, ("drip",))
_reg("clock_tick", "oneshot", clock_tick, "single clock tick", 0.1, ("tick",))
_reg("glass_clink", "oneshot", glass_clink, "glass clink", 0.6, ("clink", "cheers"))
_reg("page_turn", "oneshot", page_turn, "turning a book page", 0.7, ("page", "book_page"))
_reg("owl", "oneshot", owl, "distant owl hoots", 2.0, ("owl_hoot",))
_reg("riser", "oneshot", riser, "tension riser for transitions", 4, ("swell", "transition"))
_reg("breath", "oneshot", breath, "soft inhale (procedural breath)", 0.35, ("inhale_soft",))
_reg("singing_bowl", "oneshot", singing_bowl, "resonant singing bowl", 8, ("bowl", "gong"))
for _mood in MOODS:
    _reg(f"pad_{_mood}", "music", (lambda m: (lambda s, sr, r, **k: pad(s, sr, r, mood=m, **k)))(_mood),
         f"ambient {_mood} pad (procedural underscore)", 90, (_mood,) if _mood not in ("warm",) else ())

_ALIASES = {a: p.name for p in REGISTRY.values() for a in (p.name, *p.aliases)}


def lookup(name: str) -> Proc | None:
    key = name.lower().replace("-", "_").replace(" ", "_")
    return REGISTRY.get(_ALIASES.get(key, ""))


def render(name: str, seconds: float | None = None, sr: int = 44100, seed: str | int = 0, **params) -> np.ndarray:
    proc = lookup(name)
    if proc is None:
        raise KeyError(f"no procedural sound named '{name}'")
    secs = float(seconds if seconds else proc.default_seconds)
    return proc.fn(secs, sr, rng_for(proc.name, seed, sorted(params.items())), **params)
