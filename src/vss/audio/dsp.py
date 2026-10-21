"""Signal processing building blocks (vectorised numpy/scipy, pedalboard where it shines)."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import minimum_filter1d, uniform_filter1d
from scipy.signal import butter, lfilter, sosfilt, sosfiltfilt

EPS = 1e-12


def db_to_gain(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def gain_to_db(g: float) -> float:
    return float(20.0 * np.log10(max(abs(g), EPS)))


def apply_gain(x: np.ndarray, db: float) -> np.ndarray:
    return (x * db_to_gain(db)).astype(np.float32)


# ------------------------------------------------------------------------------ fades
def fade(x: np.ndarray, sr: int, fade_in: float = 0.0, fade_out: float = 0.0, curve: str = "cos") -> np.ndarray:
    y = np.array(x, dtype=np.float32, copy=True)
    n = len(y)
    for seconds, rising in ((fade_in, True), (fade_out, False)):
        k = min(n, int(round(seconds * sr)))
        if k <= 1:
            continue
        t = np.linspace(0.0, 1.0, k, dtype=np.float32)
        ramp = np.sin(t * np.pi / 2) ** 2 if curve == "cos" else t
        if not rising:
            ramp = ramp[::-1]
        if y.ndim == 2:
            ramp = ramp[:, None]
        if rising:
            y[:k] *= ramp
        else:
            y[n - k:] *= ramp
    return y


def declick(x: np.ndarray, sr: int, ms: float = 4.0) -> np.ndarray:
    return fade(x, sr, ms / 1000.0, ms / 1000.0, curve="lin")


def frame_rms_db(x: np.ndarray, sr: int, win: float = 0.02) -> np.ndarray:
    m = x if x.ndim == 1 else x.mean(axis=1)
    hop = max(1, int(sr * win))
    n = len(m) // hop
    if n == 0:
        return np.array([20 * np.log10(np.sqrt(np.mean(m ** 2)) + EPS)]) if len(m) else np.array([-120.0])
    frames = m[: n * hop].reshape(n, hop)
    return 20 * np.log10(np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1)) + EPS)


def trim_silence(x: np.ndarray, sr: int, threshold_db: float = -48.0, pad: float = 0.04,
                 win: float = 0.01) -> np.ndarray:
    """Trim leading/trailing silence relative to the clip's own peak level."""
    if len(x) == 0:
        return x
    db = frame_rms_db(x, sr, win)
    peak = db.max()
    active = np.where(db > max(threshold_db, peak - 45.0))[0]
    if len(active) == 0:
        return x[:0]
    hop = max(1, int(sr * win))
    start = max(0, active[0] * hop - int(pad * sr))
    end = min(len(x), (active[-1] + 1) * hop + int(pad * sr))
    return declick(x[start:end], sr)


def internal_silences(x: np.ndarray, sr: int, min_len: float = 1.2, threshold_db: float = -45.0) -> list[float]:
    """Lengths (s) of silent gaps inside a clip - long gaps often mean a broken take."""
    db = frame_rms_db(x, sr, 0.02)
    if len(db) == 0:
        return []
    quiet = db < max(threshold_db, db.max() - 40.0)
    gaps, run = [], 0
    for q in quiet[1:-1]:
        if q:
            run += 1
        else:
            if run * 0.02 >= min_len:
                gaps.append(run * 0.02)
            run = 0
    return gaps


# ------------------------------------------------------------------------------ filters
def _sos(kind: str, freq: float, sr: int, order: int = 2):
    nyq = sr / 2.0
    return butter(order, min(freq, nyq * 0.98) / nyq, btype=kind, output="sos")


def highpass(x: np.ndarray, sr: int, freq: float, order: int = 2) -> np.ndarray:
    return sosfilt(_sos("highpass", freq, sr, order), x, axis=0).astype(np.float32)


def lowpass(x: np.ndarray, sr: int, freq: float, order: int = 2) -> np.ndarray:
    return sosfilt(_sos("lowpass", freq, sr, order), x, axis=0).astype(np.float32)


def bandpass(x: np.ndarray, sr: int, lo: float, hi: float, order: int = 2, zero_phase: bool = False) -> np.ndarray:
    nyq = sr / 2.0
    sos = butter(order, [max(lo, 1.0) / nyq, min(hi, nyq * 0.98) / nyq], btype="band", output="sos")
    f = sosfiltfilt if zero_phase else sosfilt
    return f(sos, x, axis=0).astype(np.float32)


def _biquad(b, a, x):
    return lfilter(b, a, x, axis=0).astype(np.float32)


def peaking(x: np.ndarray, sr: int, freq: float, gain_db: float, q: float = 1.0) -> np.ndarray:
    """RBJ cookbook peaking EQ."""
    if abs(gain_db) < 1e-3:
        return x
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * freq / sr
    alpha = np.sin(w0) / (2 * q)
    b = np.array([1 + alpha * A, -2 * np.cos(w0), 1 - alpha * A])
    a = np.array([1 + alpha / A, -2 * np.cos(w0), 1 - alpha / A])
    return _biquad(b / a[0], a / a[0], x)


def shelf(x: np.ndarray, sr: int, freq: float, gain_db: float, high: bool = True, slope: float = 1.0) -> np.ndarray:
    """RBJ cookbook shelving EQ."""
    if abs(gain_db) < 1e-3:
        return x
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * freq / sr
    cw, sw = np.cos(w0), np.sin(w0)
    alpha = sw / 2 * np.sqrt((A + 1 / A) * (1 / slope - 1) + 2)
    sa = 2 * np.sqrt(A) * alpha
    if high:
        b = [A * ((A + 1) + (A - 1) * cw + sa), -2 * A * ((A - 1) + (A + 1) * cw), A * ((A + 1) + (A - 1) * cw - sa)]
        a = [(A + 1) - (A - 1) * cw + sa, 2 * ((A - 1) - (A + 1) * cw), (A + 1) - (A - 1) * cw - sa]
    else:
        b = [A * ((A + 1) - (A - 1) * cw + sa), 2 * A * ((A - 1) - (A + 1) * cw), A * ((A + 1) - (A - 1) * cw - sa)]
        a = [(A + 1) + (A - 1) * cw + sa, -2 * ((A - 1) + (A + 1) * cw), (A + 1) + (A - 1) * cw - sa]
    b, a = np.array(b), np.array(a)
    return _biquad(b / a[0], a / a[0], x)


# ------------------------------------------------------------------------- dynamics
def envelope(x: np.ndarray, sr: int, attack: float, release: float, frame: float = 0.001) -> np.ndarray:
    """Peak envelope follower with true attack/release, computed on short frames and
    interpolated back to the sample rate (fast enough for whole chapters)."""
    m = np.abs(x if x.ndim == 1 else np.max(np.abs(x), axis=1)).astype(np.float64)
    if len(m) == 0:
        return m.astype(np.float32)
    hop = max(1, int(sr * frame))
    n = -(-len(m) // hop)
    peaks = np.pad(m, (0, n * hop - len(m))).reshape(n, hop).max(axis=1)
    a_att = float(np.exp(-hop / max(1.0, attack * sr)))
    a_rel = float(np.exp(-hop / max(1.0, release * sr)))
    env = np.empty(n)
    e = 0.0
    for i, p in enumerate(peaks.tolist()):
        c = a_att if p > e else a_rel
        e = c * e + (1.0 - c) * p
        env[i] = e
    centers = (np.arange(n) + 0.5) * hop
    return np.interp(np.arange(len(m)), centers, env).astype(np.float32)


def compress(x: np.ndarray, sr: int, threshold_db: float = -24.0, ratio: float = 3.0, attack: float = 0.005,
             release: float = 0.08, knee_db: float = 6.0, makeup_db: float = 0.0) -> np.ndarray:
    env_db = 20 * np.log10(envelope(x, sr, attack, release) + EPS)
    over = env_db - threshold_db
    gr = np.where(
        over <= -knee_db / 2, 0.0,
        np.where(over >= knee_db / 2, over * (1 - 1 / ratio),
                 (1 - 1 / ratio) * (over + knee_db / 2) ** 2 / (2 * knee_db)),
    )
    # smooth gain changes to avoid zipper noise
    gr = uniform_filter1d(gr, size=max(1, int(sr * 0.002)))
    g = (10 ** ((makeup_db - gr) / 20)).astype(np.float32)
    return (x * (g[:, None] if x.ndim == 2 else g)).astype(np.float32)


def deess(x: np.ndarray, sr: int, freq: float = 6000.0, threshold_db: float = -30.0, max_cut_db: float = 8.0) -> np.ndarray:
    """Split-band de-esser: compresses only the sibilance band when it spikes."""
    if sr < freq * 2.2:
        return x
    hi = highpass(x, sr, freq, order=4)
    lo = x - hi
    env_db = 20 * np.log10(envelope(hi, sr, 0.001, 0.05) + EPS)
    cut = np.clip((env_db - threshold_db) * 0.6, 0.0, max_cut_db)
    g = (10 ** (-cut / 20)).astype(np.float32)
    return (lo + hi * (g[:, None] if x.ndim == 2 else g)).astype(np.float32)


def limiter(x: np.ndarray, sr: int, ceiling_db: float = -1.0, lookahead: float = 0.005, hold: float = 0.05,
            oversample: int = 4) -> np.ndarray:
    """Look-ahead brickwall limiter. Guarantees |y| <= ceiling on sample peaks; with
    ``oversample`` > 1 inter-sample peaks are estimated too (true-peak-ish).

    Gain = forward running-minimum of the required gain over ``W`` samples, followed by a
    backward moving average of the same length; this provably never exceeds the required gain
    at a peak while ramping smoothly (attack = release = W).
    """
    if len(x) == 0:
        return x
    ceiling = db_to_gain(ceiling_db)
    peak = np.abs(x if x.ndim == 1 else np.max(np.abs(x), axis=1))
    if oversample > 1:
        peak = np.maximum(peak, _intersample_peaks(x, oversample))
    need = np.minimum(1.0, ceiling / np.maximum(peak, EPS))
    if need.min() >= 1.0:
        return x.astype(np.float32)
    W = max(1, int(sr * max(lookahead, hold)))
    # forward-looking min: m[n] = min(need[n : n+W])
    padded = np.concatenate([need, np.ones(W)])
    m = minimum_filter1d(padded, size=W, origin=-(W // 2))[: len(need)]
    # backward moving average of length W: h[n] = mean(m[n-W+1 : n+1])
    c = np.cumsum(np.concatenate([np.ones(W), m]))
    h = (c[W:] - c[:-W]) / W
    h = np.minimum(h, 1.0).astype(np.float32)
    y = x * (h[:, None] if x.ndim == 2 else h)
    # numerical safety: hard clip at ceiling (should be a no-op)
    return np.clip(y, -ceiling, ceiling).astype(np.float32)


def _intersample_peaks(x: np.ndarray, factor: int, block: int = 1 << 18) -> np.ndarray:
    from scipy.signal import resample_poly

    mono_peak = np.zeros(len(x), dtype=np.float32)
    chans = [x] if x.ndim == 1 else [x[:, c] for c in range(x.shape[1])]
    for ch in chans:
        for s in range(0, len(ch), block):
            seg = ch[max(0, s - 64): s + block + 64]
            up = np.abs(resample_poly(seg, factor, 1))
            off = (s - max(0, s - 64)) * factor
            n = min(block, len(ch) - s)
            up = up[off: off + n * factor]
            if len(up) < n * factor:
                up = np.pad(up, (0, n * factor - len(up)))
            mono_peak[s: s + n] = np.maximum(mono_peak[s: s + n], up.reshape(n, factor).max(axis=1))
    return mono_peak


# ----------------------------------------------------------------------- stereo / space
def pan(mono: np.ndarray, position: float = 0.0) -> np.ndarray:
    """Constant-power pan, position -1 (left) .. +1 (right)."""
    p = (max(-1.0, min(1.0, position)) + 1) * np.pi / 4
    return np.stack([mono * np.cos(p), mono * np.sin(p)], axis=1).astype(np.float32) * np.sqrt(2).astype(np.float32)


def stereo_width(x: np.ndarray, width: float) -> np.ndarray:
    if x.ndim != 2:
        return x
    mid = (x[:, 0] + x[:, 1]) / 2
    side = (x[:, 0] - x[:, 1]) / 2 * width
    return np.stack([mid + side, mid - side], axis=1).astype(np.float32)


def time_stretch(x: np.ndarray, sr: int, rate: float, semitones: float = 0.0) -> np.ndarray:
    """High-quality tempo/pitch change (Rubber Band via pedalboard). rate>1 = faster."""
    if abs(rate - 1.0) < 1e-3 and abs(semitones) < 1e-3:
        return x
    import pedalboard

    y = pedalboard.time_stretch(x.astype(np.float32)[None, :] if x.ndim == 1 else x.T.astype(np.float32),
                                sr, stretch_factor=float(rate), pitch_shift_in_semitones=float(semitones))
    return (y[0] if x.ndim == 1 else y.T).astype(np.float32)
