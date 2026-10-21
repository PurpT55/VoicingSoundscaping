import numpy as np
import pytest

from vss.audio import acx, chain, dsp, loudness, master
from vss.sound import procedural
from vss.sound.resolve import loop_to

SR = 44100


def _speechy(seconds=4.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(SR * seconds)) / SR
    env = (np.sin(2 * np.pi * 2.5 * t) > 0).astype(np.float32)
    x = np.sin(2 * np.pi * 180 * t) * 0.4 + 0.1 * rng.standard_normal(len(t))
    return (x * env).astype(np.float32)


@pytest.mark.parametrize("seed", range(5))
def test_limiter_never_exceeds_ceiling(seed):
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal((SR * 2, 2)) * rng.uniform(0.2, 2.0)).astype(np.float32)
    y = dsp.limiter(x, SR, ceiling_db=-3.0)
    assert loudness.peak_db(y) <= -3.0 + 1e-3
    assert y.shape == x.shape


def test_fade_trim_declick():
    x = np.ones(SR, np.float32)
    y = dsp.fade(x, SR, 0.1, 0.1)
    assert y[0] == 0 and y[-1] == 0 and y[SR // 2] == 1
    padded = np.concatenate([np.zeros(SR), _speechy(1.0), np.zeros(SR)])
    trimmed = dsp.trim_silence(padded, SR)
    assert len(trimmed) < len(padded) * 0.6


def test_loudness_normalisation_accuracy():
    x = _speechy()
    y = loudness.normalize_lufs(x, SR, -20.0)
    assert loudness.lufs(y, SR) == pytest.approx(-20.0, abs=0.2)
    z = loudness.normalize_rms(x, -23.0)
    assert loudness.rms_db(z) == pytest.approx(-23.0, abs=0.05)


def test_compressor_reduces_dynamic_range():
    x = _speechy()
    x[: SR // 2] *= 4
    y = dsp.compress(x, SR, threshold_db=-20, ratio=4)
    assert loudness.peak_db(y) < loudness.peak_db(x)


@pytest.mark.parametrize("preset", list(master.PRESETS))
def test_master_presets_hit_targets(preset):
    x = np.stack([_speechy(6), _speechy(6, 1)], axis=1)
    y, rep = master.master(x, SR, preset)
    p = master.PRESETS[preset]
    measured = rep["rms_db"] if p.measure == "rms" else rep["lufs"]
    assert measured == pytest.approx(p.target, abs=0.8)
    assert rep["peak_db"] <= p.ceiling_db + 0.01


def test_acx_check_passes_compliant_file():
    body = np.stack([_speechy(20), _speechy(20, 2)], axis=1)
    y, _ = master.master(body, SR, "acx")
    tone = procedural.render("room_tone", 2.0, SR)
    tone = tone / np.sqrt(np.mean(tone ** 2)) * dsp.db_to_gain(-68)
    head = tone[: int(0.7 * SR)]
    tail = tone[: int(2.0 * SR)]
    full = np.concatenate([head, y + loop_to(tone, len(y), SR), tail])
    rep = acx.check(full, SR)
    assert rep.ok, rep.format()


def test_spaces_and_chains_shapes():
    x = _speechy(1.0)
    for name in chain.SPACES:
        y = chain.apply_space(x, SR, name)
        assert y.ndim == 2 and y.shape[1] == 2 and len(y) >= len(x)
        assert np.all(np.isfinite(y))
    for name in chain.CHAINS:
        assert chain.apply_chain(x, SR, name).shape == x.shape


@pytest.mark.parametrize("name", sorted(procedural.REGISTRY))
def test_procedural_sounds(name):
    proc = procedural.REGISTRY[name]
    secs = min(proc.default_seconds, 3.0)
    x = procedural.render(name, secs, 22050, seed=1)
    assert x.ndim == 2 and x.shape[1] == 2
    assert abs(len(x) - int(secs * 22050)) <= 1
    assert np.all(np.isfinite(x))
    assert loudness.peak_db(x) <= -0.9
    assert -30 < loudness.lufs(x, 22050) < -15
    again = procedural.render(name, secs, 22050, seed=1)
    assert np.array_equal(x, again)  # deterministic


def test_loop_to_extends_smoothly():
    x = np.random.default_rng(0).standard_normal((SR * 3, 2)).astype(np.float32) * 0.1
    y = loop_to(x, SR * 10, SR, xfade=1.0)
    assert len(y) == SR * 10
    assert np.all(np.isfinite(y))
