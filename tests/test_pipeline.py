import json
import subprocess
import sys

import numpy as np
import pytest

from conftest import HAS_FFMPEG
from vss.audio import io
from vss.config import Project
from vss.render.pipeline import Renderer
from vss.render.plan import NonverbalClip, make_plan, route_backend
from vss.script.parser import parse_files
from vss.emotion.mapping import Blend
from vss.sound.library import AssetLibrary
from vss.sound.resolve import SoundResolver
from vss.tts.base import Capabilities


def _caps_factory(**kw):
    caps = Capabilities(**kw)
    return lambda name: caps


def test_plan_splits_for_non_inline_backends(tiny_project):
    script = parse_files(tiny_project.script_files())
    ch = script.chapters[0]
    plan = make_plan(tiny_project, ch, _caps_factory(max_chars=200))
    units = plan.units
    assert units[0].text.startswith("Chapter One")             # announced title
    assert any("three roofs" in u.text for u in units)          # numbers normalised
    ana = [u for u in units if u.speaker == "ANA"]
    assert len(ana) == 2 and ana[0].blend.primary.emotion.id == "joy" and ana[1].blend.primary.emotion.id == "tired"
    assert any(isinstance(s, NonverbalClip) and s.nonverbal == "laugh" for s in plan.steps)
    ben = [u for u in units if u.speaker == "BEN"][0]
    assert [p.emotion.id for p in ben.blend.parts] == ["whisper", "afraid"]


def test_plan_keeps_inline_for_capable_backends(tiny_project):
    ch = parse_files(tiny_project.script_files()).chapters[0]
    plan = make_plan(tiny_project, ch, _caps_factory(inline_emotion=True, nonverbals="freeform", max_chars=400))
    ana = [u for u in plan.units if u.speaker == "ANA"]
    assert len(ana) == 1
    kinds = [p.kind for p in ana[0].pieces]
    assert kinds == ["speech", "nonverbal", "speech"]
    assert ana[0].pieces[2].blend.primary.emotion.id == "tired"


def test_routing():
    b = Blend.parse("scream")
    assert route_backend(b, {"scream": "fish"}, "indextts") == "fish"
    assert route_backend(Blend.parse("crying"), {"hard": "fish"}, "indextts") == "fish"
    assert route_backend(Blend.parse("joy"), {"family:joy": "qwen3"}, "indextts") == "qwen3"
    assert route_backend(Blend.parse("calm"), {"hard": "fish"}, "indextts") == "indextts"


def test_full_render_and_cache(tiny_project, tmp_path):
    out = tmp_path / "out"
    logs = []
    r = Renderer(tiny_project, log=logs.append)
    book = r.render(out_dir=out)
    ch = book["chapters"][0]
    wav = out / f"{ch['slug']}.wav"
    assert wav.exists() and (out / f"{ch['slug']}.srt").exists()
    assert (out / "render_report.json").exists() and (out / "QA_SHEET.md").exists()
    x, sr = io.load(wav)
    assert sr == 44100 and x.ndim == 2 and np.all(np.isfinite(x))
    assert ch["levels"]["lufs"] == pytest.approx(-18.0, abs=1.0)
    assert ch["levels"]["true_peak_db"] <= -1.9
    assert any("procedural:rain" in s for s in ch["sources"])
    assert any("sfx 'knock'" in s for s in ch["sources"])
    srt = (out / f"{ch['slug']}.srt").read_text()
    assert "Ana: We did it!" in srt and "three" not in srt.split("\n\n")[1]  # captions keep the written text
    # second render is served from the cache
    r2 = Renderer(Project.from_dict(tiny_project.cfg, tiny_project.root), log=lambda s: None)
    book2 = r2.render(out_dir=out)
    assert book2["cache"]["misses"] == 0 and book2["cache"]["hits"] > 0


def test_acx_preset_drops_beds_and_passes(tiny_project, tmp_path):
    tiny_project.cfg["master"]["preset"] = "acx"
    book = Renderer(tiny_project, log=lambda s: None).render(out_dir=tmp_path / "acx")
    ch = book["chapters"][0]
    assert ch["acx"]["applies"] is True
    checks = {c["name"]: c["ok"] for c in ch["acx"]["checks"]}
    assert checks["rms"] and checks["peak"] and checks["noise_floor"] and checks["head_room_tone"], ch["acx"]
    assert any("skipped" in s for s in ch["sources"])


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg required")
def test_m4b_with_chapters(tiny_project, tmp_path):
    (tiny_project.root / "script" / "02.vss").write_text("# Chapter Two\nThe end.\n")
    tiny_project.cfg["export"]["formats"] = ["wav", "mp3", "m4b"]
    book = Renderer(tiny_project, log=lambda s: None).render(out_dir=tmp_path / "book")
    m4b = book["m4b"]
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_chapters", "-of", "json", m4b], capture_output=True, text=True)
    chapters = json.loads(probe.stdout)["chapters"]
    assert [c["tags"]["title"] for c in chapters] == ["Chapter One: Test", "Chapter Two"]


def test_backend_fallback_is_reported(tiny_project, tmp_path):
    tiny_project.cfg["backends"]["default"] = "fish"
    tiny_project.cfg["backends"]["config"]["fish"] = {"url": "http://127.0.0.1:9"}
    tiny_project.cfg["backends"]["draft"] = "placeholder"
    r = Renderer(tiny_project, log=lambda s: None)
    b = r.backend("fish")
    assert b.name in ("placeholder", "espeak")
    assert "fish" in r.fallback_notes


def test_library_resolution(tmp_path):
    d = tmp_path / "sfx" / "doors"
    d.mkdir(parents=True)
    for n in ("door_creak_01.wav", "door_creak_02.wav", "wooden_door_slam.wav"):
        io.save(d / n, np.random.default_rng(0).standard_normal(4410).astype(np.float32) * 0.1, 44100)
    lib = AssetLibrary([tmp_path / "sfx"])
    assert len(lib.find_all("door_creak")) == 2
    assert lib.find("door creak", 1).name == "door_creak_02.wav"
    assert lib.find("slam").name == "wooden_door_slam.wav"
    assert lib.find("nothing_like_this") is None
    proj = Project.from_dict({"assets": {"sfx_dirs": [str(tmp_path / "sfx")]}}, tmp_path)
    res = SoundResolver(proj, 44100)
    assert res.get("sfx", "door_creak").source.startswith("library:")
    assert res.get("sfx", "thunder").source == "procedural:thunder"
    assert res.get("music", "sad farewell", seconds=5).source == "procedural:pad_melancholy"
    with pytest.raises(LookupError):
        res.get("sfx", "unobtainium_explosion")


def test_cli_smoke(tmp_path):
    def run(*args):
        return subprocess.run([sys.executable, "-m", "vss.cli", *args], capture_output=True, text=True, cwd=tmp_path,
                              env={"PYTHONPATH": str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"),
                                   "PATH": __import__("os").environ["PATH"]})

    r = run("emotions", "--backend", "fish", "indextts")
    assert r.returncode == 0 and "BEST-OF-ALL" in r.stdout
    assert "rain" in run("sounds").stdout
    r = run("new", "story", "--title", "Test Tale")
    assert r.returncode == 0, r.stderr
    assert "Test Tale" in (tmp_path / "story" / "project.yaml").read_text()
    r = run("lint", "story")
    assert r.returncode == 0, r.stdout + r.stderr
    r = run("say", "MIRA (whisper): hello there", "--backend", "placeholder", "--out", "say.wav")
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "say.wav").exists()
