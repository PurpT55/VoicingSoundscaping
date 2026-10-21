import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

HAS_FFMPEG = shutil.which("ffmpeg") is not None


@pytest.fixture
def tiny_project(tmp_path):
    """A minimal project rendering with the dependency-free placeholder backend."""
    from vss.config import Project

    (tmp_path / "script").mkdir()
    (tmp_path / "script" / "01.vss").write_text(
        "# Chapter One: Test\n"
        "## Room\n"
        "@scene ambience=\"rain:-6\" space=small_room\n"
        "@music start warm gain=-4\n"
        "The rain came down on 3 roofs.\n"
        "ANA (joy 0.7): We did it! <laugh> [tired 0.5] I need a nap.\n"
        "BEN (whisper, afraid 0.6): Quiet. <sfx knock count=2>\n"
        "@sfx chime\n"
        "@music stop fade=2\n"
    )
    data = {
        "title": "Tiny",
        "backends": {"default": "placeholder", "draft": "placeholder"},
        "voices": {"ana": {}, "ben": {}, "narrator": {}},
        "cast": {"NARRATOR": {"voice": "narrator"}, "ANA": {"voice": "ana"}, "BEN": {"voice": "ben"}},
        "render": {"sample_rate": 44100, "takes": 1, "max_takes": 1, "breaths": "procedural"},
        "master": {"preset": "audiobook"},
        "export": {"formats": ["wav"], "captions": True},
    }
    proj = Project.from_dict(data, tmp_path)
    return proj
