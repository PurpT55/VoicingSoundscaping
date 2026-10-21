"""Backend request building and wire formats, verified against mock local servers and fake engines."""

import io as _io
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import msgpack
import numpy as np
import pytest
import soundfile as sf

from vss.emotion.mapping import Blend
from vss.tts import registry
from vss.tts.base import Audio, Capabilities, Piece, SynthesisRequest, TTSBackend, VoiceSpec
from vss.tts.worker import Engine


def _wav_bytes(seconds=0.5, sr=24000):
    buf = _io.BytesIO()
    t = np.arange(int(seconds * sr)) / sr
    sf.write(buf, (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sr, format="WAV")
    return buf.getvalue()


class _Server:
    def __init__(self):
        self.requests = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"{}")

            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                outer.requests.append((self.path, self.headers.get("Content-Type"), body))
                self.send_response(200)
                self.end_headers()
                self.wfile.write(_wav_bytes())

        self.httpd = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


@pytest.fixture
def server():
    s = _Server()
    yield s
    s.close()


def _req(pieces, blend, voice=None):
    return SynthesisRequest(pieces=pieces, voice=voice or VoiceSpec("mira", language="en"), blend=blend, seed=5)


def test_fish_inline_tags_and_msgpack(server, tmp_path):
    ref = tmp_path / "ref.wav"
    ref.write_bytes(_wav_bytes(1.0))
    fish = registry.create("fish", {"url": server.url, "model": "drama-3"})
    assert fish.available()[0]
    b1, b2 = Blend.parse("whisper, afraid 0.8"), Blend.parse("sad 0.5")
    req = _req([Piece("speech", "Is anyone there?", b1), Piece("nonverbal", nonverbal="gasp", blend=b1),
                Piece("speech", "Hello?", b2)], b1, VoiceSpec("mira", ref, "reference words"))
    text = fish.render_text(req)
    assert text.startswith("[whispering] [very afraid] Is anyone there? [gasp] [sad] Hello?")
    audio = fish.synthesize(req)
    assert audio.sr == 24000 and len(audio.samples) > 0
    path, ctype, body = server.requests[-1]
    assert path == "/v1/tts" and ctype == "application/msgpack"
    payload = msgpack.unpackb(body, raw=False)
    assert payload["text"] == text and payload["single_chunk"] is True
    assert payload["references"][0]["text"] == "reference words"
    assert isinstance(payload["references"][0]["audio"], bytes)
    caps = fish.capabilities()
    assert caps.inline_emotion and caps.supports_nonverbal("sob") and caps.strong_extremes


def test_openai_speech_modes(server):
    b = Blend.parse("cheeky 0.7")
    for mode in ("instructions", "inline_tags", "none"):
        be = registry.create("openai_speech", {"url": server.url + "/v1", "model": "m", "emotion_mode": mode,
                                               "speed": True, "extra": {"language": "English"}})
        be.synthesize(_req([Piece("speech", "Who, me?", b)], b))
        payload = json.loads(server.requests[-1][2])
        assert server.requests[-1][0] == "/v1/audio/speech"
        assert payload["model"] == "m" and payload["language"] == "English" and "speed" in payload
        if mode == "instructions":
            assert "Cheeky" in payload["instructions"] and payload["input"] == "Who, me?"
        elif mode == "inline_tags":
            assert payload["input"].startswith("[cheeky]") and "instructions" not in payload
        else:
            assert payload["input"] == "Who, me?" and "instructions" not in payload


def test_indextts_args_vector_bank_and_speed(tmp_path):
    ref = tmp_path / "ref.wav"
    ref.write_bytes(_wav_bytes())
    be = registry.create("indextts", {"model_dir": str(tmp_path), "version": "2.5"})
    voice = VoiceSpec("tom", ref, "x")
    b = Blend.parse("sleepy 0.8")
    args = be.build_args(_req([Piece("speech", "Five more minutes.", b)], b, voice), "/tmp/o.wav")
    assert args["emo_vector"] and abs(sum(args["emo_vector"]) - 0.8) < 0.05
    assert args["duration_factor"] > 1.0  # sleepy speaks slower -> longer
    req = _req([Piece("speech", "x", b)], b, voice)
    req.emotion_reference = ref
    args = be.build_args(req, "/tmp/o.wav")
    assert args["emo_audio_prompt"] == str(ref) and "emo_vector" not in args and 0.4 < args["emo_alpha"] <= 1
    with pytest.raises(Exception):
        be.build_args(_req([Piece("speech", "x", b)], b, VoiceSpec("noref")), "/tmp/o.wav")


def test_qwen3_and_chatterbox_args(tmp_path):
    ref = tmp_path / "ref.wav"
    ref.write_bytes(_wav_bytes())
    b = Blend.parse("furious 0.9")
    q = registry.create("qwen3", {"mode": "custom_voice"})
    args = q.build_args(_req([Piece("speech", "Out!", b)], b, VoiceSpec("tom", options={"qwen3": {"speaker": "Ryan"}})), "o")
    assert args["speaker"] == "Ryan" and "Furious" in args["instruct"]
    qd = registry.create("qwen3", {"mode": "voice_design"})
    args = qd.build_args(_req([Piece("speech", "Out!", b)], b, VoiceSpec("tom", description="Old gravelly man")), "o")
    assert args["instruct"].startswith("Old gravelly man")
    cb = registry.create("chatterbox", {"model": "turbo"})
    req = _req([Piece("speech", "Hi", b), Piece("nonverbal", nonverbal="laugh", blend=b), Piece("nonverbal", nonverbal="sob", blend=b)],
               b, VoiceSpec("tom", ref))
    assert cb.render_text(req) == "Hi [laugh]"  # sob unsupported by Turbo -> planner uses a bank clip
    co = registry.create("chatterbox", {"model": "original"})
    args = co.build_args(req, "o")
    assert args["exaggeration"] > 0.7 and args["audio_prompt_path"] == str(ref)


class _FakeBase(TTSBackend):
    name = "fakebase"

    def capabilities(self):
        return Capabilities(emotion=frozenset({"vector8"}), sample_rate=16000)

    def synthesize(self, req):
        return Audio(np.zeros(16000, np.float32) + 0.1, 16000, {"base": True})


class _FakeEngine:
    def __init__(self):
        self.calls = []

    def call(self, op, args):
        self.calls.append((op, args))
        sf.write(args["output_path"], np.zeros(8000, np.float32) + 0.2, 16000)
        return {"path": args["output_path"], "sr": 16000}

    def close(self):
        pass


def test_step_edit_refines_emotion_and_inserts_paralinguistics():
    se = registry.backend_class("step_edit")(base_backend=_FakeBase(), model_path="/x")
    eng = _FakeEngine()
    se._engine = eng
    b = Blend.parse("whisper, happy 0.9")
    req = _req([Piece("speech", "We did it", b), Piece("nonverbal", nonverbal="sigh", blend=b)], b)
    out = se.synthesize(req)
    ops = [(a["edit_type"], a["edit_info"]) for _, a in eng.calls]
    assert ("style", "whisper") in ops and ("emotion", "happy") in ops and ops[-1][0] == "paralinguistic"
    assert eng.calls[-1][1]["target_text"] == "We did it [sigh]"
    assert len(out.samples) == 8000 and out.meta["step_edits"]
    assert se.capabilities().supports_nonverbal("sigh")


def test_worker_subprocess_protocol(tmp_path):
    import sys

    script = tmp_path / "echo_worker.py"
    script.write_text(
        "import os, sys\n"
        f"sys.path.insert(0, {str(Path(__file__).resolve().parents[1] / 'src' / 'vss' / 'tts' / 'workers')!r})\n"
        "from _runtime import serve\n"
        "class Engine:\n"
        "    def __init__(self, greeting='hi'):\n"
        "        print('noisy library output')\n"
        "        self.greeting = greeting\n"
        "    def synth(self, text):\n"
        "        return {'echo': self.greeting + ' ' + text}\n"
        "    def boom(self):\n"
        "        raise ValueError('bad')\n"
        "if __name__ == '__main__':\n"
        "    serve(Engine)\n"
    )
    eng = Engine(str(script), {"greeting": "hello"}, python=sys.executable)
    assert eng.call("synth", {"text": "world"}) == {"echo": "hello world"}
    with pytest.raises(RuntimeError, match="bad"):
        eng.call("boom", {})
    assert eng.call("synth", {"text": "again"})["echo"] == "hello again"
    eng.close()
    inproc = Engine(str(script), {"greeting": "yo"})
    assert inproc.call("synth", {"text": "x"}) == {"echo": "yo x"}


def test_placeholder_and_espeak():
    b = Blend.parse("excited 0.8")
    ph = registry.create("placeholder", {})
    a = ph.synthesize(_req([Piece("speech", "Hello there, friend.", b)], b))
    assert 0.5 < a.duration < 5
    es = registry.create("espeak", {})
    if es.available()[0]:
        a = es.synthesize(_req([Piece("speech", "Hello there.", Blend.parse("whisper"))], Blend.parse("whisper")))
        assert a.duration > 0.3 and a.meta["voice"].endswith("+whisper")
