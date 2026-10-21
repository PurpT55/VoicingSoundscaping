"""ACE-Step client flow and local-LLM annotation, against mock servers."""

import io as _io
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import soundfile as sf

from vss.llm.annotate import annotate, repair, split_manuscript
from vss.script.parser import parse_text
from vss.sound.generators import AceStep, CommandGenerator


def _serve(handler_cls):
    httpd = HTTPServer(("127.0.0.1", 0), handler_cls)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


def test_acestep_release_query_download(tmp_path):
    seen = {}

    class H(BaseHTTPRequestHandler):
        polls = 0

        def log_message(self, *a):
            pass

        def _json(self, obj):
            body = json.dumps(obj).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.startswith("/v1/models"):
                return self._json({"data": []})
            buf = _io.BytesIO()
            sf.write(buf, np.zeros(44100, np.float32), 44100, format="WAV")
            self.send_response(200)
            self.end_headers()
            self.wfile.write(buf.getvalue())

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/release_task":
                seen["release"] = payload
                return self._json({"data": {"task_id": "t1", "status": "queued"}, "code": 200})
            H.polls += 1
            status = 1 if H.polls > 1 else 0
            result = json.dumps([{"file": "/v1/audio?path=x.wav", "status": 1}])
            return self._json({"data": [{"task_id": "t1", "status": status, "result": result}], "code": 200})

    httpd, url = _serve(H)
    try:
        gen = AceStep(url=url, steps=8)
        assert gen.available()[0]
        out = gen.generate("soft piano, wonder", 30, tmp_path / "m.wav", seed=3, bpm=72)
        assert out.exists() and sf.info(str(out)).duration == 1.0
        rel = seen["release"]
        assert rel["prompt"] == "soft piano, wonder" and rel["lyrics"] == "[Instrumental]"
        assert rel["audio_duration"] == 30 and rel["bpm"] == 72 and rel["use_random_seed"] is False
    finally:
        httpd.shutdown()


def test_command_generator(tmp_path):
    import sys

    script = tmp_path / "gen.py"
    script.write_text("import sys, numpy as np, soundfile as sf\n"
                      "sf.write(sys.argv[2], np.zeros(int(float(sys.argv[1]) * 8000), np.float32), 8000)\n")
    gen = CommandGenerator(cmd=f"{sys.executable} {script} {{seconds}} {{out}}", kinds=["sfx"])
    assert gen.available()[0]
    out = gen.generate("door creak", 1.5, tmp_path / "x.wav")
    assert abs(sf.info(str(out)).duration - 1.5) < 0.01


def test_annotate_with_mock_llm():
    reply = ('MIRA (angryy 0.7): Give it back! <laughh>\n'
             'The wind howled.\n'
             'TOM (dry_humor): No.\n')

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert req["messages"][0]["role"] == "system" and "dry_humor" in req["messages"][0]["content"]
            body = json.dumps({"choices": [{"message": {"content": reply}}]}).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)

    httpd, url = _serve(H)
    try:
        text = annotate('"Give it back!" Mira shouted.\n\nThe wind howled.', url + "/v1", "local", title="Ch 1", log=lambda s: None)
    finally:
        httpd.shutdown()
    parsed = parse_text(text)
    assert not parsed.errors, [str(d) for d in parsed.diagnostics]
    assert "angry 0.7" in text and "<laugh>" in text and text.startswith("# Ch 1")


def test_repair_and_split():
    fixed, notes = repair("# C\nA (sadd, joy): hi <sighh>\n")
    assert not parse_text(fixed).errors and notes
    chunks = split_manuscript("\n\n".join(["para " * 50] * 10), max_chars=600)
    assert len(chunks) > 1 and all(len(c) <= 600 for c in chunks)
