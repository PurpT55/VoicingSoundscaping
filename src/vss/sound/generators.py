"""Clients for local generative audio tools (music, sound effects, ambience).

* ``acestep``  - ACE-Step 1.5 REST server (``uv run acestep-api``, default http://127.0.0.1:8001):
                 instrumental underscore and songs, 10-600 s, runs on < 4 GB VRAM (XL: 12-20 GB).
* ``command``  - any local CLI (MOSS-SoundEffect v2, Stable Audio Open, AudioX, TangoFlux ...)
                 driven by a command template with ``{prompt} {seconds} {seed} {out}`` placeholders.
* ``http``     - any local HTTP endpoint that takes JSON and returns audio bytes.

All of them write an audio file and return its path; the resolver handles caching, looping
and levels.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


class GeneratorError(RuntimeError):
    pass


def _http_json(url: str, payload: dict | None = None, timeout: float = 60.0) -> Any:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"},
                                 method="POST" if data is not None else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _http_get_bytes(url: str, timeout: float = 120.0) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read()


class AceStep:
    """ACE-Step 1.5 REST client (release_task -> query_result -> /v1/audio)."""

    name = "acestep"
    kinds = ("music",)

    def __init__(self, url: str = "http://127.0.0.1:8001", model: str | None = None, steps: int = 8,
                 thinking: bool = True, timeout: float = 600.0, api_key: str | None = None, **extra: Any):
        self.url = url.rstrip("/")
        self.model = model
        self.steps = steps
        self.thinking = thinking
        self.timeout = timeout
        self.api_key = api_key
        self.extra = extra

    def available(self) -> tuple[bool, str]:
        try:
            urllib.request.urlopen(self.url + "/v1/models", timeout=2.0).read()
            return True, f"ACE-Step server at {self.url}"
        except Exception as exc:  # noqa: BLE001 - any failure means "not reachable"
            return False, f"ACE-Step server not reachable at {self.url} ({exc.__class__.__name__})"

    def generate(self, prompt: str, seconds: float, out: Path, seed: int = 0, lyrics: str = "[Instrumental]",
                 **params: Any) -> Path:
        payload: dict[str, Any] = {
            "prompt": prompt, "lyrics": lyrics, "audio_duration": float(max(10.0, min(600.0, seconds))),
            "thinking": self.thinking, "inference_steps": self.steps, "batch_size": 1,
            "use_random_seed": False, "seed": int(seed) % (2 ** 31), "audio_format": "wav",
        }
        if self.model:
            payload["model"] = self.model
        if self.api_key:
            payload["ai_token"] = self.api_key
        for k in ("bpm", "key_scale", "time_signature", "guidance_scale"):
            if k in params:
                payload[k] = params[k]
        payload.update(self.extra)
        res = _http_json(self.url + "/release_task", payload)
        task_id = (res.get("data") or {}).get("task_id")
        if not task_id:
            raise GeneratorError(f"ACE-Step did not return a task id: {res}")
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            q = _http_json(self.url + "/query_result", {"task_id_list": [task_id]})
            item = (q.get("data") or [{}])[0]
            status = item.get("status")
            if status == 1:
                result = item.get("result")
                result = json.loads(result) if isinstance(result, str) else result
                file_url = result[0]["file"]
                audio = _http_get_bytes(urllib.parse.urljoin(self.url + "/", file_url.lstrip("/")))
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(audio)
                return out
            if status == 2:
                raise GeneratorError(f"ACE-Step task failed: {item}")
            time.sleep(1.5)
        raise GeneratorError("ACE-Step generation timed out")


class CommandGenerator:
    """Run a local command. Example config (MOSS-SoundEffect v2 inference script)::

        moss_sfx:
          type: command
          kinds: [sfx, ambience]
          cmd: "python ~/MOSS-TTS/moss_soundeffect_v2/infer.py --text {prompt} --duration {seconds} --seed {seed} --output {out}"
    """

    name = "command"

    def __init__(self, cmd: str, kinds: list[str] | None = None, cwd: str | None = None, timeout: float = 900.0,
                 max_seconds: float | None = None, **_: Any):
        self.cmd = cmd
        self.kinds = tuple(kinds or ("sfx", "ambience", "music"))
        self.cwd = cwd
        self.timeout = timeout
        self.max_seconds = max_seconds

    def available(self) -> tuple[bool, str]:
        exe = shlex.split(self.cmd)[0] if self.cmd else ""
        import shutil

        ok = bool(exe) and (shutil.which(Path(exe).expanduser().name) is not None or Path(exe).expanduser().exists())
        return ok, f"command '{exe}'" + ("" if ok else " not found")

    def generate(self, prompt: str, seconds: float, out: Path, seed: int = 0, **_: Any) -> Path:
        if self.max_seconds:
            seconds = min(seconds, self.max_seconds)
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = self.cmd.format(prompt=shlex.quote(prompt), seconds=f"{seconds:.2f}", seed=int(seed), out=shlex.quote(str(out)))
        proc = subprocess.run(cmd, shell=True, cwd=self.cwd, timeout=self.timeout, capture_output=True, text=True)
        if proc.returncode != 0 or not out.exists():
            raise GeneratorError(f"generator command failed ({proc.returncode}): {proc.stderr[-800:]}")
        return out


class HttpGenerator:
    """POST ``{"prompt", "seconds", "seed"}`` (+ extra fields) to ``url``; response body = audio file."""

    name = "http"

    def __init__(self, url: str, kinds: list[str] | None = None, fields: dict | None = None, timeout: float = 600.0, **_: Any):
        self.url = url
        self.kinds = tuple(kinds or ("sfx", "ambience", "music"))
        self.fields = fields or {}
        self.timeout = timeout

    def available(self) -> tuple[bool, str]:
        return True, f"http generator {self.url} (not probed)"

    def generate(self, prompt: str, seconds: float, out: Path, seed: int = 0, **_: Any) -> Path:
        payload = {"prompt": prompt, "seconds": seconds, "seed": seed, **self.fields}
        req = urllib.request.Request(self.url, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = resp.read()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(body)
        return out


def make_generator(name: str, cfg: dict[str, Any]):
    kind = cfg.get("type", name)
    cfg = {k: v for k, v in cfg.items() if k != "type"}
    if kind == "acestep":
        return AceStep(**cfg)
    if kind == "command":
        return CommandGenerator(**cfg)
    if kind == "http":
        return HttpGenerator(**cfg)
    raise GeneratorError(f"unknown generator type '{kind}' for '{name}'")
