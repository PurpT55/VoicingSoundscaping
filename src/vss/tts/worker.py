"""Run a model engine either in-process or in its own Python environment.

Most state-of-the-art TTS repos pin their own torch/transformers versions, so the robust setup
is *one virtualenv per model*. A worker is a tiny self-contained script (see ``workers/``) that
loads the model once and serves JSON-lines requests on stdin/stdout:

    -> {"id": 1, "op": "synth", "args": {...}}
    <- {"id": 1, "ok": true, "result": {"path": "/tmp/x.wav", "sr": 24000}}

Backends call :class:`Engine` the same way in both modes.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

WORKERS_DIR = Path(__file__).with_name("workers")


class WorkerError(RuntimeError):
    pass


class _Subprocess:
    def __init__(self, python: str, script: Path, init: dict[str, Any], cwd: str | None, env: dict | None):
        full_env = dict(os.environ)
        full_env.update(env or {})
        full_env.setdefault("PYTHONUNBUFFERED", "1")
        self.proc = subprocess.Popen(
            [python, str(script)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr,
            cwd=cwd, env=full_env, text=True, bufsize=1,
        )
        self.lock = threading.Lock()
        self.counter = 0
        self.call("init", init)

    def call(self, op: str, args: dict[str, Any]) -> Any:
        with self.lock:
            self.counter += 1
            msg = json.dumps({"id": self.counter, "op": op, "args": args})
            assert self.proc.stdin and self.proc.stdout
            self.proc.stdin.write(msg + "\n")
            self.proc.stdin.flush()
            while True:
                line = self.proc.stdout.readline()
                if not line:
                    raise WorkerError(f"worker exited (code {self.proc.poll()})")
                line = line.strip()
                if not line.startswith("{"):
                    continue  # stray library prints
                try:
                    reply = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if reply.get("id") != self.counter:
                    continue
                if not reply.get("ok"):
                    raise WorkerError(reply.get("error", "unknown worker error"))
                return reply.get("result")

    def close(self) -> None:
        try:
            if self.proc.poll() is None:
                self.call("shutdown", {})
        except WorkerError:
            pass
        finally:
            if self.proc.poll() is None:
                self.proc.terminate()


def _load_module(script: Path):
    spec = importlib.util.spec_from_file_location(f"vss_worker_{script.stem}", script)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(mod)
    return mod


class Engine:
    """Facade over a worker script: in-process when ``python`` is None, else a subprocess."""

    def __init__(self, worker: str, init: dict[str, Any], python: str | None = None, cwd: str | None = None,
                 env: dict | None = None):
        self.script = Path(worker) if str(worker).endswith(".py") else WORKERS_DIR / f"{worker}.py"
        self.init_args = init
        self.python = python
        self.cwd = cwd
        self.env = env
        self._impl = None

    def _ensure(self):
        if self._impl is None:
            if self.python:
                self._impl = _Subprocess(self.python, self.script, self.init_args, self.cwd, self.env)
            else:
                if self.cwd and self.cwd not in sys.path:
                    sys.path.insert(0, self.cwd)
                mod = _load_module(self.script)
                self._impl = mod.Engine(**self.init_args)
        return self._impl

    def call(self, op: str, args: dict[str, Any]) -> Any:
        impl = self._ensure()
        if isinstance(impl, _Subprocess):
            return impl.call(op, args)
        return getattr(impl, op)(**args)

    def close(self) -> None:
        if isinstance(self._impl, _Subprocess):
            self._impl.close()
        self._impl = None
