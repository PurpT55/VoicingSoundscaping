"""JSON-lines serving loop shared by worker scripts (stdlib only: workers run in model venvs)."""

import json
import sys
import traceback


def serve(engine_cls):
    engine = None
    out = sys.stdout
    sys.stdout = sys.stderr  # keep library prints off the protocol channel
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        mid, op, args = msg.get("id"), msg.get("op"), msg.get("args") or {}
        try:
            if op == "init":
                engine = engine_cls(**args)
                result = {"ready": True}
            elif op == "shutdown":
                out.write(json.dumps({"id": mid, "ok": True, "result": None}) + "\n")
                out.flush()
                break
            else:
                result = getattr(engine, op)(**args)
            out.write(json.dumps({"id": mid, "ok": True, "result": result}) + "\n")
        except Exception as exc:  # noqa: BLE001 - report every failure to the client
            traceback.print_exc(file=sys.stderr)
            out.write(json.dumps({"id": mid, "ok": False, "error": f"{exc.__class__.__name__}: {exc}"}) + "\n")
        out.flush()


def seed_everything(seed):
    if seed is None:
        return
    import random

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2 ** 32))
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
