"""Step-Audio-EditX as an emotion *refinement pass* on top of any base backend.

``step_edit`` synthesises with ``base`` (e.g. IndexTTS for the timbre) and then applies 1-3
iterations of Step-Audio-EditX emotion / style edits, plus a paralinguistic pass that inserts
nonverbals ([sigh], [laugh], [breath] ...) the base model cannot produce. StepFun reports that a
single edit iteration already lifts emotion accuracy of strong closed models, more with iterations.

    backends:
      default: step_edit
      config:
        step_edit: {base: indextts, repo_dir: ~/Step-Audio-EditX, python: ~/Step-Audio-EditX/.venv/bin/python,
                    model_path: ~/models/Step-Audio-EditX, tokenizer_path: ~/models/Step-Audio-Tokenizer}
"""

from __future__ import annotations

import os
import tempfile

import numpy as np

from ..audio import io
from ..emotion import taxonomy as tx
from ..emotion.mapping import step_edits
from .base import STEP_EDIT, Audio, Capabilities, SynthesisRequest, TTSBackend
from .worker import Engine

STEP_TAGS = frozenset(n.id for n in tx.NONVERBALS if n.step)


class StepEditBackend(TTSBackend):
    name = "step_edit"
    description = "Step-Audio-EditX refinement over a base backend (emotion/style/paralinguistics)"
    license = "Apache-2.0"

    def __init__(self, base_backend: TTSBackend | None = None, **options):
        super().__init__(**options)
        self.base = base_backend
        self._engine: Engine | None = None

    def capabilities(self) -> Capabilities:
        b = self.base.capabilities() if self.base else Capabilities()
        nv = b.nonverbals if b.nonverbals == "freeform" else frozenset(b.nonverbals) | STEP_TAGS
        return Capabilities(emotion=frozenset(set(b.emotion) | {STEP_EDIT}), nonverbals=nv,
                            inline_emotion=b.inline_emotion, emotion_reference=b.emotion_reference,
                            emotion_reference_scope=b.emotion_reference_scope, cloning=b.cloning, speed=b.speed,
                            languages=b.languages, max_chars=min(b.max_chars, 280), sample_rate=b.sample_rate,
                            strong_extremes=b.strong_extremes)

    def available(self) -> tuple[bool, str]:
        if self.base is None:
            return False, "step_edit needs a 'base' backend"
        ok, why = self.base.available()
        if not ok:
            return False, f"base {self.base.name}: {why}"
        mp = self.options.get("model_path")
        if not mp or not os.path.exists(os.path.expanduser(mp)):
            return False, f"Step-Audio-EditX model_path not found ({mp})"
        return True, f"step_edit over {self.base.name}"

    def fingerprint(self) -> dict:
        fp = super().fingerprint()
        fp["base"] = self.base.fingerprint() if self.base else None
        return fp

    def engine(self) -> Engine:
        if self._engine is None:
            o = self.options
            init = {"model_path": os.path.expanduser(o["model_path"])}
            if o.get("tokenizer_path"):
                init["tokenizer_path"] = os.path.expanduser(o["tokenizer_path"])
            init["model_source"] = o.get("model_source", "local")
            if o.get("extra"):
                init["extra"] = o["extra"]
            py = os.path.expanduser(o["python"]) if o.get("python") else None
            cwd = os.path.expanduser(o["repo_dir"]) if o.get("repo_dir") else None
            self._engine = Engine("step_edit", init, python=py, cwd=cwd)
        return self._engine

    def render_text(self, req: SynthesisRequest) -> str:
        return self.base.render_text(req) if self.base else req.text

    def paralinguistic_text(self, req: SynthesisRequest) -> str | None:
        base_caps = self.base.capabilities() if self.base else Capabilities()
        out, inserted = [], False
        for p in req.pieces:
            if p.kind == "nonverbal":
                nv = tx.get_nonverbal(p.nonverbal)
                if nv.step and not base_caps.supports_nonverbal(nv.id):
                    out.append(nv.step)
                    inserted = True
            else:
                out.append(p.text)
        return " ".join(out) if inserted else None

    def synthesize(self, req: SynthesisRequest) -> Audio:
        assert self.base is not None
        audio = self.base.synthesize(req)
        edits = [e for e in step_edits(req.blend) if not req.blend.is_neutral]
        edits = edits[: int(self.options.get("max_edits", 2))]
        para = self.paralinguistic_text(req) if self.options.get("paralinguistic", True) else None
        if not edits and not para:
            return audio
        max_it = int(self.options.get("max_iterations", 3))
        with tempfile.TemporaryDirectory() as td:
            cur = os.path.join(td, "base.wav")
            io.save(cur, audio.samples, audio.sr, subtype="PCM_16")
            for i, e in enumerate(edits):
                out = os.path.join(td, f"edit{i}.wav")
                self.engine().call("edit", {"prompt_wav": cur, "prompt_text": req.text, "edit_type": e.edit_type,
                                            "edit_info": e.edit_info, "iterations": min(max_it, e.iterations),
                                            "output_path": out, "seed": req.seed})
                cur = out
            if para:
                out = os.path.join(td, "para.wav")
                self.engine().call("edit", {"prompt_wav": cur, "prompt_text": req.text, "edit_type": "paralinguistic",
                                            "edit_info": "", "target_text": para, "iterations": 1,
                                            "output_path": out, "seed": req.seed})
                cur = out
            x, sr = io.load(cur, mono=True)
        meta = dict(audio.meta)
        meta["step_edits"] = [(e.edit_type, e.edit_info, e.iterations) for e in edits] + ([("paralinguistic", para, 1)] if para else [])
        return Audio(np.asarray(x, dtype=np.float32), sr, meta)

    def close(self) -> None:
        if self._engine:
            self._engine.close()
            self._engine = None
        if self.base:
            self.base.close()
