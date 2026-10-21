"""Fish Audio S2-Pro / Drama 3 backend via the local fish-speech API server (msgpack ``/v1/tts``).

Top of the open-weights leaderboards in 2026. Free-form inline tags (``[whispering]``,
``[very angry]``, ``[laughing]``, even ``[barely holding back tears]``) give near-complete
emotional coverage; Drama 3 renders strong emotion (crying, screams, laughter) far better than S2.

Licence (Fish Audio Community Licence, Oct 2026): free for personal/non-commercial use and for
commercial use under USD 500k annual revenue; producing audiobooks you deliver as finished files
is explicitly allowed.

Start the server::

    python tools/api_server.py --listen 127.0.0.1:8080 --llama-checkpoint-path checkpoints/s2-pro \
        --decoder-checkpoint-path checkpoints/s2-pro/codec.pth --decoder-config-name modded_dac_vq
    python tools/api_server.py --listen 127.0.0.1:8080 --llama-checkpoint-path checkpoints/drama-3 --device cuda:0
"""

from __future__ import annotations

import io as _io
import urllib.error
import urllib.request

import msgpack

from ..audio import io
from ..emotion import taxonomy as tx
from ..emotion.mapping import freeform_tags
from .base import FREEFORM, Audio, BackendUnavailable, Capabilities, SynthesisRequest, TTSBackend


class FishBackend(TTSBackend):
    name = "fish"
    description = "Fish Audio S2-Pro / Drama 3 (local API server): free-form emotion tags, cloning"
    license = "Fish Audio Community Licence (<$500k revenue commercial OK)"

    @property
    def url(self) -> str:
        return str(self.options.get("url", "http://127.0.0.1:8080")).rstrip("/")

    @property
    def drama(self) -> bool:
        return "drama" in str(self.options.get("model", "s2-pro")).lower()

    def capabilities(self) -> Capabilities:
        return Capabilities(emotion=frozenset({FREEFORM}), nonverbals="freeform", inline_emotion=True,
                            cloning=True, speed=False, languages=("*",), max_chars=900 if self.drama else 600,
                            sample_rate=44100, strong_extremes=self.drama)

    def available(self) -> tuple[bool, str]:
        try:
            urllib.request.urlopen(self.url + "/v1/health", timeout=2.0).read()
            return True, f"fish server at {self.url}"
        except Exception as exc:  # noqa: BLE001
            return False, f"fish server not reachable at {self.url} ({exc.__class__.__name__}); see docs/SETUP.md#fish"

    def render_text(self, req: SynthesisRequest) -> str:
        fmt = self.options.get("tag_format", "[{}]")
        out: list[str] = []
        last_key = None
        for p in req.pieces:
            if p.kind == "nonverbal":
                nv = tx.get_nonverbal(p.nonverbal)
                out.append(nv.fish or fmt.format(nv.describe))
                continue
            if p.blend.key() != last_key:
                tags = freeform_tags(p.blend, fmt=fmt)
                if tags:
                    out.append(tags)
                last_key = p.blend.key()
            out.append(p.text)
        return " ".join(s for s in out if s).strip()

    def payload(self, req: SynthesisRequest) -> dict:
        o = self.options
        vo = req.voice.options.get("fish", {})
        body: dict = {
            "text": self.render_text(req), "format": "wav", "normalize": bool(o.get("normalize", True)),
            "seed": int(req.seed) % (2 ** 31), "references": [], "chunk_length": int(o.get("chunk_length", 200)),
        }
        if self.drama:
            body["single_chunk"] = bool(o.get("single_chunk", True))
        for k in ("temperature", "top_p", "max_new_tokens"):
            v = req.params.get(k, o.get(k))
            if v is not None:
                body[k] = v
        ref_id = vo.get("reference_id") or (o.get("reference_ids") or {}).get(req.voice.name)
        if ref_id:
            body["reference_id"] = ref_id
        elif req.voice.reference:
            with open(req.voice.reference, "rb") as f:
                body["references"].append({"audio": f.read(), "text": req.voice.reference_text})
        return body

    def synthesize(self, req: SynthesisRequest) -> Audio:
        body = msgpack.packb(self.payload(req), use_bin_type=True)
        headers = {"Content-Type": "application/msgpack"}
        if self.options.get("api_key"):
            headers["Authorization"] = f"Bearer {self.options['api_key']}"
        r = urllib.request.Request(self.url + "/v1/tts", data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(r, timeout=float(self.options.get("timeout", 600))) as resp:
                audio = resp.read()
        except urllib.error.URLError as exc:
            raise BackendUnavailable(f"fish server error: {exc}") from exc
        x, sr = _decode(audio)
        return Audio(x, sr, {"backend": self.name, "text": self.render_text(req)})


def _decode(data: bytes):
    import soundfile as sf

    x, sr = sf.read(_io.BytesIO(data), dtype="float32", always_2d=False)
    return io.to_mono(x), sr
