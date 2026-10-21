"""Turn a plain manuscript into a draft ``.vss`` script with a LOCAL LLM (OpenAI-compatible chat API:
Ollama, llama.cpp server, LM Studio, vLLM ...). The draft is then linted and auto-repaired.

The same instructions are used by the in-session ``vss-director`` skill, where Claude does the
annotation directly - that path gives the best results; this one runs fully offline.
"""

from __future__ import annotations

import json
import re
import urllib.request

from ..emotion import taxonomy as tx
from ..script.parser import parse_text

SYSTEM = """You are an audiobook director. Convert prose into the VSS script format so a multi-voice
engine can perform it. Rules:
- Keep EVERY word of the original text, in order. Never summarise, never invent dialogue.
- Narration stays as plain paragraphs (read by the narrator).
- Quoted dialogue becomes its own line: SPEAKER (emotion intensity, ...): text
  Speaker tags are UPPERCASE names. Keep dialogue tags ("she whispered") in the narration.
- Pick emotions ONLY from this list (intensity 0.1-1.0, default 0.6): {emotions}
- Deliveries (combine with an emotion): whisper, murmur, soft, shout, scream, roar, trembling, laughing, sobbing, sing_song.
- Nonverbal sounds inline: <laugh> <chuckle> <giggle> <sigh> <gasp> <sob> <sniff> <yawn> <cough> <groan> <breath> <hmm>.
- Mid-line emotion change: [emotion 0.6] before the words it affects.
- Optional sound design lines (only when the text clearly implies them):
  @scene ambience="rain:-4, fire:-10" space=living_room
  @sfx door_knock   @music start wonder prompt="soft piano, wonder"   @music stop fade=6
- A new location/time gets a "## Scene title" line.
Return ONLY the script text, no commentary."""


def chat(url: str, model: str, system: str, user: str, temperature: float = 0.3, timeout: float = 600.0,
         api_key: str | None = None) -> str:
    body = {"model": model, "temperature": temperature,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode())
    return data["choices"][0]["message"]["content"]


def split_manuscript(text: str, max_chars: int = 3500) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, buf = [], ""
    for p in paras:
        if buf and len(buf) + len(p) + 2 > max_chars:
            chunks.append(buf)
            buf = p
        else:
            buf = f"{buf}\n\n{p}" if buf else p
    if buf:
        chunks.append(buf)
    return chunks


def repair(script_text: str) -> tuple[str, list[str]]:
    """Fix unknown emotions/events using the closest taxonomy names; drop what cannot be fixed."""
    fixes: list[str] = []
    parsed = parse_text(script_text)
    for d in parsed.diagnostics:
        m = re.search(r"unknown (emotion|event) '?<?([^'>]+)>?'?(?: \(did you mean: ([^?)]*))?", d.message)
        if not m:
            continue
        bad, sugg = m.group(2), (m.group(3) or "").split(",")[0].strip()
        if m.group(1) == "emotion":
            if sugg and tx.find(sugg):
                script_text = re.sub(rf"\b{re.escape(bad)}\b", sugg, script_text)
                fixes.append(f"emotion '{bad}' -> '{sugg}'")
            else:
                script_text = re.sub(rf",?\s*\b{re.escape(bad)}\b\s*(\d*\.?\d+)?", "", script_text)
                fixes.append(f"emotion '{bad}' removed")
        else:
            if sugg and tx.find_nonverbal(sugg):
                script_text = script_text.replace(f"<{bad}>", f"<{sugg}>")
                fixes.append(f"<{bad}> -> <{sugg}>")
            else:
                script_text = script_text.replace(f"<{bad}>", "")
                fixes.append(f"<{bad}> removed")
    script_text = re.sub(r"\(\s*,?\s*\)", "", script_text)
    return script_text, fixes


def annotate(manuscript: str, url: str, model: str, title: str = "", api_key: str | None = None, log=print) -> str:
    emotions = ", ".join(e.id for e in tx.EMOTIONS if e.kind != "delivery")
    system = SYSTEM.format(emotions=emotions)
    parts, cast = [], set()
    for i, chunk in enumerate(split_manuscript(manuscript), 1):
        hint = f"Known speakers so far: {', '.join(sorted(cast))}.\n\n" if cast else ""
        log(f"  annotating chunk {i} ({len(chunk)} chars)")
        out = chat(url, model, system, hint + chunk, api_key=api_key)
        out = re.sub(r"^```\w*\n|\n```$", "", out.strip())
        cast |= set(re.findall(r"^([A-Z][A-Z0-9 _'-]{1,30})\s*(?:\(|:)", out, flags=re.M))
        parts.append(out)
    head = f"# {title}\n\n" if title and not parts[0].lstrip().startswith("#") else ""
    text, fixes = repair(head + "\n\n".join(parts) + "\n")
    for f in fixes:
        log(f"  repaired: {f}")
    return text
