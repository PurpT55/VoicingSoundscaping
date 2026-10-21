"""Script linting against the project: speakers, voices, sounds, nonverbals and emotional coverage."""

from __future__ import annotations

from ..emotion.coverage import emotion_level
from ..script import model as m
from ..script.model import Diagnostic


def lint(project, script: m.Script, renderer=None) -> list[Diagnostic]:
    out: list[Diagnostic] = list(script.diagnostics)
    out += [Diagnostic("warning", w) for w in project.warnings]
    cast = set(project.cast)
    if not script.chapters:
        out.append(Diagnostic("error", "no chapters found (check 'script:' globs in project.yaml)"))
    for spk in sorted(script.speakers() - cast):
        out.append(Diagnostic("warning", f"speaker '{spk}' is not in cast: (an automatic voice will be used)"))
    resolver = renderer.resolver if renderer else None
    weak: dict[str, dict[str, int]] = {}
    for ch in script.chapters:
        for ev in ch.events:
            if isinstance(ev, m.Directive) and resolver is not None:
                _check_directive(ev, resolver, out)
            if isinstance(ev, m.Line):
                _check_line(project, ev, renderer, out, weak)
    for label, counts in weak.items():
        items = ", ".join(f"{k} x{v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))
        out.append(Diagnostic("info", f"on {label}: only approximated or unsupported -> {items} "
                                      "(fill an emotion bank, or route these to a stronger backend)"))
    return out


def _check_directive(d: m.Directive, resolver, out: list[Diagnostic]) -> None:
    from ..mix.timeline import parse_layers

    names: list[tuple[str, str]] = []
    if d.name == "sfx" and d.args:
        names.append(("sfx", d.args[0]))
    elif d.name == "scene" and "ambience" in d.params:
        names += [("ambience", n) for n, _ in parse_layers(d.params["ambience"])]
    elif d.name == "ambience" and d.args:
        rest = d.args[1:] if d.args[0].lower() in ("add", "remove", "stop", "set") else d.args
        names += [("ambience", n) for n, _ in parse_layers(", ".join(rest))]
    for kind, name in names:
        if resolver.describe(kind, name, d.params.get("prompt")) is None:
            out.append(Diagnostic("warning", f"{kind} '{name}' not found in libraries, generators or procedural set",
                                  d.source))


def _check_line(project, line: m.Line, renderer, out: list[Diagnostic], weak: dict) -> None:
    for it in line.items:
        if isinstance(it, m.SfxEvent) and renderer is not None:
            if renderer.resolver.describe("sfx", it.name, it.params.get("prompt")) is None:
                out.append(Diagnostic("warning", f"sfx '{it.name}' not found", line.source))
    if renderer is None:
        return
    from ..render.plan import route_backend

    member = project.member(line.speaker)
    voice = project.voices[member.voice]
    blend = member.default.merged(line.blend)
    name = str(renderer.draft_backend or line.params.get("backend") or route_backend(
        blend, project.cfg["backends"].get("routing") or {},
        member.backend or voice.backend or project.cfg["backends"]["default"]))
    try:
        backend = renderer.backend(name)
        caps = backend.capabilities()
    except Exception:  # noqa: BLE001 - availability problems are reported by the renderer
        return
    label = f"'{backend.name}'" + (f" (standing in for '{name}')" if backend.name != name else "")
    for it in line.items:
        if isinstance(it, m.NonverbalEvent) and not caps.supports_nonverbal(it.name):
            from ..render.plan import nonverbal_bank_clip

            if not nonverbal_bank_clip(project, member.voice, it.name):
                bucket = weak.setdefault(label, {})
                bucket[f"<{it.name}>"] = bucket.get(f"<{it.name}>", 0) + 1
    from ..render.plan import emotion_bank_clip

    for p in blend.parts:
        lvl = emotion_level(p.emotion, caps, bank=False)
        if lvl in ("approx", "none") and not emotion_bank_clip(project, member.voice, blend, caps):
            bucket = weak.setdefault(label, {})
            bucket[p.emotion.id] = bucket.get(p.emotion.id, 0) + 1
    if caps.cloning and not voice.reference and name in ("indextts",):
        out.append(Diagnostic("error", f"voice '{voice.name}' has no reference clip (required by {name})", line.source))
