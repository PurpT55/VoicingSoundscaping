"""``vss`` command line."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from . import __version__


def _project(path: str | None):
    from .config import Project

    p = Path(path or ".")
    if p.is_dir() and not (p / "project.yaml").exists():
        raise SystemExit(f"no project.yaml in {p.resolve()} (create one with: vss new <dir>)")
    return Project.load(p)


# ----------------------------------------------------------------------------- commands
def cmd_new(a):
    src = Path(__file__).resolve().parents[2] / "examples" / "templates" / "starter"
    dst = Path(a.dir)
    if dst.exists() and any(dst.iterdir()):
        raise SystemExit(f"{dst} exists and is not empty")
    if src.exists():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:  # installed without the examples folder: minimal skeleton
        (dst / "script").mkdir(parents=True, exist_ok=True)
        (dst / "project.yaml").write_text(f"title: {a.title}\nbackends: {{default: espeak, draft: espeak}}\n"
                                          "cast:\n  NARRATOR: {voice: narrator}\nvoices:\n  narrator: {}\n")
        (dst / "script" / "01_beginning.vss").write_text("# Chapter One\n\nOnce upon a time...\n")
    pj = dst / "project.yaml"
    pj.write_text(pj.read_text().replace("title: My Story", f"title: {a.title}"))
    print(f"created {dst}/ - edit project.yaml and script/*.vss, then: vss lint {dst} && vss render {dst} --draft")


def cmd_lint(a):
    from .render.lint import lint
    from .render.pipeline import Renderer
    from .script.parser import parse_files

    proj = _project(a.project)
    script = parse_files(proj.script_files(), proj.default_speaker)
    r = Renderer(proj, draft=a.draft, log=lambda s: None)
    issues = lint(proj, script, r)
    shown = [d for d in issues if a.verbose or d.level != "info"]
    for d in shown:
        print(d)
    n_err = sum(d.level == "error" for d in issues)
    n_warn = sum(d.level == "warning" for d in issues)
    n_info = sum(d.level == "info" for d in issues)
    lines = sum(len(c.lines) for c in script.chapters)
    print(f"{len(script.chapters)} chapter(s), {lines} line(s): {n_err} error(s), {n_warn} warning(s), {n_info} note(s)"
          + ("" if a.verbose or not n_info else " (-v to show notes)"))
    return 1 if n_err else 0


def cmd_plan(a):
    from .render.pipeline import Renderer
    from .render.plan import Cue, NonverbalClip, Pause, Unit
    from .script.parser import parse_files
    from .tts.base import SynthesisRequest

    proj = _project(a.project)
    script = parse_files(proj.script_files(), proj.default_speaker)
    r = Renderer(proj, draft=a.draft, log=lambda s: None)
    for ch in script.chapters:
        if a.chapter and ch.number not in a.chapter:
            continue
        plan = r.plan(ch)
        print(f"== chapter {ch.number}: {ch.title}")
        for s in plan.steps:
            if isinstance(s, Unit):
                b = r.backend(s.backend)
                req = SynthesisRequest(list(s.pieces), r.voice_spec(s.voice), s.blend, proj.voices[s.voice].language)
                print(f"  +{s.gap_before:.2f}s {s.speaker:<10} [{b.name}] ({s.blend}) {b.render_text(req)[:110]}")
            elif isinstance(s, Pause):
                print(f"  pause {s.seconds}s")
            elif isinstance(s, NonverbalClip):
                print(f"  <{s.nonverbal}> clip for {s.speaker}")
            elif isinstance(s, Cue):
                print(f"  @{s.kind} {' '.join(s.args)} {s.params or ''} {s.title}")
        for w in plan.warnings:
            print("  !", w)
    r.close()


def cmd_render(a):
    from .render.pipeline import Renderer

    proj = _project(a.project)
    r = Renderer(proj, draft=a.draft, takes=a.takes, use_cache=not a.no_cache, preset=a.preset)
    book = r.render(a.chapter, Path(a.out) if a.out else None, force=a.force)
    for ch in book["chapters"]:
        ok = ("PASS" if ch["acx"].get("ok") else "FAIL") + ("" if ch["acx"].get("applies") else " (n/a for this preset)")
        flagged = sum(u["flagged"] for u in ch["units"])
        print(f"chapter {ch['number']}: {ch['duration'] / 60:.1f} min, ACX {ok}, {flagged} flagged take(s) -> "
              + ", ".join(Path(v).name for v in ch["files"].values()))
    if book.get("m4b"):
        print(f"book: {book['m4b']}")
    out = Path(a.out) if a.out else proj.out_dir
    print(f"report: {out / 'render_report.json'}  listening sheet: {out / 'QA_SHEET.md'}")


def cmd_say(a):
    from .config import Project
    from .mix.timeline import layout, mixdown
    from .render.pipeline import Renderer
    from .script.parser import parse_text

    proj = _project(a.project) if a.project else Project.from_dict(
        {"backends": {"default": a.backend or "espeak"}, "render": {"announce_chapters": False}})
    if a.backend:
        proj.cfg["backends"]["default"] = a.backend
    proj.cfg["render"]["announce_chapters"] = False
    script = parse_text("# say\n" + a.line, "<say>", default_speaker=proj.default_speaker)
    for d in script.diagnostics:
        print(d)
    r = Renderer(proj, takes=a.takes, log=print)
    ch = script.chapters[0]
    plan = r.plan(ch)
    clips = {}
    for u in plan.units:
        clips[u.uid], res = r.render_unit(u)
        print(f"{u.speaker} [{res.backend}] ({u.blend}) score {res.score}: {res.render_text}")
    tl = layout(plan, clips, r.sr, proj.cfg, r.resolver, nonverbal_audio=r.nonverbal_audio)
    mix, _, _ = mixdown(tl, r.resolver, proj.cfg)
    from .audio import io
    from .audio.master import master

    y, _ = master(mix, r.sr, a.preset or "podcast")
    io.save(a.out, y, r.sr)
    r.close()
    print(f"wrote {a.out}")


def cmd_doctor(a):
    from .emotion import coverage
    from .tts import registry

    print(f"vss {__version__}  python {sys.version.split()[0]}")
    print(f"ffmpeg: {shutil.which('ffmpeg') or 'MISSING (needed for mp3/m4b)'}")
    try:
        import torch

        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
        print(f"torch {torch.__version__}, CUDA GPU: {gpu}")
    except ImportError:
        print("torch: not installed in this env (fine: heavy models run in their own venvs via workers)")
    proj = None
    if a.project or Path("project.yaml").exists():
        proj = _project(a.project)
    reps = []
    print("\nbackends:")
    for name in registry.BACKENDS:
        try:
            b = registry.create(name, None, proj) if proj else registry.create(name, {})
            ok, why = b.available()
            caps = b.capabilities()
            reps.append(coverage.report(name, caps))
            print(f"  {'OK ' if ok else '-- '} {name:<14} {why}")
        except Exception as exc:  # noqa: BLE001
            print(f"  ERR {name:<14} {exc}")
    print("\nemotional coverage (controllability, not a quality score):")
    print(coverage.format_table(reps))
    if proj:
        from .voices.bank import coverage as bank_cov

        shared = bank_cov(proj.bank_dir("emotions"))
        print(f"\nshared emotion bank: {sum(shared.values())}/{len(shared)} emotions ({proj.bank_dir('emotions')})")
        for v in proj.voices.values():
            if v.emotion_bank:
                c = bank_cov(v.emotion_bank)
                print(f"  voice {v.name}: {sum(c.values())}/{len(c)}")
            ref = "ok" if v.reference and v.reference.exists() else ("MISSING " + str(v.reference) if v.reference else "none")
            print(f"  voice {v.name:<14} reference: {ref}")


def cmd_emotions(a):
    from .emotion import coverage
    from .emotion import taxonomy as tx
    from .tts import registry

    if a.markdown:
        print(emotions_markdown())
        return
    if a.list:
        for e in tx.EMOTIONS:
            al = ", ".join(e.aliases[:4])
            print(f"{e.id:<14} {e.kind:<9} {e.family:<10} {e.describe}  [{al}]")
        print("\nnonverbals: " + ", ".join(f"<{n.id}>" for n in tx.NONVERBALS))
        return
    names = a.backend or list(registry.BACKENDS)
    reps = []
    for n in names:
        opts = {}
        if "=" in n:
            n, _, spec = n.partition("=")
            opts = dict(kv.split(":") for kv in spec.split(","))
        b = registry.create(n, opts)
        reps.append(coverage.report(n + (f"({','.join(f'{k}={v}' for k, v in opts.items())})" if opts else ""),
                                    b.capabilities(), bank=a.bank))
    if len(reps) > 1:
        reps.append(coverage.combine("BEST-OF-ALL", reps))
    print(coverage.format_table(reps, show_rows=a.rows))


def emotions_markdown() -> str:
    """Taxonomy + coverage tables (used to generate docs/EMOTIONS.md)."""
    from .emotion import coverage
    from .emotion import taxonomy as tx
    from .tts import registry

    out = ["| id | kind | family | acting note | aliases | Step-Audio-EditX |", "|---|---|---|---|---|---|"]
    for e in tx.EMOTIONS:
        step = f"{e.step[0]}:{e.step[1]}" if e.step else ""
        out.append(f"| `{e.id}` | {e.kind} | {e.family} | {e.describe} | {', '.join(e.aliases[:5])} | {step} |")
    out += ["", "| nonverbal | Fish tag | Chatterbox Turbo | Step-Audio-EditX |", "|---|---|---|---|"]
    for n in tx.NONVERBALS:
        out.append(f"| `<{n.id}>` | {n.fish} | {n.chatterbox or '-'} | {n.step or '-'} |")
    setups = [("indextts", {}, False), ("indextts", {}, True), ("fish", {"model": "s2-pro"}, False),
              ("fish", {"model": "drama-3"}, False), ("qwen3", {"mode": "custom_voice"}, False),
              ("openai_speech", {"emotion_mode": "inline_tags"}, False), ("chatterbox", {"model": "original"}, False),
              ("chatterbox", {"model": "turbo"}, False), ("step_edit", {"base": "indextts"}, False),
              ("kokoro", {}, False), ("espeak", {}, False)]
    out += ["", "| backend setup | native+bank | weighted | nonverbals |", "|---|---|---|---|"]
    for name, opts, bank in setups:
        r = coverage.report(name, registry.create(name, opts).capabilities(), bank=bank)
        label = name + (f" ({', '.join(f'{k}={v}' for k, v in opts.items())})" if opts else "") + (" + full emotion bank" if bank else "")
        out.append(f"| {label} | {r.native_pct:.1f}% | {r.weighted_pct:.1f}% | {r.nonverbal_pct:.1f}% |")
    return "\n".join(out)


def cmd_sounds(a):
    from .sound import procedural

    for p in procedural.REGISTRY.values():
        print(f"{p.name:<16} {p.kind:<8} {p.description}" + (f"  (aka {', '.join(p.aliases)})" if p.aliases else ""))


def cmd_sound(a):
    from .audio import io
    from .sound import procedural

    x = procedural.render(a.name, a.seconds, a.sr, seed=a.seed)
    io.save(a.out, x, a.sr)
    print(f"wrote {a.out} ({len(x) / a.sr:.1f}s)")


def cmd_acx(a):
    from .audio import io
    from .audio.acx import check

    x, sr = io.load(a.file)
    rep = check(x, sr)
    print(rep.format())
    return 0 if rep.ok else 1


def cmd_master(a):
    from .audio import io
    from .audio.master import master

    x, sr = io.load(a.input)
    y, rep = master(x, sr, a.preset)
    io.save(a.output, y, sr)
    print(json.dumps(rep, indent=1))


def cmd_bank(a):
    from .tts import registry
    from .tts.base import VoiceSpec
    from .voices import bank

    proj = _project(a.project)
    if a.action == "list":
        dirs = {"shared": proj.bank_dir("emotions")}
        dirs.update({f"voice:{v.name}": v.emotion_bank for v in proj.voices.values() if v.emotion_bank})
        for label, d in dirs.items():
            cov = bank.coverage(d)
            missing = [k for k, v in cov.items() if not v]
            print(f"{label}: {sum(cov.values())}/{len(cov)} ({d})")
            if a.verbose:
                print("  missing: " + ", ".join(missing))
        return
    backend = registry.create(a.backend, None, proj)
    ok, why = backend.available()
    if not ok:
        raise SystemExit(f"backend {a.backend} unavailable: {why}")
    if a.voice:
        v = proj.voices[a.voice]
        out = v.emotion_bank or proj.root / "voices" / v.name / "emotions"
        spec = VoiceSpec(v.name, v.reference, v.reference_text, v.description, v.language, v.options)
    else:
        out = proj.bank_dir("emotions") or proj.root / "banks" / "emotions"
        vname = a.reference_voice or next(iter(proj.voices))
        v = proj.voices[vname]
        spec = VoiceSpec(v.name, v.reference, v.reference_text, v.description, v.language, v.options)
    emotions = [e.strip() for e in a.emotions.split(",")] if a.emotions else None
    made = bank.build(backend, spec, Path(out), emotions, intensity=a.intensity, overwrite=a.overwrite)
    backend.close()
    print(f"{len(made)} clip(s) written to {out}")


def cmd_annotate(a):
    from .llm.annotate import annotate

    text = Path(a.manuscript).read_text(encoding="utf-8")
    out = annotate(text, a.llm, a.model, title=a.title or "", api_key=a.api_key)
    Path(a.out).write_text(out, encoding="utf-8")
    print(f"wrote {a.out} - review it, then run vss lint")


# ----------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="vss", description="VoicingSoundscaping: local audiobook & storytelling engine")
    ap.add_argument("--version", action="version", version=f"vss {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("new", help="create a project from the starter template")
    s.add_argument("dir")
    s.add_argument("--title", default="My Story")
    s.set_defaults(fn=cmd_new)

    s = sub.add_parser("lint", help="check scripts against the project")
    s.add_argument("project", nargs="?")
    s.add_argument("-v", "--verbose", action="store_true")
    s.add_argument("--draft", action="store_true", help="lint against the draft backend")
    s.set_defaults(fn=cmd_lint)

    s = sub.add_parser("plan", help="show how each line will be performed (backend, emotion, model text)")
    s.add_argument("project", nargs="?")
    s.add_argument("--chapter", type=int, nargs="*")
    s.add_argument("--draft", action="store_true")
    s.set_defaults(fn=cmd_plan)

    s = sub.add_parser("render", help="render chapters (and the m4b) to the output folder")
    s.add_argument("project", nargs="?")
    s.add_argument("--chapter", type=int, nargs="*")
    s.add_argument("--draft", action="store_true", help="use the fast draft backend for everything")
    s.add_argument("--takes", type=int)
    s.add_argument("--preset", choices=["acx", "audiobook", "podcast", "cinematic", "bedtime"])
    s.add_argument("--out")
    s.add_argument("--no-cache", action="store_true")
    s.add_argument("--force", action="store_true", help="render despite script errors")
    s.set_defaults(fn=cmd_render)

    s = sub.add_parser("say", help="audition one line in script syntax")
    s.add_argument("line", help='e.g. "MIRA (whisper, afraid 0.7): Is anyone there?"')
    s.add_argument("--project")
    s.add_argument("--backend")
    s.add_argument("--takes", type=int, default=1)
    s.add_argument("--preset")
    s.add_argument("--out", default="say.wav")
    s.set_defaults(fn=cmd_say)

    s = sub.add_parser("doctor", help="check environment, backends and coverage")
    s.add_argument("project", nargs="?")
    s.set_defaults(fn=cmd_doctor)

    s = sub.add_parser("emotions", help="emotion taxonomy and per-backend coverage")
    s.add_argument("--backend", nargs="*", help="backend names, optionally name=key:val,key:val")
    s.add_argument("--bank", action="store_true", help="assume emotion banks are complete")
    s.add_argument("--rows", action="store_true", help="show the per-emotion matrix")
    s.add_argument("--list", action="store_true", help="list all emotions with descriptions")
    s.add_argument("--markdown", action="store_true", help="print taxonomy and coverage tables as markdown")
    s.set_defaults(fn=cmd_emotions)

    s = sub.add_parser("sounds", help="list procedural sounds")
    s.set_defaults(fn=cmd_sounds)

    s = sub.add_parser("sound", help="render one procedural sound")
    s.add_argument("name")
    s.add_argument("--seconds", type=float)
    s.add_argument("--sr", type=int, default=44100)
    s.add_argument("--seed", default="0")
    s.add_argument("--out", default="sound.wav")
    s.set_defaults(fn=cmd_sound)

    s = sub.add_parser("acx", help="check an audio file against ACX/Audible requirements")
    s.add_argument("file")
    s.set_defaults(fn=cmd_acx)

    s = sub.add_parser("master", help="master any audio file with a preset")
    s.add_argument("input")
    s.add_argument("output")
    s.add_argument("--preset", default="acx")
    s.set_defaults(fn=cmd_master)

    s = sub.add_parser("bank", help="emotion banks: list coverage or build clips with a backend")
    s.add_argument("action", choices=["list", "build"])
    s.add_argument("project", nargs="?")
    s.add_argument("--backend", default="fish")
    s.add_argument("--voice", help="build a per-voice bank (else the shared bank)")
    s.add_argument("--reference-voice", help="voice used to perform the shared bank")
    s.add_argument("--emotions", help="comma-separated subset")
    s.add_argument("--intensity", type=float, default=0.8)
    s.add_argument("--overwrite", action="store_true")
    s.add_argument("-v", "--verbose", action="store_true")
    s.set_defaults(fn=cmd_bank)

    s = sub.add_parser("annotate", help="draft a script from a manuscript with a local LLM")
    s.add_argument("manuscript")
    s.add_argument("--llm", default="http://127.0.0.1:11434/v1", help="OpenAI-compatible base URL (Ollama default)")
    s.add_argument("--model", default="qwen3:14b")
    s.add_argument("--api-key")
    s.add_argument("--title")
    s.add_argument("--out", default="draft.vss")
    s.set_defaults(fn=cmd_annotate)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rc = args.fn(args)
    return int(rc or 0)


if __name__ == "__main__":
    raise SystemExit(main())
