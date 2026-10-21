import pytest

from vss.emotion import coverage
from vss.emotion import taxonomy as tx
from vss.emotion.mapping import (
    Blend, chatterbox_params, freeform_tags, index_vector, instruct_text, prosody, step_edits, step_is_exact,
)
from vss.tts import registry


def test_taxonomy_size_and_kinds():
    assert len(tx.EMOTIONS) >= 90
    kinds = {e.kind for e in tx.EMOTIONS}
    assert kinds == set(tx.KINDS)
    assert len(tx.NONVERBALS) >= 20


def test_requested_range_is_present():
    # the emotions explicitly asked for in the project brief
    for name in ["cry", "whisper", "scream", "anger", "tired", "bored", "cheeky", "dry humor", "peaky", "peppy",
                 "frustrated", "joy", "contentment", "sleepy", "peeved", "sad", "laughing", "sobbing"]:
        assert tx.find(name) is not None, name


def test_aliases_do_not_collide():
    seen = {}
    for e in tx.EMOTIONS:
        for a in (e.id, *e.aliases):
            a = a.lower()
            assert seen.get(a, e.id) == e.id, f"alias {a} used by {seen.get(a)} and {e.id}"
            seen[a] = e.id


def test_entries_are_complete():
    for e in tx.EMOTIONS:
        assert e.describe and e.line and e.tag
        assert len(e.vec) == 8 and all(0 <= v <= 1 for v in e.vec)
        assert -1 <= e.valence <= 1 and -1 <= e.arousal <= 1 and -1 <= e.dominance <= 1
        assert 0.6 <= e.rate <= 1.3
        assert e.e2v in tx.E2V_CLASSES
        if e.step:
            assert e.step[1] in (tx.STEP_EMOTIONS if e.step[0] == "emotion" else tx.STEP_STYLES)


def test_blend_parse_and_merge():
    b = Blend.parse("whisper, fear 0.7")
    assert [p.emotion.id for p in b.parts] == ["whisper", "afraid"]
    assert b.parts[1].intensity == pytest.approx(0.7)
    merged = Blend.parse("content 0.5, whisper").merged(Blend.parse("angry"))
    ids = [p.emotion.id for p in merged.parts]
    assert "content" not in ids and "angry" in ids and "whisper" in ids  # delivery persists
    with pytest.raises(KeyError):
        Blend.parse("not_an_emotion")


def test_index_vector_capped_and_scaled():
    v = index_vector(Blend.parse("furious 1.0, terrified 1.0"), cap=0.8)
    assert sum(v) == pytest.approx(0.8, abs=1e-3)
    weak = index_vector(Blend.parse("sad 0.2"))
    strong = index_vector(Blend.parse("sad 0.8"))
    assert weak[2] < strong[2]
    assert index_vector(Blend()) is None


def test_text_renderings():
    b = Blend.parse("whisper, afraid 0.85", "barely breathing")
    assert "Whispering" in instruct_text(b) and "strong" in instruct_text(b) and "barely breathing" in instruct_text(b)
    tags = freeform_tags(b)
    assert tags.startswith("[whispering]") and "[very afraid]" in tags and "[barely breathing]" in tags
    assert instruct_text(Blend()) == ""


def test_chatterbox_and_prosody():
    calm = chatterbox_params(Blend.parse("calm 0.6"))
    angry = chatterbox_params(Blend.parse("furious 0.9"))
    assert angry["exaggeration"] > calm["exaggeration"]
    assert angry["cfg_weight"] <= calm["cfg_weight"]
    assert prosody(Blend.parse("whisper")).level <= -6
    assert prosody(Blend.parse("sleepy 0.8")).rate < 0.9


def test_step_edits():
    edits = step_edits(Blend.parse("whisper 0.5, happy 0.9"))
    assert ("style", "whisper") in [(e.edit_type, e.edit_info) for e in edits]
    assert all(e.iterations >= 1 for e in edits)
    assert step_is_exact(tx.get("happy")) and not step_is_exact(tx.get("crying"))


def test_coverage_targets():
    fish_s2 = coverage.report("fish", registry.create("fish", {"model": "s2-pro"}).capabilities())
    drama = coverage.report("fish", registry.create("fish", {"model": "drama-3"}).capabilities())
    index_bank = coverage.report("indextts", registry.create("indextts", {}).capabilities(), bank=True)
    index = coverage.report("indextts", registry.create("indextts", {}).capabilities())
    espeak = coverage.report("espeak", registry.create("espeak", {}).capabilities())
    assert fish_s2.native_pct >= 80          # project goal: 80-90 % of the range
    assert drama.native_pct == 100.0
    assert index_bank.native_pct == 100.0
    assert index.native_pct < index_bank.native_pct
    assert espeak.native_pct == 0.0
    combo = coverage.combine("routed", [index, fish_s2])
    assert combo.native_pct >= fish_s2.native_pct
    step = coverage.report("step", registry.create("step_edit", {"base": "indextts"}).capabilities())
    assert step.native_pct >= index.native_pct  # refinement never loses coverage
