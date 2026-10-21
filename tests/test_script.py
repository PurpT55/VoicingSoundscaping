from vss.script import model as m
from vss.script.normalize import ChunkPolicy, chunk, normalize, split_sentences
from vss.script.parser import parse_text

SAMPLE = """---
title: Book
---
# Chapter One: The Light
## Cliff
@scene ambience="sea_waves:-6, wind_soft" space=outdoor_open
@music start wonder prompt="harp, wonder" gain=-20
Prose line one
continues here.

MIRA (whisper, afraid 0.6, "afraid to wake something", rate=0.95): Is anyone there? <gasp> [sad 0.7] Hello?
OLD TOM (dry_humor): Only me. <chuckle> {Aoife|EE-fa}, the *kettle* is on. <sfx kettle gain=-8> <pause 1.2> \\[not an emotion\\]
// a comment
@pause 2
# Chapter Two
Second chapter text.
"""


def test_structure():
    s = parse_text(SAMPLE, "t.vss")
    assert not s.errors, [str(d) for d in s.diagnostics]
    assert s.meta["title"] == "Book"
    assert [c.title for c in s.chapters] == ["Chapter One: The Light", "Chapter Two"]
    ev = s.chapters[0].events
    assert isinstance(ev[0], m.Scene) and ev[0].title == "Cliff"
    assert isinstance(ev[1], m.Directive) and ev[1].params["space"] == "outdoor_open"
    assert ev[2].args == ["start", "wonder"] and ev[2].params["gain"] == -20
    prose = ev[3]
    assert prose.narration and prose.speaker == "NARRATOR"
    assert prose.spoken_text == "Prose line one continues here."


def test_speaker_line_details():
    s = parse_text(SAMPLE, "t.vss")
    lines = s.chapters[0].lines
    mira, tom = lines[1], lines[2]
    assert mira.speaker == "MIRA" and mira.params == {"rate": 0.95}
    assert [p.emotion.id for p in mira.blend.parts] == ["whisper", "afraid"]
    assert mira.blend.direction == "afraid to wake something"
    kinds = [type(i).__name__ for i in mira.items]
    assert kinds == ["Text", "NonverbalEvent", "EmotionShift", "Text"]
    assert tom.speaker == "OLD TOM"
    texts = [i for i in tom.items if isinstance(i, m.Text)]
    assert "EE-fa" in texts[1].text and "Aoife" in texts[1].display and "*" not in texts[1].text
    sfx = [i for i in tom.items if isinstance(i, m.SfxEvent)][0]
    assert sfx.name == "kettle" and sfx.params["gain"] == -8
    assert any(isinstance(i, m.PauseEvent) and i.seconds == 1.2 for i in tom.items)
    assert "[not an emotion]" in tom.display_text


def test_errors_are_collected_with_suggestions():
    s = parse_text("# C\nBOB (angryy): hi <laughh> [very_wrong]\n@bogus x\n", "e.vss")
    msgs = [d.message for d in s.errors]
    assert any("angryy" in x and "angry" in x for x in msgs)
    assert any("laughh" in x for x in msgs)
    assert any("very_wrong" in x for x in msgs)
    assert any("@bogus" in x for x in msgs)
    assert all(d.source.file == "e.vss" for d in s.errors)


def test_prose_with_colon_is_not_a_speaker():
    s = parse_text("# C\nThen she said: hello.\n", "p.vss")
    line = s.chapters[0].lines[0]
    assert line.narration and line.spoken_text == "Then she said: hello."


def test_normalize_numbers_and_abbreviations():
    out = normalize("Dr. Smith paid $5.50 at 3:15 p.m. on the 21st, in 1999. Mrs. Lee had 45% of 1,200.")
    assert "Doctor" in out and "five dollars and fifty cents" in out
    assert "three fifteen p.m." in out and "p.m.." not in out
    assert "twenty-first" in out and "nineteen ninety-nine" in out
    assert "forty-five percent" in out and "one thousand two hundred" in out
    assert normalize("Aoife came", lexicon={"Aoife": "EE-fa"}) == "EE-fa came"
    assert "o'clock" in normalize("at 7:00")


def test_chunking_limits():
    text = " ".join(f"Sentence number {i} is right here." for i in range(40))
    pol = ChunkPolicy(target_chars=120, max_chars=160)
    chunks = chunk(text, pol)
    assert all(len(c) <= 160 for c in chunks)
    assert " ".join(chunks) == text
    long = "word " * 200
    assert all(len(c) <= 160 for c in chunk(long.strip(), pol))
    assert split_sentences('He left. "Wait!" she cried. Then quiet.') == ["He left.", '"Wait!" she cried.', "Then quiet."]
