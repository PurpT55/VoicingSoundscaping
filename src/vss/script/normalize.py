"""Text normalisation and sentence chunking for speech synthesis.

Models read written text inconsistently ("Dr." "1999" "£5" "3:15"), so everything is turned
into speakable words *before* synthesis, with a per-project pronunciation lexicon on top.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

try:
    from num2words import num2words as _n2w
except ImportError:  # pragma: no cover - dependency is declared, keep a graceful fallback
    _n2w = None

ABBREVIATIONS = {
    "Mr.": "Mister", "Mrs.": "Missus", "Ms.": "Miz", "Dr.": "Doctor", "Prof.": "Professor",
    "St.": "Saint", "Mt.": "Mount", "Capt.": "Captain", "Sgt.": "Sergeant", "Lt.": "Lieutenant",
    "Col.": "Colonel", "Gen.": "General", "Jr.": "Junior", "Sr.": "Senior", "vs.": "versus",
    "etc.": "et cetera", "e.g.": "for example", "i.e.": "that is", "approx.": "approximately",
    "No.": "Number",
}
_CURRENCY = {"$": ("dollar", "dollars", "cent", "cents"), "£": ("pound", "pounds", "penny", "pence"),
             "€": ("euro", "euros", "cent", "cents")}


def _words(n: float | int, lang: str = "en", to: str = "cardinal") -> str:
    if _n2w is None:
        return str(n)
    try:
        out = _n2w(n, lang=lang, to=to)
    except (NotImplementedError, OverflowError, ValueError):
        out = _n2w(n, lang="en", to=to)
    return out.replace(",", "")


def _year(y: int, lang: str) -> str:
    if lang == "en" and 1100 <= y <= 2099 and y % 1000 >= 10 and not (2000 <= y <= 2009):
        return _words(y, lang, "year")
    return _words(y, lang)


def expand_numbers(text: str, lang: str = "en") -> str:
    def money(m: re.Match) -> str:
        sym, whole, frac = m.group(1), m.group(2).replace(",", ""), m.group(3)
        one, many, sub1, subn = _CURRENCY[sym]
        w = int(whole)
        out = f"{_words(w, lang)} {one if w == 1 else many}"
        if frac and int(frac):
            c = int(frac.ljust(2, "0")[:2])
            out += f" and {_words(c, lang)} {sub1 if c == 1 else subn}"
        return out

    text = re.sub(r"([$£€])(\d[\d,]*)(?:\.(\d{1,2}))?", money, text)
    text = re.sub(r"\b(\d{1,2}):(\d{2})\s*([ap])\.?m\.?(?!\w)",
                  lambda m: _clock(int(m.group(1)), int(m.group(2)), lang) + f" {m.group(3).lower()}.m.", text, flags=re.I)
    text = re.sub(r"\b(\d{1,2}):(\d{2})\b", lambda m: _clock(int(m.group(1)), int(m.group(2)), lang), text)
    text = re.sub(r"\b(\d+)(st|nd|rd|th)\b", lambda m: _words(int(m.group(1)), lang, "ordinal"), text)
    text = re.sub(r"\b(\d+(?:\.\d+)?)\s?%", lambda m: f"{_num(m.group(1), lang)} percent", text)
    text = re.sub(r"\b(1[1-9]\d\d|20\d\d)s\b", lambda m: _plural(_year(int(m.group(1)), lang)), text)
    text = re.sub(r"(?<![\d,.])\b(1[1-9]\d\d|20\d\d)\b(?![\d,.]\d)", lambda m: _year(int(m.group(1)), lang), text)
    text = re.sub(r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b", lambda m: _num(m.group(0).replace(",", ""), lang), text)
    text = re.sub(r"\b\d+\.\d+\b", lambda m: _num(m.group(0), lang), text)
    text = re.sub(r"\b\d+\b", lambda m: _words(int(m.group(0)), lang), text)
    return text


def _num(s: str, lang: str) -> str:
    if "." in s:
        whole, frac = s.split(".", 1)
        return f"{_words(int(whole or 0), lang)} point " + " ".join(_words(int(d), lang) for d in frac)
    return _words(int(s), lang)


def _clock(h: int, m: int, lang: str) -> str:
    if m == 0:
        return f"{_words(h, lang)} o'clock" if lang == "en" else _words(h, lang)
    if m < 10:
        return f"{_words(h, lang)} oh {_words(m, lang)}" if lang == "en" else f"{_words(h, lang)} {_words(m, lang)}"
    return f"{_words(h, lang)} {_words(m, lang)}"


def _plural(words: str) -> str:
    if words.endswith("y"):
        return words[:-1] + "ies"
    return words + "s"


def apply_lexicon(text: str, lexicon: dict[str, str]) -> str:
    """Whole-word, case-sensitive replacements (longest keys first)."""
    for key in sorted(lexicon, key=len, reverse=True):
        text = re.sub(rf"(?<!\w){re.escape(key)}(?!\w)", lexicon[key], text)
    return text


def clean_unicode(text: str) -> str:
    text = text.replace("…", "...").replace(" ", " ")
    text = re.sub(r"\s*[—–]\s*", " — ", text)  # spaced em dash
    text = re.sub(r"\.{4,}", "...", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize(text: str, lang: str = "en", lexicon: dict[str, str] | None = None) -> str:
    if lexicon:
        text = apply_lexicon(text, lexicon)
    for abbr, full in ABBREVIATIONS.items():
        text = re.sub(rf"(?<!\w){re.escape(abbr)}", full, text)
    text = expand_numbers(text, lang)
    return clean_unicode(text)


# ---------------------------------------------------------------------------- chunking
_SENT_END = re.compile(r"(?<=[.!?…])[\"'”’)\]]*\s+(?=[\"'“‘(\[]?[A-Z0-9À-ɏ])")


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENT_END.split(text) if p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


@dataclass
class ChunkPolicy:
    target_chars: int = 220   # merge short sentences up to this size
    max_chars: int = 300      # hard limit; longer sentences are split at commas/dashes


def chunk(text: str, policy: ChunkPolicy = ChunkPolicy()) -> list[str]:
    sentences: list[str] = []
    for s in split_sentences(text):
        sentences.extend(_split_long(s, policy.max_chars))
    chunks: list[str] = []
    buf = ""
    for s in sentences:
        if not buf:
            buf = s
        elif len(buf) + 1 + len(s) <= policy.target_chars:
            buf = f"{buf} {s}"
        else:
            chunks.append(buf)
            buf = s
    if buf:
        chunks.append(buf)
    return chunks


def _split_long(s: str, max_chars: int) -> list[str]:
    if len(s) <= max_chars:
        return [s]
    pieces = re.split(r"(?<=[,;:—])\s+", s)
    out, buf = [], ""
    for p in pieces:
        if buf and len(buf) + 1 + len(p) > max_chars:
            out.append(buf)
            buf = p
        else:
            buf = f"{buf} {p}".strip()
    if buf:
        out.append(buf)
    final = []
    for p in out:  # still too long (no punctuation): split on spaces
        while len(p) > max_chars:
            cut = p.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            final.append(p[:cut].strip())
            p = p[cut:].strip()
        if p:
            final.append(p)
    return final
