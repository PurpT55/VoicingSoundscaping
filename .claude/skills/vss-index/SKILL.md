---
name: vss-index
description: Session protocol for the VoicingSoundscaping project - read and update 0_Index.txt (goals, workstreams, progress, next actions, decisions, session log). Use at the start of ANY session in this repo, before ending a session, and whenever work items are started, finished, blocked or discovered.
---

# VSS index protocol

`0_Index.txt` at the repo root is the single source of truth for what VSS is, where it stands and
what comes next. It must stay **compact** (< 300 lines), **current** and **readable cold** by a new
session with no other context.

## At session start
1. Read `0_Index.txt` fully (it is short by design).
2. Read the skill for the area you'll touch (`vss-director`, `vss-voices`, `vss-emotion`,
   `vss-soundscape`, `vss-mastering-qa`, `vss-backends`, `vss-dev`).
3. Confirm the tree matches the index: `git log --oneline -10`, `.venv/bin/python -m pytest -q`
   (or `pip install -e ".[dev]" && pytest -q` in a fresh checkout).

## While working
* Use workstream IDs (`W3.2`) in commit messages and notes.
* New gap found? Add it under its workstream as `[ ]` with one line of context. Don't let it vanish
  into chat.

## Before ending a session (or after any meaningful milestone)
Update, in place:
* **STATUS** line (date, one-sentence state, test count).
* Workstream items: `[x]` done, `[~]` in progress / partial (say what's missing), `[ ]` todo,
  `[!]` blocked (say on what).
* **NEXT** - the 3-7 highest-value actions, ordered, each actionable by a fresh session.
* **DECISIONS** - append (never rewrite) one line per decision with the date and the reason.
* **SESSION LOG** - append one line: date, what changed, what was verified (tests/renders).

## Style rules
* One line per item; details belong in docs/ or code, not the index.
* Paths relative to the repo root; commands copy-pasteable.
* Never claim something works unless it was run; write "untested on GPU" etc. explicitly.
* Prefer editing over appending; prune finished noise from NEXT.

## Status legend
`[x]` done & verified · `[~]` partial · `[ ]` todo · `[!]` blocked · `(gpu)` needs a GPU machine to verify
