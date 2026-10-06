"""The profile update: new entries in, a proposed me.md out, for you to review.

Nothing changes until you accept it (keel review). Every new line has to
point at real entries, and lines under Hold me to and Corrections can only be
added or removed by you.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from datetime import datetime

from . import ollama
from .config import Config
from .harness import entries_block
from .profile import LOCKED, SECTIONS, Profile, cited, insert_line, sections
from .store import Store

SCHEMA = {
    "type": "object",
    "properties": {
        "profile": {"type": "string"},
        "changes": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["profile", "changes"],
}

INSTRUCTIONS = """You keep a person's self-profile, me.md, up to date from their journal entries.

Rules:
- Write in the first person, as them ("I ..."). Short bullet lines starting with "- ".
- Keep these sections, in this order: {sections}.
- Every line you add or change must end with the tags of the entries it comes from, like [e:12] or [e:3, e:9]. Only use tags from the entries given. Never add anything the entries don't support.
- Keep lines that are still true. Update "Right now" so it reflects the newest entries; move things that are over out of it.
- Never add, remove or change lines under "Hold me to" or "Corrections". Only the person does that.
- Lines under "Corrections" override everything. Never write anything that contradicts them.
- No diagnoses or labels. Describe patterns in their own terms.
- At most 8 lines per section. Prefer fewer, sharper lines.
- In "changes", list each change in a few plain words (for example "added: Ana's Sunday call")."""


@dataclass
class Proposal:
    text: str
    changes: list[str]
    dropped: list[str]
    new_entries: int


def validate(current: str, proposed: str, valid_ids: set[int]) -> tuple[str, list[str]]:
    """Drop new lines without real citations; put back any locked line the model removed."""
    old_lines = {line.strip() for line in current.splitlines()}
    kept, dropped = [], []
    section = None
    for line in proposed.splitlines():
        if line.startswith("## "):
            section = line[3:].strip()
        bullet = line.strip().startswith("- ")
        if bullet and line.strip() not in old_lines:
            ids = cited(line)
            if section in LOCKED or not ids or not ids <= valid_ids:
                dropped.append(line.strip())
                continue
        kept.append(line)
    text = "\n".join(kept).rstrip() + "\n"
    have = sections(text)
    for name in SECTIONS:
        if name not in have:
            text = text.rstrip("\n") + f"\n\n## {name}\n"
    for name in LOCKED:
        present = set(sections(text).get(name, []))
        for line in sections(current).get(name, []):
            if line not in present:
                text = insert_line(text, name, line)
    return text, dropped


def last_review(cfg: Config) -> float | None:
    try:
        return float(cfg.review_marker.read_text().strip())
    except (FileNotFoundError, ValueError):
        return None


def propose(cfg: Config, store: Store, profile: Profile, *, now: datetime | None = None, llm=ollama) -> Proposal | None:
    now = now or datetime.now().astimezone()
    new = store.entries(after=last_review(cfg))
    if not new:
        return None
    current = profile.read()
    content = (
        f"Today is {now:%A, %B} {now.day}, {now.year}.\n\n"
        f"The current me.md:\n{current}\n\n"
        f"New entries since the last update:\n{entries_block(new, now)}"
    )
    data = llm.chat_json(
        cfg.chat,
        [
            {"role": "system", "content": INSTRUCTIONS.format(sections=", ".join(SECTIONS))},
            {"role": "user", "content": content},
        ],
        SCHEMA,
    )
    valid = {row["id"] for row in store.entries()}
    text, dropped = validate(current, data["profile"], valid)
    cfg.proposal_path.write_text(text)
    return Proposal(text, list(data.get("changes", [])), dropped, len(new))


def diff(cfg: Config, profile: Profile) -> str:
    if not cfg.proposal_path.exists():
        return ""
    return "".join(
        difflib.unified_diff(
            profile.read().splitlines(keepends=True),
            cfg.proposal_path.read_text().splitlines(keepends=True),
            "me.md (now)",
            "me.md (proposed)",
        )
    )


def accept(cfg: Config, profile: Profile, *, now: datetime | None = None) -> bool:
    now = now or datetime.now().astimezone()
    changed = profile.write(cfg.proposal_path.read_text(), "profile update", when=now)
    cfg.review_marker.write_text(str(now.timestamp()))
    cfg.proposal_path.unlink()
    return changed


def reject(cfg: Config) -> None:
    cfg.proposal_path.unlink(missing_ok=True)
