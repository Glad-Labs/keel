"""Deciding what from your past bears on what you just said.

Left alone with your whole profile, a model brings up whatever it can see:
the pricing decision in a reply about cutting a feature, a commitment
stretched to fit. So a short first step picks what's relevant, and the reply
only ever sees what was picked. It can't drag in what it can't see.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from . import ollama
from .profile import TAG, cited, sections

# Shown with every reply, because they shape how Keel talks, not what it brings up.
ALWAYS = ("People", "How to talk to me")
MAX_PICKS = 3

SCHEMA = {
    "type": "object",
    "properties": {"thinking": {"type": "string"}, "pick": {"type": "array", "items": {"type": "string"}}},
    "required": ["thinking", "pick"],
}

INSTRUCTIONS = """You decide what from someone's past, if anything, bears on what they just said to their journal. You see their message and a list of things from their journal and their profile, each with a key like e:12 or p:3.

Pick the keys that bear directly on what they said: at most three, usually one, none if nothing does.
- A "Hold me to" line counts only if what they said goes against it: doing what it says not to, or skipping what it says to do. Don't stretch a line to fit.
- A pattern counts only if what they said is a clear example of it.
- A correction counts if what they said touches the thing they corrected.
- If they ask about something specific, pick what answers it, and nothing if nothing does.
- If they're low, prefer a time they got through something like it, or a person who helped.
{note}"""

MODE_NOTES = {
    "listen": "They asked to be listened to: pick at most one, and only if it helps them feel understood.",
    "care": "Something heavy came up: pick at most one, and only something comforting.",
    "past": "They're talking to their past self: pick what that past self would bring up in answer.",
}


@dataclass
class Item:
    key: str
    label: str
    entry_ids: set[int] = field(default_factory=set)
    row: object = None
    section: str | None = None
    line: str | None = None


def entry_label(row, now: datetime) -> str:
    from .harness import ago_text

    created = datetime.fromisoformat(row["created_at"])
    asked = f"(answering: {row['prompt']}) " if row["prompt"] else ""
    return f"{created:%a %b} {created.day}, {created.year} ({ago_text(created, now)}): {asked}{row['text']}"


def candidates(rows, profile_text: str, now: datetime) -> list[Item]:
    items = [Item(f"e:{r['id']}", entry_label(r, now), {r["id"]}, row=r) for r in rows]
    n = 0
    for section, lines in sections(profile_text).items():
        if section in ALWAYS:
            continue
        for line in lines:
            n += 1
            text = line[2:].strip()
            items.append(Item(f"p:{n}", f"({section}) {TAG.sub('', text).strip()}", cited(text), section=section, line=text))
    return items


def choose(ep, said: str, mode: str, items: list[Item], earlier: str = "", llm=ollama) -> tuple[list[Item], str]:
    if not items:
        return [], ""
    listing = "\n".join(f"{item.key} {item.label}" for item in items)
    content = f"What they just said:\n{said}\n\n"
    if earlier:
        content += f"Earlier in this conversation:\n{earlier}\n\n"
    content += f"From their past:\n{listing}"
    data = llm.chat_json(
        ep,
        [
            {"role": "system", "content": INSTRUCTIONS.format(note=MODE_NOTES.get(mode, ""))},
            {"role": "user", "content": content},
        ],
        SCHEMA,
    )
    by_key = {item.key: item for item in items}
    picked: list[Item] = []
    for key in data.get("pick", []):
        item = by_key.get(str(key).strip().strip("[]"))
        if item and item not in picked:
            picked.append(item)
    limit = 1 if mode in ("listen", "care") else MAX_PICKS
    return picked[:limit], str(data.get("thinking", ""))
