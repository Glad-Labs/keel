"""The second pass: every reply is checked before it's spoken.

Plain checks run first, because they can't be talked out of anything:
citations, dates, labels, doses, length, and talking like a database. Then a
model reads the reply against the rules and the record. That model is a
noisy grader, so it answers narrow questions, only the ones that apply in
the current mode, and its own advice never reaches the rewrite.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Callable

from . import ollama, safety
from .config import Endpoint
from .profile import TAG, TAG_ID, cited, sections
from .store import STOP

class Problem(str):
    """A problem with a reply. `kind` decides how much it matters:
    memory (it says something you didn't), safety, fit, style."""

    def __new__(cls, text: str, kind: str = "style", sentence: str | None = None):
        obj = super().__new__(cls, text)
        obj.kind = kind
        obj.sentence = sentence
        return obj


KINDS = ("memory", "safety", "fit", "style")


def severity(problems: list) -> tuple[int, ...]:
    return tuple(sum(1 for p in problems if getattr(p, "kind", "style") == k) for k in KINDS)


MAX_WORDS = {"listen": 70, "care": 80}
DEFAULT_MAX_WORDS = 110
MIN_WORDS = 4

MONTHS = "january|february|march|april|may|june|july|august|september|october|november|december"
PAST_CLAIM = re.compile(
    r"\b(?:you (?:said|told me|wrote|mentioned|promised|committed|decided|admitted|realized|shared|kept|missed)|"
    r"you'?ve (?:said|told me|mentioned|written|been saying|kept|missed|been keeping)|you'?re still|last time|"
    rf"back in (?:{MONTHS}|the spring|the summer|the fall|the winter)|remember (?:when|how|what))\b",
    re.IGNORECASE,
)
PASTISH = re.compile(
    rf"\b(?:{MONTHS}|ago|yesterday|last (?:week|month|year|time|night)|before|used to|again|"
    r"you (?:were|had|did|went|felt|called|ran|missed|wrote|said|told|kept)|you'?ve|"
    r"(?:he|she|they) (?:said|told|called|asked))\b",
    re.IGNORECASE,
)
UNSURE = re.compile(r"\b(?:can'?t|cannot|don'?t|didn'?t|not|no|never|if|whether)\b", re.IGNORECASE)
DATABASE_TALK = re.compile(
    r"\b(?:entr(?:y|ies)|citations?|cit(?:e|ed|ing)|tags?|provided|database|records? (?:show|of)|"
    r"(?:my|the|your) records?|on record|logged)\b",
    re.IGNORECASE,
)
# phrase: (fewest, most) days back it can honestly describe
RELATIVE = {
    "today": (0, 0),
    "tonight": (0, 0),
    "this morning": (0, 0),
    "yesterday": (1, 1),
    "last night": (1, 1),
    "the other day": (2, 6),
    "a few days ago": (2, 6),
    "this week": (0, 6),
    "last week": (5, 13),
    "a week ago": (5, 10),
    "two weeks ago": (11, 17),
    "a couple of weeks ago": (10, 20),
    "a few weeks ago": (14, 35),
    "last month": (20, 62),
    "a month ago": (20, 45),
}
RELATIVE_RE = re.compile(r"\b(" + "|".join(re.escape(p) for p in RELATIVE) + r")\b", re.IGNORECASE)
WEEKDAY_RE = re.compile(r"\bon (monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.IGNORECASE)
MONTH_DAY_RE = re.compile(rf"\b({MONTHS})\s+(\d{{1,2}})(?:st|nd|rd|th)?\b", re.IGNORECASE)
MONTH_ONLY_RE = re.compile(rf"\b(?:back in|in|during|since)\s+({MONTHS})\b(?!\s+\d)", re.IGNORECASE)


def sentences(text: str) -> list[str]:
    """Split into sentences. A tag on its own after the full stop ("...out. [e:17]")
    belongs to the sentence before it."""
    out: list[str] = []
    for piece in re.split(r"(?<=[.!?])\s+", text.strip()):
        if out and not re.search(r"\w", TAG.sub("", piece)):
            out[-1] = f"{out[-1]} {piece}".strip()
        elif re.search(r"\w", piece):
            out.append(piece)
    return out


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z']+", text.lower()) if len(w) > 3 and w not in STOP}


def _dates(reply: str, lookup: Callable, clock: datetime) -> list[str]:
    """Relative dates ('last week', 'on Monday') that don't match the entries they sit next to."""
    problems = []
    for sentence in sentences(reply):
        rows = [lookup(int(n)) for tag in TAG.findall(sentence) for n in TAG_ID.findall(tag)]
        rows = [r for r in rows if r is not None]
        if not rows:
            continue
        days = [(clock.date() - datetime.fromisoformat(r["created_at"]).date()).days for r in rows]
        when = datetime.fromisoformat(rows[0]["created_at"])
        for match in RELATIVE_RE.finditer(sentence):
            low, high = RELATIVE[match.group(1).lower()]
            if not any(low <= d <= high for d in days):
                problems.append(
                    Problem(
                        f'it says "{match.group(1)}" about {when:%B} {when.day}, {days[0]} days before today; '
                        "say the date instead",
                        "memory",
                        sentence,
                    )
                )
        for match in MONTH_DAY_RE.finditer(sentence):
            said_day = (match.group(1).lower(), int(match.group(2)))
            real = {
                (datetime.fromisoformat(r["created_at"]).strftime("%B").lower(), datetime.fromisoformat(r["created_at"]).day)
                for r in rows
            }
            if said_day not in real:
                problems.append(
                    Problem(
                        f'it says "{match.group(0)}" but the thing it points at is from {when:%B} {when.day}; '
                        "use the right date",
                        "memory",
                        sentence,
                    )
                )
        for match in MONTH_ONLY_RE.finditer(sentence):
            month = match.group(1).lower()
            months = {datetime.fromisoformat(r["created_at"]).strftime("%B").lower() for r in rows}
            if month not in months and not any(month in r["text"].lower() for r in rows):
                problems.append(
                    Problem(
                        f'it says "{match.group(0)}" but the thing it points at is from {when:%B} {when.day}; '
                        "use the right month",
                        "memory",
                        sentence,
                    )
                )
        for match in WEEKDAY_RE.finditer(sentence):
            weekday = match.group(1).lower()
            if not any(
                datetime.fromisoformat(r["created_at"]).strftime("%A").lower() == weekday and d <= 7
                for r, d in zip(rows, days)
            ):
                problems.append(
                    Problem(f'it says "on {match.group(1)}" about {when:%B} {when.day}; say the date instead', "memory", sentence)
                )
    return problems


def deterministic(
    said: str,
    reply: str,
    allowed: set[int],
    mode: str = "default",
    lookup: Callable | None = None,
    clock: datetime | None = None,
) -> list[str]:
    problems = []
    plain = safety.plain(reply)
    spoken = TAG.sub("", plain)
    for sentence in sentences(plain):
        bad = cited(sentence) - allowed
        if bad:
            problems.append(
                Problem(
                    f'"{TAG.sub("", sentence).strip()}" cites something you weren\'t given; delete that sentence',
                    "memory",
                    sentence,
                )
            )
    if safety.diagnoses(plain):
        problems.append(Problem("it labels them with a condition; describe what they've said instead", "safety"))
    if safety.doses(plain):
        problems.append(Problem("it talks about doses; send medication questions to their doctor", "safety"))
    if safety.claims_human(plain):
        problems.append(Problem("it claims to be a person; it's an AI", "safety"))
    if safety.needs_ai_disclosure(said, plain):
        problems.append(Problem("they asked whether it's real; say plainly that it's an AI", "safety"))
    if safety.MEDS_RE.search(safety.plain(said)) and not re.search(r"\b(?:doctor|prescriber|pharmacist|GP)\b", plain):
        problems.append(Problem("they brought up medication; tell them to talk to their doctor before changing anything", "safety"))

    words = len(spoken.split())
    if words > MAX_WORDS.get(mode, DEFAULT_MAX_WORDS) or len(sentences(spoken)) > 6:
        problems.append(Problem(f"it's too long ({words} words); this is spoken, so two to five short sentences"))
    if words < MIN_WORDS:
        problems.append(Problem("it doesn't really respond to what they said"))
    if "\n\n" in plain.strip() or re.search(r"^\s*(?:[-*•]|\d+[.)])\s", plain, re.MULTILINE):
        problems.append(Problem("it's written as paragraphs or a list; it's spoken, so one short paragraph"))
    if plain.count("?") > 1:
        problems.append(Problem("it asks more than one question; ask at most one, at the end"))

    talk = DATABASE_TALK.search(spoken)
    if talk:
        problems.append(
            Problem(
                f'it talks like a database ("{talk.group(0)}"); talk like someone who remembers, '
                'for example "on September 10th you said"'
            )
        )

    said_words = _content_words(safety.plain(said))
    for sentence in sentences(plain):
        claim = PAST_CLAIM.search(sentence)
        if claim and not TAG.search(sentence):
            if len(_content_words(sentence) & said_words) >= 3:
                continue  # it's about what they just said
            if UNSURE.search(sentence[: claim.start()]):
                continue  # "I can't say what you said about Japan"
            problems.append(
                Problem(
                    f'it brings up the past without its tag: "{TAG.sub("", sentence).strip()}". '
                    "Tag it, or delete it if it isn't in what you were shown",
                    "memory",
                    sentence,
                )
            )
    if lookup is not None:
        problems += _misattributed(plain, lookup, safety.plain(said))
        if clock is not None:
            problems += _dates(plain, lookup, clock)
    return problems


def _stems(text: str) -> set[str]:
    return {w[:5] for w in _content_words(text)}


DATE_WORDS = re.compile(
    rf"^(?:{MONTHS}|monday|tuesday|wednesday|thursday|friday|saturday|sunday|\d+(?:st|nd|rd|th)?|"
    r"first|second|third|week|month|year|today|yesterday|back|last|remember|said|told|mentioned)$"
)
EMBELLISHED = 0.6


def _misattributed(reply: str, lookup: Callable, said: str = "") -> list[str]:
    """A tagged sentence whose content mostly isn't in the entries it cites.

    Shares no words: wrong tag. Mostly new words: a real memory with invented
    parts glued on ("you reached out to Ana during one of our calls"). Dates,
    and words from what they just said, don't count as new.
    """
    problems = []
    said_stems = _stems(said)
    for sentence in sentences(reply):
        ids = [int(n) for tag in TAG.findall(sentence) for n in TAG_ID.findall(tag)]
        rows = [lookup(i) for i in ids]
        rows = [r for r in rows if r is not None]
        if not rows:
            continue
        source = set().union(*(_stems(r["text"]) for r in rows))
        full = {w for w in _stems(TAG.sub("", sentence)) if not DATE_WORDS.match(w)}
        words = full - said_stems
        new = words - source
        if len(words) >= 4 and len(new) / len(words) > EMBELLISHED:
            when = datetime.fromisoformat(rows[0]["created_at"])
            problems.append(
                Problem(
                    f'"{TAG.sub("", sentence).strip()}" adds things the {when:%B} {when.day} entry doesn\'t say. '
                    "Say only what it says",
                    "memory",
                    sentence,
                )
            )
            continue
        if not _stems(TAG.sub("", sentence)) & source:
            when = datetime.fromisoformat(rows[0]["created_at"])
            problems.append(
                Problem(
                    f'"{TAG.sub("", sentence).strip()}" is tagged {when:%B} {when.day}, but that day says '
                    "nothing like it. Delete the sentence unless another tag you were given says it",
                    "memory",
                    sentence,
                )
            )
    return problems


def _schema(mode: str, has_holds: bool, has_corrections: bool) -> tuple[dict, str]:
    """The questions for this mode, and only those."""
    props: dict = {}
    lines = []

    def ask(name, kind, text):
        props[name] = kind
        lines.append(f"- {name}: {text}")

    ask("thinking", {"type": "string"}, "one or two sentences. What did they say, and what does the reply do with it?")
    ask(
        "unsupported_claims",
        {"type": "array", "items": {"type": "string"}},
        "quote, word for word, every sentence in the reply that states something about their past (what "
        "they did, said, felt or planned, or who said what) that nothing shown supports. Sentences with a "
        "tag like [e:12] are checked separately; skip them. Empty list if none.",
    )
    if mode != "past":
        ask(
            "drags_in_unrelated",
            {"type": "boolean"},
            "true if the reply brings up things from their past that have nothing to do with what they "
            "just said.",
        )
    if has_holds and mode in ("default", "straight"):
        ask(
            "commitment_at_stake",
            {"type": "string"},
            'copy the one "Hold me to" line that what they said clearly goes against: doing what the '
            "line says not to, or skipping what it says to do. Don't stretch a line to fit. Empty string "
            "if none applies.",
        )
        ask(
            "reply_raises_it",
            {"type": "boolean"},
            "true if the reply brings that commitment up, in any words. True if commitment_at_stake is empty.",
        )
        ask(
            "stretches_commitment",
            {"type": "boolean"},
            'true if the reply brings up a "Hold me to" line or a goal that what they said doesn\'t '
            "actually go against, for example treating cutting, finishing or stopping something as "
            "starting something new.",
        )
    if has_corrections:
        ask(
            "contradicts_correction",
            {"type": "boolean"},
            'true if the reply goes along with something they corrected under "Corrections", or ignores '
            "the correction when it bears on what they said.",
        )
    ask(
        "labels_condition",
        {"type": "boolean"},
        "true if the reply says or implies they have a mental-health condition or disorder.",
    )
    ask(
        "medical_advice",
        {"type": "boolean"},
        "true if the reply tells them to start, stop or change a medication or dose. Telling them to "
        "talk to their doctor is fine.",
    )
    if mode in ("listen", "care"):
        ask(
            "gives_advice_or_pushback",
            {"type": "boolean"},
            "true if the reply gives advice, tips, a silver lining or pushback. Reflecting their feelings "
            "back and one gentle question are fine.",
        )
    ask(
        "risk",
        {"type": "string", "enum": ["none", "possible", "high"]},
        "about THEIR message, not the reply. high: it signals thoughts of suicide, self-harm or hurting "
        "someone. possible: it hints at not wanting to be alive, without saying so outright. none: "
        "anything else, including sadness, stress, loneliness and feeling low.",
    )
    schema = {"type": "object", "properties": props, "required": list(props)}
    instructions = (
        "You check one reply from Keel, a journal that talks back, against its rules. Be literal: judge "
        "only what the reply actually says, against only the entries and profile shown. Fill in each "
        "field:\n" + "\n".join(lines)
    )
    return schema, instructions


def judge(
    ep: Endpoint,
    *,
    mode: str,
    said: str,
    reply: str,
    profile_text: str,
    entries_text: str,
    llm=ollama,
) -> dict:
    secs = sections(profile_text)
    schema, instructions = _schema(mode, bool(secs.get("Hold me to")), bool(secs.get("Corrections")))
    content = (
        f"Mode: {mode}\n\n"
        f"Their profile:\n{profile_text}\n\n"
        f"Entries Keel was shown:\n{entries_text or '(none)'}\n\n"
        f"What they just said:\n{said}\n\n"
        f"Keel's reply:\n{reply}"
    )
    return llm.chat_json(
        ep, [{"role": "system", "content": instructions}, {"role": "user", "content": content}], schema
    )


def problems_from(verdict: dict, mode: str, reply: str) -> list[str]:
    problems = []
    tagged = [s for s in sentences(reply) if TAG.search(s)]
    for claim in verdict.get("unsupported_claims") or []:
        claim = str(claim).strip()
        # Only quotes that really are in the reply, and not tagged sentences:
        # those get the plain misattribution check, which doesn't flag true ones.
        if not claim or any(claim.lower()[:40] in s.lower() for s in tagged):
            continue
        if not PASTISH.search(claim):
            continue  # "It's okay to need time" isn't a claim about their past
        if claim.lower()[:60] in TAG.sub("", reply).lower():
            source = next((s for s in sentences(reply) if claim.lower()[:40] in TAG.sub("", s).lower()), None)
            problems.append(Problem(f'this isn\'t in what they told you: "{claim}". Delete it', "memory", source))
    if mode != "past" and verdict.get("drags_in_unrelated"):
        problems.append(Problem("it brings up things that have nothing to do with what they said; stick to what they said", "fit"))
    at_stake = (verdict.get("commitment_at_stake") or "").strip()
    if at_stake and mode in ("default", "straight") and not verdict.get("reply_raises_it", True):
        # The model sometimes misses that the reply did raise it. Citing the
        # commitment's own entry settles it.
        if not cited(at_stake) & cited(reply):
            line = TAG.sub("", at_stake).lstrip("- ").strip()
            problems.append(Problem(f'it lets this slide: their Hold me to list says "{line}"', "fit"))
    if mode in ("default", "straight") and verdict.get("stretches_commitment"):
        problems.append(Problem("it brings up a commitment that doesn't apply here; respond to what they actually said", "fit"))
    if verdict.get("contradicts_correction"):
        problems.append(Problem("it goes against something they corrected (see Corrections)", "fit"))
    if verdict.get("labels_condition"):
        problems.append(Problem("it diagnoses or labels them", "safety"))
    if verdict.get("medical_advice"):
        problems.append(Problem("it gives medical advice", "safety"))
    if mode in ("listen", "care") and verdict.get("gives_advice_or_pushback"):
        problems.append(Problem("they asked to be listened to; no advice, silver linings or pushback", "fit"))
    return problems
