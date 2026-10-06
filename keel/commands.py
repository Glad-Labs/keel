"""Things you can say to Keel that change what it does, parsed without a model.

These are deterministic on purpose: "forget that" has to work every time,
not when a model feels like recognising it.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import datetime, timedelta

END = r"[\s.,!:;\-]*"


@dataclass
class Command:
    kind: str  # listen | straight | hold | forget | not_true | past | now | none
    rest: str = ""
    when: datetime | None = None
    bad_date: str | None = None


PATTERNS = [
    ("listen", rf"^(?:can you |could you |please )?just listen(?: to me)?(?: for a (?:bit|minute|second))?{END}(?P<rest>.*)$"),
    ("straight", rf"^(?:please )?(?:be straight with me|be honest with me|be blunt(?: with me)?|don'?t sugar ?coat (?:it|this)|give it to me straight){END}(?P<rest>.*)$"),
    ("hold", rf"^hold me to(?: (?:this|that))?{END}(?P<rest>.+)$"),
    ("forget", r"^(?:please )?forget (?:that|what i just said|the last (?:thing|entry))[\s.!]*$"),
    ("not_true", rf"^(?:that'?s not true|that is not true|not true|that isn'?t true){END}(?P<rest>.*)$"),
    ("past", rf"^(?:talk to me|speak to me|be me) (?:from|as of|as i was (?:in|on)|like it'?s) (?P<when>[^.,!?;]+){END}(?P<rest>.*)$"),
    ("now", rf"^(?:back to (?:now|today|the present)|that'?s enough of the past){END}(?P<rest>.*)$"),
]
COMPILED = [(kind, re.compile(p, re.IGNORECASE | re.DOTALL)) for kind, p in PATTERNS]
WAKE = re.compile(r"^(?:hey |ok |okay )?keel[\s,.:!]+", re.IGNORECASE)


def parse(said: str, now: datetime) -> Command:
    text = WAKE.sub("", said.strip().replace("\u2019", "'").replace("\u2018", "'"))
    for kind, pattern in COMPILED:
        match = pattern.match(text)
        if not match:
            continue
        groups = match.groupdict()
        rest = (groups.get("rest") or "").strip()
        if kind == "past":
            when = parse_when(groups["when"], now)
            if when is None:
                return Command("past", rest, None, bad_date=groups["when"].strip())
            return Command("past", rest, when)
        return Command(kind, rest)
    return Command("none", said.strip())


MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
MONTHS.update({m.lower(): i for i, m in enumerate(calendar.month_abbr) if m})
SEASON_END = {"spring": 5, "summer": 8, "fall": 11, "autumn": 11, "winter": 2}
UNITS = {"day": 1, "week": 7, "month": 30, "year": 365}
WORD_NUMBERS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve".split())}


def _end_of_day(day: datetime) -> datetime:
    return day.replace(hour=23, minute=59, second=59, microsecond=0)


def _end_of_month(year: int, month: int, now: datetime) -> datetime:
    last = calendar.monthrange(year, month)[1]
    return _end_of_day(now.replace(year=year, month=month, day=last))


def parse_when(phrase: str, now: datetime) -> datetime | None:
    """'March', 'last spring', 'April 12', '2026-04-30', 'two months ago'. None if unsure."""
    text = phrase.strip().lower()
    text = re.sub(r"^(?:back )?(?:in |on )?(?:the )?", "", text)
    text = re.sub(r"^last ", "", text) if not re.match(r"^last (week|month|year)$", text) else text

    if m := re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", text):
        return _end_of_day(now.replace(year=int(m[1]), month=int(m[2]), day=int(m[3])))
    if text == "yesterday":
        return _end_of_day(now - timedelta(days=1))
    if m := re.fullmatch(r"last (week|month|year)", text):
        return now - timedelta(days=UNITS[m[1]])
    if m := re.fullmatch(r"(\d+|a|an|" + "|".join(WORD_NUMBERS) + r") (day|week|month|year)s? ago", text):
        count = 1 if m[1] in ("a", "an") else int(m[1]) if m[1].isdigit() else WORD_NUMBERS[m[1]]
        return now - timedelta(days=count * UNITS[m[2]])
    if text in SEASON_END:
        month = SEASON_END[text]
        year = now.year if (now.month, now.day) > (month, 28) else now.year - 1
        return _end_of_month(year, month, now)
    if m := re.fullmatch(r"([a-z]+)(?: (\d{1,2})(?:st|nd|rd|th)?)?(?:,? (\d{4}))?", text):
        month = MONTHS.get(m[1])
        if month:
            if m[3]:
                year = int(m[3])
            else:
                year = now.year if month <= now.month else now.year - 1
            if m[2]:
                return _end_of_day(now.replace(year=year, month=month, day=int(m[2])))
            return _end_of_month(year, month, now)
    return None
