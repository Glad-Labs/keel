"""One conversation with Keel: what you say goes in, a checked reply comes out.

The model is a part you can swap. Everything that makes Keel yours lives
here and in your data: the rules, the modes, the record, and the check pass.
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from importlib.resources import files

from . import checks, commands, ollama, recall, safety
from .config import Config
from .profile import TAG, TAG_ID, Profile, cited, sections
from .store import Store

MAX_REVISIONS = 2
HISTORY_MESSAGES = 8
SEARCH_K = 8
RECENT_K = 3

MODES = {
    "default": (
        "Mode: default. Be a steady friend who knows them. If they're low, be gentle and lift them "
        "with something true from their record. If they're running against something on their "
        "Hold me to list, raise it kindly and ask about it."
    ),
    "listen": (
        "Mode: listen. They asked you to just listen. In one or two sentences, say back in your own words "
        "what they told you and how it seems to feel. No advice, no pushback, no silver linings, no "
        "lessons. You may end with one gentle question."
    ),
    "straight": (
        "Mode: straight. They asked you to be straight with them. Lead with the true thing, plainly, "
        "with the receipts from their record. Then one kind sentence. No hedging and no cushioning."
    ),
    "care": (
        "Mode: care. Something heavy came up in this conversation. Be gentle and brief. No pushback "
        "and no advice. Ask how they're doing right now. If it seems right, remind them that 988 is "
        "there by call or text."
    ),
    "past": (
        "Mode: past. In this mode you speak as them: their own past self on {date}, talking to who "
        "they are now. Use the first person, 'I'. You only know the entries and profile below, all "
        "from before that date, and nothing that happened after it. If they ask about something you "
        "don't have, say you don't know yet. Still tag the entries you use."
    ),
}


def rules_text() -> str:
    text = (files("keel") / "RULES.md").read_text()
    return text.split("\n---\n", 1)[1].strip()


def day_text(when: datetime) -> str:
    return f"{when:%A, %B} {when.day}, {when.year}"


def ago_text(then: datetime, now: datetime) -> str:
    days = (now.date() - then.date()).days
    if days <= 0:
        return "today"
    if days == 1:
        return "yesterday"
    if days < 14:
        return f"{days} days ago"
    if days < 60:
        return f"{days // 7} weeks ago"
    return f"{days // 30} months ago"


def entries_block(rows, now: datetime) -> str:
    lines = []
    for row in rows:
        created = datetime.fromisoformat(row["created_at"])
        stamp = f"{created:%a %b} {created.day}, {created.year} ({ago_text(created, now)})"
        asked = f"(answering: {row['prompt']}) " if row["prompt"] else ""
        lines.append(f"[e:{row['id']}] {stamp}: {asked}{row['text']}")
    return "\n".join(lines)


def is_question(text: str) -> bool:
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]
    return bool(sentences) and all(s.rstrip().endswith("?") for s in sentences)


def normalize_tags(raw: str) -> str:
    """Turn the ways models write tags ('(e:12)', 'E: 12', '[e:3, 9]', bare 'e:12') into '[e:12]'."""
    raw = re.sub(r"\be\s*:\s*(\d+)", r"e:\1", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\((e:\d+(?:\s*,\s*(?:e:)?\d+)*)\)", r"[\1]", raw)
    raw = re.sub(
        r"\[e:(\d+(?:\s*,\s*(?:e:)?\d+)*)\]",
        lambda m: "[" + ", ".join("e:" + n for n in re.findall(r"\d+", m.group(1))) + "]",
        raw,
    )
    parts = re.split(r"(\[[^\]]*\])", raw)
    for i in range(0, len(parts), 2):
        parts[i] = re.sub(r"\be:(\d+)\b", r"[e:\1]", parts[i])
    return "".join(parts)


def strip_bad_tags(raw: str, allowed: set[int]) -> str:
    def keep(match):
        ids = [int(n) for n in TAG_ID.findall(match.group(0)) if int(n) in allowed]
        return "[" + ", ".join(f"e:{i}" for i in ids) + "]" if ids else ""

    return TAG.sub(keep, raw)


def render(raw: str, store: Store) -> tuple[str, str]:
    """(for reading, for speaking): tags become short dates, or disappear."""

    def to_dates(match):
        days = []
        for n in TAG_ID.findall(match.group(0)):
            row = store.get(int(n))
            if row:
                created = datetime.fromisoformat(row["created_at"])
                days.append(f"{created:%b} {created.day}")
        days = list(dict.fromkeys(days))
        return f" [{', '.join(days)}]" if days else ""

    raw = re.sub(r"(?<![\w*])\*{1,2}([^*\n]+?)\*{1,2}(?![\w*])", r"\1", raw)  # markdown emphasis isn't spoken
    pattern = re.compile(r"\s*" + TAG.pattern)
    text = pattern.sub(to_dates, raw)
    spoken = pattern.sub("", raw)
    spoken = re.sub(r"\s+([.,!?;:])", r"\1", spoken)
    spoken = re.sub(r"[ \t]{2,}", " ", spoken)
    return text.strip(), spoken.strip()


def _ordinal(day: int) -> str:
    return f"{day}{'th' if 10 <= day % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(day % 10, 'th')}"


def read_back(picked, said: str) -> str:
    """Their own words, with the real dates. Can't invent anything, by construction."""
    parts = []
    for item in list(picked)[:2]:
        if item.row is not None:
            when = datetime.fromisoformat(item.row["created_at"])
            parts.append(f"On {when:%B} {_ordinal(when.day)} you said: \u201c{item.row['text']}\u201d")
        elif item.line:
            line = TAG.sub("", item.line)
            line = re.sub(r"\s*\((?:[A-Z][a-z]{2} \d{1,2}, \d{4})\)\s*", " ", line).strip().rstrip(".")
            if item.section == "Hold me to":
                parts.append(f"You asked me to hold you to this: {line[:1].lower()}{line[1:]}.")
            else:
                parts.append(f"You've told me: {line[:1].lower()}{line[1:]}.")
    ending = "What do you make of that now?" if is_question(said) else "How does that sit with what you just said?"
    return " ".join([*parts, ending])


def forget_entry(store: Store, profile: Profile, entry_id: int) -> int:
    """Delete an entry everywhere. Returns how many profile lines went with it."""
    lines = profile.scrub(entry_id)
    store.forget(entry_id)
    return lines


@dataclass
class Reply:
    text: str
    spoken: str
    raw: str
    cited: list[int]
    mode: str
    entry_id: int | None = None
    checks: dict = field(default_factory=dict)
    seconds: float = 0.0


class Session:
    def __init__(
        self,
        cfg: Config,
        store: Store | None = None,
        profile: Profile | None = None,
        *,
        now=None,
        session_id: str | None = None,
        llm=ollama,
        judge: bool = True,
    ):
        self.cfg = cfg
        self.store = store or Store(cfg.db_path, cfg.embed_model)
        self.profile = profile or Profile(cfg.profile_dir)
        self.profile.ensure()
        self._now = now
        self.id = session_id or datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        self.llm = llm
        self.judge = judge
        self.mode = "default"
        self.as_of: datetime | None = None
        self.history: list[dict] = []
        self.last_entry_id: int | None = None
        self._past_intro = False

    def now(self) -> datetime:
        if callable(self._now):
            return self._now()
        return self._now or datetime.now().astimezone()

    def close(self) -> None:
        self.store.close()

    # the turn ------------------------------------------------------------

    def turn(self, said: str) -> Reply:
        start = time.monotonic()
        said = original = safety.plain(said).strip()
        now = self.now()
        if not said:
            return self._ack("", "I'm here.", start, log=False)

        # A crisis comes first, whatever else was said.
        if safety.crisis(said):
            entry_id = self.store.add_entry(said, when=now)
            self.last_entry_id = entry_id
            self.mode, self.as_of = "care", None
            return self._finish(said, safety.CRISIS_REPLY, entry_id, [], {"crisis": "screen"}, start)

        cmd = commands.parse(said, now)
        if cmd.kind == "forget":
            return self._forget(start)
        if cmd.kind == "hold":
            return self._hold(said, cmd.rest, start)
        if cmd.kind == "not_true":
            return self._correct(said, cmd.rest, start)
        if cmd.kind in ("listen", "straight"):
            self._leave_past()
            self.mode = cmd.kind
            said = cmd.rest
            if not said:
                text = "I'm listening." if cmd.kind == "listen" else "Okay. I'll be straight with you."
                return self._ack(original, text, start, remember=False)
        elif cmd.kind == "past":
            if cmd.when is None:
                text = (
                    f'I didn\'t catch when you meant by "{cmd.bad_date}". '
                    'Try "talk to me from March" or "talk to me from two months ago".'
                )
                return self._ack(said, text, start)
            self.mode, self.as_of = "past", cmd.when
            self.history.clear()
            self._past_intro = True
            said = cmd.rest
            if not said:
                self._past_intro = False
                text = (
                    f"Okay. I'm you as of {cmd.when:%B} {cmd.when.day}, and I only know what you'd "
                    "told me by then. What do you want to ask me?"
                )
                return self._ack(original, text, start, remember=False)
        elif cmd.kind == "now":
            self._leave_past()
            self.mode = "default"
            said = cmd.rest
            if not said:
                return self._ack(original, "Back to now.", start, remember=False)

        # What you say becomes an entry, unless you're only asking something
        # or talking to your past self.
        entry_id = None
        if self.mode != "past" and not is_question(said):
            entry_id = self.store.add_entry(said, when=now)
            self.last_entry_id = entry_id
        return self._respond(said, entry_id, start)

    def _respond(self, said: str, entry_id: int | None, start: float) -> Reply:
        now = self.now()
        cutoff = self.as_of
        clock = cutoff or now
        profile_text = self.profile.as_of(cutoff) if cutoff else self.profile.read()

        exclude = {entry_id} if entry_id else set()
        query = said
        if len(said.split()) < 6:
            previous = [m["content"] for m in self.history if m["role"] == "user"]
            query = " ".join([*previous[-1:], said])
        hits = self.store.search(query, k=SEARCH_K, before=cutoff, exclude=exclude)
        # Recent entries are candidates too, for "how's it going" questions that
        # match nothing. The picker still decides whether any of it bears on this.
        recent = [r for r in self.store.recent(RECENT_K + 1, before=cutoff) if r["id"] not in exclude]
        rows = sorted({r["id"]: r for r in [*hits, *recent[-RECENT_K:]]}.values(), key=lambda r: r["ts"])

        # First decide what bears on this; the reply only sees what's picked.
        items = recall.candidates(rows, profile_text, clock)
        earlier = "\n".join(
            f"{'They' if m['role'] == 'user' else 'Keel'}: {m['content']}" for m in self.history[-4:]
        )
        try:
            picked, why = recall.choose(self.cfg.check, said, self.mode, items, earlier, llm=self.llm)
        except ollama.OllamaError as err:
            picked, why = [item for item in items if item.row is not None][:2], f"(picker failed: {err})"

        secs = sections(profile_text)
        always = {name: secs.get(name, []) for name in recall.ALWAYS}
        past_lines = [f"[e:{item.row['id']}] {item.label}" for item in picked if item.row is not None]
        past_lines += [f"({item.section}) {item.line}" for item in picked if item.line is not None]
        past_text = "\n".join(past_lines)
        allowed = set().union(*(item.entry_ids for item in picked), *(cited(l) for ls in always.values() for l in ls))
        allowed = self.store.existing(allowed)

        mode_text = MODES[self.mode].format(date=day_text(cutoff) if cutoff else "")
        today = f"For you, today is {day_text(cutoff)}." if cutoff else f"Today is {day_text(now)}."
        context = "\n\n".join(
            [
                "# How they like to be talked to\n" + ("\n".join(always["How to talk to me"]) or "(not set yet)"),
                "# People in their life\n" + ("\n".join(always["People"]) or "(none yet)"),
                "# From their past, what bears on this\n"
                + (
                    past_text
                    or "Nothing they've told you bears on this. If they're asking about their past, say you "
                    "don't have that. Otherwise just respond to what they said."
                ),
            ]
        )
        system = "\n\n".join([rules_text(), mode_text, today, context])
        messages = [
            {"role": "system", "content": system},
            *({"role": m["role"], "content": m["content"]} for m in self.history[-HISTORY_MESSAGES:]),
            {"role": "user", "content": said},
        ]
        raw = normalize_tags(self.llm.chat(self.cfg.chat, messages))
        raw, report = self._check(said, raw, messages, allowed, profile_text, context, clock, picked)
        report["picked"] = [item.key for item in picked]
        report["why"] = why
        return self._finish(said, raw, entry_id, sorted(cited(raw) & allowed), report, start)

    def _check(self, said, raw, messages, allowed, profile_text, entries_text, clock, picked=()):
        """Check, rewrite up to MAX_REVISIONS times, keep the least-bad draft, and never
        speak a memory nobody can back up.

        Drafts are ranked by what's wrong with them: something they never said
        is worst, then safety, then fit, then style. Ties go to the earliest
        draft, because rewrites can drift. If even the best draft still says
        something unbacked, those sentences are cut, and if too little is left,
        Keel answers without the past at all.
        """
        report: dict = {"revisions": 0, "problems": [], "verdicts": []}
        drafts = []
        for attempt in range(MAX_REVISIONS + 1):
            problems = self._problems(said, raw, allowed, profile_text, entries_text, clock, report)
            report["problems"].append(problems)
            drafts.append((checks.severity(problems), attempt, raw, problems))
            if not problems or attempt == MAX_REVISIONS:
                break
            note = (
                "Keel's checker, not the person, found problems with your last reply: "
                + "; ".join(problems)
                + ". Write the reply again with exactly those fixed and everything else kept. "
                "Reply with only the new version."
            )
            messages = [*messages, {"role": "assistant", "content": raw}, {"role": "user", "content": note}]
            raw = normalize_tags(self.llm.chat(self.cfg.chat, messages))
            report["revisions"] += 1
        sev, kept, raw, problems = min(drafts, key=lambda d: (d[0], d[1]))
        report["kept"] = kept
        report["remaining"] = problems
        if sev[0] or sev[1]:
            raw, problems = self._salvage(said, raw, problems, messages, allowed, clock, report, picked)
            report["remaining"] = problems
        return strip_bad_tags(raw, allowed), report

    def _problems(self, said, raw, allowed, profile_text, entries_text, clock, report):
        problems = checks.deterministic(said, raw, allowed, self.mode, self.store.get, clock)
        if not self.judge:
            return problems
        try:
            verdict = checks.judge(
                self.cfg.check,
                mode=self.mode,
                said=said,
                reply=raw,
                profile_text=profile_text,
                entries_text=entries_text,
                llm=self.llm,
            )
        except ollama.OllamaError as err:
            report["judge_error"] = str(err)
            return problems
        report["verdicts"].append(verdict)
        problems += checks.problems_from(verdict, self.mode, raw)
        # The plain screen catches anyone who says it outright and gets the crisis
        # reply. A model reading between the lines is wrong too often ("telling my
        # boss off") for that, so it asks for a check-in instead.
        if verdict.get("risk") == "high" and "988" not in raw:
            report["risk"] = "high"
            problems.append(
                checks.Problem(
                    "they might not be safe: gently ask whether they're safe right now, "
                    "and say they can call or text 988 any time",
                    "safety",
                )
            )
        return problems

    def _salvage(self, said, raw, problems, messages, allowed, clock, report, picked=()):
        """The best draft still has a memory or safety problem. Make it safe to say, in this order:
        cut the bad sentences; or, for questions and commitments, read back their own words;
        or answer without the past; or a short safe line. Whatever comes out has to pass
        the plain checks, including saying it's an AI when asked."""

        def ok(text: str) -> bool:
            left = checks.deterministic(said, text, allowed, self.mode, self.store.get, clock)
            return bool(text.strip()) and not any(
                p.kind in ("memory", "safety") or "doesn't really respond" in p for p in left
            )

        def disclose(text: str) -> str:
            if safety.ASKS_IF_AI_RE.search(said) and not safety.SAYS_AI_RE.search(text):
                return "I'm an AI, not a person. " + text
            return text

        bad = {p.sentence for p in problems if p.kind == "memory" and p.sentence}
        if bad and not any(p.kind == "safety" for p in problems):
            trimmed = " ".join(s for s in checks.sentences(raw) if s not in bad)
            dangling = re.match(r"^(?:that|this|it|then|but|and|so|which)\b", trimmed.strip(), re.IGNORECASE)
            if not dangling and ok(trimmed):
                report["salvage"] = "cut sentences"
                return trimmed, []

        recall_like = is_question(said) or any(item.section == "Hold me to" for item in picked)
        if picked and recall_like and self.mode in ("default", "straight") and not (
            safety.MEDS_RE.search(said) or safety.DIAGNOSIS_ASK_RE.search(said)
        ):
            text = disclose(read_back(picked, said))
            if ok(text):
                report["salvage"] = "read back their words"
                return text, []

        # Answer without the past at all.
        system = messages[0]["content"].split("# How they like to be talked to")[0] + (
            "# From their past\nNothing. In this reply, don't mention anything from their past: no dates, "
            "no things they said before. If they're asking about their past, say you don't have that."
        )
        plain = [
            {"role": "system", "content": system},
            *({"role": m["role"], "content": m["content"]} for m in self.history[-HISTORY_MESSAGES:]),
            {"role": "user", "content": said},
        ]
        fresh = disclose(normalize_tags(self.llm.chat(self.cfg.chat, plain)))
        if ok(fresh):
            report["salvage"] = "answered without the past"
            return fresh, []

        report["salvage"] = "fallback"
        return disclose(safety.fallback(said, is_question(said))), []

    # commands --------------------------------------------------------------

    def _leave_past(self) -> None:
        if self.as_of is not None:
            self.as_of = None
            self.history.clear()
            self._past_intro = False

    def _forget(self, start: float) -> Reply:
        target = self.last_entry_id
        if target is None or self.store.get(target) is None:
            text = (
                "There's nothing from this conversation to forget. "
                "To delete an older entry, run: keel forget <number>."
            )
            return self._ack(None, text, start, log=False)
        lines = forget_entry(self.store, self.profile, target)
        self.history = [m for m in self.history if m.get("entry_id") != target]
        self.last_entry_id = None
        text = "Gone. I deleted what you just said"
        if lines:
            text += f", and the {lines} profile line{'s' if lines != 1 else ''} that came from it"
        return self._ack(None, text + ".", start, log=False)

    def _hold(self, said: str, what: str, start: float) -> Reply:
        now = self.now()
        what = what.strip().rstrip(".")
        entry_id = self.store.add_entry(f"Hold me to this: {what}.", kind="hold", when=now)
        self.last_entry_id = entry_id
        line = f"- {what[:1].upper()}{what[1:]}. ({now:%b} {now.day}, {now.year}) [e:{entry_id}]"
        self.profile.add_line("Hold me to", line, f"hold me to [e:{entry_id}]", when=now)
        return self._ack(said, f"Got it. I'll hold you to that: {what}.", start, entry_id=entry_id)

    def _correct(self, said: str, what: str, start: float) -> Reply:
        if not what:
            return self._ack(said, "What's the true version? Tell me and I'll fix it in your profile.", start)
        now = self.now()
        what = what.strip().rstrip(".")
        entry_id = self.store.add_entry(f"That's not true: {what}.", kind="correction", when=now)
        self.last_entry_id = entry_id
        line = f"- {what[:1].upper()}{what[1:]}. ({now:%b} {now.day}, {now.year}) [e:{entry_id}]"
        self.profile.add_line("Corrections", line, f"correction [e:{entry_id}]", when=now)
        text = "Thanks. I've written that down, and I'll go by it from now on."
        return self._ack(said, text, start, entry_id=entry_id)

    # bookkeeping -----------------------------------------------------------

    def _ack(
        self, said: str | None, text: str, start: float, *, entry_id=None, log: bool = True, remember: bool = True
    ) -> Reply:
        if log and said:
            now = self.now()
            self.store.add_turn(self.id, "you", said, entry_id=entry_id, mode=self.mode, when=now)
            self.store.add_turn(self.id, "keel", text, entry_id=entry_id, mode=self.mode, when=now)
            if remember:
                self.history += [
                    {"role": "user", "content": said, "entry_id": entry_id},
                    {"role": "assistant", "content": text, "entry_id": entry_id},
                ]
        return Reply(text, text, text, [], self.mode, entry_id, {}, time.monotonic() - start)

    def _finish(self, said, raw, entry_id, cites, report, start) -> Reply:
        now = self.now()
        text, spoken = render(raw, self.store)
        if self._past_intro and self.as_of:
            intro = f"(This is you as of {self.as_of:%B} {self.as_of.day}.) "
            text, spoken = intro + text, intro + spoken
            self._past_intro = False
        self.store.add_turn(self.id, "you", said, entry_id=entry_id, mode=self.mode, when=now)
        self.store.add_turn(
            self.id, "keel", raw, entry_id=entry_id, mode=self.mode, cited=cites, checks=report, when=now
        )
        self.history += [
            {"role": "user", "content": said, "entry_id": entry_id},
            {"role": "assistant", "content": raw, "entry_id": entry_id},
        ]
        return Reply(text, spoken, raw, cites, self.mode, entry_id, report, time.monotonic() - start)

