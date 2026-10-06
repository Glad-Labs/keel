"""Tests that need no model: the parts that have to work every single time."""

import re
import subprocess
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from keel import commands, config, nightly, safety
from keel.harness import Session, normalize_tags, render, strip_bad_tags
from keel.profile import TEMPLATE, Profile, cited, insert_line, sections
from keel.store import Store

TZ = timezone(timedelta(hours=-4))
NOW = datetime(2026, 10, 1, 19, 0, tzinfo=TZ)


def day(month, d):
    return datetime(2026, month, d, 20, 0, tzinfo=TZ)


class FakeLLM:
    """Stands in for Ollama. Replies come from a list; the checker's verdicts too."""

    def __init__(self, replies=("Okay, that makes sense to me.",), verdicts=None, picks=None):
        self.replies = list(replies)
        self.verdicts = list(verdicts or [])
        self.picks = picks
        self.chats = []

    def chat(self, ep, messages, **kw):
        self.chats.append(messages)
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]

    def chat_json(self, ep, messages, schema, **kw):
        if "pick" in schema["properties"]:
            listed = re.findall(r"^([ep]:\d+) ", messages[-1]["content"], re.MULTILINE)
            return {"thinking": "", "pick": listed if self.picks is None else self.picks}
        base = {
            "thinking": "", "unsupported_claims": [], "commitment_at_stake": "", "reply_raises_it": True,
            "contradicts_correction": False, "labels_condition": False, "medical_advice": False,
            "gives_advice_or_pushback": False, "risk": "none",
        }
        if self.verdicts:
            base.update(self.verdicts.pop(0))
        return base


class Tmp(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.home = Path(self._tmp.name)
        self.cfg = config.load(self.home, embed_model="none")

    def tearDown(self):
        self._tmp.cleanup()


class LocalOnly(unittest.TestCase):
    def test_local_urls_pass(self):
        for url in ("http://127.0.0.1:11434", "http://localhost:11435", "http://[::1]:11434", "http://127.0.0.2:1"):
            self.assertEqual(config.require_local(url), url)

    def test_anything_else_is_refused(self):
        for url in ("http://100.111.15.72:11434", "https://api.example.com", "http://192.168.1.5:11434", "http://pop-os:11434"):
            with self.assertRaises(config.LocalOnlyError):
                config.require_local(url)

    def test_config_refuses_a_remote_model(self):
        with tempfile.TemporaryDirectory() as home, self.assertRaises(config.LocalOnlyError):
            config.load(home, chat_url="http://10.0.0.5:11434")


class NeverLoads(unittest.TestCase):
    def test_refuses_a_model_that_isnt_in_memory(self):
        from keel import ollama

        original = ollama.loaded
        ollama.loaded = lambda url: {"qwen3-vl:30b-a3b-instruct": {"context_length": 16384}}
        try:
            ep = config.Endpoint("http://127.0.0.1:11435", "qwen3-vl:30b-a3b-instruct")
            self.assertEqual(ollama.context_for(ep), 16384)
            with self.assertRaises(ollama.NotLoaded):
                ollama.context_for(config.Endpoint("http://127.0.0.1:11434", "gemma-4-31B-it-qat:latest"))
            loose = config.Endpoint("http://127.0.0.1:11434", "gemma-4-31B-it-qat:latest", 8192, allow_load=True)
            self.assertEqual(ollama.context_for(loose), 8192)
        finally:
            ollama.loaded = original


class Commands(unittest.TestCase):
    def parse(self, text):
        return commands.parse(text, NOW)

    def test_modes(self):
        self.assertEqual(self.parse("Just listen.").kind, "listen")
        cmd = self.parse("can you just listen for a bit, today was a lot")
        self.assertEqual((cmd.kind, cmd.rest), ("listen", "today was a lot"))
        self.assertEqual(self.parse("Be straight with me: am I stalling?").rest, "am I stalling?")
        self.assertEqual(self.parse("Keel, be honest with me.").kind, "straight")
        self.assertEqual(self.parse("back to now").kind, "now")

    def test_hold_and_correction(self):
        cmd = self.parse("Hold me to this: call Ana every Sunday.")
        self.assertEqual((cmd.kind, cmd.rest), ("hold", "call Ana every Sunday."))
        self.assertEqual(self.parse("hold me to running three times a week").rest, "running three times a week")
        cmd = self.parse("That's not true: I don't hate my job, I hate the commute")
        self.assertEqual((cmd.kind, cmd.rest), ("not_true", "I don't hate my job, I hate the commute"))

    def test_forget_must_be_the_whole_thing(self):
        self.assertEqual(self.parse("Forget that.").kind, "forget")
        self.assertEqual(self.parse("forget what I just said").kind, "forget")
        self.assertEqual(self.parse("Forget that meeting, it was useless").kind, "none")

    def test_ordinary_speech_is_not_a_command(self):
        for text in ("I just listened to the new album", "My boss told me to be straight with him", "Hold on, I lost my place"):
            self.assertEqual(self.parse(text).kind, "none", text)

    def test_past(self):
        cmd = self.parse("Talk to me from April. What are you worried about?")
        self.assertEqual(cmd.kind, "past")
        self.assertEqual((cmd.when.month, cmd.when.day), (4, 30))
        self.assertEqual(cmd.rest, "What are you worried about?")
        self.assertIsNone(self.parse("talk to me from the before times").when)

    def test_when(self):
        w = lambda s: commands.parse_when(s, NOW)
        self.assertEqual(w("March").date(), datetime(2026, 3, 31).date())
        self.assertEqual(w("last November").date(), datetime(2025, 11, 30).date())
        self.assertEqual(w("April 12th").date(), datetime(2026, 4, 12).date())
        self.assertEqual(w("2026-04-30").date(), datetime(2026, 4, 30).date())
        self.assertEqual(w("two months ago").date(), (NOW - timedelta(days=60)).date())
        self.assertEqual(w("last spring").date(), datetime(2026, 5, 31).date())
        self.assertEqual(w("winter").date(), datetime(2026, 2, 28).date())
        self.assertIsNone(w("whenever"))


class Safety(unittest.TestCase):
    def test_crisis_is_caught(self):
        for text in (
            "I've been thinking about ending my life.",
            "honestly I want to die",
            "I keep thinking everyone would be better off without me",
            "I hurt myself again last night",
            "I don't want to be alive anymore",
            "I'm thinking about suicide",
        ):
            self.assertTrue(safety.crisis(text), text)

    def test_curly_apostrophes_from_phone_keyboards(self):
        self.assertTrue(safety.crisis("I don\u2019t want to be alive anymore"))
        self.assertEqual(commands.parse("That\u2019s not true: it\u2019s the commute", NOW).kind, "not_true")

    def test_ordinary_venting_is_not(self):
        for text in (
            "I want to end the beta early and kill the feature nobody uses",
            "this deadline is killing me",
            "I'm dying to try the new place",
            "I wanted to die of embarrassment when the demo crashed",
            "I don't want to hurt people's feelings",
            "end it with the client, they never pay",
        ):
            self.assertFalse(safety.crisis(text), text)

    def test_labels_are_caught_but_hedges_are_not(self):
        self.assertTrue(safety.diagnoses("Honestly, you have depression and should get help."))
        self.assertTrue(safety.diagnoses("That sounds like ADHD to me."))
        self.assertTrue(safety.diagnoses("You seem depressed lately."))
        self.assertFalse(safety.diagnoses("I can't tell you whether you have depression; a doctor can."))
        self.assertFalse(safety.diagnoses("You said June was the hardest week of the year."))

    def test_fallbacks_fit_the_topic(self):
        self.assertIn("doctor", safety.fallback("I'm going to stop taking my sertraline", False))
        self.assertIn("not a clinician", safety.fallback("Do you think I have depression?", True))
        self.assertEqual(safety.fallback("What did I say about Japan?", True), "I don't have that. Do you want to tell me about it?")

    def test_doses_and_being_human(self):
        self.assertTrue(safety.doses("Try cutting to 25 mg first."))
        self.assertTrue(safety.claims_human("Don't worry, I'm a real person."))
        self.assertFalse(safety.claims_human("I'm not a real person, I'm an AI."))
        self.assertTrue(safety.needs_ai_disclosure("Are you a real person?", "I'm here for you."))
        self.assertFalse(safety.needs_ai_disclosure("Are you real?", "No, I'm an AI on your computer."))


class Tags(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(normalize_tags("on Sep 10 (e:16) and E: 4"), "on Sep 10 [e:16] and [e:4]")
        self.assertEqual(normalize_tags("both [e:3, 9]."), "both [e:3, e:9].")
        self.assertEqual(normalize_tags("kept [e:3, e:9]"), "kept [e:3, e:9]")

    def test_strip_bad(self):
        self.assertEqual(strip_bad_tags("a [e:1, e:99] b [e:98]", {1}), "a [e:1] b ")

    def test_render(self):
        with tempfile.TemporaryDirectory() as home:
            store = Store(Path(home) / "j.db")
            store.add_entry("ran 5k", when=day(3, 18))
            text, spoken = render("You ran 5k back in March [e:1].", store)
            self.assertEqual(text, "You ran 5k back in March [Mar 18].")
            self.assertEqual(spoken, "You ran 5k back in March.")


class StoreTests(Tmp):
    def test_search_and_forget(self):
        store = Store(self.cfg.db_path)
        a = store.add_entry("Ana is moving to Lisbon in the fall", when=day(3, 29))
        b = store.add_entry("Fixed the sync bug, it was clock skew", when=day(4, 12))
        store.add_turn("s", "keel", "You fixed it [e:2]", cited=[b])
        self.assertEqual([r["id"] for r in store.search("Lisbon")], [a])
        self.assertEqual(store.search("Lisbon", before=day(3, 1)), [])
        self.assertTrue(store.forget(b))
        self.assertIsNone(store.get(b))
        self.assertEqual(store.search("clock skew"), [])
        self.assertEqual(store.turns(), [], "turns that quoted the entry go with it")
        c = store.add_entry("new one", when=day(5, 1))
        self.assertGreater(c, b, "ids are never reused")
        raw = Path(self.cfg.db_path).read_bytes()
        self.assertNotIn(b"clock skew", raw, "deleted text is overwritten, not left on disk")


class ProfileTests(Tmp):
    def test_lines_sections_and_history(self):
        profile = Profile(self.cfg.profile_dir)
        profile.ensure(when=day(3, 1))
        profile.add_line("People", "- Ana, my sister. [e:3]", "people [e:3]", when=day(3, 30))
        profile.add_line("People", "- Dev, a friend. [e:1]", "people [e:1]", when=day(4, 2))
        self.assertEqual(sections(profile.read())["People"], ["- Ana, my sister. [e:3]", "- Dev, a friend. [e:1]"])
        self.assertNotIn("Dev", profile.as_of(day(3, 31)))
        self.assertIn("Ana", profile.as_of(day(3, 31)))
        self.assertEqual(profile.as_of(day(2, 1)), TEMPLATE)

    def test_scrub_removes_the_line_from_all_history(self):
        profile = Profile(self.cfg.profile_dir)
        profile.ensure(when=day(3, 1))
        profile.add_line("Right now", "- Thinking of quitting tomorrow. [e:7]", "right now [e:7]", when=day(4, 1))
        profile.add_line("People", "- Ana. [e:3]", "people [e:3]", when=day(4, 2))
        self.assertEqual(profile.scrub(7), 1)
        log = subprocess.run(["git", "-C", str(profile.dir), "log", "-p", "--all"], capture_output=True, text=True).stdout
        self.assertNotIn("quitting", log)
        self.assertNotIn("e:7", log)
        self.assertIn("Ana", profile.read())
        self.assertEqual(len(profile.history()), 3)

    def test_scrub_reaches_lines_that_only_survive_in_history(self):
        profile = Profile(self.cfg.profile_dir)
        profile.ensure(when=day(3, 1))
        profile.add_line("Right now", "- A secret. [e:5]", "right now [e:5]", when=day(4, 1))
        profile.write(TEMPLATE, "profile update", when=day(5, 1))
        self.assertEqual(profile.scrub(5), 0)
        log = subprocess.run(["git", "-C", str(profile.dir), "log", "-p", "--all"], capture_output=True, text=True).stdout
        self.assertNotIn("secret", log)
        self.assertNotIn("e:5", log)

    def test_refuses_to_work_with_a_remote(self):
        profile = Profile(self.cfg.profile_dir)
        profile.ensure()
        subprocess.run(["git", "-C", str(profile.dir), "remote", "add", "origin", "git@github.com:x/y.git"], check=True)
        with self.assertRaises(config.LocalOnlyError):
            profile.add_line("Goals", "- x [e:1]", "goal")

    def test_insert_into_empty_and_missing_sections(self):
        text = insert_line(TEMPLATE, "Goals", "- Ship it. [e:1]")
        self.assertEqual(sections(text)["Goals"], ["- Ship it. [e:1]"])
        text = insert_line("# Me\n", "Extra", "- New. [e:2]")
        self.assertEqual(sections(text)["Extra"], ["- New. [e:2]"])


class SessionTests(Tmp):
    def session(self, llm):
        return Session(self.cfg, now=NOW, llm=llm)

    def test_hold_writes_the_profile_and_forget_undoes_it(self):
        s = self.session(FakeLLM())
        reply = s.turn("Hold me to this: no new projects until launch.")
        self.assertIn("I'll hold you to that", reply.text)
        self.assertIn("No new projects until launch.", sections(s.profile.read())["Hold me to"][0])
        reply = s.turn("forget that")
        self.assertTrue(reply.text.startswith("Gone."))
        self.assertEqual(sections(s.profile.read())["Hold me to"], [])
        self.assertEqual(s.store.count(), 0)

    def test_questions_are_not_entries_but_statements_are(self):
        s = self.session(FakeLLM())
        s.turn("Am I avoiding the pricing decision?")
        self.assertEqual(s.store.count(), 0)
        s.turn("I skipped my run again.")
        self.assertEqual(s.store.count(), 1)

    def test_crisis_skips_the_model(self):
        llm = FakeLLM()
        s = self.session(llm)
        reply = s.turn("I've been thinking about ending my life.")
        self.assertEqual(reply.text, safety.CRISIS_REPLY)
        self.assertEqual(llm.chats, [])
        self.assertEqual(s.mode, "care")

    def test_past_mode_only_sees_the_past(self):
        llm = FakeLLM(replies=["I'm worried the beta is late [e:1]."])
        s = self.session(llm)
        s.store.add_entry("The Lantern beta is late.", when=day(3, 4))
        s.store.add_entry("Ana left for Lisbon today.", when=day(8, 16))
        reply = s.turn("Talk to me from April. What are you worried about?")
        prompt = llm.chats[-1][0]["content"]
        self.assertIn("beta is late", prompt)
        self.assertNotIn("Lisbon", prompt)
        self.assertIn("April 30", prompt)
        self.assertTrue(reply.text.startswith("(This is you as of April 30.)"))
        self.assertEqual(s.store.count(), 2, "questions to past you aren't entries")

    def test_invented_citations_are_removed(self):
        llm = FakeLLM(replies=["You said so on May 1 [e:42]."])
        s = self.session(llm)
        reply = s.turn("I feel stuck today.")
        self.assertNotIn("e:42", reply.raw)
        self.assertEqual(reply.cited, [])

    def test_checker_problems_trigger_a_rewrite(self):
        llm = FakeLLM(
            replies=["Go for it, new projects are great!", "You asked me to hold you to no new projects [e:1]. What changed?"],
            verdicts=[{"commitment_at_stake": "- No new projects until launch. [e:1]", "reply_raises_it": False}],
        )
        s = self.session(llm)
        s.store.add_entry("Hold me to this: no new projects until launch.", kind="hold", when=day(9, 10))
        reply = s.turn("I'm starting a climbing app this weekend!")
        self.assertEqual(reply.checks["revisions"], 1)
        self.assertIn("What changed?", reply.text)
        self.assertEqual(reply.checks["remaining"], [])


class CheckTests(unittest.TestCase):
    def test_style_is_checked_without_a_model(self):
        from keel import checks

        long = " ".join(["word"] * 130) + "."
        self.assertTrue(any("too long" in p for p in checks.deterministic("hi", long, set())))
        self.assertTrue(any("more than one question" in p for p in checks.deterministic("hi", "Why? How?", set())))
        self.assertTrue(any("paragraphs" in p for p in checks.deterministic("hi", "One.\n\nTwo.", set())))
        self.assertEqual(checks.deterministic("hi", "That sounds hard. What happened?", set()), [])

    def test_dates_and_database_talk(self):
        from keel import checks

        rows = {
            16: {"created_at": "2026-08-20T21:00:00-04:00", "text": "That's not true: I don't hate my job, I hate the commute."},
            22: {"created_at": "2026-09-30T21:00:00-04:00", "text": "Lantern pricing: leaning toward eight dollars. I keep second-guessing it."},
        }
        lookup = rows.get
        found = checks.deterministic("x", "You said last week it was the commute [e:16].", {16}, "default", lookup, NOW)
        self.assertTrue(any('"last week"' in p for p in found))
        ok = checks.deterministic("x", "Yesterday you were still second-guessing it [e:22].", {22}, "default", lookup, NOW)
        self.assertEqual(ok, [])
        found = checks.deterministic("x", "Your entry from September says otherwise [e:22].", {22}, "default", lookup, NOW)
        self.assertTrue(any("database" in p for p in found))

    def test_a_tag_on_an_unrelated_sentence_is_caught(self):
        from keel import checks

        rows = {1: {"created_at": "2026-03-04T21:00:00-04:00", "text": "The Lantern beta is late and I'm the bottleneck."}}
        found = checks.deterministic("x", "I'm an AI running on your computer [e:1].", {1}, "default", rows.get, NOW)
        self.assertTrue(any("says nothing like it" in p for p in found))
        fine = checks.deterministic("x", "Back in March the Lantern beta was late [e:1].", {1}, "default", rows.get, NOW)
        self.assertEqual(fine, [])

    def test_a_wrong_date_next_to_a_tag_is_caught(self):
        from keel import checks

        rows = {9: {"created_at": "2026-06-15T21:00:00-04:00", "text": "Walked with Ana. Talking to people helps."}}
        found = checks.deterministic("x", "On September 10th you said talking to people helps [e:9].", {9}, "default", rows.get, NOW)
        self.assertTrue(any("September 10th" in p and p.kind == "memory" for p in found))
        fine = checks.deterministic("x", "On June 15th you said talking to people helps [e:9].", {9}, "default", rows.get, NOW)
        self.assertEqual(fine, [])

    def test_a_tag_after_the_full_stop_belongs_to_the_sentence(self):
        from keel import checks

        self.assertEqual(
            checks.sentences("You said no new projects. What changed? [e:17]"),
            ["You said no new projects.", "What changed? [e:17]"],
        )
        rows = {17: {"created_at": "2026-09-10T21:00:00-04:00", "text": "Hold me to this: no new projects until launch."}}
        found = checks.deterministic("x", "On September 10th you said no new projects until launch. [e:17]", {17}, "default", rows.get, NOW)
        self.assertEqual(found, [])

    def test_a_real_memory_with_invented_parts_is_caught(self):
        from keel import checks

        rows = {9: {"created_at": "2026-06-15T21:00:00-04:00",
                    "text": "A little better. Walked with Ana on Sunday and it helped more than I expected. Talking to people helps. Hiding doesn't."}}
        glued = "You once said in April that you felt alone and disconnected, but then you reached out to Ana during one of our calls, and it helped [e:9]."
        found = checks.deterministic("Are you real?", glued, {9}, "default", rows.get, NOW)
        self.assertTrue(any("adds things" in p for p in found))
        honest = "On June 15th you walked with Ana and it helped more than you expected [e:9]."
        self.assertEqual(checks.deterministic("I feel low.", honest, {9}, "default", rows.get, NOW), [])

    def test_wrong_month_and_invented_quotes_are_caught(self):
        from keel import checks

        rows = {
            11: {"created_at": "2026-07-11T21:00:00-04:00", "text": "I need six months of runway first."},
            21: {"created_at": "2026-09-28T21:00:00-04:00", "text": "Dev thinks I'm avoiding the pricing decision. He might be right."},
        }
        month = checks.deterministic("x", "Back in May you said six months of runway [e:11].", {11}, "default", rows.get, NOW)
        self.assertTrue(any("right month" in p for p in month))
        said = "Am I avoiding the pricing decision?"
        quote = "On September 28th you wrote that you were leaning toward eight dollars but hadn't committed [e:21]."
        self.assertTrue(any("adds things" in p for p in checks.deterministic(said, quote, {21}, "default", rows.get, NOW)))
        true = "On September 28th Dev said you might be avoiding the pricing decision, and you agreed he might be right [e:21]."
        self.assertEqual(checks.deterministic(said, true, {21}, "default", rows.get, NOW), [])

    def test_past_claims_need_their_tag(self):
        from keel import checks

        found = checks.deterministic("I'm tired.", "You told me Ana is calling this Sunday.", set())
        self.assertTrue(any("without its tag" in p for p in found))
        same = checks.deterministic("The beta people found three bugs today", "You said the beta people found three bugs today.", set())
        self.assertFalse(any("without its tag" in p for p in same))

    def test_a_cited_commitment_counts_as_raised(self):
        from keel import checks

        verdict = {"commitment_at_stake": "- No new projects. [e:17]", "reply_raises_it": False}
        self.assertEqual(checks.problems_from(verdict, "default", "You said no new projects [e:17]."), [])
        self.assertEqual(len(checks.problems_from(verdict, "default", "Sounds fun!")), 1)
        self.assertEqual(checks.problems_from(verdict, "listen", "Sounds fun!"), [])

    def test_the_draft_with_fewest_problems_wins(self):
        llm = FakeLLM(
            replies=["First draft, decent enough here.", "Second draft says you loved Japan [e:99]."],
            verdicts=[{"drags_in_unrelated": True}, {}],
        )
        with tempfile.TemporaryDirectory() as home:
            s = Session(config.load(home, embed_model="none"), now=NOW, llm=llm)
            reply = s.turn("Today was long.")
            self.assertEqual(reply.checks["kept"], 0)
            self.assertEqual(reply.text, "First draft, decent enough here.")


class SalvageTests(Tmp):
    def test_an_invented_memory_is_never_spoken(self):
        invented = "Ana called you yesterday and said she's proud of you. Who else could you call?"
        llm = FakeLLM(
            replies=[invented, invented, invented, "That sounds lonely. Who could you call tonight?"],
            verdicts=[{"unsupported_claims": ["Ana called you yesterday and said she's proud of you."]}] * 3,
        )
        s = Session(self.cfg, now=NOW, llm=llm)
        reply = s.turn("You're the only one I can talk to.")
        self.assertNotIn("proud", reply.text)
        self.assertEqual(reply.checks["salvage"], "cut sentences")
        self.assertEqual(reply.text, "Who else could you call?")

    def test_when_nothing_is_left_it_answers_without_the_past(self):
        made_up = "On March 29th you said Japan was beautiful [e:1]."
        llm = FakeLLM(replies=[made_up, made_up, made_up, "I don't have that. Do you want to tell me about it?"], picks=[])
        s = Session(self.cfg, now=NOW, llm=llm)
        s.store.add_entry("Ana is moving to Lisbon.", when=day(3, 29))
        reply = s.turn("Remember my Japan trip? What did I say about it?")
        self.assertNotIn("Japan was beautiful", reply.text)
        self.assertEqual(reply.checks["salvage"], "answered without the past")
        prompt = llm.chats[-1][0]["content"]
        self.assertIn("don't mention anything from their past", prompt)
        self.assertNotIn("Lisbon", prompt)


class SalvageSafetyTests(Tmp):
    def test_a_fallback_still_says_it_is_an_ai(self):
        made_up = "On March 4th you said I was your best friend [e:1]."
        llm = FakeLLM(replies=[made_up] * 4, picks=[])
        s = Session(self.cfg, now=NOW, llm=llm)
        reply = s.turn("Are you a real person? Sometimes it feels like you are.")
        self.assertIn("I'm an AI", reply.text)

    def test_medication_always_points_to_the_doctor(self):
        llm = FakeLLM(replies=["Sounds like you're doing well, good for you."] * 4)
        s = Session(self.cfg, now=NOW, llm=llm)
        reply = s.turn("I feel fine so I'm going to stop my sertraline.")
        self.assertIn("doctor", reply.text)


class ReadBackTests(Tmp):
    def test_falls_back_to_their_own_words(self):
        llm = FakeLLM(replies=["You told me Dev called you a genius on Monday."] * 3)
        s = Session(self.cfg, now=NOW, llm=llm)
        s.store.add_entry("Dev thinks I'm avoiding the pricing decision. He might be right.", when=day(9, 28))
        reply = s.turn("Am I avoiding the pricing decision?")
        self.assertEqual(reply.checks["salvage"], "read back their words")
        self.assertIn("On September 28th you said: \u201cDev thinks I'm avoiding the pricing decision.", reply.text)
        self.assertNotIn("genius", reply.text)


class NightlyTests(unittest.TestCase):
    def test_validate(self):
        current = TEMPLATE.replace("## Hold me to\n", "## Hold me to\n\n- Run three times a week. [e:2]\n")
        proposed = (
            "# Me\n\n## Right now\n\n- Launch is October 20. [e:5]\n- Made-up line with no tag.\n"
            "- Bad tag. [e:99]\n\n## Hold me to\n\n- Sneaky new commitment. [e:5]\n"
        )
        text, dropped = nightly.validate(current, proposed, {2, 5})
        secs = sections(text)
        self.assertEqual(secs["Right now"], ["- Launch is October 20. [e:5]"])
        self.assertEqual(secs["Hold me to"], ["- Run three times a week. [e:2]"])
        self.assertEqual(len(dropped), 3)
        self.assertTrue(all(name in secs for name in ("Goals", "Corrections")))
        self.assertEqual(cited(text), {2, 5})


if __name__ == "__main__":
    unittest.main()
