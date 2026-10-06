"""The lines Keel doesn't cross, checked with plain patterns before any model is trusted.

The crisis screen runs on what you say. The rest run on what Keel is about to
say back. A model double-checks both (see checks.py), but these never depend
on one.
"""

from __future__ import annotations

import re

QUOTES = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"'})


def plain(text: str) -> str:
    """Straight quotes. Phone keyboards type 'don\u2019t', and every pattern here expects 'don't'."""
    return text.translate(QUOTES)


CRISIS = [
    r"\bsuicid(?:e|al)\b",
    r"\bkill(?:ing)?\s+myself\b",
    r"\bend(?:ing)?\s+(?:my\s+(?:own\s+)?life|it\s+all)\b",
    r"\b(?:take|taking)\s+my\s+(?:own\s+)?life\b",
    r"\b(?:want|wanna|wanted)\s+to\s+die\b(?!\s+(?:of|from)\s+(?:embarrass\w*|laugh\w*|shame|boredom|cringe))",
    r"\bwish\s+i\s+(?:was|were)\s+dead\b",
    r"\bbetter\s+off\s+(?:dead|without\s+me)\b",
    r"\bno\s+(?:reason|point)\s+(?:to|in)\s+(?:live|living|being\s+alive|going\s+on)\b",
    r"\bdon'?t\s+want\s+to\s+(?:be\s+alive|live\s+anymore|exist|be\s+here\s+anymore)\b",
    r"\bnot\s+worth\s+living\b",
    r"\b(?:hurt|harm|cut)(?:ing)?\s+myself\b",
    r"\bself[-\s]?harm\b",
    r"\boverdos(?:e|ing)\b",
]
CRISIS_RE = re.compile("|".join(CRISIS), re.IGNORECASE)
# Hurting someone else. "people's feelings" and "I don't want to hurt him" don't count.
OTHERS_RE = re.compile(
    r"\b(?:want|wanna|going|gonna|thinking\s+about|thought\s+about)\s+(?:to\s+)?(?:kill|hurt)\s+"
    r"(?:him|her|them|someone|somebody|people)\b(?!'s)",
    re.IGNORECASE,
)
NEGATED_RE = re.compile(r"\b(?:don'?t|do\s+not|never|wouldn'?t|won'?t|not)\s+$", re.IGNORECASE)

CRISIS_REPLY = (
    "I'm an AI, so I'm not the right help for this, and I don't want you alone with it. "
    "If you're thinking about ending your life or hurting yourself, please call or text 988 right now. "
    "That's the Suicide and Crisis Lifeline, open all day and all night. "
    "If you or anyone else is in danger this minute, call 911. "
    "And if there's someone you trust, tell them what you just told me. I'll still be here after."
)


def crisis(said: str) -> bool:
    said = plain(said)
    if CRISIS_RE.search(said):
        return True
    return any(
        not NEGATED_RE.search(said[max(0, m.start() - 15) : m.start()]) for m in OTHERS_RE.finditer(said)
    )


CONDITION = (
    r"(?:clinical(?:ly)?\s+)?(?:depress(?:ion|ed|ive)|bipolar|adhd|ptsd|ocd|anxiety\s+disorder|"
    r"generali[sz]ed\s+anxiety|borderline|narcissis\w*|personality\s+disorder|autis\w*|schizo\w*|"
    r"psychos[ie]s|psychotic|mania|manic|eating\s+disorder|anorexi\w*|bulimi\w*)"
)
DIAGNOSIS = [
    rf"\byou(?:'re| are| have| might have| may have| probably have| could have| likely have| seem to have|"
    rf" show signs of| sound| seem| look)\s+(?:like\s+)?(?:a\s+|an\s+|some\s+)?(?:mild\s+|severe\s+|major\s+|"
    rf"high-functioning\s+)?{CONDITION}",
    rf"\b(?:this|that|it)\s+(?:is|sounds like|looks like|could be|might be|seems like)\s+(?:a\s+|an\s+)?"
    rf"(?:classic\s+)?(?:case\s+of\s+|sign\s+of\s+)?{CONDITION}",
    rf"\bsymptoms?\s+of\s+{CONDITION}",
]
DIAGNOSIS_RE = re.compile("|".join(DIAGNOSIS), re.IGNORECASE)
HEDGE_RE = re.compile(r"\b(?:whether|if|can'?t|cannot|not|isn'?t|no\s+way|won'?t)\b", re.IGNORECASE)

DOSE_RE = re.compile(r"\b\d+(?:\.\d+)?\s?(?:mg|milligrams?|mcg|micrograms?)\b", re.IGNORECASE)
HUMAN_RE = re.compile(r"\bI(?:'m| am)\s+(?:a\s+)?(?:real\s+)?(?:human|person)\b", re.IGNORECASE)
ASKS_IF_AI_RE = re.compile(
    r"\b(?:are|r)\s+(?:you|u)\s+(?:a\s+|an\s+)?(?:real|human|person|bot|robot|ai|machine|alive)\b",
    re.IGNORECASE,
)
SAYS_AI_RE = re.compile(r"\bAI\b|\bartificial\b|\bnot\s+a\s+(?:real\s+)?(?:person|human)\b", re.IGNORECASE)


def diagnoses(reply: str) -> list[str]:
    """Places the reply says or implies you have a condition. Hedged mentions don't count."""
    reply = plain(reply)
    found = []
    for match in DIAGNOSIS_RE.finditer(reply):
        before = reply[max(0, match.start() - 40) : match.start()]
        if not HEDGE_RE.search(before):
            found.append(match.group(0))
    return found


def doses(reply: str) -> list[str]:
    return DOSE_RE.findall(plain(reply))


def claims_human(reply: str) -> bool:
    reply = plain(reply)
    for match in HUMAN_RE.finditer(reply):
        if not re.search(r"\bnot\b", reply[max(0, match.start() - 20) : match.end()], re.IGNORECASE):
            return True
    return False


def needs_ai_disclosure(said: str, reply: str) -> bool:
    return bool(ASKS_IF_AI_RE.search(plain(said))) and not SAYS_AI_RE.search(plain(reply))


MEDS_RE = re.compile(
    r"\b(?:meds?|medications?|medicines?|pills?|prescri\w+|doses?|dosage|sertraline|zoloft|prozac|lexapro|"
    r"ssris?|antidepressants?|adderall|xanax|lithium|insulin)\b",
    re.IGNORECASE,
)
DIAGNOSIS_ASK_RE = re.compile(
    rf"\b(?:i have|i've got|am i|i'm|is (?:this|it|that))\b.*?{CONDITION}", re.IGNORECASE
)


def fallback(said: str, question: bool) -> str:
    """What Keel says when every draft failed its checks. Safe, short, honest."""
    said = plain(said)
    if MEDS_RE.search(said):
        return (
            "That's one to talk through with your doctor before you change anything. "
            "I'm an AI, so I can't advise on medication."
        )
    if DIAGNOSIS_ASK_RE.search(said):
        return (
            "I can't tell you that. I'm an AI, not a clinician. A doctor or therapist can, "
            "and I can tell you what you've told me if that would help."
        )
    if question:
        return "I don't have that. Do you want to tell me about it?"
    return "I'm with you. Tell me more about what's going on."
