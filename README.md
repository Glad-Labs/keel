# Keel

A journal that talks back. You tell it about your day, and it remembers. When
it brings up your past, it uses your own words and their dates. When you're
running against something you asked it to hold you to, it says so, kindly.
It runs on your own machine, and nothing leaves it.

"Keel" is a working name.

This is version 0. You talk to it by typing. Voice and the phone app come next.

## Try it

```bash
cd keel
uv venv && uv pip install -e ".[embed]"
.venv/bin/keel doctor      # checks the models, search, and the profile repo
.venv/bin/keel interview   # fourteen questions about your life, then your first profile
.venv/bin/keel chat
```

Things you can say in a chat:

| Say | What happens |
|---|---|
| "just listen" | It reflects back what you said. No advice and no pushback. |
| "be straight with me" | It gives you the plain answer, with your own words as evidence. |
| "hold me to this: ..." | Adds a line to your profile. It raises it when you run against it. |
| "that's not true: ..." | Adds a correction, which overrides everything else. |
| "talk to me from March" | Your March self answers, knowing only what you'd said by then. |
| "back to now" | Leaves past mode. |
| "forget that" | Deletes what you just said, everywhere. |

Other commands:
- `keel nightly` drafts profile changes from your new entries, and `keel review` shows the diff and lets you accept it.
- `keel forget 12` deletes entry 12.
- `keel profile --as-of "two months ago"` prints your profile as it was then.

Your journal lives in `~/.keel` and is never stored in this repo.

## What it promises, and how

| Promise | How it's kept |
|---|---|
| Nothing leaves this machine | It only talks to models on localhost and refuses any other address. Search runs on the CPU, inside the process. Your profile's git repo refuses to be used if it has a remote. |
| It never says something you didn't | Every claim about your past has to carry a tag pointing at a real entry. Plain checks confirm the tag exists, the date matches, and the sentence says what the entry says. If a reply still fails after two rewrites, the bad sentences are cut. If too little is left, it reads your own words back with their dates, or answers without the past. |
| "Forget that" deletes | It removes the entry, every turn that quoted it, and every profile line that came from it. That includes the profile's git history, which is rebuilt without them. SQLite's `secure_delete` and a search-index merge keep the text from lingering on disk. It can't do anything about how an SSD handles deleted blocks. |
| A crisis gets a real answer | A plain pattern screen runs before any model sees the message, and it returns a fixed reply with 988 and 911. If the model only suspects risk, it asks whether you're safe instead, because the model misreads ordinary venting. |
| It doesn't play doctor | No diagnoses or labels. Medication questions always point to your doctor. Asked whether it's a person, it says it's an AI. |
| It doesn't get in Poindexter's way | It never loads a model, uses the context length a model is already loaded with, and never sends keep_alive. |

The rules it follows are in [keel/RULES.md](keel/RULES.md). That file is sent
to the model word for word, so editing it changes Keel.

## How a reply is made

1. Keyword and meaning search finds candidate entries and profile lines.
2. A short first step picks the one to three that actually bear on what you said. The reply only sees those, plus your people and how you like to be talked to. It can't drag in what it can't see.
3. The model writes a reply.
4. Plain checks review it: tags, dates, how much is invented, labels, doses, length, and talking like a database. Then a model checks it against your record.
5. If anything is wrong, it gets up to two rewrites. The least-bad draft wins, and nothing that fails the memory or safety checks is ever spoken.

## What the tests show

There are 20 test conversations in `evals/`, run against a fictional
journal. They cover commitments, corrections, lifting you up, recall, an
invented-memory trap, the modes, crisis handling, medication and deletion. The
latest report is in `evals/results/`.

Results on `qwen3-vl:30b-a3b-instruct`, the model kept loaded on the 3090, on 2026-10-06:

- **The hard lines held in every run:** the crisis reply, deletion, "hold me to", corrections, medication pointing to a doctor, and saying it's an AI.
- **Grading:** plain checks passed 18 of 20 and the grader model passed 17. I read every reply myself: 15 good, 4 middling, 1 bad.
- **Speed:** the median reply takes 6.5 seconds.
- **What still goes wrong:**
  - It stretches commitments, treating "cut a feature" as breaking "no new projects".
  - Now and then it adds a detail that isn't in the journal, in a sentence the checks can't tie to an entry.
  - The fallbacks are safe but flat.
- **Don't trust the grader model alone.** It passed replies with invented memories until it was given the journal to check against, and it still misses some. Read the transcripts.

To run the tests: `.venv/bin/python -m evals.run --model qwen3-vl:30b-a3b-instruct`.
Plain unit tests, which need no model: `.venv/bin/python -m unittest discover -s tests -t .`.

## Which model to use

Keel only uses models that are already loaded. On this machine that means the
3090's qwen3-vl 30B-A3B, which is fast and always there. But it invents
memories when it has nothing to go on, which is why most of the checks above
exist.

Gemma 4 31B, GLM 4.7 and Qwen 3.6 27B are installed, but they load on the
5090, which Poindexter keeps busy. On 2026-10-06 a test load found 1.3 GB
free there, spilled onto the CPU, and stalled that Ollama server for ten
minutes. One Poindexter embedding job waited about four minutes and finished
with no failures. To compare models, pick a quiet window and run:

```bash
.venv/bin/python -m evals.run --model gemma-4-31B-it-qat:latest --allow-load
```

## Not built yet

- **Voice.** Speech in and out, and the phone app over Tailscale. Kokoro and Whisper are already running here behind the speech server on port 8001.
- **A schedule for the profile update.** `keel nightly` and `keel review` only run when you call them.
- **A LoRA**, for a mode where it talks as you.

## Layout

| Path | What |
|---|---|
| `keel/harness.py` | One conversation: commands, crisis, picking, drafting, checking, salvage |
| `keel/recall.py` | Picks what from your past bears on what you said |
| `keel/checks.py` | Every check a reply goes through |
| `keel/safety.py` | Crisis screen, labels, doses, fallbacks |
| `keel/store.py` | Entries, turns, search, deletion (SQLite) |
| `keel/profile.py` | `me.md` and its git history |
| `keel/nightly.py` | Drafts profile changes for you to review |
| `keel/RULES.md` | The rules, as the model reads them |
| `evals/` | The fictional journal, the 20 test conversations, the runner, the results |
