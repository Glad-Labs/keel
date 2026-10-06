"""keel: talk to your journal from the terminal."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

from . import commands, config, embed, nightly, ollama
from .harness import Session, forget_entry
from .interview import QUESTIONS
from .profile import Profile
from .store import Store

HELP = (
    'Say "just listen", "be straight with me", "hold me to this: ...", "that\'s not true: ...", '
    '"talk to me from March", "back to now", or "forget that". Ctrl-D to stop.'
)


def _open(cfg: config.Config) -> tuple[Store, Profile]:
    store = Store(cfg.db_path, cfg.embed_model)
    profile = Profile(cfg.profile_dir)
    profile.ensure()
    return store, profile


def cmd_chat(cfg, args) -> int:
    session = Session(cfg)
    print(f"Keel, using {cfg.chat.model} on this machine. {HELP}\n")
    while True:
        try:
            said = input("you > ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not said.strip():
            continue
        try:
            reply = session.turn(said)
        except ollama.OllamaError as err:
            print(f"keel > (I couldn't answer: {err})\n")
            continue
        mode = "" if reply.mode == "default" else f" ({reply.mode})"
        print(f"keel{mode} > {reply.text}")
        if args.debug:
            print(json.dumps({"seconds": round(reply.seconds, 1), **reply.checks}, indent=2))
        print()


def cmd_say(cfg, args) -> int:
    try:
        reply = Session(cfg).turn(" ".join(args.text))
    except ollama.OllamaError as err:
        print(f"I couldn't answer: {err}", file=sys.stderr)
        return 1
    print(reply.spoken if args.spoken else reply.text)
    return 0


def _read_answer() -> str:
    lines = []
    while True:
        try:
            line = input("  ")
        except EOFError:
            break
        if not line.strip():
            break
        lines.append(line)
    return " ".join(lines).strip()


def cmd_interview(cfg, args) -> int:
    store, profile = _open(cfg)
    print(
        "The kickoff interview: fourteen questions about your life so far. Answer in as much or as "
        "little as you like, then press Enter on an empty line. Say skip to skip one, or stop to finish early.\n"
    )
    saved = 0
    for question in QUESTIONS:
        print(question)
        answer = _read_answer()
        if answer.lower() in ("stop", "quit"):
            break
        if answer and answer.lower() != "skip":
            store.add_entry(answer, kind="interview", prompt=question)
            saved += 1
        print()
    print(f"Saved {saved} answers.")
    if saved:
        return cmd_nightly(cfg, args)
    return 0


def cmd_nightly(cfg, args) -> int:
    store, profile = _open(cfg)
    print("Reading new entries and drafting profile changes...")
    proposal = nightly.propose(cfg, store, profile)
    if proposal is None:
        print("No new entries since the last update.")
        return 0
    print(f"From {proposal.new_entries} new entries:")
    for change in proposal.changes:
        print(f"  - {change}")
    if proposal.dropped:
        print(f"Left out {len(proposal.dropped)} line(s) that didn't point at real entries.")
    print("Run `keel review` to see the diff and accept or reject it.")
    return 0


def cmd_review(cfg, args) -> int:
    _, profile = _open(cfg)
    while True:
        diff = nightly.diff(cfg, profile)
        if not diff:
            print("Nothing to review. Run `keel nightly` first.")
            return 0
        print(diff)
        choice = input("Accept these changes? [y]es, [n]o, or [e]dit first: ").strip().lower()
        if choice.startswith("e"):
            subprocess.run([os.environ.get("EDITOR", "nano"), str(cfg.proposal_path)])
            continue
        if choice.startswith("y"):
            nightly.accept(cfg, profile)
            print("Saved. Your profile is updated.")
        else:
            nightly.reject(cfg)
            print("Thrown away. Your profile is unchanged.")
        return 0


def cmd_profile(cfg, args) -> int:
    _, profile = _open(cfg)
    if args.as_of:
        when = commands.parse_when(args.as_of, datetime.now().astimezone())
        if when is None:
            print(f"Couldn't read the date {args.as_of!r}.", file=sys.stderr)
            return 1
        print(profile.as_of(when))
    else:
        print(profile.read())
    return 0


def cmd_entries(cfg, args) -> int:
    store, _ = _open(cfg)
    for row in store.entries(limit=args.last):
        created = datetime.fromisoformat(row["created_at"])
        print(f"[e:{row['id']}] {created:%Y-%m-%d %H:%M} {row['kind']}: {row['text']}")
    return 0


def cmd_forget(cfg, args) -> int:
    store, profile = _open(cfg)
    if store.get(args.entry) is None:
        print(f"There's no entry {args.entry}.")
        return 1
    lines = forget_entry(store, profile, args.entry)
    print(f"Deleted entry {args.entry}" + (f" and {lines} profile line(s) that came from it." if lines else "."))
    return 0


def cmd_reindex(cfg, args) -> int:
    store, _ = _open(cfg)
    print(f"Embedded {store.reindex()} entries.")
    return 0


def cmd_doctor(cfg, args) -> int:
    ok = True
    print(f"Journal: {cfg.home}")
    for name, ep in (("Talks with", cfg.chat), ("Checks with", cfg.check)):
        try:
            have = ep.model in ollama.installed(ep.url)
            loaded = ollama.loaded(ep.url).get(ep.model)
            if loaded:
                state = f"loaded, context {loaded['context_length']}"
            elif ep.allow_load:
                state = "not loaded; Keel will load it, pushing out whatever is there"
            else:
                state = "NOT LOADED, and Keel won't load it (set allow_load to let it)"
                ok = False
            print(f"{name}: {ep.model} at {ep.url} ({'installed' if have else 'NOT INSTALLED'}, {state})")
            ok &= have
        except ollama.OllamaError as err:
            print(f"{name}: {ep.model} at {ep.url}: {err}")
            ok = False
    search = "keyword and meaning" if cfg.embed_model and embed.available() else "keyword only"
    print(f"Search: {search}" + (f" ({cfg.embed_model}, on the CPU)" if search != "keyword only" else ""))
    try:
        store, profile = _open(cfg)
        print(f"Entries: {store.count()}. Profile versions: {len(profile.history())}. No git remote: yes.")
    except config.LocalOnlyError as err:
        print(f"Profile: {err}")
        ok = False
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="keel", description="A journal that talks back, on your own machine.")
    sub = parser.add_subparsers(dest="command", required=True)
    chat = sub.add_parser("chat", help="talk to your journal")
    chat.add_argument("--debug", action="store_true", help="show what the check pass found")
    say = sub.add_parser("say", help="one turn, for scripts")
    say.add_argument("text", nargs="+")
    say.add_argument("--spoken", action="store_true", help="print the version meant for speaking")
    sub.add_parser("interview", help="the kickoff interview")
    sub.add_parser("nightly", help="draft profile changes from new entries")
    sub.add_parser("review", help="accept or reject the drafted profile changes")
    prof = sub.add_parser("profile", help="print your profile")
    prof.add_argument("--as-of", help='e.g. "March" or "two months ago"')
    ent = sub.add_parser("entries", help="list entries")
    ent.add_argument("--last", type=int, default=20)
    forget = sub.add_parser("forget", help="delete an entry everywhere")
    forget.add_argument("entry", type=int)
    sub.add_parser("reindex", help="embed entries that are missing one")
    sub.add_parser("doctor", help="check models, search and the profile repo")

    args = parser.parse_args(argv)
    try:
        cfg = config.load()
        handler = globals()[f"cmd_{args.command}"]
        return handler(cfg, args)
    except config.LocalOnlyError as err:
        print(err, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
