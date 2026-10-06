"""me.md: what you've told Keel about yourself.

It lives in its own git repo so every version is kept, and that repo must
never have a remote. Keel checks before every write and refuses if one shows
up, because your journal leaving this machine is the one thing it exists to
prevent.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

from .config import LocalOnlyError

SECTIONS = ("Hold me to", "Right now", "People", "Goals", "Patterns", "How to talk to me", "Corrections")
# Only you add or remove lines here. The nightly update can't drop them.
LOCKED = ("Hold me to", "Corrections")

HEADER = (
    "# Me\n\n"
    "Written from what I've told Keel. Each line ends with the entries it came from, like [e:12].\n"
    "I can edit this by hand, and my edits win.\n"
)
TEMPLATE = HEADER + "".join(f"\n## {name}\n" for name in SECTIONS)

TAG = re.compile(r"\[e:\d+(?:\s*,\s*e:\d+)*\]")
TAG_ID = re.compile(r"e:(\d+)")


def cited(text: str) -> set[int]:
    return {int(n) for tag in TAG.findall(text) for n in TAG_ID.findall(tag)}


def sections(text: str) -> dict[str, list[str]]:
    """Bullet lines under each '## ' heading."""
    out: dict[str, list[str]] = {}
    current = None
    for line in text.splitlines():
        if line.startswith("## "):
            current = line[3:].strip()
            out.setdefault(current, [])
        elif current is not None and line.strip().startswith("- "):
            out[current].append(line.strip())
    return out


def insert_line(text: str, section: str, line: str) -> str:
    """Add a bullet at the end of a section, creating the section if needed."""
    lines = text.rstrip("\n").split("\n")
    try:
        start = next(i for i, l in enumerate(lines) if l.strip() == f"## {section}")
    except StopIteration:
        return text.rstrip("\n") + f"\n\n## {section}\n\n{line}\n"
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    last = max((i for i in range(start + 1, end) if lines[i].strip()), default=start)
    if last == start:
        lines[start + 1 : start + 1] = ["", line]
    else:
        lines.insert(last + 1, line)
    return "\n".join(lines) + "\n"


def drop_lines_citing(text: str, entry_id: int) -> tuple[str, int]:
    pattern = re.compile(rf"(?<!\d)e:{entry_id}(?!\d)")
    kept, dropped = [], 0
    for line in text.split("\n"):
        if pattern.search(line):
            dropped += 1
        else:
            kept.append(line)
    return "\n".join(kept), dropped


class Profile:
    def __init__(self, directory: Path | str):
        self.dir = Path(directory)
        self.path = self.dir / "me.md"

    def _git(self, *args: str, env: dict | None = None, cwd: Path | None = None) -> str:
        result = subprocess.run(
            ["git", "-C", str(cwd or self.dir), *args], capture_output=True, text=True, env=env
        )
        if result.returncode:
            raise RuntimeError(f"git {' '.join(args)}: {result.stderr.strip()}")
        return result.stdout

    def _init(self, where: Path) -> None:
        where.mkdir(parents=True, exist_ok=True)
        self._git("init", "-q", "-b", "main", cwd=where)
        self._git("config", "user.name", "Keel", cwd=where)
        self._git("config", "user.email", "keel@localhost", cwd=where)
        self._git("config", "commit.gpgsign", "false", cwd=where)

    def ensure(self, when: datetime | None = None) -> None:
        if not (self.dir / ".git").exists():
            self._init(self.dir)
        self.assert_no_remote()
        if not self.path.exists():
            self.path.write_text(TEMPLATE)
            self.commit("start the profile", when)

    def assert_no_remote(self) -> None:
        if self._git("remote").strip():
            raise LocalOnlyError(
                f"{self.dir} has a git remote, so Keel won't write to it. "
                f"Remove it with: git -C {self.dir} remote remove <name>"
            )

    def read(self) -> str:
        return self.path.read_text() if self.path.exists() else TEMPLATE

    def _has_commits(self) -> bool:
        return subprocess.run(
            ["git", "-C", str(self.dir), "rev-parse", "--verify", "-q", "HEAD"], capture_output=True
        ).returncode == 0

    def commit(self, message: str, when: datetime | None = None) -> bool:
        self.assert_no_remote()
        self._git("add", "me.md")
        if not self._git("status", "--porcelain", "me.md").strip():
            return False
        env = None
        if when is not None:
            stamp = when.isoformat()
            env = {**os.environ, "GIT_AUTHOR_DATE": stamp, "GIT_COMMITTER_DATE": stamp}
        self._git("commit", "-q", "-m", message, env=env)
        return True

    def write(self, text: str, message: str, when: datetime | None = None) -> bool:
        self.path.write_text(text)
        return self.commit(message, when)

    def add_line(self, section: str, line: str, message: str, when: datetime | None = None) -> bool:
        return self.write(insert_line(self.read(), section, line), message, when)

    def as_of(self, when: datetime) -> str:
        """The profile as it stood at a moment in the past."""
        if not self._has_commits():
            return TEMPLATE
        rev = self._git("rev-list", "-1", f"--before={when.isoformat()}", "HEAD").strip()
        return self._git("show", f"{rev}:me.md") if rev else TEMPLATE

    def history(self) -> list[tuple[str, str]]:
        if not self._has_commits():
            return []
        out = self._git("log", "--format=%aI%x1f%s")
        return [tuple(line.split("\x1f", 1)) for line in out.splitlines()]

    def scrub(self, entry_id: int) -> int:
        """Remove every line citing an entry, from me.md and from all of its history.

        Git keeps old versions forever, so this rebuilds the repo without the
        lines and deletes the old one. Commit messages never contain your
        words, only entry numbers, and those are rewritten too. Returns how
        many lines left the current profile.
        """
        current, dropped = drop_lines_citing(self.read(), entry_id)
        in_history = 0
        versions = []
        if self._has_commits():
            log = self._git("log", "--reverse", "--format=%H%x1f%aI%x1f%cI%x1f%s")
            for line in log.splitlines():
                sha, author_date, commit_date, subject = line.split("\x1f")
                text, n = drop_lines_citing(self._git("show", f"{sha}:me.md"), entry_id)
                cleaned = re.sub(rf"(?<!\d)e:{entry_id}(?!\d)", "a forgotten entry", subject)
                in_history += n + (cleaned != subject)
                versions.append((author_date, commit_date, text, cleaned))
        if not dropped and not in_history:
            return 0

        fresh = Path(tempfile.mkdtemp(prefix=".profile-", dir=self.dir.parent))
        self._init(fresh)
        for author_date, commit_date, text, subject in versions:
            (fresh / "me.md").write_text(text)
            self._git("add", "me.md", cwd=fresh)
            if self._git("status", "--porcelain", "me.md", cwd=fresh).strip():
                env = {**os.environ, "GIT_AUTHOR_DATE": author_date, "GIT_COMMITTER_DATE": commit_date}
                self._git("commit", "-q", "-m", subject, env=env, cwd=fresh)
        (fresh / "me.md").write_text(current)
        self._git("add", "me.md", cwd=fresh)
        if self._git("status", "--porcelain", "me.md", cwd=fresh).strip():
            self._git("commit", "-q", "-m", "forget an entry", cwd=fresh)

        old = self.dir.with_name(self.dir.name + ".old")
        self.dir.rename(old)
        fresh.rename(self.dir)
        shutil.rmtree(old)
        return dropped
