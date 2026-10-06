"""The journal: entries in your own words, the conversation log, and search.

Entries are only ever your words. Keel's replies go in the turns log and are
never treated as facts about you.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from . import embed

SCHEMA = """
CREATE TABLE IF NOT EXISTS entries(
  id INTEGER PRIMARY KEY AUTOINCREMENT,  -- never reused, so a tag like [e:12] can't change meaning
  created_at TEXT NOT NULL,              -- local time with offset, for people
  ts REAL NOT NULL,                      -- seconds since epoch, for ordering
  kind TEXT NOT NULL DEFAULT 'checkin',  -- checkin | interview | hold
  prompt TEXT,                           -- the question an interview entry answered
  text TEXT NOT NULL,
  audio_path TEXT,
  embedding BLOB,
  embed_model TEXT
);
CREATE VIRTUAL TABLE IF NOT EXISTS entries_fts
  USING fts5(text, content='entries', content_rowid='id', tokenize='porter unicode61');
CREATE TRIGGER IF NOT EXISTS entries_ai AFTER INSERT ON entries BEGIN
  INSERT INTO entries_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS entries_ad AFTER DELETE ON entries BEGIN
  INSERT INTO entries_fts(entries_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;
CREATE TABLE IF NOT EXISTS turns(
  id INTEGER PRIMARY KEY,
  session TEXT NOT NULL,
  created_at TEXT NOT NULL,
  role TEXT NOT NULL,      -- you | keel
  text TEXT NOT NULL,
  entry_id INTEGER,        -- the entry this exchange created
  mode TEXT,
  cited TEXT,              -- JSON list of entry ids the reply used
  checks TEXT              -- JSON: what the check pass found
);
CREATE TABLE IF NOT EXISTS forgotten(entry_id INTEGER PRIMARY KEY, forgotten_at TEXT NOT NULL);
"""

STOP = frozenset(
    """a about after again all also am an and any are as at be because been before being but by
    can could did do does doing done don't for from get got had has have having he her here him his
    how i i'd i'll i'm i've if in into is it it's its just me more most my myself no nor not now of
    off on once only or other our out over own same she should so some such than that that's the
    their them then there these they this those through to too under until up very was we were
    what when where which while who why will with would you you're your yours""".split()
)


def _now() -> datetime:
    return datetime.now().astimezone()


def fts_query(text: str) -> str:
    words = [w.strip("'") for w in re.findall(r"[a-z0-9']+", text.lower())]
    words = [w for w in words if len(w) > 2 and w not in STOP]
    return " OR ".join(f'"{w}"' for w in list(dict.fromkeys(words))[:24])


class Store:
    def __init__(self, path: Path | str, embed_model: str | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        # Deleted rows are overwritten with zeros instead of left in free pages.
        self.db.execute("PRAGMA secure_delete = ON")
        self.db.executescript(SCHEMA)
        self.embed_model = embed_model if embed_model and embed.available() else None

    def close(self) -> None:
        self.db.close()

    # entries -----------------------------------------------------------

    def add_entry(
        self,
        text: str,
        *,
        kind: str = "checkin",
        prompt: str | None = None,
        when: datetime | None = None,
        audio_path: str | None = None,
        embed_now: bool = True,
    ) -> int:
        when = when or _now()
        cur = self.db.execute(
            "INSERT INTO entries(created_at, ts, kind, prompt, text, audio_path) VALUES (?,?,?,?,?,?)",
            (when.isoformat(timespec="seconds"), when.timestamp(), kind, prompt, text, audio_path),
        )
        self.db.commit()
        if embed_now and self.embed_model:
            self._embed([cur.lastrowid])
        return cur.lastrowid

    def _embed(self, ids: list[int]) -> None:
        rows = [self.get(i) for i in ids]
        vecs = embed.passages(self.embed_model, [r["text"] for r in rows])
        self.db.executemany(
            "UPDATE entries SET embedding = ?, embed_model = ? WHERE id = ?",
            [(embed.to_blob(v), self.embed_model, r["id"]) for r, v in zip(rows, vecs)],
        )
        self.db.commit()

    def reindex(self) -> int:
        """Embed every entry that's missing one. Returns how many it did."""
        if not self.embed_model:
            return 0
        ids = [
            r[0]
            for r in self.db.execute(
                "SELECT id FROM entries WHERE embedding IS NULL OR embed_model IS NOT ?",
                (self.embed_model,),
            )
        ]
        for start in range(0, len(ids), 64):
            self._embed(ids[start : start + 64])
        return len(ids)

    def get(self, entry_id: int) -> sqlite3.Row | None:
        return self.db.execute("SELECT * FROM entries WHERE id = ?", (entry_id,)).fetchone()

    def existing(self, ids) -> set[int]:
        ids = list(ids)
        if not ids:
            return set()
        marks = ",".join("?" * len(ids))
        return {r[0] for r in self.db.execute(f"SELECT id FROM entries WHERE id IN ({marks})", ids)}

    def entries(
        self,
        *,
        before: datetime | None = None,
        after: float | None = None,
        limit: int | None = None,
    ) -> list[sqlite3.Row]:
        """Entries oldest first. `after` is a timestamp, `before` a datetime."""
        sql, args = "SELECT * FROM entries WHERE 1=1", []
        if before is not None:
            sql += " AND ts < ?"
            args.append(before.timestamp())
        if after is not None:
            sql += " AND ts > ?"
            args.append(after)
        sql += " ORDER BY ts"
        rows = self.db.execute(sql, args).fetchall()
        return rows[-limit:] if limit else rows

    def recent(self, n: int, *, before: datetime | None = None) -> list[sqlite3.Row]:
        return self.entries(before=before, limit=n)

    def count(self) -> int:
        return self.db.execute("SELECT count(*) FROM entries").fetchone()[0]

    def search(
        self, text: str, *, k: int = 6, before: datetime | None = None, exclude=()
    ) -> list[sqlite3.Row]:
        """Keyword and meaning search, merged by reciprocal rank fusion."""
        cutoff = before.timestamp() if before else None
        rankings: list[list[int]] = []

        query = fts_query(text)
        if query:
            sql = (
                "SELECT e.id FROM entries_fts JOIN entries e ON e.id = entries_fts.rowid "
                "WHERE entries_fts MATCH ?"
            )
            args: list = [query]
            if cutoff is not None:
                sql += " AND e.ts < ?"
                args.append(cutoff)
            sql += " ORDER BY bm25(entries_fts) LIMIT 30"
            rankings.append([r[0] for r in self.db.execute(sql, args)])

        if self.embed_model:
            qvec = embed.query(self.embed_model, text)
            sql, args = "SELECT id, embedding FROM entries WHERE embed_model = ?", [self.embed_model]
            if cutoff is not None:
                sql += " AND ts < ?"
                args.append(cutoff)
            scored = [
                (embed.dot(qvec, embed.from_blob(r["embedding"])), r["id"])
                for r in self.db.execute(sql, args)
                if r["embedding"]
            ]
            scored.sort(reverse=True)
            rankings.append([i for _, i in scored[:30]])

        score: dict[int, float] = defaultdict(float)
        for ranking in rankings:
            for rank, entry_id in enumerate(ranking):
                score[entry_id] += 1.0 / (60 + rank)
        ids = [i for i in sorted(score, key=score.get, reverse=True) if i not in set(exclude)]
        return [self.get(i) for i in ids[:k]]

    def forget(self, entry_id: int) -> bool:
        """Delete an entry, every turn that came from it or quoted it, and its audio."""
        row = self.get(entry_id)
        if row is None:
            return False
        self.db.execute("DELETE FROM turns WHERE entry_id = ?", (entry_id,))
        self.db.execute(
            "DELETE FROM turns WHERE id IN "
            "(SELECT t.id FROM turns t, json_each(t.cited) j WHERE t.cited IS NOT NULL AND j.value = ?)",
            (entry_id,),
        )
        self.db.execute("DELETE FROM entries WHERE id = ?", (entry_id,))
        self.db.execute(
            "INSERT OR REPLACE INTO forgotten(entry_id, forgotten_at) VALUES (?, ?)",
            (entry_id, _now().isoformat(timespec="seconds")),
        )
        self.db.commit()
        # Merge the search index so the deleted words don't linger in old segments.
        self.db.execute("INSERT INTO entries_fts(entries_fts) VALUES ('optimize')")
        self.db.commit()
        if row["audio_path"]:
            Path(row["audio_path"]).unlink(missing_ok=True)
        return True

    # turns -------------------------------------------------------------

    def add_turn(
        self,
        session: str,
        role: str,
        text: str,
        *,
        entry_id: int | None = None,
        mode: str | None = None,
        cited: list[int] | None = None,
        checks: dict | None = None,
        when: datetime | None = None,
    ) -> int:
        cur = self.db.execute(
            "INSERT INTO turns(session, created_at, role, text, entry_id, mode, cited, checks) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (
                session,
                (when or _now()).isoformat(timespec="seconds"),
                role,
                text,
                entry_id,
                mode,
                json.dumps(cited) if cited is not None else None,
                json.dumps(checks) if checks is not None else None,
            ),
        )
        self.db.commit()
        return cur.lastrowid

    def turns(self, session: str | None = None) -> list[sqlite3.Row]:
        if session is None:
            return self.db.execute("SELECT * FROM turns ORDER BY id").fetchall()
        return self.db.execute("SELECT * FROM turns WHERE session = ? ORDER BY id", (session,)).fetchall()
