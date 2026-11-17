"""Conversation history: every chat is a session whose full message history is checkpointed by LangGraph's
SqliteSaver in <workspace>/.lca/sessions.db, so sessions survive restarts and can be resumed.

This module keeps the session index (title, timestamps) next to the checkpoints and prunes old checkpoints, so the
database stays small: only the latest state of each session is needed to resume it.
"""

from __future__ import annotations

import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from lca.memory.store import ensure_private_dir


@dataclass
class SessionInfo:
    id: str
    title: str
    created: float
    updated: float
    turns: int

    def as_dict(self) -> dict:
        return {"id": self.id, "title": self.title, "created": self.created, "updated": self.updated, "turns": self.turns}


class SessionStore:
    def __init__(self, workspace_root: Path, db_path: Path | None = None) -> None:
        self.path = db_path or ensure_private_dir(workspace_root) / "sessions.db"
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS lca_sessions (id TEXT PRIMARY KEY, title TEXT NOT NULL DEFAULT '', "
            "created REAL NOT NULL, updated REAL NOT NULL, turns INTEGER NOT NULL DEFAULT 0)"
        )
        self._db.commit()
        # LangGraph uses its own connection; SQLite in WAL mode handles both safely.
        self._saver_conn = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
        self.checkpointer = SqliteSaver(self._saver_conn)
        self.checkpointer.setup()

    def create(self, title: str = "") -> SessionInfo:
        now = time.time()
        info = SessionInfo(uuid.uuid4().hex[:12], title[:120], now, now, 0)
        with self._lock:
            self._db.execute("INSERT INTO lca_sessions VALUES (?, ?, ?, ?, ?)", (info.id, info.title, now, now, 0))
            self._db.commit()
        return info

    def touch(self, session_id: str, first_message: str = "") -> None:
        title = " ".join(first_message.split())[:80]
        with self._lock:
            self._db.execute(
                "UPDATE lca_sessions SET updated = ?, turns = turns + 1, title = CASE WHEN title = '' THEN ? ELSE title END WHERE id = ?",
                (time.time(), title, session_id),
            )
            self._db.commit()
        self.prune(session_id)

    def get(self, session_id: str) -> SessionInfo | None:
        with self._lock:
            row = self._db.execute("SELECT id, title, created, updated, turns FROM lca_sessions WHERE id = ?", (session_id,)).fetchone()
        return SessionInfo(*row) if row else None

    def latest(self) -> SessionInfo | None:
        items = self.list(1)
        return items[0] if items else None

    def list(self, limit: int = 50) -> list[SessionInfo]:
        with self._lock:
            rows = self._db.execute(
                "SELECT id, title, created, updated, turns FROM lca_sessions WHERE turns > 0 ORDER BY updated DESC LIMIT ?", (limit,)
            ).fetchall()
        return [SessionInfo(*r) for r in rows]

    def delete(self, session_id: str) -> None:
        self.checkpointer.delete_thread(session_id)
        with self._lock:
            self._db.execute("DELETE FROM lca_sessions WHERE id = ?", (session_id,))
            self._db.commit()

    def prune(self, session_id: str) -> None:
        """Keep only the newest checkpoint of a session (it contains the full message history)."""
        with self._lock:
            try:
                latest = self._db.execute(
                    "SELECT checkpoint_id FROM checkpoints WHERE thread_id = ? AND checkpoint_ns = '' ORDER BY checkpoint_id DESC LIMIT 1",
                    (session_id,),
                ).fetchone()
                if not latest:
                    return
                self._db.execute("DELETE FROM checkpoints WHERE thread_id = ? AND checkpoint_id != ?", (session_id, latest[0]))
                self._db.execute("DELETE FROM writes WHERE thread_id = ? AND checkpoint_id != ?", (session_id, latest[0]))
                self._db.commit()
            except sqlite3.Error:
                pass  # pruning is housekeeping only

    def close(self) -> None:
        for conn in (self._db, self._saver_conn):
            try:
                conn.close()
            except sqlite3.Error:
                pass
