"""Checkpoints: the state of every file the agent changes, saved before the change, per user message.

Each user message (a task or /build) opens a checkpoint. Before a file tool writes, edits or deletes a file, the
file's exact bytes (or "did not exist") are saved in that checkpoint, once per file. Rewinding to a checkpoint puts
every file changed since then back as it was before that message, and can also cut the conversation back to it.

Stored as JSON in <workspace>/.lca/checkpoints/<session>/<seq>.json. Changes made by shell commands (npm install,
code generators) are not tracked; only the agent's file tools are.
"""

from __future__ import annotations

import base64
import json
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from lca.memory.store import ensure_private_dir

MAX_FILE_BYTES = 5_000_000  # larger files are noted but not saved
MAX_DIR_FILES = 2000

RewindMode = Literal["both", "code", "conversation"]


@dataclass
class Checkpoint:
    session: str
    seq: int
    prompt: str
    created: float
    message_id: str  # id of the user message this checkpoint belongs to
    files: dict[str, dict] = field(default_factory=dict)  # path -> {"data": base64 | None, "skipped": reason}

    @property
    def id(self) -> str:
        return f"{self.session}:{self.seq}"

    def summary(self) -> dict:
        return {"id": self.id, "seq": self.seq, "prompt": self.prompt[:300], "created": self.created, "message_id": self.message_id, "files": sorted(self.files)}


@dataclass
class RewindResult:
    restored: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)  # files the agent had created
    skipped: list[str] = field(default_factory=list)  # could not be restored (too large, unreadable)
    prompt: str = ""

    def as_dict(self) -> dict:
        return {"restored": self.restored, "removed": self.removed, "skipped": self.skipped, "prompt": self.prompt}


class CheckpointStore:
    def __init__(self, workspace_root: Path) -> None:
        self.root = workspace_root
        self.base = ensure_private_dir(workspace_root) / "checkpoints"
        self._lock = threading.Lock()

    def _dir(self, session: str) -> Path:
        return self.base / session

    def _path(self, session: str, seq: int) -> Path:
        return self._dir(session) / f"{seq:05d}.json"

    def _save(self, cp: Checkpoint) -> None:
        d = self._dir(cp.session)
        d.mkdir(parents=True, exist_ok=True)
        data = {"session": cp.session, "seq": cp.seq, "prompt": cp.prompt, "created": cp.created, "message_id": cp.message_id, "files": cp.files}
        tmp = self._path(cp.session, cp.seq).with_suffix(".tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        tmp.replace(self._path(cp.session, cp.seq))

    def _load(self, path: Path) -> Checkpoint | None:
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
            return Checkpoint(d["session"], int(d["seq"]), d.get("prompt", ""), float(d["created"]), d.get("message_id", ""), d.get("files", {}))
        except (OSError, ValueError, KeyError, TypeError):
            return None

    # ------------------------------------------------------------ recording

    def begin(self, session: str, prompt: str, message_id: str) -> Checkpoint:
        """Open the checkpoint for a new user message. Every message gets one, so the conversation can be rewound to
        any message even when it changed no files."""
        with self._lock:
            seqs = [int(p.stem) for p in self._dir(session).glob("*.json") if p.stem.isdigit()] if self._dir(session).is_dir() else []
            cp = Checkpoint(session, max(seqs, default=0) + 1, prompt, time.time(), message_id)
            self._save(cp)
            return cp

    def record(self, cp: Checkpoint, rel_path: str, *, is_dir: bool = False) -> None:
        """Save the current state of `rel_path` (a file, or every file in a folder) unless already saved in `cp`."""
        target = self.root / rel_path
        paths = [rel_path]
        if is_dir and target.is_dir():
            paths = [p.relative_to(self.root).as_posix() for p in sorted(target.rglob("*")) if p.is_file()][:MAX_DIR_FILES]
        with self._lock:
            changed = False
            for rel in paths:
                if rel in cp.files:
                    continue
                cp.files[rel] = self._snapshot(self.root / rel)
                changed = True
            if changed:
                self._save(cp)

    @staticmethod
    def _snapshot(path: Path) -> dict:
        if not path.exists():
            return {"data": None}
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                return {"data": None, "skipped": "too large to save"}
            return {"data": base64.b64encode(path.read_bytes()).decode("ascii")}
        except OSError as e:
            return {"data": None, "skipped": f"unreadable: {e.strerror or e}"}

    # ------------------------------------------------------------ listing and rewinding

    def list(self, session: str) -> list[Checkpoint]:
        d = self._dir(session)
        if not d.is_dir():
            return []
        cps = [cp for p in sorted(d.glob("*.json")) if (cp := self._load(p)) is not None]
        return sorted(cps, key=lambda c: c.seq)

    def find(self, checkpoint_id: str) -> Checkpoint | None:
        session, _, seq = checkpoint_id.rpartition(":")
        if not session or not seq.isdigit():
            return None
        return self._load(self._path(session, int(seq)))

    def rewind(self, checkpoint_id: str, *, restore_code: bool = True) -> RewindResult:
        """Restore every file changed at or after this checkpoint to its state before it, then drop those checkpoints."""
        target = self.find(checkpoint_id)
        if target is None:
            raise KeyError(f"No checkpoint {checkpoint_id!r}.")
        later = [cp for cp in self.list(target.session) if cp.seq >= target.seq]
        result = RewindResult(prompt=target.prompt)
        if restore_code:
            earliest: dict[str, dict] = {}
            for cp in later:  # oldest first: the first saved state of a file is the one before `target`
                for rel, snap in cp.files.items():
                    earliest.setdefault(rel, snap)
            for rel, snap in sorted(earliest.items()):
                self._restore(rel, snap, result)
        with self._lock:
            for cp in later:
                self._path(cp.session, cp.seq).unlink(missing_ok=True)
        return result

    def _restore(self, rel: str, snap: dict, result: RewindResult) -> None:
        path = self.root / rel
        if snap.get("skipped"):
            result.skipped.append(f"{rel} ({snap['skipped']})")
            return
        try:
            if snap.get("data") is None:
                if path.is_file() or path.is_symlink():
                    path.unlink()
                    result.removed.append(rel)
                elif path.is_dir():
                    shutil.rmtree(path)
                    result.removed.append(rel)
                return
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.is_dir():
                shutil.rmtree(path)
            path.write_bytes(base64.b64decode(snap["data"]))
            result.restored.append(rel)
        except OSError as e:
            result.skipped.append(f"{rel} ({e.strerror or e})")

    def delete_session(self, session: str) -> None:
        shutil.rmtree(self._dir(session), ignore_errors=True)
