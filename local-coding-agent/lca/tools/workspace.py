"""The authorized workspace: path validation, secret protection and gitignore-aware file enumeration."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from collections.abc import Iterator
from functools import cached_property
from pathlib import Path, PurePosixPath

from lca.security import is_secret_path

# Directories skipped when the workspace is not a git repository (or for untracked files).
DEFAULT_IGNORED_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv", "env", ".tox", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", "dist", "build", "target", ".next", ".nuxt", "out", "coverage",
    ".idea", ".vs", ".gradle", ".lca", "bin", "obj",
}

MAX_ENUMERATED_FILES = 200_000

# Entries under <workspace>/.lca that hold the agent's own state and are never useful to the model.
INTERNAL_LCA_ENTRIES = {"checkpoints", "trash", "logs"}


class WorkspaceError(Exception):
    """A path request the agent is not allowed to make. The message is shown to the model."""


class Workspace:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        if not self.root.is_dir():
            raise WorkspaceError(f"Workspace root is not a directory: {root}")

    # ------------------------------------------------------------ paths

    def resolve(self, path: str | os.PathLike[str], *, allow_secret: bool = False) -> Path:
        """Turn a model-supplied path into an absolute path inside the workspace, or raise WorkspaceError."""
        raw = str(path).strip()
        if not raw:
            raise WorkspaceError("Path is empty.")
        if "\0" in raw:
            raise WorkspaceError("Path contains a NUL byte.")
        if os.name == "nt":
            # Git Bash / MSYS style (/c/Users/...) or WSL style (/mnt/c/Users/...) → C:/Users/...
            m = re.match(r"^(?:/mnt)?/([a-zA-Z])(/.*)?$", raw.replace("\\", "/"))
            if m:
                raw = f"{m.group(1).upper()}:{m.group(2) or '/'}"
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        # resolve() follows symlinks, so a link pointing outside the workspace is rejected too
        resolved = candidate.resolve()
        if resolved != self.root and not resolved.is_relative_to(self.root):
            rescued = self._rescue(raw)
            if rescued is None:
                raise WorkspaceError(self._outside(raw))
            resolved = rescued
        if not allow_secret and self.is_secret(resolved):
            raise WorkspaceError(f"Access denied: '{self.rel(resolved)}' may contain secrets and cannot be read.")
        if self.is_internal(resolved):
            raise WorkspaceError(
                f"'{self.rel(resolved)}' is the agent's own internal data (checkpoints, sessions, trash), not part of the "
                "project. It never helps with the task: work with the project's files instead."
            )
        return resolved

    def is_internal(self, path: Path) -> bool:
        """The agent's own bookkeeping under .lca/ (checkpoints, session database, trash). Small models wander into it
        and read checkpoint JSON instead of the project; config, memory, commands and skills stay accessible."""
        if path == self.root or not path.is_relative_to(self.root):
            return False
        parts = path.relative_to(self.root).parts
        if len(parts) < 2 or parts[0] != ".lca":
            return False
        return parts[1] in INTERNAL_LCA_ENTRIES or parts[1].startswith("sessions.db")

    def _rescue(self, raw: str) -> Path | None:
        """Small models often invent an absolute prefix ('/Users/Project/src/app.py'). If the end of the path names
        something that exists in the workspace ('src/app.py'), use that. The result is always inside the workspace."""
        parts = [p for p in Path(raw.replace("\\", "/")).parts if p not in ("/", "\\") and not p.endswith(":") and not p.endswith(":\\")]
        if parts and parts[-1] == self.root.name:
            return self.root
        for i in range(len(parts)):
            candidate = (self.root / Path(*parts[i:])).resolve()
            if candidate.is_relative_to(self.root) and candidate.exists():
                return candidate
        return None

    def _outside(self, raw: str) -> str:
        try:
            entries = sorted(self.root.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except OSError:
            entries = []
        shown = ", ".join([f"{e.name}{'/' if e.is_dir() else ''}" for e in entries if not e.name.startswith(".")][:15])
        return (
            f"'{raw}' is outside the workspace. Paths are relative to the workspace root: use '.' for the root itself, "
            f"and e.g. '{next((e.name for e in entries if e.is_dir() and not e.name.startswith('.')), 'src')}/...' for "
            f"files below it. Never guess absolute paths. The root contains: {shown or '(nothing)'}."
        )

    def rel(self, path: Path) -> str:
        if path == self.root:
            return "."
        return path.relative_to(self.root).as_posix()

    def is_secret(self, path: Path) -> bool:
        if path == self.root:
            return False
        return is_secret_path(path.relative_to(self.root).parts)

    # ------------------------------------------------------------ git

    @cached_property
    def git(self) -> str | None:
        return shutil.which("git")

    @cached_property
    def is_git_repo(self) -> bool:
        if not self.git:
            return False
        try:
            r = subprocess.run(
                [self.git, "-C", str(self.root), "rev-parse", "--is-inside-work-tree"],
                stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=10, check=False,
            )
            return r.returncode == 0 and r.stdout.strip() == "true"
        except (OSError, subprocess.SubprocessError):
            return False

    def run_git(self, *args: str, timeout: float = 30) -> str | None:
        if not self.is_git_repo or not self.git:
            return None
        try:
            r = subprocess.run(
                # stdin=DEVNULL: under `lca serve` stdin is the protocol pipe, and on Windows a child that inherits
                # it hangs while the main thread is blocked reading it
                [self.git, "-C", str(self.root), *args],
                stdin=subprocess.DEVNULL, capture_output=True, timeout=timeout, check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if r.returncode != 0:
            return None
        return r.stdout.decode("utf-8", "replace")

    # ------------------------------------------------------------ enumeration

    def iter_files(self, base: Path | None = None) -> Iterator[Path]:
        """Yield files under `base`, honoring .gitignore when in a git repo. Never yields secret files."""
        base = base or self.root
        rel_base = self.rel(base)
        listed = None
        if self.is_git_repo:
            pathspec = [] if rel_base == "." else ["--", rel_base]
            listed = self.run_git("ls-files", "-z", "--cached", "--others", "--exclude-standard", *pathspec, timeout=60)
        if listed is not None:
            count = 0
            for entry in listed.split("\0"):
                if not entry:
                    continue
                parts = PurePosixPath(entry).parts
                if any(p in DEFAULT_IGNORED_DIRS for p in parts[:-1]) or is_secret_path(parts):
                    continue
                path = self.root / entry
                if path.is_file():
                    yield path
                    count += 1
                    if count >= MAX_ENUMERATED_FILES:
                        return
            return
        yield from self._walk(base)

    def suggest(self, wanted: str, limit: int = 5) -> list[str]:
        """Closest existing paths to a path that does not exist (small models often guess paths)."""
        import difflib

        wanted_posix = wanted.replace("\\", "/").strip("/").lower()
        name = PurePosixPath(wanted_posix).name
        stem = name.split(".")[0]
        files: list[str] = []
        dirs: set[str] = set()
        for i, f in enumerate(self.iter_files()):
            if i >= 20_000:
                break
            rel = self.rel(f)
            files.append(rel)
            parent = str(PurePosixPath(rel).parent)
            while parent not in (".", ""):
                dirs.add(parent + "/")
                parent = str(PurePosixPath(parent).parent)
        candidates = files + sorted(dirs)
        exact_name = [c for c in candidates if PurePosixPath(c.rstrip("/")).name.lower() == name]
        same_stem = [c for c in candidates if stem and PurePosixPath(c.rstrip("/")).name.lower().split(".")[0] == stem]
        close = difflib.get_close_matches(wanted_posix, [c.lower() for c in candidates], n=limit, cutoff=0.5)
        by_lower = {c.lower(): c for c in candidates}
        out: list[str] = []
        for c in [*exact_name, *same_stem, *(by_lower[x] for x in close)]:
            if c not in out:
                out.append(c)
        return out[:limit]

    def not_found(self, wanted: str, kind: str = "File") -> str:
        hints = self.suggest(wanted)
        msg = f"{kind} not found: {wanted}."
        if hints:
            msg += " Did you mean: " + ", ".join(hints) + "?"
        else:
            msg += " Use list_files or search_code to find the right path."
        return msg

    def _walk(self, base: Path) -> Iterator[Path]:
        count = 0
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(d for d in dirnames if d not in DEFAULT_IGNORED_DIRS and not d.startswith("."))
            current = Path(dirpath)
            for name in sorted(filenames):
                path = current / name
                if self.is_secret(path):
                    continue
                yield path
                count += 1
                if count >= MAX_ENUMERATED_FILES:
                    return
