"""File-changing tools: write_file, edit_file, delete_file.

Rules that protect the user's work:
- An existing file must have been read in this run before it can be changed.
- If the file changed on disk since the agent read it (e.g. the user edited it), the change is refused.
- The change is computed first and shown as a diff for approval; it is re-checked right before writing.
- Deleted files are moved to .lca/trash/<timestamp>/ instead of being destroyed.
"""

from __future__ import annotations

import hashlib
import re
import shutil
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from lca.tools.base import FileChange, FileWriteTool, ToolContext, ToolError
from lca.tools.diff import diff_stats

MAX_EDIT_BYTES = 5_000_000


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def record_snapshot(ctx: ToolContext, rel: str, path: Path) -> None:
    try:
        ctx.snapshots[rel] = sha256(path.read_bytes())
    except OSError:
        pass


def _read_text(path: Path, rel: str) -> tuple[str, str]:
    """Returns (text with \\n line endings, original newline style)."""
    data = path.read_bytes()
    if len(data) > MAX_EDIT_BYTES:
        raise ToolError(f"{rel} is too large to edit ({len(data)} bytes).")
    if b"\0" in data[:8192]:
        raise ToolError(f"{rel} is a binary file and cannot be edited.")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as e:
        raise ToolError(f"{rel} is not valid UTF-8; refusing to edit it to avoid corrupting it.") from e
    newline = "\r\n" if "\r\n" in text else "\n"
    return text.replace("\r\n", "\n"), newline


def _require_fresh(ctx: ToolContext, path: Path, rel: str) -> None:
    snapshot = ctx.snapshots.get(rel)
    if snapshot is None:
        raise ToolError(f"Read {rel} with read_file before changing it, so you know its current contents.")
    if sha256(path.read_bytes()) != snapshot:
        raise ToolError(f"{rel} changed on disk since you read it (possibly edited by the user). Read it again first.")


def _check_parents(ctx: ToolContext, path: Path) -> None:
    """A new file's parent folders are created on write; fail early if one of them is an existing file."""
    for parent in path.parents:
        if parent == ctx.workspace.root or ctx.workspace.root not in parent.parents:
            return
        if parent.exists() and not parent.is_dir():
            raise ToolError(f"Cannot create {ctx.workspace.rel(path)}: {ctx.workspace.rel(parent)} is a file, not a folder.")


def _write(ctx: ToolContext, path: Path, rel: str, text: str, newline: str) -> None:
    _check_parents(ctx, path)
    data = (text.replace("\n", "\r\n") if newline == "\r\n" else text).encode("utf-8")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    except OSError as e:
        raise ToolError(f"Could not write {rel}: {e.strerror or e}") from e
    ctx.snapshots[rel] = sha256(data)
    ctx.files_changed.add(rel)


def _summary(verb: str, change: FileChange) -> str:
    diff = change.diff()
    added, removed = diff_stats(diff)
    shown = diff if len(diff) <= 3000 else diff[:3000] + "\n[... diff truncated ...]\n"
    return f"{verb} {change.path} (+{added} -{removed} lines).\n{shown}".rstrip()


def _verify_unchanged(change: FileChange, path: Path) -> None:
    """Re-check right before writing: the user may have edited the file while the approval was pending."""
    if change.action == "create":
        if path.exists():
            raise ToolError(f"{change.path} was created by someone else meanwhile. Read it and try again.")
        return
    if not path.exists():
        raise ToolError(f"{change.path} no longer exists.")
    if change.is_dir:
        return
    current, _ = _read_text(path, change.path)
    if current != change.before:
        raise ToolError(f"{change.path} changed while waiting for approval. Read it again and redo the change.")


# ---------------------------------------------------------------- write_file


class WriteFileArgs(BaseModel):
    path: str = Field(description="File path relative to the workspace root")
    content: str = Field(description="The complete new file content (no placeholders)")


class WriteFile(FileWriteTool):
    name = "write_file"
    description = (
        "Create a new file, or replace an existing file's entire content. Parent folders are created. "
        "For small changes to an existing file prefer edit_file. Existing files must be read first."
    )
    Args = WriteFileArgs

    def prepare(self, args: WriteFileArgs, ctx: ToolContext) -> FileChange:
        ws = ctx.workspace
        path = ws.resolve(args.path)
        rel = ws.rel(path)
        if path.is_dir():
            raise ToolError(f"{rel} is a directory.")
        after = args.content.replace("\r\n", "\n")
        if not path.exists():
            _check_parents(ctx, path)
            return FileChange(path=rel, action="create", before=None, after=after)
        _require_fresh(ctx, path, rel)
        before, newline = _read_text(path, rel)
        if before == after:
            raise ToolError(f"{rel} already has exactly this content; nothing to change.")
        return FileChange(path=rel, action="modify", before=before, after=after, newline=newline)

    def apply(self, change: FileChange, ctx: ToolContext) -> str:
        path = ctx.workspace.resolve(change.path)
        _verify_unchanged(change, path)
        _write(ctx, path, change.path, change.after or "", change.newline)
        return _summary("Created" if change.action == "create" else "Rewrote", change)


# ---------------------------------------------------------------- edit_file


class EditFileArgs(BaseModel):
    path: str = Field(description="File path relative to the workspace root")
    old_string: str = Field(description="Exact text to replace, copied from read_file output without the line-number prefix")
    new_string: str = Field(description="Replacement text")
    replace_all: bool = Field(default=False, description="Replace every occurrence instead of requiring a unique match")


_LINE_NO = re.compile(r"^\s*\d+\t")


def _strip_line_numbers(s: str) -> str:
    lines = s.split("\n")
    if lines and all(_LINE_NO.match(line) for line in lines if line.strip()):
        return "\n".join(_LINE_NO.sub("", line) for line in lines)
    return s


def _fuzzy_find(text: str, needle: str) -> str | None:
    """Match ignoring leading/trailing whitespace per line; returns the exact original span if unique."""
    want = [line.strip() for line in needle.split("\n")]
    while want and not want[0]:
        want.pop(0)
    while want and not want[-1]:
        want.pop()
    if not want:
        return None
    lines = text.split("\n")
    hits = [i for i in range(len(lines) - len(want) + 1) if all(lines[i + j].strip() == want[j] for j in range(len(want)))]
    if len(hits) != 1:
        return None
    return "\n".join(lines[hits[0] : hits[0] + len(want)])


def _occurrence_lines(text: str, needle: str, limit: int = 5) -> list[int]:
    out, start = [], 0
    while len(out) < limit:
        idx = text.find(needle, start)
        if idx < 0:
            break
        out.append(text.count("\n", 0, idx) + 1)
        start = idx + 1
    return out


class EditFile(FileWriteTool):
    name = "edit_file"
    description = (
        "Replace an exact piece of text in an existing file. old_string must match the current file exactly "
        "(including indentation) and be unique; include surrounding lines to make it unique. Read the file first."
    )
    Args = EditFileArgs

    def prepare(self, args: EditFileArgs, ctx: ToolContext) -> FileChange:
        rel, text, newline = _open_for_edit(ctx, args.path)
        after = _replace(text, rel, args.old_string, args.new_string, args.replace_all)
        return FileChange(path=rel, action="modify", before=text, after=after, newline=newline)

    def apply(self, change: FileChange, ctx: ToolContext) -> str:
        path = ctx.workspace.resolve(change.path)
        _verify_unchanged(change, path)
        _write(ctx, path, change.path, change.after or "", change.newline)
        return _summary("Edited", change)


def _open_for_edit(ctx: ToolContext, wanted: str) -> tuple[str, str, str]:
    """(relative path, text with \\n line endings, newline style) of an existing file the agent has read."""
    ws = ctx.workspace
    path = ws.resolve(wanted)
    rel = ws.rel(path)
    if not path.is_file():
        raise ToolError(ws.not_found(wanted) + " Use write_file to create a new file.")
    _require_fresh(ctx, path, rel)
    text, newline = _read_text(path, rel)
    return rel, text, newline


def _replace(text: str, rel: str, old_string: str, new_string: str, replace_all: bool, label: str = "old_string") -> str:
    old = old_string.replace("\r\n", "\n")
    new = new_string.replace("\r\n", "\n")
    if not old:
        raise ToolError(f"{label} is empty. Use write_file to replace a whole file.")
    if old == new:
        raise ToolError(f"{label} and its new_string are identical; nothing to change.")

    count = text.count(old)
    if count == 0:
        stripped = _strip_line_numbers(old)
        if stripped != old and text.count(stripped):
            old, new = stripped, _strip_line_numbers(new)
            count = text.count(old)
    if count == 0:
        fuzzy = _fuzzy_find(text, old)
        if fuzzy is not None:
            old, count = fuzzy, 1
    if count == 0:
        raise ToolError(
            f"{label} was not found in {rel}. It must match the current text exactly, including indentation. "
            "Read the file again and copy the lines you want to change."
        )
    if count > 1 and not replace_all:
        lines = ", ".join(str(n) for n in _occurrence_lines(text, old))
        raise ToolError(
            f"{label} occurs {count} times in {rel} (lines {lines}). Include more surrounding lines to make it unique, "
            "or set replace_all=true."
        )
    return text.replace(old, new) if replace_all else text.replace(old, new, 1)


# ---------------------------------------------------------------- multi_edit


class OneEdit(BaseModel):
    old_string: str = Field(description="Exact text to replace (as it is after the previous edits in this list)")
    new_string: str = Field(description="Replacement text")
    replace_all: bool = Field(default=False)


class MultiEditArgs(BaseModel):
    path: str = Field(description="File path relative to the workspace root")
    edits: list[OneEdit] = Field(min_length=1, max_length=30, description="Edits applied in order, each to the result of the previous one")


class MultiEdit(FileWriteTool):
    name = "multi_edit"
    description = (
        "Make several edits to one file in one step (one diff, one approval). Each edit works like edit_file and applies "
        "to the result of the previous one. If any edit fails, none is applied. Read the file first."
    )
    Args = MultiEditArgs

    def prepare(self, args: MultiEditArgs, ctx: ToolContext) -> FileChange:
        rel, text, newline = _open_for_edit(ctx, args.path)
        after = text
        for i, e in enumerate(args.edits, 1):
            after = _replace(after, rel, e.old_string, e.new_string, e.replace_all, label=f"edits[{i}].old_string")
        return FileChange(path=rel, action="modify", before=text, after=after, newline=newline)

    def apply(self, change: FileChange, ctx: ToolContext) -> str:
        path = ctx.workspace.resolve(change.path)
        _verify_unchanged(change, path)
        _write(ctx, path, change.path, change.after or "", change.newline)
        return _summary("Edited", change)


# ---------------------------------------------------------------- delete_file


class DeleteFileArgs(BaseModel):
    path: str = Field(description="File or folder to delete, relative to the workspace root")


class DeleteFile(FileWriteTool):
    name = "delete_file"
    description = "Delete a file or folder. It is moved to .lca/trash so the user can restore it."
    Args = DeleteFileArgs

    def prepare(self, args: DeleteFileArgs, ctx: ToolContext) -> FileChange:
        ws = ctx.workspace
        path = ws.resolve(args.path)
        rel = ws.rel(path)
        if path == ws.root:
            raise ToolError("Refusing to delete the workspace root.")
        if not path.exists():
            raise ToolError(f"Not found: {args.path}")
        if path.is_dir():
            files = sum(1 for p in path.rglob("*") if p.is_file())
            return FileChange(path=rel, action="delete", before=f"[folder with {files} file(s)]", after=None, is_dir=True)
        try:
            before, _ = _read_text(path, rel)
        except ToolError:
            before = f"[binary or non-text file, {path.stat().st_size} bytes]"
        return FileChange(path=rel, action="delete", before=before, after=None)

    def apply(self, change: FileChange, ctx: ToolContext) -> str:
        ws = ctx.workspace
        path = ws.resolve(change.path)
        if not path.exists():
            raise ToolError(f"{change.path} no longer exists.")
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        dest = ws.root / ".lca" / "trash" / stamp / change.path
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.rename(dest)
        except OSError:
            if path.is_dir():
                shutil.copytree(path, dest)
                shutil.rmtree(path)
            else:
                shutil.copy2(path, dest)
                path.unlink()
        ctx.snapshots.pop(change.path, None)
        ctx.files_changed.add(change.path)
        kind = "folder" if change.is_dir else "file"
        return f"Deleted {kind} {change.path}. A copy is kept in .lca/trash/{stamp}/{change.path}."
