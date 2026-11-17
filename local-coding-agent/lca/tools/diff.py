"""Unified diffs for reviewing file changes."""

from __future__ import annotations

import difflib


def unified_diff(before: str | None, after: str | None, path: str, context: int = 3) -> str:
    a = (before or "").splitlines(keepends=True)
    b = (after or "").splitlines(keepends=True)
    lines = difflib.unified_diff(
        a,
        b,
        fromfile=f"a/{path}" if before is not None else "/dev/null",
        tofile=f"b/{path}" if after is not None else "/dev/null",
        n=context,
    )
    return "".join(line if line.endswith("\n") else line + "\n" for line in lines)


def diff_stats(diff: str) -> tuple[int, int]:
    added = removed = 0
    for line in diff.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    return added, removed
