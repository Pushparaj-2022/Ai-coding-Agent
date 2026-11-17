"""Code search: ripgrep when installed, otherwise a pure-Python scan of the (gitignore-aware) file list."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from lca.security import SECRET_DIRS, SECRET_FILE_PATTERNS
from lca.tools.base import Tool, ToolContext, ToolError
from lca.tools.filesystem import _is_binary, glob_to_regex

PYTHON_SEARCH_SECONDS = 30
MAX_FILE_BYTES = 2_000_000


@dataclass
class FileHits:
    path: str
    lines: list[tuple[int, str, bool]] = field(default_factory=list)  # (line, text, is_match)


class SearchCodeArgs(BaseModel):
    pattern: str = Field(min_length=1, description="Regular expression (or plain text with fixed_string=true)")
    path: str = Field(default=".", description="Directory or file to search")
    glob: str | None = Field(default=None, description="Only search matching files, e.g. '*.py'")
    fixed_string: bool = Field(default=False, description="Treat pattern as literal text")
    case_sensitive: bool = False
    context_lines: int = Field(default=0, ge=0, le=5)
    max_results: int = Field(default=80, ge=1, le=500, description="Maximum matching lines")


class SearchCode(Tool):
    name = "search_code"
    description = (
        "Search file contents across the workspace (respects .gitignore). Returns matching lines grouped by file "
        "with line numbers. Use it to find definitions, usages and error messages."
    )
    Args = SearchCodeArgs

    def run(self, args: SearchCodeArgs, ctx: ToolContext) -> str:
        base = ctx.workspace.resolve(args.path)
        if not base.exists():
            raise ToolError(ctx.workspace.not_found(args.path, "Path"))
        rg = shutil.which("rg")
        hits, total, note = (self._ripgrep(rg, args, ctx) if rg else self._python(args, ctx))
        if not hits:
            return f"No matches for {args.pattern!r}." + (f" {note}" if note else "")
        out: list[str] = []
        for fh in hits:
            out.append(fh.path)
            prev = None
            for line_no, text, is_match in fh.lines:
                if prev is not None and line_no > prev + 1:
                    out.append("    --")
                out.append(f"  {line_no:>5}{':' if is_match else '-'} {text[:300]}")
                prev = line_no
        shown = sum(1 for fh in hits for *_, m in fh.lines if m)
        summary = f"\n{total} matching line(s) in {len(hits)} file(s)"
        if total > shown:
            summary += f"; showing {shown}. Narrow with 'path' or 'glob'"
        return "\n".join(out) + summary + (f". {note}" if note else ".")

    # ------------------------------------------------------------ ripgrep backend

    def _ripgrep(self, rg: str, args: SearchCodeArgs, ctx: ToolContext) -> tuple[list[FileHits], int, str]:
        ws = ctx.workspace
        cmd = [rg, "--json", "--max-columns", "400", "--max-filesize", "2M", "-C", str(args.context_lines)]
        cmd.append("-s" if args.case_sensitive else "-i")
        if args.fixed_string:
            cmd.append("-F")
        if args.glob:
            cmd += ["--glob", args.glob]
        for d in SECRET_DIRS:
            cmd += ["--glob", f"!**/{d}/**"]
        for p in SECRET_FILE_PATTERNS:
            cmd += ["--iglob", f"!{p}"]
        cmd += ["--", args.pattern, ws.rel(ws.resolve(args.path))]
        try:
            proc = subprocess.Popen(cmd, cwd=ws.root, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except OSError as e:
            raise ToolError(f"Could not start ripgrep: {e}") from e
        unregister = ctx.cancel.on_cancel(proc.kill)
        hits: dict[str, FileHits] = {}
        total = 0
        try:
            assert proc.stdout is not None
            for raw in proc.stdout:
                try:
                    ev = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if ev.get("type") not in ("match", "context"):
                    continue
                data = ev["data"]
                path = (data.get("path") or {}).get("text")
                text = ((data.get("lines") or {}).get("text") or "").rstrip("\r\n")
                if not path:
                    continue
                is_match = ev["type"] == "match"
                if is_match:
                    total += 1
                    if total > args.max_results:
                        continue
                elif total >= args.max_results:
                    continue
                fh = hits.setdefault(path, FileHits(path=path.replace("\\", "/")))
                fh.lines.append((data["line_number"], text, is_match))
            stderr = proc.stderr.read().decode("utf-8", "replace") if proc.stderr else ""
            code = proc.wait(timeout=60)
        finally:
            unregister()
            if proc.poll() is None:
                proc.kill()
        ctx.cancel.raise_if_cancelled()
        if code == 2 and not hits:
            raise ToolError(f"Search failed: {stderr.strip()[:400] or 'invalid pattern'}")
        return [h for h in hits.values() if any(m for *_, m in h.lines)], total, ""

    # ------------------------------------------------------------ python fallback

    def _python(self, args: SearchCodeArgs, ctx: ToolContext) -> tuple[list[FileHits], int, str]:
        ws = ctx.workspace
        flags = 0 if args.case_sensitive else re.I
        try:
            regex = re.compile(re.escape(args.pattern) if args.fixed_string else args.pattern, flags)
        except re.error as e:
            raise ToolError(f"Invalid regular expression: {e}") from e
        matcher = glob_to_regex(args.glob) if args.glob else None
        base = ws.resolve(args.path)
        files = [base] if base.is_file() else ws.iter_files(base)
        deadline = time.monotonic() + PYTHON_SEARCH_SECONDS
        hits: list[FileHits] = []
        total = 0
        note = ""
        for f in files:
            ctx.cancel.raise_if_cancelled()
            if time.monotonic() > deadline:
                note = f"Search stopped after {PYTHON_SEARCH_SECONDS}s (install ripgrep for fast search)"
                break
            if matcher and not matcher.match(f.relative_to(base).as_posix() if base.is_dir() else f.name):
                continue
            try:
                if f.stat().st_size > MAX_FILE_BYTES or _is_binary(f):
                    continue
                lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            match_idx = [i for i, line in enumerate(lines) if regex.search(line)]
            if not match_idx:
                continue
            fh = FileHits(path=ws.rel(f))
            wanted: dict[int, bool] = {}
            for i in match_idx:
                total += 1
                if total > args.max_results:
                    continue
                for j in range(max(0, i - args.context_lines), min(len(lines), i + args.context_lines + 1)):
                    wanted[j] = wanted.get(j, False) or j == i
            for j in sorted(wanted):
                fh.lines.append((j + 1, lines[j], wanted[j]))
            if fh.lines:
                hits.append(fh)
        return hits, total, note
