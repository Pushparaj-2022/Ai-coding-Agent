"""Read-only filesystem tools: repository overview, file listing and ranged file reads."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from pydantic import BaseModel, Field

from lca.tools.base import Tool, ToolContext, ToolError

MAX_READ_LINES = 1000
MAX_READ_CHARS = 60_000

LANGUAGES = {
    ".py": "Python", ".js": "JavaScript", ".mjs": "JavaScript", ".cjs": "JavaScript", ".jsx": "JavaScript",
    ".ts": "TypeScript", ".tsx": "TypeScript", ".java": "Java", ".kt": "Kotlin", ".go": "Go", ".rs": "Rust",
    ".c": "C", ".h": "C/C++", ".cpp": "C++", ".cc": "C++", ".hpp": "C++", ".cs": "C#", ".rb": "Ruby",
    ".php": "PHP", ".swift": "Swift", ".scala": "Scala", ".dart": "Dart", ".lua": "Lua", ".sql": "SQL",
    ".sh": "Shell", ".ps1": "PowerShell", ".html": "HTML", ".css": "CSS", ".scss": "SCSS", ".vue": "Vue",
    ".svelte": "Svelte", ".md": "Markdown", ".json": "JSON", ".yaml": "YAML", ".yml": "YAML", ".toml": "TOML",
}

KEY_FILES = {
    "README.md", "README", "AGENTS.md", "LCA.md", "CLAUDE.md", "package.json", "pyproject.toml", "setup.py",
    "requirements.txt", "Cargo.toml", "go.mod", "pom.xml", "build.gradle", "build.gradle.kts", "Makefile",
    "Dockerfile", "docker-compose.yml", "compose.yaml", "tsconfig.json", "vite.config.ts", "next.config.js",
    "pytest.ini", "tox.ini", "Gemfile", "composer.json", ".github",
}


def glob_to_regex(pattern: str) -> re.Pattern[str]:
    """Glob with ** support. Patterns without '/' match the file name in any directory."""
    p = pattern.replace("\\", "/").removeprefix("./")
    if "/" not in p:
        p = "**/" + p
    out = []
    i = 0
    while i < len(p):
        ch = p[i]
        if ch == "*":
            if p[i : i + 3] == "**/":
                out.append("(?:.*/)?")
                i += 3
                continue
            if p[i : i + 2] == "**":
                out.append(".*")
                i += 2
                continue
            out.append("[^/]*")
        elif ch == "?":
            out.append("[^/]")
        elif ch == "{" and "}" in p[i:]:
            end = p.index("}", i)
            out.append("(?:" + "|".join(re.escape(x) for x in p[i + 1 : end].split(",")) + ")")
            i = end
        else:
            out.append(re.escape(ch))
        i += 1
    return re.compile("^" + "".join(out) + "$", re.I)


def _is_binary(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            return b"\0" in f.read(8192)
    except OSError:
        return False


class RepoOverviewArgs(BaseModel):
    path: str = Field(default=".", description="Directory to summarize, relative to the workspace root")
    depth: int = Field(default=2, ge=1, le=4, description="How many directory levels to show")


class RepoOverview(Tool):
    name = "repo_overview"
    description = (
        "Summarize the repository: directory structure with file counts, languages, key config files and git status. "
        "Call this first when you do not know the project yet."
    )
    Args = RepoOverviewArgs

    def run(self, args: RepoOverviewArgs, ctx: ToolContext) -> str:
        ws = ctx.workspace
        base = ws.resolve(args.path)
        if not base.is_dir():
            raise ToolError(f"Not a directory: {args.path}")
        langs: Counter[str] = Counter()
        dir_counts: Counter[str] = Counter()
        top_files: list[str] = []
        key_files: set[str] = set()
        total = 0
        for f in ws.iter_files(base):
            ctx.cancel.raise_if_cancelled()
            total += 1
            rel_parts = f.relative_to(base).parts
            if lang := LANGUAGES.get(f.suffix.lower()):
                langs[lang] += 1
            if f.name in KEY_FILES or (rel_parts and rel_parts[0] in KEY_FILES):
                key_files.add(ws.rel(f) if len(rel_parts) <= 2 else rel_parts[0])
            if len(rel_parts) == 1:
                top_files.append(rel_parts[0])
            for level in range(1, min(len(rel_parts), args.depth + 1)):
                dir_counts["/".join(rel_parts[:level])] += 1

        lines = [f"Repository: {ws.rel(base) if base != ws.root else ws.root.name}"]
        if ws.is_git_repo:
            branch = (ws.run_git("rev-parse", "--abbrev-ref", "HEAD") or "?").strip()
            status = ws.run_git("status", "--porcelain") or ""
            changed = [line for line in status.splitlines() if line.strip()]
            modified = sum(1 for line in changed if not line.startswith("??"))
            untracked = sum(1 for line in changed if line.startswith("??"))
            lines.append(f"Git: branch {branch}, {modified} modified, {untracked} untracked")
        else:
            lines.append("Git: not a git repository")
        lines.append(f"Files: {total}")
        if langs:
            lines.append("Languages: " + ", ".join(f"{k} {v}" for k, v in langs.most_common(8)))
        if key_files:
            lines.append("Key files: " + ", ".join(sorted(key_files)[:25]))
        lines.append("Structure:")
        shown = 0
        for d in sorted(dir_counts):
            depth = d.count("/")
            lines.append(f"{'  ' * (depth + 1)}{d.rsplit('/', 1)[-1]}/  ({dir_counts[d]} files)")
            shown += 1
            if shown >= 120:
                lines.append("  ... (more directories; call repo_overview on a subdirectory)")
                break
        for name in sorted(top_files)[:40]:
            lines.append(f"  {name}")
        if len(top_files) > 40:
            lines.append(f"  ... and {len(top_files) - 40} more files at this level")
        return "\n".join(lines)


class ListFilesArgs(BaseModel):
    path: str = Field(default=".", description="Directory to list, relative to the workspace root")
    glob: str | None = Field(default=None, description="Filter, e.g. '*.py', 'src/**/*.ts', '*.{js,json}'")
    max_results: int = Field(default=200, ge=1, le=1000)


class ListFiles(Tool):
    name = "list_files"
    description = "List files under a directory (recursively, respecting .gitignore), optionally filtered by a glob."
    Args = ListFilesArgs

    def run(self, args: ListFilesArgs, ctx: ToolContext) -> str:
        ws = ctx.workspace
        base = ws.resolve(args.path)
        if not base.exists():
            raise ToolError(ws.not_found(args.path, "Directory"))
        if base.is_file():
            return ws.rel(base)
        matcher = glob_to_regex(args.glob) if args.glob else None
        results: list[str] = []
        total = 0
        for f in ws.iter_files(base):
            ctx.cancel.raise_if_cancelled()
            rel_to_base = f.relative_to(base).as_posix()
            if matcher and not matcher.match(rel_to_base):
                continue
            total += 1
            if len(results) < args.max_results:
                results.append(ws.rel(f))
        if not results:
            return "No files matched." if matcher else "Directory is empty."
        out = "\n".join(results)
        if total > len(results):
            out += f"\n... and {total - len(results)} more files ({total} total). Narrow with 'path' or 'glob'."
        return out


class ReadFileArgs(BaseModel):
    path: str = Field(description="File path relative to the workspace root")
    start_line: int = Field(default=1, ge=1, description="First line to read (1-based)")
    end_line: int | None = Field(default=None, ge=1, description="Last line to read (inclusive). Default: start_line + 399")


class ReadFile(Tool):
    name = "read_file"
    description = (
        "Read a text file with line numbers. Reads at most 1000 lines per call; use start_line/end_line "
        "to read large files in parts. The line-number prefix is not part of the file."
    )
    Args = ReadFileArgs

    def run(self, args: ReadFileArgs, ctx: ToolContext) -> str:
        ws = ctx.workspace
        path = ws.resolve(args.path)
        if not path.exists():
            raise ToolError(ws.not_found(args.path))
        if path.is_dir():
            raise ToolError(f"{args.path} is a directory. Use list_files.")
        if _is_binary(path):
            raise ToolError(f"{args.path} is a binary file.")
        start = args.start_line
        end = args.end_line or start + 399
        if end < start:
            raise ToolError("end_line must be greater than or equal to start_line.")
        end = min(end, start + MAX_READ_LINES - 1)

        selected: list[str] = []
        total = 0
        chars = 0
        cut_at: int | None = None
        with path.open("r", encoding="utf-8", errors="replace", newline=None) as f:
            for total, line in enumerate(f, start=1):
                if start <= total <= end and cut_at is None:
                    text = line.rstrip("\n")
                    if len(text) > 2000:  # minified bundles, data blobs
                        text = text[:2000] + f" …[line truncated, {len(text)} chars]"
                    chars += len(text)
                    if chars > MAX_READ_CHARS:
                        cut_at = total - 1
                        continue
                    selected.append(f"{total:>6}\t{text}")
                if total % 5000 == 0:
                    ctx.cancel.raise_if_cancelled()
        rel = ws.rel(path)
        ctx.files_read.add(rel)
        from lca.tools.edit import record_snapshot  # local import: edit imports this module's siblings

        record_snapshot(ctx, rel, path)
        if total == 0:
            return f"{rel} is empty."
        if start > total:
            raise ToolError(f"start_line {start} is past the end of {rel} ({total} lines).")
        last = cut_at if cut_at is not None else min(end, total)
        header = f"{rel} (lines {start}-{last} of {total})"
        footer = ""
        if last < total:
            footer = f"\n[{total - last} more lines; read again with start_line={last + 1}]"
        return header + "\n" + "\n".join(selected) + footer
