"""Permission rules: allow / ask / deny patterns for tool calls, checked in code before every call.

    run_command                 any command          Bash(...) is an alias
    run_command(npm test)       exactly `npm test`
    run_command(npm test:*)     `npm test` and anything after it (`npm test -- --watch`)
    run_command(git log *)      glob on the command
    edit_file(src/**)           file tools: glob on the workspace-relative path; Edit(...) = write/edit/delete
    read_file(secrets/**)       Read(...) = read_file, list_files, search_code, repo_overview

Precedence: deny > ask > allow. A command made of several parts (`a && b`, `a | b`, `a; b`) is allowed by rules only
if every part is allowed, and denied if any part is denied. Commands with redirection or substitution (`>`, `<`,
`$(...)`, backticks) are never auto-allowed by a rule. Commands the safety check marks as risky (git push, rm -rf, secrets)
always ask, and blocked commands never run, whatever the rules say.

Sources: [permissions] in ~/.lca/config.toml (all three lists), the project's .lca/config.toml (ask and deny only:
a repository must not be able to allow things), and the rules you choose with "Always allow" (kept per project in
~/.lca/permissions/, outside the repository).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from lca.config import lca_home

Kind = Literal["allow", "ask", "deny"]
KINDS: tuple[Kind, ...] = ("deny", "ask", "allow")  # precedence order

FILE_TOOLS = {"read_file", "list_files", "search_code", "repo_overview", "write_file", "edit_file", "multi_edit", "delete_file"}
ALIASES = {
    "bash": ["run_command"],
    "shell": ["run_command"],
    "edit": ["write_file", "edit_file", "multi_edit", "delete_file"],
    "write": ["write_file"],
    "delete": ["delete_file"],
    "read": ["read_file", "list_files", "search_code", "repo_overview"],
}
INTERPRETERS = {"node", "python", "python3", "py", "bash", "sh", "zsh", "powershell", "pwsh", "cmd", "ruby", "perl", "deno", "bun", "php"}
_RULE = re.compile(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:\((.*)\))?\s*", re.S)


@dataclass(frozen=True)
class Rule:
    tool: str
    pattern: str | None
    raw: str

    @classmethod
    def parse(cls, text: str) -> Rule:
        m = _RULE.fullmatch(text or "")
        if not m:
            raise ValueError(f"Invalid permission rule {text!r}: use Tool or Tool(pattern), e.g. run_command(npm test:*)")
        pattern = m.group(2).strip() if m.group(2) is not None else None
        tool = m.group(1)
        raw = f"{tool}({pattern})" if pattern else tool
        return cls(tool, pattern or None, raw)

    def tools(self) -> list[str]:
        return ALIASES.get(self.tool.lower(), [self.tool])

    def matches(self, tool: str, target: str | None, parts: list[str] | None = None) -> bool:
        """`target` is the command or path; `parts` the command's parts (for run_command)."""
        if tool not in self.tools():
            return False
        if self.pattern is None:
            return True
        if target is None:
            return False
        if tool == "run_command":
            return any(_command_matches(self.pattern, p) for p in (parts or [target]))
        return _path_regex(self.pattern).fullmatch(target) is not None


def _command_matches(pattern: str, command: str) -> bool:
    command = " ".join(command.split())
    pattern = " ".join(pattern.split())
    if pattern.endswith(":*"):
        prefix = pattern[:-2].rstrip()
        return command == prefix or command.startswith(prefix + " ")
    if "*" in pattern or "?" in pattern:
        return re.fullmatch(".*".join(re.escape(s) for s in pattern.split("*")).replace(r"\?", "."), command) is not None
    return command == pattern


def _path_regex(pattern: str) -> re.Pattern[str]:
    p = pattern.replace("\\", "/").removeprefix("./")
    if p.endswith("/"):
        p += "**"
    out = ""
    i = 0
    while i < len(p):
        if p.startswith("**/", i):
            out += "(?:.*/)?"
            i += 3
        elif p.startswith("**", i):
            out += ".*"
            i += 2
        elif p[i] == "*":
            out += "[^/]*"
            i += 1
        elif p[i] == "?":
            out += "[^/]"
            i += 1
        else:
            out += re.escape(p[i])
            i += 1
    return re.compile(out)


def split_command(command: str) -> tuple[list[str], bool]:
    """Split on &&, ||, ;, |, & and newlines outside quotes. Returns (parts, unsafe): unsafe when the command uses
    redirection or substitution, which a rule must never approve on the strength of its first word."""
    parts: list[str] = []
    buf = ""
    quote = ""
    unsafe = "$(" in command or "`" in command
    i = 0
    while i < len(command):
        c = command[i]
        if quote:
            buf += c
            if c == quote:
                quote = ""
        elif c in "'\"":
            quote = c
            buf += c
        elif c in "<>":
            unsafe = True
            buf += c
        elif command.startswith(("&&", "||"), i):
            parts.append(buf)
            buf = ""
            i += 1
        elif c in ";|&\n":
            parts.append(buf)
            buf = ""
        else:
            buf += c
        i += 1
    parts.append(buf)
    return [" ".join(p.split()) for p in parts if p.strip()], unsafe


def suggest_rule(command: str) -> str | None:
    """A reusable allow rule for a command the user approved, e.g. `npm run build -- --x` → run_command(npm run build:*).
    None when no rule would be both useful and safe (several parts, redirection, a bare interpreter)."""
    parts, unsafe = split_command(command)
    if unsafe or len(parts) != 1:
        return None
    tokens = parts[0].split()
    kept = [tokens[0]]
    for tok in tokens[1:3]:
        if tok in ("-m", "run", "exec", "x") or re.fullmatch(r"[A-Za-z][\w:-]*", tok):
            kept.append(tok)
            if tok not in ("-m", "run", "exec", "x"):
                break
        else:
            break
    if kept[-1] in ("-m", "run", "exec", "x"):
        kept.pop()
    if len(kept) == 1 and kept[0].lower().removesuffix(".exe") in INTERPRETERS:
        return None
    return f"run_command({' '.join(kept)}:*)"


@dataclass
class Match:
    kind: Kind
    rule: Rule
    source: str


@dataclass
class RuleSet:
    rules: dict[Kind, list[tuple[Rule, str]]] = field(default_factory=lambda: {"allow": [], "ask": [], "deny": []})

    def add(self, kind: Kind, rule: Rule, source: str) -> None:
        if not any(r.raw == rule.raw for r, _ in self.rules[kind]):
            self.rules[kind].append((rule, source))

    def remove(self, kind: Kind, raw: str) -> None:
        self.rules[kind] = [(r, s) for r, s in self.rules[kind] if r.raw != raw]

    def listing(self) -> list[dict[str, str]]:
        return [{"kind": k, "rule": r.raw, "source": s} for k in KINDS for r, s in self.rules[k]]

    def decide(self, tool: str, args: Any, change: Any = None) -> Match | None:
        """The rule that applies to this call (deny > ask > allow), or None."""
        target, parts, unsafe = _target(tool, args, change)
        for kind in ("deny", "ask"):
            for rule, source in self.rules[kind]:
                if rule.matches(tool, target, parts):  # for commands: any part
                    return Match(kind, rule, source)
        allows = self.rules["allow"]
        if tool == "run_command":
            # every part must be allowed, and nothing may hide in redirections or substitutions
            if unsafe or not parts:
                return None
            first: tuple[Rule, str] | None = None
            for p in parts:
                hit = next(((r, s) for r, s in allows if r.matches(tool, p, [p])), None)
                if hit is None:
                    return None
                first = first or hit
            return Match("allow", *first) if first else None
        for rule, source in allows:
            if rule.matches(tool, target, parts):
                return Match("allow", rule, source)
        return None


def _target(tool: str, args: Any, change: Any) -> tuple[str | None, list[str], bool]:
    if tool == "run_command":
        command = str(getattr(args, "command", "") or "")
        parts, unsafe = split_command(command)
        return command, parts, unsafe
    if tool in FILE_TOOLS:
        path = getattr(change, "path", None) or getattr(args, "path", None)
        if path is None:
            return None, [], False
        p = str(path).replace("\\", "/").removeprefix("./")
        return (p or "."), [], False
    return None, [], False


# ---------------------------------------------------------------- rules the user chose ("Always allow"), per project


class PermissionStore:
    """Rules saved from approvals and `lca permissions`, kept outside the repository so a project cannot plant them."""

    def __init__(self, workspace_root: Path) -> None:
        key = hashlib.sha256(str(workspace_root.resolve()).lower().encode()).hexdigest()[:16]
        self.path = lca_home() / "permissions" / f"{key}.json"
        self.root = workspace_root

    def load(self) -> dict[Kind, list[str]]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"allow": [], "ask": [], "deny": []}
        return {k: [str(r) for r in data.get(k, []) if isinstance(r, str)] for k in ("allow", "ask", "deny")}  # type: ignore[misc]

    def _save(self, data: dict[Kind, list[str]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"workspace": str(self.root), **data}, indent=2), encoding="utf-8")

    def add(self, kind: Kind, raw: str) -> str:
        rule = Rule.parse(raw)
        data = self.load()
        if rule.raw not in data[kind]:
            data[kind].append(rule.raw)
            self._save(data)
        return rule.raw

    def remove(self, kind: Kind, raw: str) -> bool:
        data = self.load()
        normalized = Rule.parse(raw).raw
        if normalized not in data[kind]:
            return False
        data[kind].remove(normalized)
        self._save(data)
        return True


def build_ruleset(config: dict[str, list[str]], store: PermissionStore | None, warnings: list[str]) -> RuleSet:
    """Rules from the config files (project allow rules were already dropped by lca.config) and the saved ones."""
    rs = RuleSet()

    def add_all(lists: dict[str, list[str]], source: str) -> None:
        for kind in KINDS:
            for raw in lists.get(kind, []) or []:
                try:
                    rs.add(kind, Rule.parse(str(raw)), source)
                except ValueError as e:
                    warnings.append(f"{source}: {e}")

    add_all(config, "config")
    if store is not None:
        add_all(store.load(), "saved")  # type: ignore[arg-type]
    return rs
