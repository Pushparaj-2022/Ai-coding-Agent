"""Hooks: your own shell commands that run at points in the agent's work, like Claude Code's hooks.

    [[hooks.PreToolUse]]                 # before a tool call; can block it
    matcher = "run_command|Edit"         # regex on the tool name (aliases: Bash, Edit, Write, Read); empty = all
    command = "python .lca/hooks/check.py"
    timeout = 60

Events: PreToolUse, PostToolUse, UserPromptSubmit, Stop, SessionStart. Each hook gets a JSON object on stdin
(`event`, `workspace`, `session_id`, plus `tool` / `arguments` / `output` / `prompt` depending on the event) and the
environment variables LCA_HOOK_EVENT and LCA_WORKSPACE. Exit codes:

    0      fine. For UserPromptSubmit and SessionStart, stdout is added as context for the agent; for PostToolUse,
           stdout is shown to the agent with the tool result.
    2      block: PreToolUse refuses the call, UserPromptSubmit refuses the message, Stop sends the agent back to
           work, PostToolUse reports a problem to the agent. stderr is the reason the agent sees.
    other  the hook failed; this is reported as a warning and the agent continues.

A hook can instead print JSON: {"decision": "block", "reason": "..."}.

Hooks from ~/.lca/config.toml always run. Hooks from a project's .lca/config.toml run only after you trust that exact
hook configuration with `lca hooks --trust` (a repository must not be able to run commands on your machine by itself);
when the project's hooks change, they need trusting again.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from lca.agent.trust import is_trusted, read_project_section, trust_section

log = logging.getLogger("lca.hooks")

HookEvent = Literal["PreToolUse", "PostToolUse", "UserPromptSubmit", "Stop", "SessionStart"]
EVENTS: tuple[str, ...] = ("PreToolUse", "PostToolUse", "UserPromptSubmit", "Stop", "SessionStart")
TOOL_ALIASES = {
    "run_command": ["Bash", "Shell"], "write_file": ["Write", "Edit"], "edit_file": ["Edit"], "multi_edit": ["Edit"], "delete_file": ["Edit", "Delete"],
    "read_file": ["Read"], "list_files": ["Read"], "search_code": ["Read", "Grep"], "repo_overview": ["Read"],
}
MAX_OUTPUT = 4000


@dataclass(frozen=True)
class Hook:
    event: str
    command: str
    matcher: str = ""
    timeout: int = 60
    source: str = "user"

    def matches(self, tool: str | None) -> bool:
        if not self.matcher or tool is None:
            return True
        names = [tool, *TOOL_ALIASES.get(tool, [])]
        try:
            return any(re.fullmatch(self.matcher, n) for n in names)
        except re.error:
            return False


@dataclass
class HookResult:
    blocked: bool = False
    reason: str = ""  # why it blocked (stderr or JSON reason)
    context: list[str] = field(default_factory=list)  # stdout of successful hooks
    warnings: list[str] = field(default_factory=list)


def parse_hooks(data: Any, source: str, warnings: list[str]) -> list[Hook]:
    hooks: list[Hook] = []
    if not isinstance(data, dict):
        return hooks
    for event, entries in data.items():
        if event not in EVENTS:
            warnings.append(f"Unknown hook event {event!r} in the {source} config (use {', '.join(EVENTS)}).")
            continue
        for entry in entries if isinstance(entries, list) else [entries]:
            if not isinstance(entry, dict) or not str(entry.get("command", "")).strip():
                warnings.append(f"A {event} hook in the {source} config has no command.")
                continue
            hooks.append(Hook(event, str(entry["command"]), str(entry.get("matcher", "") or ""), int(entry.get("timeout", 60) or 60), source))
    return hooks


def read_project_hooks(root: Path) -> Any:
    return read_project_section(root, "hooks")


def trust_project_hooks(root: Path) -> str | None:
    """Trust the project's current hook configuration. Returns its digest, or None when it has no hooks."""
    return trust_section(root, "hooks")


def load_hooks(root: Path, user_data: Any, warnings: list[str]) -> list[Hook]:
    hooks = parse_hooks(user_data, "user", warnings)
    project = read_project_hooks(root)
    if project:
        if is_trusted(root, "hooks", project):
            hooks += parse_hooks(project, "project", warnings)
        else:
            warnings.append("This project's .lca/config.toml defines hooks that are not trusted, so they do not run. "
                            "Review them, then run `lca hooks --trust` to enable them.")
    return hooks


class HookRunner:
    def __init__(self, hooks: list[Hook], workspace: Path, shell: Any = None) -> None:
        self.hooks = hooks
        self.workspace = workspace
        self.shell = shell  # lca.tools.terminal.Shell; hooks run in the same shell as the agent's commands

    def has(self, event: str) -> bool:
        return any(h.event == event for h in self.hooks)

    def run(self, event: str, payload: dict[str, Any], tool: str | None = None) -> HookResult:
        result = HookResult()
        for hook in (h for h in self.hooks if h.event == event and h.matches(tool)):
            self._run_one(hook, {"event": event, "workspace": str(self.workspace), **payload}, result)
            if result.blocked:
                break
        return result

    def _run_one(self, hook: Hook, payload: dict[str, Any], result: HookResult) -> None:
        env = {**os.environ, "LCA_HOOK_EVENT": hook.event, "LCA_WORKSPACE": str(self.workspace), "PYTHONIOENCODING": "utf-8"}
        argv: Any = self.shell.argv(hook.command) if self.shell is not None else hook.command
        try:
            proc = subprocess.run(
                argv, shell=self.shell is None, cwd=self.workspace, input=json.dumps(payload, default=str),
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=hook.timeout, env=env,
            )
        except subprocess.TimeoutExpired:
            result.warnings.append(f"{hook.event} hook timed out after {hook.timeout}s: {hook.command}")
            return
        except OSError as e:
            result.warnings.append(f"{hook.event} hook could not start ({e}): {hook.command}")
            return
        out, err = proc.stdout.strip()[:MAX_OUTPUT], proc.stderr.strip()[:MAX_OUTPUT]
        log.info("hook", extra={"fields": {"event": hook.event, "command": hook.command, "exit": proc.returncode}})
        decision = _json_decision(out)
        if proc.returncode == 2 or (decision and decision.get("decision") == "block"):
            result.blocked = True
            result.reason = (decision or {}).get("reason") or err or out or f"blocked by the hook `{hook.command}`"
            return
        if proc.returncode != 0:
            result.warnings.append(f"{hook.event} hook failed (exit {proc.returncode}): {hook.command}{': ' + err.splitlines()[0] if err else ''}")
            return
        if out and decision is None:
            result.context.append(out)
        elif decision and decision.get("context"):
            result.context.append(str(decision["context"])[:MAX_OUTPUT])


def _json_decision(out: str) -> dict | None:
    if not out.startswith("{"):
        return None
    try:
        data = json.loads(out)
    except ValueError:
        return None
    return data if isinstance(data, dict) and ("decision" in data or "context" in data) else None
