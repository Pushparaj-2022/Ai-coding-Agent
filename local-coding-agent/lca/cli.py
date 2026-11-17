"""Command-line interface: `lca run`, `lca build`, `lca chat`, `lca serve`, `lca doctor`, `lca tools`."""

from __future__ import annotations

import atexit
import json
import os
import platform
import shutil
import sys
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar

import typer
from pydantic import BaseModel

from lca import __version__
from lca.agent.build import BuildPipeline
from lca.agent.events import AgentEvent, AgentResult, BuildFinished, Diagnostic
from lca.agent.loop import AgentLoop
from lca.graph.agent_graph import Asker, DiagnosticsSource
from lca.agent.permissions import READ_ONLY_MODES, ApprovalRequest, ApprovalResult, Approver, Mode, PermissionPolicy
from lca.agent.mentions import expand_mentions
from lca.agent.prompts import build_system_prompt, git_context
from lca.agent.statusline import StatusLine
from lca.images import MAX_IMAGES_PER_MESSAGE, image_data_url, valid_data_urls
from lca.agent.commands import expand_input, load_skills, slash_items
from lca.agent.hooks import Hook, HookRunner, load_hooks, read_project_hooks, trust_project_hooks
from lca.agent.trust import read_project_section, trust_section
from lca.agent.rules import KINDS, PermissionStore, Rule, RuleSet, build_ruleset
from lca.memory.checkpoints import CheckpointStore
from lca.cancel import CancelToken
from lca.config import ConfigError, Settings, lca_home, load_settings
from lca.logging_setup import setup_logging
from lca.memory.instructions import load_instructions
from lca.messages import Message
from lca.providers import create_provider
from lca.providers.base import Provider, ServerInfo, ToolSpec
from lca.graph.agent_graph import transcript
from lca.memory.sessions import SessionStore
from lca.memory.store import MemoryStore
from lca.mcp import McpError, McpManager, load_mcp_configs
from lca.tools import INTERACTION_TOOLS, MEMORY_TOOLS, READ_TOOLS, WEB_TOOLS, ToolRegistry, Workspace, WorkspaceError, default_registry
from lca.tools.interaction import MARKS

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Local-first coding agent.")

# Windows consoles and pipes often default to a legacy code page; never let output encoding break a run.
for _stream in (sys.stdout, sys.stderr, sys.stdin):
    if hasattr(_stream, "reconfigure") and (getattr(_stream, "encoding", "") or "").lower().replace("-", "") != "utf8":
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass

EXIT_CODES = {"completed": 0, "blocked": 3, "max_iterations": 4, "cancelled": 130, "error": 1}
FALLBACK_CONTEXT = 8192
T = TypeVar("T")


class Output(str, Enum):
    text = "text"
    json = "json"


class SetupError(Exception):
    pass


# ---------------------------------------------------------------------- rendering


def _err(msg: str) -> None:
    typer.secho(msg, fg=typer.colors.RED, err=True)


def _warn(msg: str) -> None:
    typer.secho(msg, fg=typer.colors.YELLOW, err=True)


def _arg_summary(args: dict[str, Any]) -> str:
    parts = []
    for k, v in args.items():
        if k in ("content", "old_string", "new_string", "description"):
            parts.append(f"{k}=<{len(str(v))} chars>")
            continue
        s = json.dumps(v) if not isinstance(v, str) else v
        parts.append(s if k in ("path", "pattern") else f"{k}={s}")
    text = " ".join(parts)
    return text if len(text) <= 100 else text[:99] + "…"


class TextRenderer:
    def __init__(self, verbose: bool) -> None:
        self.verbose = verbose
        self.background_hint = False  # set by lca chat when Ctrl+B works
        self.at_line_start = True
        self.lock = threading.Lock()

    def _write(self, s: str) -> None:
        typer.echo(s, nl=False)  # strips color codes when stdout is not a terminal
        sys.stdout.flush()
        if s:
            self.at_line_start = s.endswith("\n")

    def _line(self, s: str) -> None:
        if not self.at_line_start:
            self._write("\n")
        self._write(s + "\n")

    def __call__(self, e: AgentEvent) -> None:
        with self.lock:
            self._render(e)

    def _render(self, e: AgentEvent) -> None:
        if e.type == "text_delta":
            self._write(e.text)
        elif e.type == "reasoning_delta" and self.verbose:
            self._write(typer.style(e.text, dim=True))
        elif e.type == "turn_started" and self.verbose:
            self._line(typer.style(f"── turn {e.iteration}", dim=True))
        elif e.type == "tool_call_started":
            who = typer.style(f"[{e.agent}] ", fg=typer.colors.MAGENTA) if e.agent else ""
            self._line(who + typer.style("● ", fg=typer.colors.CYAN) + typer.style(e.name, bold=True) + " " + typer.style(_arg_summary(e.arguments), dim=True))
            if e.name == "run_command" and not e.arguments.get("background") and self.background_hint:
                self._line(typer.style("  (Ctrl+B moves it to the background)", dim=True))
        elif e.type == "stage_started":
            who = f"  · {e.agent} agent" if e.agent else ""
            self._line("\n" + typer.style(f"━━ {e.title}", fg=typer.colors.CYAN, bold=True) + typer.style(who, dim=True))
        elif e.type == "build_finished":
            color = {"passed": typer.colors.GREEN, "cancelled": typer.colors.YELLOW}.get(e.status, typer.colors.RED)
            self._line("\n" + typer.style(f"━━ Build {e.status} in {e.duration_s}s", fg=color, bold=True))
            self._line(f"   {e.summary}")
            if e.files_changed:
                self._line(typer.style(f"   files changed: {', '.join(e.files_changed)}", fg=typer.colors.CYAN))
            for c in e.commands[-8:]:
                ok = c.get("exit_code") == 0
                self._line(f"   ran: {c['command']}  [{typer.style('ok', fg=typer.colors.GREEN) if ok else typer.style('exit ' + str(c.get('exit_code')), fg=typer.colors.RED)}]")
        elif e.type == "run_finished" and e.agent:
            pass  # sub-agent runs inside /build: the build summary covers them
        elif e.type == "tool_output_delta" and self.verbose:
            self._write(typer.style(e.text, dim=True))
        elif e.type == "tool_call_finished":
            text = f"  ⎿ {e.summary[:150]}"
            self._line(typer.style(text, fg=typer.colors.RED) if not e.ok else typer.style(text, dim=True))
        elif e.type == "notice":
            if e.quiet:
                if self.verbose:
                    self._line(typer.style(f"  ({e.message})", dim=True))
            else:
                self._line(typer.style(f"! {e.message}", fg=typer.colors.YELLOW))
        elif e.type == "mode":
            self._line(typer.style(f"Plan approved: permission mode is now {e.mode}.", fg=typer.colors.GREEN))
        elif e.type == "plan_updated":
            self._line(typer.style("Plan", fg=typer.colors.CYAN, bold=True))
            for i, s in enumerate(e.steps, 1):
                self._line(typer.style(f"  {MARKS[s.status]} {i}. {s.step}", dim=s.status in ("done", "skipped")))
        elif e.type == "run_finished":
            self._summary(e.result)

    def _summary(self, r: AgentResult) -> None:
        color = {"completed": typer.colors.GREEN, "cancelled": typer.colors.YELLOW}.get(r.status, typer.colors.RED)
        failed = sum(1 for t in r.tool_calls if not t.ok)
        self._line("")
        self._line(typer.style(f"── {r.status} after {r.iterations} iteration(s) in {r.duration_s}s", fg=color, bold=True))
        if r.status != "completed" and r.final_message:
            self._line(typer.style(f"   {r.final_message}", fg=color))
        self._line(typer.style(f"   tool calls executed: {len(r.tool_calls)} ({failed} failed)", dim=True))
        if r.files_changed:
            self._line(typer.style(f"   files changed: {', '.join(r.files_changed)}", fg=typer.colors.CYAN))
        commands = [t for t in r.tool_calls if t.name == "run_command" and t.exit_code is not None]
        for t in commands[-6:]:
            mark = typer.style("ok", fg=typer.colors.GREEN) if t.exit_code == 0 else typer.style(f"exit {t.exit_code}", fg=typer.colors.RED)
            self._line(f"   ran: {t.arguments.get('command', '')}  [{mark}]")
        rep = r.report
        if rep.verification.status != "not_needed":
            v = rep.verification
            color = {"passed": typer.colors.GREEN, "failed": typer.colors.RED}.get(v.status, typer.colors.YELLOW)
            self._line(typer.style(f"   verification: {v.status}", fg=color) + (f" ({v.note})" if v.note else ""))
        for d in rep.problems[:10]:
            self._line(typer.style(f"   problem: {d.render()}", fg=typer.colors.RED))
        for item in rep.remaining:
            self._line(typer.style(f"   remaining: {item}", fg=typer.colors.YELLOW))
        for item in rep.needs_approval:
            self._line(typer.style(f"   needs your approval: {item}", fg=typer.colors.YELLOW))
        if r.files_read:
            files = ", ".join(r.files_read[:12]) + (f" (+{len(r.files_read) - 12} more)" if len(r.files_read) > 12 else "")
            self._line(typer.style(f"   files read: {files}", dim=True))
        if r.usage.prompt_tokens or r.usage.completion_tokens:
            self._line(typer.style(f"   tokens: {r.usage.prompt_tokens} prompt, {r.usage.completion_tokens} completion", dim=True))


def _json_renderer(e: AgentEvent) -> None:
    sys.stdout.write(e.model_dump_json() + "\n")
    sys.stdout.flush()


def _terminal_approver(renderer: TextRenderer) -> Approver:
    def approve(req: ApprovalRequest) -> ApprovalResult:
        if req.plan is not None:
            return approve_plan(req.plan)
        with renderer.lock:
            renderer._line(typer.style(f"? {req.description}", fg=typer.colors.YELLOW, bold=True))
            if req.command:
                renderer._line(typer.style(f"    $ {req.command}", bold=True))
            if req.change is not None:
                diff = req.change.diff().splitlines()
                for line in diff[:80]:
                    color = typer.colors.GREEN if line.startswith("+") else typer.colors.RED if line.startswith("-") else None
                    renderer._line(typer.style("    " + line, fg=color) if color else typer.style("    " + line, dim=True))
                if len(diff) > 80:
                    renderer._line(typer.style(f"    ... {len(diff) - 80} more diff lines", dim=True))
            try:
                remember = "" if req.level == "sensitive" else (" / [a]ll commands this session" if req.command else " / [a]ll edits this session")
                if req.suggested_rule:
                    remember += f" / [p] always allow {req.suggested_rule} in this project"
                answer = input(f"  Allow? [y]es{remember} / [n]o / or type instructions: ").strip()
            except EOFError:
                answer = "n"
            renderer.at_line_start = True
        low = answer.lower()
        if low in ("y", "yes", ""):
            return ApprovalResult(True)
        if low == "p" and req.suggested_rule:
            return ApprovalResult(True, always=True)
        if low in ("a", "all", "always"):
            return ApprovalResult(True, remember=True)
        if low in ("n", "no"):
            return ApprovalResult(False)
        return ApprovalResult(False, feedback=answer)

    def approve_plan(plan: str) -> ApprovalResult:
        with renderer.lock:
            renderer._line(typer.style("━━ Plan for approval", fg=typer.colors.CYAN, bold=True))
            for line in plan.strip().splitlines():
                renderer._line("  " + line)
            try:
                answer = input("  Implement it? [y]es, approve each edit / [a]uto-accept edits / [n]o / or type changes: ").strip()
            except EOFError:
                answer = "n"
            renderer.at_line_start = True
        low = answer.lower()
        if low in ("y", "yes", ""):
            return ApprovalResult(True, mode="ask")
        if low in ("a", "auto"):
            return ApprovalResult(True, mode="auto-edit")
        if low in ("n", "no"):
            return ApprovalResult(False)
        return ApprovalResult(False, feedback=answer)

    return approve


def _terminal_asker(renderer: TextRenderer) -> Asker:
    def ask(question: str, options: list[str]) -> str:
        with renderer.lock:
            renderer._line(typer.style(f"? {question}", fg=typer.colors.CYAN, bold=True))
            for i, o in enumerate(options, 1):
                renderer._line(f"    {i}. {o}")
            try:
                answer = input(f"  Answer{' (number or text)' if options else ''}: ").strip()
            except EOFError:
                answer = ""
            renderer.at_line_start = True
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return options[int(answer) - 1]
        return answer

    return ask


# ---------------------------------------------------------------------- setup


@dataclass
class Session:
    settings: Settings
    provider: Provider
    workspace: Workspace
    info: ServerInfo
    context_window: int
    warnings: list[str]
    rules: RuleSet
    permission_store: PermissionStore
    hooks: list[Hook] = field(default_factory=list)
    mcp: McpManager | None = None

    def policy(self, mode: Mode, approver: Approver | None = None) -> PermissionPolicy:
        return PermissionPolicy(mode, approver, self.rules, self.permission_store)


def _setup(root: Path, overrides: dict[str, Any]) -> Session:
    """Validate config, workspace and model server. Raises SetupError with a user-facing message."""
    try:
        workspace = Workspace(root)
        settings, warnings = load_settings(workspace.root, overrides)
    except (ConfigError, WorkspaceError) as e:
        raise SetupError(str(e)) from e
    setup_logging(lca_home() / "logs", settings.log_level)
    provider = create_provider(settings.provider)
    if not provider.is_local:
        if not settings.allow_remote:
            raise SetupError(
                f"Refusing to use the non-local model server {settings.provider.base_url}.\n"
                "Prompts, file contents and tool output would leave this machine. To allow it, pass --allow-remote, "
                "set LCA_ALLOW_REMOTE=1, or set allow_remote = true in ~/.lca/config.toml."
            )
        warnings.append(f"Remote inference enabled: code and tool output will be sent to {settings.provider.base_url}")
    info = provider.server_info()
    if not info.reachable:
        hint = (
            "Start it with e.g.:  llama-server -m <model.gguf> --port 8080 --jinja -ngl 99"
            if settings.provider.kind == "openai"
            else "Start it with:  ollama serve"
        )
        raise SetupError(f"Cannot reach the model server at {settings.provider.base_url} ({info.detail}).\n{hint}")
    if not settings.provider.model and info.models:
        settings.provider.model = info.models[0]
    context_window = settings.provider.context_window or info.context_window or FALLBACK_CONTEXT
    if settings.provider.vision is None:  # auto: what the server reports; servers that do not say get the benefit of the doubt
        settings.provider.vision = info.vision if info.vision is not None else True
    store = PermissionStore(workspace.root)
    rules = build_ruleset(settings.permissions.model_dump(), store, warnings)
    hooks = load_hooks(workspace.root, settings.hooks, warnings)
    mcp = _start_mcp(workspace.root, settings, warnings)
    return Session(settings, provider, workspace, info, context_window, warnings, rules, store, hooks, mcp)


def _start_mcp(root: Path, settings: Settings, warnings: list[str]) -> McpManager | None:
    configs = load_mcp_configs(root, settings.mcp, warnings)
    if not configs:
        return None
    manager = McpManager(configs, root)
    manager.start()  # in the background; tools appear as servers get ready
    atexit.register(manager.stop)
    return manager


def _setup_or_exit(root: Path, overrides: dict[str, Any], *, quiet: bool) -> Session:
    try:
        s = _setup(root, overrides)
    except SetupError as e:
        _err(str(e))
        raise typer.Exit(2) from e
    for w in s.warnings:
        _warn(f"! {w}")
    if not quiet:
        shown_model = Path(s.settings.provider.model).name if s.settings.provider.model else "(default model)"
        typer.secho(
            f"lca {__version__} · {shown_model} · {s.settings.provider.base_url} · ctx {s.context_window} · {s.workspace.root}",
            dim=True,
            err=True,
        )
    return s


def _registry(s: Session, memory: bool = False) -> ToolRegistry:
    a = s.settings.agent
    return default_registry(a.tool_output_chars, a.shell, a.passthrough_env, memory=memory, interaction=True, web=s.settings.web)


def _make_agent(
    s: Session,
    events: Callable[[AgentEvent], None],
    policy: PermissionPolicy,
    *,
    role: str | None = None,
    registry: ToolRegistry | None = None,
    max_iterations: int | None = None,
    sessions: SessionStore | None = None,
    thread_id: str | None = None,
    asker: Asker | None = None,
    diagnostics: DiagnosticsSource | None = None,
    checkpoints: CheckpointStore | None = None,
) -> AgentLoop:
    registry = registry or _registry(s, memory=role is None)
    memory_tools = [n for n in MEMORY_TOOLS if registry.get(n) is not None]
    instructions = load_instructions(s.workspace.root)
    run_tool = registry.get("run_command")
    shell = run_tool.shell.label if run_tool is not None and hasattr(run_tool, "shell") else ""

    def names() -> list[str]:
        """The tools for the current permission mode (it can change during a session, e.g. after plan approval)."""
        if s.mcp is not None:  # servers that became ready since the last run
            for tool in s.mcp.tools():
                if registry.get(tool.name) is None:
                    registry.register(tool)
        read_only = policy.mode in READ_ONLY_MODES
        web = [n for n in WEB_TOOLS if registry.get(n) is not None and policy.mode == Mode.PLAN]  # planning may read the web
        # read-only and plan modes write nothing, memory included (a review must not store its guesses as facts)
        base = [*READ_TOOLS, "run_command", *web] if read_only else [n for n in registry.names if n not in INTERACTION_TOOLS]
        # planning is for the chat agent (build specialists get a plan from the planner); questions need someone to
        # answer them, diagnostics need an editor, and sub-agents do not start sub-agents of their own
        unavailable = {
            "update_plan": role is not None or read_only,
            "ask_user": role is not None or asker is None,
            "get_diagnostics": diagnostics is None,
            "task": role is not None,
            "exit_plan_mode": role is not None or policy.mode != Mode.PLAN,
            "load_skill": not load_skills(s.workspace.root),
            "view_image": s.settings.provider.vision is False,
        }
        return base + [n for n in INTERACTION_TOOLS if registry.get(n) is not None and not unavailable[n]]

    git_cache: dict[str, Any] = {"at": 0.0, "text": ""}

    def git() -> str:
        # the prompt is rebuilt for every model call; the repository state is refreshed at most every 15 seconds
        if role is None and time.monotonic() - git_cache["at"] > 15:
            git_cache.update(at=time.monotonic(), text=git_context(s.workspace))
        return git_cache["text"]

    def prompt() -> str:
        return build_system_prompt(
            s.workspace, names(), instructions, read_only=policy.mode in READ_ONLY_MODES, shell=shell, role=role,
            plan_mode=policy.mode == Mode.PLAN and role is None,
            skills=[(k.name, k.description) for k in load_skills(s.workspace.root).values()],
            git=git(),
        )

    def spawn(agent_type: str) -> AgentLoop:
        """task tool: explore and plan agents only read; a general agent works under the same permissions as the user's."""
        sub_role = SUBAGENT_ROLES[agent_type]
        sub_policy = policy if agent_type == "general" else PermissionPolicy(Mode.READ_ONLY, rules=s.rules)
        return _make_agent(
            s, events, sub_policy, role=sub_role, registry=registry, max_iterations=ROLE_BUDGETS.get(sub_role), diagnostics=diagnostics,
        )

    return AgentLoop(
        s.provider,
        registry,
        s.workspace,
        system_prompt=prompt,
        settings=s.settings.agent.model_copy(update={"max_iterations": max_iterations}) if max_iterations else s.settings.agent,
        policy=policy,
        tool_names=names,
        events=events,
        context_window=s.context_window,
        label=role,
        checkpointer=sessions.checkpointer if sessions else None,
        thread_id=thread_id,
        memory=MemoryStore(s.workspace.root),
        asker=asker if role is None else None,
        diagnostics=diagnostics,
        checkpoints=checkpoints if role is None else None,
        spawner=spawn if role is None else None,
        hooks=HookRunner(s.hooks, s.workspace.root, getattr(run_tool, "shell", None)) if s.hooks else None,
    )


# turn budgets per specialist, so no single agent can loop for long
ROLE_BUDGETS = {"planner": 12, "coder": 30, "tester": 20, "debugger": 20, "reviewer": 12, "explore": 20, "general": 30}
SUBAGENT_ROLES = {"explore": "explore", "plan": "planner", "general": "general"}


def _make_pipeline(
    s: Session, events: Callable[[AgentEvent], None], policy: PermissionPolicy, diagnostics: DiagnosticsSource | None = None
) -> BuildPipeline:
    """Specialist agents share one tool registry (and so one set of background processes)."""
    registry = _registry(s)
    read_only = PermissionPolicy(Mode.READ_ONLY, rules=s.rules)

    def factory(role: str) -> AgentLoop:
        return _make_agent(
            s, events, read_only if role in ("planner", "reviewer") else policy,
            role=role, registry=registry, max_iterations=ROLE_BUDGETS.get(role), diagnostics=diagnostics,
        )

    return BuildPipeline(factory, registry, s.workspace, policy, events)


class StatusTracker:
    """What a status line command is told about the session (see lca.agent.statusline)."""

    def __init__(self, s: Session) -> None:
        self.s = s
        cfg = s.settings.status_line
        run_tool = default_registry().get("run_command")
        self.line = StatusLine(cfg.command, s.workspace.root, getattr(run_tool, "shell", None), cfg.timeout_s) if cfg.command else None
        self.prompt_tokens = self.completion_tokens = 0
        self.last_run: dict[str, Any] = {}

    def record(self, result: AgentResult) -> None:
        self.prompt_tokens += result.usage.prompt_tokens
        self.completion_tokens += result.usage.completion_tokens
        self.last_run = {"status": result.status, "duration_s": result.duration_s, "files_changed": result.files_changed}

    def render(self, agent: AgentLoop, mode: Mode, session_id: str) -> str:
        if self.line is None:
            return ""
        window = agent.budget.window
        branch = (self.s.workspace.run_git("rev-parse", "--abbrev-ref", "HEAD") or "").strip() if self.s.workspace.is_git_repo else ""
        return self.line.render({
            "model": Path(self.s.settings.provider.model).name if self.s.settings.provider.model else "",
            "workspace": str(self.s.workspace.root), "session_id": session_id, "mode": mode.value, "git_branch": branch,
            "context": {"used": agent.context_used, "window": window, "percent": round(100 * agent.context_used / window) if window else 0},
            "tokens": {"prompt": self.prompt_tokens, "completion": self.completion_tokens},
            "last_run": self.last_run, "version": __version__,
        })


def _key_reader() -> Callable[[], str | None] | None:
    """Non-blocking single-key reads from the console (Windows), used for Ctrl+B while a task runs."""
    if os.name != "nt" or not sys.stdin.isatty():
        return None
    import msvcrt

    def read() -> str | None:
        return msvcrt.getwch() if msvcrt.kbhit() else None

    return read


def _run_cancellable(
    fn: Callable[[CancelToken], T], keys: dict[str, Callable[[], None]] | None = None, paused: Callable[[], bool] | None = None
) -> T:
    """Run `fn` in a worker thread so Ctrl+C is handled promptly even while waiting on the model. `keys` maps single
    keys (e.g. Ctrl+B) to actions while it runs; `paused` says when the worker itself is reading the keyboard."""
    token = CancelToken()
    box: dict[str, Any] = {}

    def target() -> None:
        try:
            box["result"] = fn(token)
        except BaseException as e:  # re-raised in the main thread
            box["error"] = e

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    interrupts = 0
    read_key = _key_reader() if keys else None
    while worker.is_alive():
        try:
            worker.join(0.1 if read_key else 0.2)
            if read_key and keys and not (paused and paused()):
                key = read_key()
                if key in keys:
                    keys[key]()
        except KeyboardInterrupt:
            interrupts += 1
            token.cancel()
            if interrupts == 1:
                _warn("\nCancelling… (press Ctrl+C again to quit immediately)")
            else:
                raise typer.Exit(130) from None
    if "error" in box:
        raise box["error"]
    return box["result"]


def _overrides(
    base_url: str | None, model: str | None, provider: str | None, tool_mode: str | None, max_iterations: int | None, allow_remote: bool
) -> dict[str, Any]:
    p: dict[str, Any] = {}
    if provider:
        p["kind"] = provider
        if not base_url:
            p["base_url"] = ""  # re-derive the default URL for this provider kind
    if base_url:
        p["base_url"] = base_url
    if model:
        p["model"] = model
    if tool_mode:
        p["tool_mode"] = tool_mode
    out: dict[str, Any] = {"provider": p} if p else {}
    if max_iterations:
        out["agent"] = {"max_iterations": max_iterations}
    if allow_remote:
        out["allow_remote"] = True
    return out


def _parse_mode(mode: str, yes: bool) -> Mode:
    if yes:
        return Mode.AUTO_EDIT
    try:
        return Mode(mode)
    except ValueError:
        raise typer.BadParameter(f"--mode must be one of: {', '.join(m.value for m in Mode)}") from None


RootOpt = typer.Option(Path("."), "--root", "-r", help="Project directory the agent may access.")
UrlOpt = typer.Option(None, "--base-url", help="Model server URL (default http://127.0.0.1:8080/v1).")
ModelOpt = typer.Option(None, "--model", "-m", help="Model name (default: first model on the server).")
ProviderOpt = typer.Option(None, "--provider", help="openai (llama.cpp, LM Studio, vLLM) or ollama.")
ToolModeOpt = typer.Option(None, "--tool-mode", help="auto | native | text")
IterOpt = typer.Option(None, "--max-iterations", help="Safety limit on model turns.")
RemoteOpt = typer.Option(False, "--allow-remote", help="Allow sending code to a non-local model server.")
ModeOpt = typer.Option("ask", "--mode", help="plan (research, then approve a plan) | read-only | ask (approve each change) | auto-edit | full")
YesOpt = typer.Option(False, "--yes", "-y", help="Apply file changes without asking (same as --mode auto-edit).")


# ---------------------------------------------------------------------- commands


@app.command()
def run(
    task: str = typer.Argument(..., help="What the agent should do."),
    root: Path = RootOpt,
    base_url: str | None = UrlOpt,
    model: str | None = ModelOpt,
    provider: str | None = ProviderOpt,
    tool_mode: str | None = ToolModeOpt,
    max_iterations: int | None = IterOpt,
    allow_remote: bool = RemoteOpt,
    mode: str = ModeOpt,
    yes: bool = YesOpt,
    output: Output = typer.Option(Output.text, "--output", "-o", help="text, or json (one event per line)."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show model reasoning and turn markers."),
    image: list[Path] = typer.Option([], "--image", "-i", help="Attach an image (screenshot, mockup) for a vision model."),
) -> None:
    """Run one task to completion and exit (exit code reflects the outcome)."""
    perm = _parse_mode(mode, yes)
    s = _setup_or_exit(root, _overrides(base_url, model, provider, tool_mode, max_iterations, allow_remote), quiet=output == Output.json)
    asker: Asker | None = None
    if output == Output.json:
        renderer: Callable[[AgentEvent], None] = _json_renderer
        approver = None  # no interactive approvals in JSON mode: use --yes, or `lca serve`
    else:
        renderer = TextRenderer(verbose)
        approver = _terminal_approver(renderer) if sys.stdin.isatty() else None
        asker = _terminal_asker(renderer) if sys.stdin.isatty() else None
    if s.mcp is not None:
        s.mcp.wait(60)
    agent = _make_agent(s, renderer, s.policy(perm, approver), asker=asker)
    try:
        task, images = _prepare_input(s, task)
        images += [image_data_url(p) for p in image]
    except KeyError as e:
        _err(f"Unknown command /{e.args[0]}. `lca commands` lists the custom commands and skills.")
        raise typer.Exit(2) from e
    except (ValueError, OSError) as e:
        _err(str(e))
        raise typer.Exit(2) from e
    result = _run_cancellable(lambda token: agent.run(task, token, images))
    raise typer.Exit(EXIT_CODES[result.status])


@app.command()
def build(
    task: str = typer.Argument(..., help="What to build or change."),
    root: Path = RootOpt,
    base_url: str | None = UrlOpt,
    model: str | None = ModelOpt,
    provider: str | None = ProviderOpt,
    tool_mode: str | None = ToolModeOpt,
    allow_remote: bool = RemoteOpt,
    mode: str = ModeOpt,
    yes: bool = YesOpt,
    output: Output = typer.Option(Output.text, "--output", "-o", help="text, or json (one event per line)."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Multi-agent pipeline: plan → implement → install → test ⇄ debug → review. Exit 0 only if verification passed."""
    perm = _parse_mode(mode, yes)
    s = _setup_or_exit(root, _overrides(base_url, model, provider, tool_mode, None, allow_remote), quiet=output == Output.json)
    if output == Output.json:
        renderer: Callable[[AgentEvent], None] = _json_renderer
        approver = None
    else:
        renderer = TextRenderer(verbose)
        approver = _terminal_approver(renderer) if sys.stdin.isatty() else None
    pipeline = _make_pipeline(s, renderer, s.policy(perm, approver))
    result: BuildFinished = _run_cancellable(lambda token: pipeline.run(task, token))
    raise typer.Exit({"passed": 0, "unverified": 4, "cancelled": 130}.get(result.status, 1))


@app.command()
def chat(
    root: Path = RootOpt,
    base_url: str | None = UrlOpt,
    model: str | None = ModelOpt,
    provider: str | None = ProviderOpt,
    tool_mode: str | None = ToolModeOpt,
    max_iterations: int | None = IterOpt,
    allow_remote: bool = RemoteOpt,
    mode: str = ModeOpt,
    yes: bool = YesOpt,
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    resume: str | None = typer.Option(None, "--resume", help="Continue a saved session: its id, or 'last'."),
) -> None:
    """Interactive multi-turn session with saved history.
    Commands: /sessions, /resume <id>, /rewind [n] [code|chat|both], /permissions, /clear, /exit."""
    perm = _parse_mode(mode, yes)
    s = _setup_or_exit(root, _overrides(base_url, model, provider, tool_mode, max_iterations, allow_remote), quiet=False)
    renderer = TextRenderer(verbose)
    renderer.background_hint = _key_reader() is not None
    policy = s.policy(perm, _terminal_approver(renderer))
    store = SessionStore(s.workspace.root)
    checkpoints = CheckpointStore(s.workspace.root)
    status = StatusTracker(s)

    def open_session(wanted: str | None) -> tuple[AgentLoop, str]:
        info = store.latest() if wanted == "last" else store.get(wanted) if wanted else None
        if wanted and info is None:
            _warn(f"No saved session {wanted!r}; starting a new one.")
        info = info or store.create()
        agent = _make_agent(
            s, renderer, policy, sessions=store, thread_id=info.id, asker=_terminal_asker(renderer), checkpoints=checkpoints
        )
        if wanted:
            items = transcript(agent.history())
            typer.secho(f"Resumed session {info.id}: {info.title or '(untitled)'} ({len(items)} items)", dim=True)
            for item in items[-6:]:
                if item["role"] in ("user", "assistant"):
                    typer.secho(f"  {item['role']}: {item['text'][:200]}", dim=True)
        return agent, info.id

    agent, session_id = open_session(resume)
    typer.secho(
        f"Mode: {perm.value}. Session {session_id}. /sessions lists saved sessions, /resume <id> continues one, "
        "/rewind undoes messages, /permissions shows rules, /clear starts a new one, /exit quits. Ctrl+C cancels a task.",
        dim=True,
    )
    while True:
        if status.line is not None and (text := status.render(agent, policy.mode, session_id)):
            typer.secho("\n" + text, dim=True)
        try:
            line = input("\n› ").strip()
        except (EOFError, KeyboardInterrupt):
            typer.echo()
            return
        if not line:
            continue
        if line in ("/exit", "/quit"):
            return
        if line == "/clear":
            agent, session_id = open_session(None)
            typer.secho(f"New session {session_id}.", dim=True)
            continue
        if line == "/sessions":
            _print_sessions(store)
            continue
        if line.startswith("/resume"):
            agent, session_id = open_session(line.split(maxsplit=1)[1] if " " in line else "last")
            continue
        if line == "/permissions":
            _print_rules(s.rules, s.permission_store)
            continue
        if line == "/commands":
            _print_commands(s.workspace.root)
            continue
        if line == "/mcp":
            _print_mcp(s.mcp.status() if s.mcp else [])
            continue
        if line.split()[0] == "/rewind":
            _terminal_rewind(agent, checkpoints, line.split()[1:])
            continue
        try:
            prompt, images = _prepare_input(s, line)
        except KeyError as e:
            _warn(f"Unknown command /{e.args[0]}. /commands lists the custom commands and skills.")
            continue
        keys = {"\x02": lambda: (agent.background_event.set(), renderer._line(typer.style("  (moving the command to the background)", dim=True)))}
        result = _run_cancellable(lambda token: agent.run(prompt, token, images), keys=keys, paused=renderer.lock.locked)
        status.record(result)
        store.touch(session_id, line)


REWIND_MODES = {"both": "both", "code": "code", "chat": "conversation", "conversation": "conversation"}


def _terminal_rewind(agent: AgentLoop, checkpoints: CheckpointStore, args: list[str]) -> None:
    """/rewind lists the messages; /rewind <n> [code|chat|both] goes back to before message n."""
    cps = checkpoints.list(agent.thread_id)
    if not cps:
        typer.secho("Nothing to rewind yet.", dim=True)
        return
    if not args:
        for i, cp in enumerate(cps, 1):
            files = f"  ({len(cp.files)} file(s) changed)" if cp.files else ""
            typer.echo(f"  {i:>2}. {' '.join(cp.prompt.split())[:90]}{typer.style(files, dim=True)}")
        typer.secho("/rewind <n> goes back to before message n: code and conversation. Add `code` or `chat` for only one.", dim=True)
        return
    if not args[0].isdigit() or not 1 <= int(args[0]) <= len(cps):
        _warn(f"Pick a message number from 1 to {len(cps)}.")
        return
    mode = REWIND_MODES.get(args[1] if len(args) > 1 else "both")
    if mode is None:
        _warn("The mode is code, chat or both.")
        return
    try:
        result = agent.rewind(cps[int(args[0]) - 1].id, mode)  # type: ignore[arg-type]
    except ValueError as e:
        _warn(str(e))
        return
    _print_rewind(result.as_dict(), mode)


def _print_rewind(r: dict[str, Any], mode: str) -> None:
    if mode != "conversation":
        typer.secho(f"Restored {len(r['restored'])} file(s), removed {len(r['removed'])} the agent had created.", fg=typer.colors.GREEN)
        for f in r["restored"]:
            typer.secho(f"  restored {f}", dim=True)
        for f in r["removed"]:
            typer.secho(f"  removed  {f}", dim=True)
        for f in r["skipped"]:
            _warn(f"  not restored: {f}")
    if mode != "code":
        typer.secho(f"The conversation is back to before: {r['prompt'][:120]}", fg=typer.colors.GREEN)


def _print_rules(rules: RuleSet, store: PermissionStore) -> None:
    items = rules.listing()
    if not items:
        typer.secho("No permission rules. Add some with `lca permissions --allow 'run_command(npm test:*)'`, or choose "
                    "\"always allow\" when approving a command.", dim=True)
    for item in items:
        color = {"allow": typer.colors.GREEN, "ask": typer.colors.YELLOW, "deny": typer.colors.RED}[item["kind"]]
        kind = typer.style(f"{item['kind']:<5}", fg=color)
        typer.echo(f"  {kind}  {item['rule']}  {typer.style('(' + item['source'] + ')', dim=True)}")
    typer.secho(f"Saved rules: {store.path}", dim=True)


def _print_sessions(store: SessionStore) -> None:
    import datetime as _dt

    items = store.list()
    if not items:
        typer.secho("No saved sessions yet.", dim=True)
    for i in items:
        when = _dt.datetime.fromtimestamp(i.updated).strftime("%Y-%m-%d %H:%M")
        typer.echo(f"  {i.id}  {when}  {i.turns:>3} turn(s)  {i.title or '(untitled)'}")


@app.command()
def sessions(root: Path = RootOpt, delete: str | None = typer.Option(None, "--delete", help="Delete a session by id.")) -> None:
    """List (or delete) the saved conversations of a project."""
    store = SessionStore(Workspace(root).root)
    if delete:
        store.delete(delete)
        typer.echo(f"Deleted session {delete}.")
        return
    _print_sessions(store)


@app.command()
def memory(
    root: Path = RootOpt,
    add: str | None = typer.Option(None, "--add", help="Add a fact to project memory."),
    forget: str | None = typer.Option(None, "--forget", help="Delete entries containing this text."),
) -> None:
    """Show or edit the agent's long-term memory (project and user)."""
    store = MemoryStore(Workspace(root).root)
    if add:
        typer.echo(store.add(add))
    if forget:
        typer.echo(f"Removed {store.remove(forget)} entr(y/ies).")
    for scope in ("user", "project"):
        typer.secho(f"{scope} memory: {store.files[scope]}", bold=True)
        for e in store.entries(scope):  # type: ignore[arg-type]
            typer.echo(f"  - {e}")


@app.command()
def permissions(
    root: Path = RootOpt,
    allow: list[str] = typer.Option([], "--allow", help="Add an allow rule, e.g. 'run_command(npm test:*)'."),
    ask: list[str] = typer.Option([], "--ask", help="Add an ask rule: always ask, even in auto modes."),
    deny: list[str] = typer.Option([], "--deny", help="Add a deny rule, e.g. 'run_command(git push:*)' or 'edit_file(migrations/**)'."),
    remove: list[str] = typer.Option([], "--remove", help="Remove a saved rule (from any list)."),
) -> None:
    """Show or edit the permission rules for a project (saved outside the repository, in ~/.lca/permissions)."""
    try:
        ws = Workspace(root)
        settings, warnings = load_settings(ws.root)
    except (ConfigError, WorkspaceError) as e:
        _err(str(e))
        raise typer.Exit(2) from e
    store = PermissionStore(ws.root)
    try:
        for kind, rules in (("allow", allow), ("ask", ask), ("deny", deny)):
            for raw in rules:
                typer.echo(f"Added {kind} rule {store.add(kind, raw)}.")  # type: ignore[arg-type]
        for raw in remove:
            gone = [k for k in KINDS if store.remove(k, raw)]
            typer.echo(f"Removed {raw} ({', '.join(gone)})." if gone else f"No saved rule {raw}.")
    except ValueError as e:
        _err(str(e))
        raise typer.Exit(2) from e
    rules = build_ruleset(settings.permissions.model_dump(), store, warnings)
    for w in warnings:
        _warn(f"! {w}")
    _print_rules(rules, store)


def _prepare_input(s: Session, text: str) -> tuple[str, list[str]]:
    """/command expansion, then @file attachments. Returns (task, images). Raises KeyError for an unknown /command."""
    text = expand_input(s.workspace.root, text)[0]
    task, _, images = expand_mentions(s.workspace, text, images=s.settings.provider.vision is not False)
    return task, images


def _print_commands(root: Path) -> None:
    items = slash_items(root)
    if not items:
        typer.secho("No custom commands or skills. Add .lca/commands/<name>.md or .lca/skills/<name>/SKILL.md "
                    "(or the same under ~/.lca/ for every project).", dim=True)
    for i in items:
        hint = f" {i['argument_hint']}" if i["argument_hint"] else ""
        typer.echo(f"  /{i['name']}{hint}  {typer.style(i['kind'] + ', ' + i['source'], dim=True)}  {i['description']}")


@app.command()
def commands(root: Path = RootOpt) -> None:
    """List custom slash commands (.lca/commands/*.md) and skills (.lca/skills/*/SKILL.md)."""
    _print_commands(Workspace(root).root)


@app.command()
def hooks(
    root: Path = RootOpt,
    trust: bool = typer.Option(False, "--trust", help="Allow this project's hooks (as they are now) to run."),
) -> None:
    """Show the hooks that run for a project, and trust a project's own hooks."""
    try:
        ws = Workspace(root)
        settings, warnings = load_settings(ws.root)
    except (ConfigError, WorkspaceError) as e:
        _err(str(e))
        raise typer.Exit(2) from e
    if trust:
        project = read_project_hooks(ws.root)
        if not project:
            _warn("This project's .lca/config.toml has no [hooks].")
        else:
            typer.secho("Trusting these project hooks (they run commands on this machine):", bold=True)
            typer.echo(json.dumps(project, indent=2))
            trust_project_hooks(ws.root)
            warnings = [w for w in warnings if "not trusted" not in w]
    found = load_hooks(ws.root, settings.hooks, warnings)
    for w in warnings:
        _warn(f"! {w}")
    if not found:
        typer.secho("No hooks. Add [[hooks.PreToolUse]] etc. to ~/.lca/config.toml (see docs/EXTENDING.md).", dim=True)
    for h in found:
        matcher = f" [{h.matcher}]" if h.matcher else ""
        typer.echo(f"  {h.event}{matcher}  {h.command}  {typer.style('(' + h.source + ')', dim=True)}")


def _print_mcp(servers: list[dict]) -> None:
    if not servers:
        typer.secho("No MCP servers. Add [mcp.servers.<name>] to ~/.lca/config.toml (see docs/EXTENDING.md).", dim=True)
    for srv in servers:
        color = {"ready": typer.colors.GREEN, "starting": typer.colors.YELLOW}.get(srv["status"], typer.colors.RED)
        typer.echo(f"  {typer.style(srv['status'], fg=color):<10} {srv['name']}  {typer.style('(' + srv['transport'] + ', ' + srv['source'] + ')', dim=True)}")
        if srv["error"]:
            typer.secho(f"      {srv['error']}", fg=typer.colors.RED)
        if srv["tools"]:
            typer.secho(f"      tools: {', '.join(srv['tools'])}", dim=True)


@app.command()
def mcp(
    root: Path = RootOpt,
    trust: bool = typer.Option(False, "--trust", help="Allow this project's MCP servers (as configured now) to start."),
    login: str | None = typer.Option(None, "--login", help="Sign in to an HTTP MCP server (OAuth, in your browser)."),
    logout: str | None = typer.Option(None, "--logout", help="Forget the sign-in of an HTTP MCP server."),
) -> None:
    """Connect to the configured MCP servers and show their status and tools."""
    try:
        ws = Workspace(root)
        settings, warnings = load_settings(ws.root)
    except (ConfigError, WorkspaceError) as e:
        _err(str(e))
        raise typer.Exit(2) from e
    if trust:
        project = read_project_section(ws.root, "mcp")
        if not project:
            _warn("This project's .lca/config.toml has no [mcp] servers.")
        else:
            typer.secho("Trusting these project MCP servers (they run programs on this machine):", bold=True)
            typer.echo(json.dumps(project, indent=2))
            trust_section(ws.root, "mcp")
    warnings = []
    manager = McpManager(load_mcp_configs(ws.root, settings.mcp, warnings), ws.root)
    for w in warnings:
        _warn(f"! {w}")
    if logout:
        from lca.mcp_auth import forget_tokens

        srv = next((x for x in manager.servers if x.cfg.name == logout), None)
        if srv is None or not srv.cfg.url:
            _err(f"No HTTP MCP server named {logout!r}.")
            raise typer.Exit(2)
        forget_tokens(srv.cfg.url)
        typer.echo(f"Signed out of {logout}.")
    manager.start()
    manager.wait(90)
    if login:
        import webbrowser

        def open_url(url: str) -> None:
            typer.echo(f"Opening your browser to sign in to {login}. If it does not open, visit:\n  {url}")
            webbrowser.open(url)

        try:
            manager.login(login, open_url)
            typer.secho(f"Signed in to {login}.", fg=typer.colors.GREEN)
        except McpError as e:
            _err(f"Sign-in failed: {e}")
    _print_mcp(manager.status())
    manager.stop()


def _with_editor_context(text: str, context: Any) -> str:
    if not isinstance(context, dict):
        return text
    parts = []
    sel = context.get("selection")
    file = context.get("file")
    if isinstance(sel, dict) and sel.get("text") and file:
        code = str(sel["text"])[:12000]
        parts.append(f"[The user selected lines {sel.get('start')}-{sel.get('end')} of {file} in the editor:]\n```\n{code}\n```")
    elif file:
        parts.append(f"[For reference only: the user has {file} open in the editor. Ignore this unless the request is about it.]")
    return text + ("\n\n" + "\n".join(parts) if parts else "")


@app.command()
def serve(
    root: Path = RootOpt,
    base_url: str | None = UrlOpt,
    model: str | None = ModelOpt,
    provider: str | None = ProviderOpt,
    tool_mode: str | None = ToolModeOpt,
    max_iterations: int | None = IterOpt,
    allow_remote: bool = RemoteOpt,
    mode: str = ModeOpt,
    editor_diagnostics: bool = typer.Option(
        False, "--diagnostics", help="The client answers diagnostics_request with the editor's problems (VS Code does)."
    ),
) -> None:
    """JSON-lines server on stdin/stdout for editor integrations (see docs/PROTOCOL.md)."""
    out_lock = threading.Lock()

    def send(obj: BaseModel | dict[str, Any]) -> None:
        line = obj.model_dump_json() if isinstance(obj, BaseModel) else json.dumps(obj)
        with out_lock:
            sys.stdout.write(line + "\n")
            sys.stdout.flush()

    try:
        s = _setup(root, _overrides(base_url, model, provider, tool_mode, max_iterations, allow_remote))
    except SetupError as e:
        send({"type": "fatal", "message": str(e)})
        raise typer.Exit(2) from e
    for w in s.warnings:
        send({"type": "notice", "level": "warning", "message": w})

    pending: dict[str, dict[str, Any]] = {}
    state: dict[str, Any] = {"token": None, "thread": None}

    def request(payload: dict[str, Any], cancelled_type: str | None = None, timeout_s: float | None = None) -> Any:
        """Send a request to the client and wait for its reply. None when the task is cancelled or time runs out."""
        rid = uuid.uuid4().hex
        slot: dict[str, Any] = {"event": threading.Event(), "result": None}
        pending[rid] = slot
        send({**payload, "request_id": rid})
        token: CancelToken | None = state["token"]
        deadline = time.monotonic() + timeout_s if timeout_s else None
        try:
            while not slot["event"].wait(0.2):
                if token is not None and token.cancelled:
                    if cancelled_type:
                        send({"type": cancelled_type, "request_id": rid})
                    return None
                if deadline is not None and time.monotonic() > deadline:
                    return None
            return slot["result"]
        finally:
            pending.pop(rid, None)

    def answer(rid: Any, result: Any) -> None:
        slot = pending.get(str(rid))
        if slot:
            slot["result"] = result
            slot["event"].set()

    def asker(question: str, options: list[str]) -> str:
        return request({"type": "question", "question": question, "options": options}, "question_cancelled") or ""

    def diagnostics(paths: list[str]) -> list[Diagnostic] | None:
        items = request({"type": "diagnostics_request", "paths": paths}, timeout_s=15)
        if not isinstance(items, list):
            return None
        out = []
        for item in items:
            try:
                out.append(Diagnostic.model_validate(item))
            except ValueError:
                continue
        return out

    diag_source = diagnostics if editor_diagnostics else None

    def approver(req: ApprovalRequest) -> ApprovalResult:
        change = req.change
        result = request({
            "type": "approval_requested",
            "tool": req.tool,
            "risk": req.risk,
            "description": req.description,
            "command": req.command,
            "level": req.level,
            "reason": req.reason,
            "suggested_rule": req.suggested_rule,
            "plan": req.plan,
            "change": None if change is None else {
                "path": change.path,
                "action": change.action,
                "is_dir": change.is_dir,
                "before": (change.before or "")[:500_000],
                "after": (change.after or "")[:500_000],
                "diff": change.diff()[:200_000],
            },
        }, "approval_cancelled")
        return result if isinstance(result, ApprovalResult) else ApprovalResult(False, feedback="cancelled")

    policy = s.policy(_parse_mode(mode, False), approver)
    store = SessionStore(s.workspace.root)
    checkpoints = CheckpointStore(s.workspace.root)
    first = store.create()

    def chat_agent(thread_id: str) -> AgentLoop:
        return _make_agent(
            s, send, policy, sessions=store, thread_id=thread_id, asker=asker, diagnostics=diag_source, checkpoints=checkpoints
        )

    def history_items(agent: AgentLoop) -> list[dict]:
        """The transcript, with the checkpoint of each user message (for rewind buttons)."""
        ids = {cp.message_id: cp.id for cp in checkpoints.list(agent.thread_id)}
        items = transcript(agent.history())
        for item in items:
            if item["role"] == "user":
                item["checkpoint_id"] = ids.get(item.pop("id", None) or "")
        return items

    def send_permissions() -> None:
        send({"type": "permissions", "rules": s.rules.listing(), "file": str(s.permission_store.path)})

    agent_box: dict[str, Any] = {"agent": chat_agent(first.id), "session": first.id}

    def switch_session(session_id: str | None) -> None:
        info = store.get(session_id) if session_id else None
        if session_id and info is None:
            send({"type": "notice", "level": "error", "message": f"Session {session_id} no longer exists."})
            return
        info = info or store.create()
        agent_box["agent"] = chat_agent(info.id)
        agent_box["session"] = info.id
        if session_id:
            send({"type": "history", "session_id": info.id, "title": info.title, "items": history_items(agent_box["agent"])})
        else:
            send({"type": "cleared", "session_id": info.id})
        push_status_line()

    tracker = StatusTracker(s)
    status_lock = threading.Lock()

    def push_status_line() -> None:
        """Run the user's status line command in the background and send its output."""
        if tracker.line is None:
            return

        def work() -> None:
            if not status_lock.acquire(blocking=False):
                return  # one at a time; the next change triggers another
            try:
                send({"type": "status_line", "text": tracker.render(agent_box["agent"], policy.mode, agent_box["session"])})
            finally:
                status_lock.release()

        threading.Thread(target=work, daemon=True).start()

    def busy() -> bool:
        t = state["thread"]
        return bool(t and t.is_alive())

    def start_task(text: str, build: bool = False, images: list[str] | None = None) -> None:
        token = CancelToken()
        state["token"] = token

        def work() -> None:
            try:
                if build:
                    message_id, hook = agent_box["agent"].begin_checkpoint(f"/build {text}")
                    pipeline = _make_pipeline(s, send, policy, diag_source)
                    pipeline.change_hook = hook
                    result = pipeline.run(text, token)
                    # let the chat agent know what the build did, for follow-up questions
                    agent_box["agent"].record_exchange(
                        f"/build {text}",
                        f"Build {result.status}. {result.summary}\nFiles changed: {', '.join(result.files_changed) or 'none'}",
                        message_id=message_id,
                    )
                else:
                    tracker.record(agent_box["agent"].run(text, token, images))
                store.touch(agent_box["session"], text)
            except Exception as e:  # report instead of dying silently
                send({"type": "notice", "level": "error", "message": f"Agent crashed: {type(e).__name__}: {e}"})
            finally:
                send({"type": "idle"})
                push_status_line()

        t = threading.Thread(target=work, daemon=True)
        state["thread"] = t
        send({"type": "busy"})
        t.start()

    send({
        "type": "ready",
        "version": __version__,
        "model": Path(s.settings.provider.model).name if s.settings.provider.model else "",
        "base_url": s.settings.provider.base_url,
        "local": s.provider.is_local,
        "context_window": s.context_window,
        "workspace": str(s.workspace.root),
        "mode": policy.mode.value,
        "session_id": first.id,
        "vision": s.settings.provider.vision is not False,
    })
    push_status_line()

    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            msg = json.loads(raw)
            kind = msg.get("type")
        except (json.JSONDecodeError, AttributeError):
            send({"type": "notice", "level": "error", "message": "Ignored a malformed request (not a JSON object)."})
            continue
        if kind in ("task", "build"):
            if busy():
                send({"type": "notice", "level": "warning", "message": "A task is already running. Cancel it first."})
            elif not str(msg.get("text", "")).strip():
                send({"type": "notice", "level": "warning", "message": "Empty task."})
            else:
                text = str(msg["text"])
                images = valid_data_urls(msg.get("images")) if s.settings.provider.vision is not False else []
                if msg.get("images") and s.settings.provider.vision is False:
                    send({"type": "notice", "level": "warning", "message": "Images were not sent: this model cannot read images (the server reports no vision support). Use a vision model, e.g. llama-server with --mmproj."})
                try:
                    if kind == "build":
                        text = expand_mentions(s.workspace, text, images=False)[0]
                    else:
                        text, mentioned = _prepare_input(s, text)
                        images = (images + mentioned)[:MAX_IMAGES_PER_MESSAGE]
                except KeyError as e:
                    send({"type": "notice", "level": "error", "message": f"Unknown command /{e.args[0]}. Type / to see the commands."})
                    send({"type": "idle"})
                    continue
                start_task(_with_editor_context(text, msg.get("context")), build=kind == "build", images=images)
        elif kind == "background":
            agent_box["agent"].background_event.set()  # the running command (if any) moves to the background
        elif kind == "cancel":
            if state["token"]:
                state["token"].cancel()
        elif kind == "approval":
            answer(msg.get("request_id"), ApprovalResult(
                bool(msg.get("approved")), str(msg.get("feedback") or ""), bool(msg.get("remember")), bool(msg.get("always")),
                str(msg["mode"]) if msg.get("mode") else None,
            ))
        elif kind == "answer":
            answer(msg.get("request_id"), str(msg.get("text") or ""))
        elif kind == "diagnostics":
            answer(msg.get("request_id"), msg.get("items") if isinstance(msg.get("items"), list) else [])
        elif kind == "set_mode":
            try:
                policy.mode = Mode(msg.get("mode"))
                send({"type": "mode", "mode": policy.mode.value})
                push_status_line()
            except ValueError:
                send({"type": "notice", "level": "error", "message": f"Unknown mode {msg.get('mode')!r}."})
        elif kind == "clear":
            if busy():
                send({"type": "notice", "level": "warning", "message": "Cannot clear while a task is running."})
            else:
                switch_session(None)
        elif kind == "list_sessions":
            send({"type": "sessions", "current": agent_box["session"], "items": [i.as_dict() for i in store.list()]})
        elif kind == "resume":
            if busy():
                send({"type": "notice", "level": "warning", "message": "Stop the running task before switching sessions."})
            else:
                switch_session(str(msg.get("session_id") or ""))
        elif kind == "delete_session":
            sid = str(msg.get("session_id") or "")
            if sid and sid != agent_box["session"]:
                store.delete(sid)
                checkpoints.delete_session(sid)
            send({"type": "sessions", "current": agent_box["session"], "items": [i.as_dict() for i in store.list()]})
        elif kind == "list_checkpoints":
            send({"type": "checkpoints", "items": [cp.summary() for cp in checkpoints.list(agent_box["agent"].thread_id)]})
        elif kind == "rewind":
            rewind_mode = str(msg.get("mode") or "both")
            if busy():
                send({"type": "notice", "level": "warning", "message": "Stop the running task before rewinding."})
            elif rewind_mode not in ("both", "code", "conversation"):
                send({"type": "notice", "level": "error", "message": f"Unknown rewind mode {rewind_mode!r}."})
            else:
                try:
                    result = agent_box["agent"].rewind(str(msg.get("checkpoint_id") or ""), rewind_mode)
                except ValueError as e:
                    send({"type": "notice", "level": "error", "message": str(e)})
                else:
                    send({"type": "rewound", "mode": rewind_mode, "checkpoint_id": msg.get("checkpoint_id"), **result.as_dict()})
                    if rewind_mode != "code":
                        info = store.get(agent_box["session"])
                        send({
                            "type": "history", "session_id": agent_box["session"], "title": info.title if info else "",
                            "items": history_items(agent_box["agent"]), "rewound": True,
                        })
        elif kind == "list_permissions":
            send_permissions()
        elif kind == "list_mcp":
            send({"type": "mcp", "servers": s.mcp.status() if s.mcp else []})
        elif kind == "mcp_login":
            name = str(msg.get("server") or "")

            def sign_in(name: str = name) -> None:
                try:
                    if s.mcp is None:
                        raise McpError("no MCP servers are configured.")
                    # the editor opens the page (it also works when the agent runs on a remote machine)
                    s.mcp.login(name, lambda url: send({"type": "open_url", "url": url}))
                    send({"type": "notice", "level": "info", "message": f"Signed in to {name}; its tools are available from the next message."})
                except McpError as e:
                    send({"type": "notice", "level": "error", "message": f"Sign-in to {name} failed: {e}"})
                send({"type": "mcp", "servers": s.mcp.status() if s.mcp else []})

            threading.Thread(target=sign_in, daemon=True).start()
        elif kind == "list_commands":
            send({"type": "commands", "items": slash_items(s.workspace.root)})
        elif kind in ("add_permission", "remove_permission"):
            rule_kind = str(msg.get("kind") or "allow")
            try:
                if rule_kind not in KINDS:
                    raise ValueError(f"Unknown rule kind {rule_kind!r}: use allow, ask or deny.")
                if kind == "add_permission":
                    raw = s.permission_store.add(rule_kind, str(msg.get("rule") or ""))  # type: ignore[arg-type]
                    s.rules.add(rule_kind, Rule.parse(raw), "saved")  # type: ignore[arg-type]
                else:
                    raw = Rule.parse(str(msg.get("rule") or "")).raw
                    if not s.permission_store.remove(rule_kind, raw):  # type: ignore[arg-type]
                        raise ValueError(f"{raw} is not a saved rule; rules from config files are edited there.")
                    s.rules.remove(rule_kind, raw)  # type: ignore[arg-type]
            except ValueError as e:
                send({"type": "notice", "level": "error", "message": str(e)})
            send_permissions()
        elif kind == "shutdown":
            break
        else:
            send({"type": "notice", "level": "error", "message": f"Unknown request type {kind!r}."})

    if state["token"]:
        state["token"].cancel()
    if state["thread"]:
        state["thread"].join(5)


@app.command()
def doctor(
    root: Path = RootOpt,
    base_url: str | None = UrlOpt,
    model: str | None = ModelOpt,
    provider: str | None = ProviderOpt,
    allow_remote: bool = RemoteOpt,
    probe: bool = typer.Option(True, "--probe/--no-probe", help="Send a tiny tool-calling request to the model."),
) -> None:
    """Check the environment: Python, git, ripgrep, model server, context size and tool calling."""
    ok = True

    def row(name: str, good: bool | None, detail: str) -> None:
        mark = {True: typer.style("ok  ", fg=typer.colors.GREEN), False: typer.style("FAIL", fg=typer.colors.RED), None: typer.style("warn", fg=typer.colors.YELLOW)}[good]
        typer.echo(f"  {mark} {name:<14} {detail}")

    typer.echo(f"lca {__version__}")
    row("python", sys.version_info >= (3, 11), f"{platform.python_version()} ({sys.executable})")
    git = shutil.which("git")
    row("git", bool(git) or None, git or "not found (repository features limited)")
    rg = shutil.which("rg")
    row("ripgrep", bool(rg) or None, rg or "not found: using slower built-in search (winget install BurntSushi.ripgrep.MSVC)")
    try:
        ws = Workspace(root)
        row("workspace", True, f"{ws.root}{' (git repository)' if ws.is_git_repo else ''}")
    except WorkspaceError as e:
        row("workspace", False, f"{e} - pass an existing folder with --root, or cd into it")
        ok = False
    try:
        settings, warnings = load_settings(root.resolve(), _overrides(base_url, model, provider, None, None, allow_remote))
    except ConfigError as e:
        row("config", False, str(e))
        raise typer.Exit(1) from e
    for w in warnings:
        row("config", None, w)
    p = create_provider(settings.provider)
    row("provider", True, f"{settings.provider.kind} {settings.provider.base_url} ({'local' if p.is_local else 'REMOTE'})")
    if not p.is_local and not settings.allow_remote:
        row("remote", False, "non-local server and remote inference is not allowed (--allow-remote)")
        raise typer.Exit(1)
    info = p.server_info()
    row("server", info.reachable, ", ".join(Path(m).name for m in info.models[:5]) if info.reachable else info.detail)
    if not info.reachable:
        raise typer.Exit(1)
    settings.provider.model = settings.provider.model or (info.models[0] if info.models else "")
    row("context", None if not info.context_window else True, f"{info.context_window or 'unknown (set provider.context_window)'} tokens")
    if probe:
        spec = ToolSpec(
            name="get_time",
            description="Returns the current time.",
            parameters={"type": "object", "properties": {"timezone": {"type": "string"}}, "required": ["timezone"]},
        )
        try:
            resp = p.chat(
                [Message.system("Use tools when asked."), Message.user("Call the get_time tool for timezone UTC. Do not answer in text.")],
                [spec],
            )
            if resp.tool_calls and resp.tool_calls[0].name == "get_time" and not resp.tool_calls[0].parse_error:
                row("tool calling", True, f"{resp.tool_call_source} format (mode now: {p.effective_tool_mode})")
            else:
                ok = False
                row("tool calling", None, f"model did not produce a valid tool call: {resp.content[:120]!r}")
        except Exception as e:  # report, do not crash
            ok = False
            row("tool calling", False, f"{type(e).__name__}: {e}")
    raise typer.Exit(0 if ok else 1)


@app.command()
def tools() -> None:
    """List the agent's tools and their argument schemas."""
    registry = default_registry(memory=True, interaction=True)
    for spec in registry.specs():
        t = registry.get(spec.name)
        typer.secho(f"{spec.name}  [{t.risk if t else '?'}]", bold=True)
        typer.echo(f"  {spec.description}")
        typer.secho(f"  {json.dumps(spec.parameters)}", dim=True)


def _show_version(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(False, "--version", callback=_show_version, is_eager=True, help="Show version and exit."),
) -> None:
    """Local-first coding agent."""
