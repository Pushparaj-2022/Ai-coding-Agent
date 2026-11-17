"""Tool interface, execution context and registry (argument validation, permissions, safe output)."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ValidationError

from lca.cancel import Cancelled, CancelToken
from lca.messages import ToolCall
from lca.providers.base import ToolSpec
from lca.security import redact_secrets
from lca.tools.workspace import Workspace, WorkspaceError

log = logging.getLogger("lca.tools")

Risk = Literal["read", "write", "execute", "network"]


class ToolError(Exception):
    """An expected failure; the message is returned to the model so it can correct itself."""


class ToolOutput(BaseModel):
    content: str
    is_error: bool = False
    declined: bool = False  # the user rejected this action
    feedback: str = ""
    exit_code: int | None = None  # for commands
    by_mode: bool = False  # refused by the permission mode


class FileChange(BaseModel):
    """A proposed change, computed before approval so the user can review the exact diff."""

    path: str
    action: Literal["create", "modify", "delete"]
    before: str | None = None  # None for a new file
    after: str | None = None  # None for a deletion
    newline: str = "\n"  # line ending to write (existing CRLF files stay CRLF)
    is_dir: bool = False

    def diff(self) -> str:
        from lca.tools.diff import unified_diff

        return unified_diff(self.before, self.after, self.path)


@dataclass
class ToolContext:
    workspace: Workspace
    cancel: CancelToken
    policy: Any  # lca.agent.permissions.PermissionPolicy (kept loose to avoid an import cycle)
    files_read: set[str] = field(default_factory=set)
    files_changed: set[str] = field(default_factory=set)
    snapshots: dict[str, str] = field(default_factory=dict)  # path -> sha256 of the content the agent last saw
    progress: Any = None  # optional callback(str) for live output of the running tool
    plan: list[Any] = field(default_factory=list)  # lca.agent.events.PlanStep list, set by update_plan
    ask: Any = None  # optional callback(question, options) -> answer; None when no user can answer
    diagnostics: Any = None  # optional callback(paths) -> list[Diagnostic] | None, from the editor
    on_change: Any = None  # optional callback(FileChange) called right before a file change is applied (checkpoints)
    spawn: Any = None  # optional callback(agent_type, description, prompt) -> report, runs a sub-agent (task tool)
    exit_plan: Any = None  # optional callback(plan) -> ToolOutput, asks the user to approve a plan (plan mode)
    extract: Any = None  # optional callback(text, prompt) -> the parts of a long text that answer prompt (model call)
    background: Any = None  # optional threading.Event: set to move the running command to the background
    pending_images: list[Any] = field(default_factory=list)  # (path, data URL) from view_image, shown after the tool results


class Tool(ABC):
    name: ClassVar[str]
    description: ClassVar[str]
    Args: ClassVar[type[BaseModel]]
    risk: ClassVar[Risk] = "read"

    def spec(self) -> ToolSpec:
        return ToolSpec(name=self.name, description=self.description, parameters=_clean_schema(self.Args.model_json_schema()))

    def describe_call(self, args: BaseModel) -> str:
        """Short human-readable description, used for approval prompts and logs."""
        return f"{self.name} {args.model_dump(exclude_defaults=True)}"

    @abstractmethod
    def run(self, args: Any, ctx: ToolContext) -> str | ToolOutput: ...


class FileWriteTool(Tool):
    """A tool that changes files. The registry calls prepare() → permission check (with the diff) → apply()."""

    risk: ClassVar[Risk] = "write"

    @abstractmethod
    def prepare(self, args: Any, ctx: ToolContext) -> FileChange: ...

    @abstractmethod
    def apply(self, change: FileChange, ctx: ToolContext) -> str: ...

    def run(self, args: Any, ctx: ToolContext) -> str:
        return self.apply(self.prepare(args, ctx), ctx)


def _clean_schema(schema: Any) -> Any:
    """Compact pydantic schemas for small models: drop titles, turn Optional[X] into plain X."""
    if isinstance(schema, dict):
        variants = schema.get("anyOf")
        if isinstance(variants, list):
            non_null = [v for v in variants if v != {"type": "null"}]
            if len(non_null) == 1 and len(non_null) < len(variants):
                merged = {k: v for k, v in schema.items() if k != "anyOf" and not (k == "default" and v is None)}
                return _clean_schema({**non_null[0], **merged})
        return {k: _clean_schema(v) for k, v in schema.items() if k != "title" and not (k == "default" and v is None)}
    if isinstance(schema, list):
        return [_clean_schema(v) for v in schema]
    return schema


_BOUND_ERRORS = {"less_than_equal": "le", "less_than": "lt", "greater_than_equal": "ge", "greater_than": "gt"}


# argument names small models use instead of the real ones: canonical -> aliases
ARG_ALIASES: dict[str, tuple[str, ...]] = {
    "pattern": ("query", "regex", "search", "text", "keyword"),
    "path": ("file", "file_path", "filepath", "filename", "dir", "directory", "folder"),
    "command": ("cmd", "shell", "script"),
    "content": ("contents", "text", "code", "data"),
    "old_string": ("old", "old_text", "search", "find"),
    "new_string": ("new", "new_text", "replace", "replacement"),
    "start_line": ("start", "line_start", "from_line", "offset"),
    "end_line": ("end", "line_end", "to_line"),
}


def _apply_aliases(model: type[BaseModel], arguments: dict[str, Any]) -> dict[str, Any]:
    fields = model.model_fields
    out = dict(arguments)
    for canonical, aliases in ARG_ALIASES.items():
        if canonical not in fields or canonical in out:
            continue
        alias = next((a for a in aliases if a in out and a not in fields), None)
        if alias is not None:
            out[canonical] = out.pop(alias)
    return out


def _validate_lenient(model: type[BaseModel], arguments: dict[str, Any]) -> BaseModel:
    """Validate, but clamp out-of-range numbers (e.g. timeout_s=3600 → the maximum) instead of failing, and accept
    common wrong argument names (query → pattern). Small models do both often; refusing the call wastes a turn."""
    arguments = _apply_aliases(model, arguments)
    try:
        return model.model_validate(arguments)
    except ValidationError as e:
        fixed = dict(arguments)
        for err in e.errors():
            kind = _BOUND_ERRORS.get(err["type"])
            if not kind or len(err["loc"]) != 1 or not isinstance(err["loc"][0], str):
                raise
            limit = (err.get("ctx") or {}).get(kind)
            if not isinstance(limit, (int, float)):
                raise
            if kind in ("lt", "gt"):  # strict bounds: step one unit inside
                limit = limit - 1 if kind == "lt" else limit + 1
            fixed[err["loc"][0]] = limit
        return model.model_validate(fixed)


def _format_validation_error(e: ValidationError) -> str:
    problems = []
    for err in e.errors()[:6]:
        loc = ".".join(str(x) for x in err["loc"]) or "arguments"
        problems.append(f"{loc}: {err['msg']}")
    return "; ".join(problems)


def _log_args(arguments: dict[str, Any]) -> dict[str, Any]:
    """Arguments for the log: long values (file contents) are shortened; secrets are redacted by the formatter."""
    out: dict[str, Any] = {}
    for k, v in arguments.items():
        text = v if isinstance(v, str) else repr(v)
        out[k] = text if len(text) <= 200 else f"<{len(text)} chars>"
    return out


def truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = int(limit * 0.7)
    tail = limit - head
    return f"{text[:head]}\n\n[... {len(text) - limit} characters omitted; narrow your request to see more ...]\n\n{text[-tail:]}"


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] = (), *, output_limit: int = 12000) -> None:
        self._tools: dict[str, Tool] = {}
        self.output_limit = output_limit
        for t in tools:
            self.register(t)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Duplicate tool name: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def specs(self, names: Iterable[str] | None = None) -> list[ToolSpec]:
        selected = self._tools.values() if names is None else [self._tools[n] for n in names]
        return [t.spec() for t in selected]

    def execute(self, call: ToolCall, ctx: ToolContext) -> ToolOutput:
        """Validate, authorize and run one tool call. Never raises except for cancellation."""
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolOutput(content=f"Error: unknown tool '{call.name}'. Available tools: {', '.join(self._tools)}.", is_error=True)
        try:
            args = _validate_lenient(tool.Args, call.arguments)
        except ValidationError as e:
            return ToolOutput(content=f"Error: invalid arguments for {tool.name}: {_format_validation_error(e)}", is_error=True)

        change: FileChange | None = None
        if isinstance(tool, FileWriteTool):
            try:
                change = tool.prepare(args, ctx)
            except (ToolError, WorkspaceError) as e:
                return ToolOutput(content=f"Error: {e}", is_error=True)

        precheck = getattr(tool, "precheck", None)
        if precheck is not None and (message := precheck(args, ctx)):
            return ToolOutput(content=message, is_error=True)
        assessment = tool.assess(args, ctx) if hasattr(tool, "assess") else None
        if assessment is not None and assessment.level == "blocked":
            return ToolOutput(
                content=f"Blocked: this command {assessment.reason}, so it is never run. Do not try to work around this.",
                is_error=True,
            )
        decision = ctx.policy.check(tool, args, change, assessment) if ctx.policy else None
        if decision is not None and not decision.allowed:
            return ToolOutput(
                content=f"Permission denied: {decision.reason}",
                is_error=True,
                declined=decision.declined_by_user,
                feedback=decision.feedback,
                by_mode=decision.by_mode,
            )

        ctx.cancel.raise_if_cancelled()
        if change is not None and ctx.on_change is not None:
            try:
                ctx.on_change(change)
            except Exception:  # never lose the change itself because the checkpoint could not be written
                log.exception("checkpoint failed", extra={"fields": {"path": change.path}})
        started = time.monotonic()
        try:
            out = tool.apply(change, ctx) if isinstance(tool, FileWriteTool) and change is not None else tool.run(args, ctx)
            output = out if isinstance(out, ToolOutput) else ToolOutput(content=str(out))
        except Cancelled:
            raise
        except (ToolError, WorkspaceError) as e:
            output = ToolOutput(content=f"Error: {e}", is_error=True)
        except Exception as e:  # a buggy tool must not crash the agent
            log.exception("tool crashed", extra={"fields": {"tool": tool.name}})
            output = ToolOutput(content=f"Error: internal error in {tool.name}: {type(e).__name__}: {e}", is_error=True)
        fields: dict[str, Any] = {
            "tool": tool.name,
            "ok": not output.is_error,
            "ms": int((time.monotonic() - started) * 1000),
            "args": _log_args(call.arguments),
        }
        if output.is_error and output.content:
            fields["error"] = output.content.splitlines()[0][:300]
        log.info("tool call", extra={"fields": fields})
        content = truncate(redact_secrets(output.content), self.output_limit)
        return ToolOutput(
            content=content or "(no output)", is_error=output.is_error, exit_code=output.exit_code,
            declined=output.declined, feedback=output.feedback,
        )
