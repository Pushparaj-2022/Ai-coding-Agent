"""Events emitted by the agent loop and the build pipeline. The CLI renders them as text or JSON lines; the
VS Code extension consumes the same JSON-lines stream. See docs/PROTOCOL.md."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, Field

from lca.messages import Usage

RunStatus = Literal["completed", "blocked", "max_iterations", "cancelled", "error"]


class ToolCallRecord(BaseModel):
    name: str
    arguments: dict[str, Any]
    ok: bool
    summary: str
    duration_ms: int
    exit_code: int | None = None


class PlanStep(BaseModel):
    step: str
    status: Literal["pending", "in_progress", "done", "skipped"] = "pending"


class Diagnostic(BaseModel):
    """A problem reported by the editor's language services (VS Code diagnostics)."""

    path: str
    line: int
    column: int = 1
    severity: Literal["error", "warning", "info", "hint"] = "error"
    message: str
    source: str = ""
    code: str = ""

    def render(self) -> str:
        tag = f" [{self.source}{' ' + self.code if self.code else ''}]" if self.source or self.code else ""
        return f"{self.path}:{self.line}:{self.column} {self.severity}: {self.message}{tag}"


class Verification(BaseModel):
    """Automatic verification after the agent changed files: not_needed (no changes), passed, failed, or
    unverified (no command found, declined, or turned off)."""

    status: Literal["not_needed", "passed", "failed", "unverified"] = "not_needed"
    commands: list[dict[str, Any]] = Field(default_factory=list)  # [{"command": ..., "exit_code": ...}]
    note: str = ""


class RunReport(BaseModel):
    """What a run did, assembled by code (not by the model) so it can be trusted."""

    plan: list[PlanStep] = Field(default_factory=list)
    files_changed: list[str] = Field(default_factory=list)
    verification: Verification = Field(default_factory=Verification)
    problems: list[Diagnostic] = Field(default_factory=list)  # editor errors left in changed files
    remaining: list[str] = Field(default_factory=list)
    needs_approval: list[str] = Field(default_factory=list)  # actions the user declined or that still wait for them


class AgentResult(BaseModel):
    status: RunStatus
    final_message: str
    iterations: int
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    files_read: list[str] = Field(default_factory=list)
    files_changed: list[str] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    duration_s: float = 0.0
    error: str | None = None
    report: RunReport = Field(default_factory=RunReport)


class Event(BaseModel):
    # which agent produced the event: None for the main agent, e.g. "coder" for a build sub-agent
    agent: str | None = None


class TurnStarted(Event):
    type: Literal["turn_started"] = "turn_started"
    iteration: int


class TextDelta(Event):
    type: Literal["text_delta"] = "text_delta"
    text: str


class ReasoningDelta(Event):
    type: Literal["reasoning_delta"] = "reasoning_delta"
    text: str


class ToolCallStarted(Event):
    type: Literal["tool_call_started"] = "tool_call_started"
    call_id: str
    name: str
    arguments: dict[str, Any]


class ToolCallFinished(Event):
    type: Literal["tool_call_finished"] = "tool_call_finished"
    call_id: str
    name: str
    ok: bool
    summary: str
    output: str
    duration_ms: int
    exit_code: int | None = None


class ToolOutputDelta(Event):
    """Live output from a running command."""

    type: Literal["tool_output_delta"] = "tool_output_delta"
    call_id: str
    text: str


class ContextUsage(Event):
    """How much of the model's context window the conversation uses (sent after every model call)."""

    type: Literal["context"] = "context"
    used: int
    window: int


class Notice(Event):
    type: Literal["notice"] = "notice"
    level: Literal["info", "warning", "error"] = "info"
    message: str
    quiet: bool = False  # housekeeping (e.g. context trimming): show unobtrusively


class CheckpointCreated(Event):
    """A user message started; files the agent changes from here on can be rewound to their state before it."""

    type: Literal["checkpoint"] = "checkpoint"
    checkpoint_id: str
    message_id: str
    prompt: str


class ModeChanged(Event):
    """The permission mode changed during a task (e.g. a plan was approved in plan mode)."""

    type: Literal["mode"] = "mode"
    mode: str


class PlanUpdated(Event):
    """The agent created or updated its plan for the current task."""

    type: Literal["plan_updated"] = "plan_updated"
    steps: list[PlanStep]


class RunFinished(Event):
    type: Literal["run_finished"] = "run_finished"
    result: AgentResult


class StageStarted(Event):
    """A step of the /build pipeline (plan, implement, install, verify, fix, review)."""

    type: Literal["stage_started"] = "stage_started"
    stage: str
    title: str


class BuildFinished(Event):
    type: Literal["build_finished"] = "build_finished"
    status: Literal["passed", "failed", "unverified", "cancelled", "error"]
    summary: str
    files_changed: list[str] = Field(default_factory=list)
    commands: list[dict[str, Any]] = Field(default_factory=list)  # [{"command": ..., "exit_code": ...}]
    review: str = ""
    duration_s: float = 0.0


AgentEvent = (
    TurnStarted | TextDelta | ReasoningDelta | ToolCallStarted | ToolOutputDelta | ToolCallFinished
    | ContextUsage | Notice | CheckpointCreated | ModeChanged | PlanUpdated | RunFinished | StageStarted | BuildFinished
)
EventSink = Callable[[AgentEvent], None]
