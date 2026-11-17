"""Tools for working with the user and the editor: plan the task, ask a question, read editor diagnostics."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from lca.agent.events import Diagnostic, PlanStep
from lca.tools.base import Tool, ToolContext, ToolError

MARKS = {"pending": "[ ]", "in_progress": "[~]", "done": "[x]", "skipped": "[-]"}


def render_plan(steps: list[PlanStep]) -> str:
    return "\n".join(f"{MARKS[s.status]} {i}. {s.step}" for i, s in enumerate(steps, 1))


def format_diagnostics(items: list[Diagnostic], limit: int = 40) -> str:
    shown = [d.render() for d in items[:limit]]
    if len(items) > limit:
        shown.append(f"… and {len(items) - limit} more")
    return "\n".join(shown)


class UpdatePlanArgs(BaseModel):
    steps: list[PlanStep] = Field(
        min_length=1, max_length=20,
        description="The whole plan, in order. Each step: {step: what you will do (files, commands), status: pending|in_progress|done|skipped}",
    )

    @field_validator("steps", mode="before")
    @classmethod
    def _plain_strings(cls, v: Any) -> Any:  # small models often send ["step one", "step two"]
        if isinstance(v, list):
            return [{"step": s} if isinstance(s, str) else s for s in v]
        return v


class UpdatePlan(Tool):
    name = "update_plan"
    description = (
        "Record your plan for a task that changes the project, after you inspected the relevant code and before the "
        "first file change. Send the whole list each time; mark steps in_progress / done as you work."
    )
    Args = UpdatePlanArgs

    def run(self, args: UpdatePlanArgs, ctx: ToolContext) -> str:
        ctx.plan = list(args.steps)
        return "Plan recorded:\n" + render_plan(ctx.plan)


class AskUserArgs(BaseModel):
    question: str = Field(min_length=3, description="One specific question")
    options: list[str] = Field(default_factory=list, max_length=6, description="Suggested answers, if there are clear choices")


class AskUser(Tool):
    name = "ask_user"
    description = (
        "Ask the user a question and wait for the answer. Only for decisions the project cannot answer (e.g. which of "
        "two behaviours they want, a missing requirement). Never ask what you can find out with tools."
    )
    Args = AskUserArgs

    def run(self, args: AskUserArgs, ctx: ToolContext) -> str:
        if ctx.ask is None:
            return "No user is available to answer. Make the most reasonable assumption, say so in your final answer, and continue."
        answer = (ctx.ask(args.question, args.options) or "").strip()
        ctx.cancel.raise_if_cancelled()
        return f"The user answered: {answer}" if answer else "The user gave no answer. Make the most reasonable assumption and say so."


SUBAGENT_TYPES = ("explore", "plan", "general")


class TaskArgs(BaseModel):
    description: str = Field(min_length=3, max_length=80, description="3-8 word summary shown to the user")
    prompt: str = Field(
        min_length=10,
        description="Everything the sub-agent needs: the goal, what you already know, files to look at, and what its report must contain. It cannot see this conversation.",
    )
    agent_type: str = Field(
        default="explore",
        description="explore: read-only research of the codebase. plan: read-only design of an implementation. general: a self-contained task that may change files and run commands",
    )


class Task(Tool):
    name = "task"
    description = (
        "Delegate a self-contained job to a sub-agent with its own fresh context; you get back only its final report. "
        "Use explore for broad searches (where is X handled, how does Y work across many files) so the file contents do "
        "not fill your context. Do not delegate what one or two read_file/search_code calls can answer."
    )
    Args = TaskArgs

    def run(self, args: TaskArgs, ctx: ToolContext) -> str:
        kind = args.agent_type.strip().lower()
        if kind not in SUBAGENT_TYPES:
            raise ToolError(f"agent_type must be one of {', '.join(SUBAGENT_TYPES)}.")
        if ctx.spawn is None:
            raise ToolError("sub-agents are not available here. Do the work yourself.")
        return ctx.spawn(kind, args.description, args.prompt)


class ExitPlanModeArgs(BaseModel):
    plan: str = Field(
        min_length=20,
        description="The implementation plan in Markdown: files to change and what changes, new dependencies, how it will be verified. Use a numbered list for the steps",
    )


class ExitPlanMode(Tool):
    name = "exit_plan_mode"
    description = (
        "In plan mode: present your finished plan to the user for approval. If they approve, the mode changes so you "
        "can implement it in this same task; if not, you get their feedback."
    )
    Args = ExitPlanModeArgs

    def run(self, args: ExitPlanModeArgs, ctx: ToolContext) -> Any:
        if ctx.exit_plan is None:
            raise ToolError("you are not in plan mode.")
        return ctx.exit_plan(args.plan)


def parse_plan_steps(plan: str) -> list[PlanStep]:
    """Numbered or bulleted top-level lines of a Markdown plan, as plan steps."""
    import re

    steps = []
    for line in plan.splitlines():
        m = re.match(r"^(?:\d+[.)]|[-*])\s+(.+)", line)
        if m:
            steps.append(PlanStep(step=re.sub(r"\*\*(.+?)\*\*", r"\1", m.group(1)).strip()[:300]))
    return steps[:20]


class ViewImageArgs(BaseModel):
    path: str = Field(description="Image file relative to the workspace root (png, jpg, gif, webp, bmp), e.g. a screenshot")


class ViewImage(Tool):
    name = "view_image"
    description = (
        "Look at an image file in the workspace: a screenshot, a diagram, a rendered chart, UI output. The image is shown "
        "to you in the next message."
    )
    Args = ViewImageArgs

    def run(self, args: ViewImageArgs, ctx: ToolContext) -> str:
        from lca.images import image_data_url

        path = ctx.workspace.resolve(args.path)
        rel = ctx.workspace.rel(path)
        if not path.is_file():
            raise ToolError(ctx.workspace.not_found(args.path, "Image"))
        try:
            url = image_data_url(path)
        except ValueError as e:
            raise ToolError(str(e)) from e
        ctx.pending_images.append((rel, url))
        return f"Showing {rel} ({path.stat().st_size // 1000} KB); the image follows."


class LoadSkillArgs(BaseModel):
    name: str = Field(min_length=1, description="The skill's name, from the Skills list")
    file: str = Field(default="", description="Optional: another file of the skill to load, as listed when you loaded it")


class LoadSkill(Tool):
    name = "load_skill"
    description = "Load a skill's full instructions (or one of its files) before doing a task the skill covers."
    Args = LoadSkillArgs

    def run(self, args: LoadSkillArgs, ctx: ToolContext) -> str:
        from lca.agent.commands import load_skills, skill_text

        skills = load_skills(ctx.workspace.root)
        skill = skills.get(args.name.strip().lower())
        if skill is None:
            raise ToolError(f"no skill named {args.name!r}. Available: {', '.join(sorted(skills)) or 'none'}.")
        try:
            return skill_text(skill, args.file.strip())
        except ValueError as e:
            raise ToolError(str(e)) from e


class GetDiagnosticsArgs(BaseModel):
    paths: list[str] = Field(default_factory=list, description="Files to check, relative to the workspace root. Empty = every file with problems")


class GetDiagnostics(Tool):
    name = "get_diagnostics"
    description = "Errors and warnings the editor (type checker, linter, compiler) currently reports, with path:line:column."
    Args = GetDiagnosticsArgs

    def run(self, args: GetDiagnosticsArgs, ctx: ToolContext) -> str:
        if ctx.diagnostics is None:
            return "Editor diagnostics are not available here (no editor connected). Run the project's type checker or linter instead."
        paths = [ctx.workspace.rel(ctx.workspace.resolve(p)) for p in args.paths]
        items = ctx.diagnostics(paths)
        if items is None:
            raise ToolError("the editor did not answer in time. Run the project's type checker or linter instead.")
        if not items:
            scope = ", ".join(paths) if paths else "the workspace"
            return f"No errors or warnings reported for {scope}."
        errors = sum(1 for d in items if d.severity == "error")
        return f"{errors} error(s), {len(items) - errors} warning(s):\n" + format_diagnostics(items)
