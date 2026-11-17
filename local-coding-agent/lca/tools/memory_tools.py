"""Tools the agent uses to manage its long-term memory (see lca.memory.store)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from lca.memory.store import MemoryStore
from lca.tools.base import Tool, ToolContext


class RememberArgs(BaseModel):
    fact: str = Field(min_length=1, max_length=300, description="One short, durable fact, e.g. 'Tests run with: py -m pytest -q'")
    scope: Literal["project", "user"] = Field(
        default="project", description="project: about this codebase; user: a preference of the user for every project"
    )


class Remember(Tool):
    name = "remember"
    description = (
        "Save a durable fact for future sessions: project conventions, commands that work, architecture notes, or the "
        "user's preferences. Do not save task progress, guesses, or secrets."
    )
    Args = RememberArgs

    def run(self, args: RememberArgs, ctx: ToolContext) -> str:
        return MemoryStore(ctx.workspace.root).add(args.fact, args.scope)


class ForgetArgs(BaseModel):
    text: str = Field(min_length=3, description="Text contained in the memory entries to delete")


class Forget(Tool):
    name = "forget"
    description = "Delete memory entries that are wrong or outdated (every entry containing the text)."
    Args = ForgetArgs

    def run(self, args: ForgetArgs, ctx: ToolContext) -> str:
        n = MemoryStore(ctx.workspace.root).remove(args.text)
        return f"Removed {n} memory entr{'y' if n == 1 else 'ies'}." if n else "No memory entry contains that text."
