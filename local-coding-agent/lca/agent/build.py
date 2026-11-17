"""The /build pipeline as a LangGraph state graph: specialist agents plus deterministic verification.

    plan → implement → install → find_verify ─cmds─→ verify ─fail, rounds left─→ fix ─→ verify
                                     └─none─→ review ←─passed / out of rounds─┘
    review ─CHANGES_REQUIRED─→ revise → verify (again) → END

Success is decided by real command exit codes, never by an agent saying it is done.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from lca.agent.events import AgentEvent, BuildFinished, Notice, StageStarted, ToolCallFinished, ToolCallStarted, ToolOutputDelta
from lca.agent.loop import AgentLoop
from lca.agent.permissions import PermissionPolicy
from lca.agent.project import install_commands, verify_commands
from lca.cancel import Cancelled, CancelToken
from lca.messages import ToolCall
from lca.providers.base import new_call_id
from lca.tools.base import ToolContext, ToolOutput, ToolRegistry, truncate
from lca.tools.workspace import Workspace

AgentFactory = Callable[[str], AgentLoop]  # role -> fresh agent for that role
VERIFY_RE = re.compile(r"VERIFY_COMMAND:\s*`?([^`\n]+?)`?\s*$", re.M | re.I)


class BuildState(TypedDict, total=False):
    task: str
    plan: str
    cmds: list[str]
    passed: bool | None
    failure: str
    fix_round: int
    review: str
    reviewed: bool
    revised: bool


class StopBuild(Exception):
    def __init__(self, status: str, reason: str) -> None:
        super().__init__(reason)
        self.status = status


class BuildPipeline:
    def __init__(
        self,
        make_agent: AgentFactory,
        registry: ToolRegistry,
        workspace: Workspace,
        policy: PermissionPolicy,
        emit: Callable[[AgentEvent], None],
        *,
        max_fix_rounds: int = 3,
        command_timeout_s: int = 600,
    ) -> None:
        self.make_agent = make_agent
        self.registry = registry
        self.workspace = workspace
        self.policy = policy
        self.emit = emit
        self.max_fix_rounds = max_fix_rounds
        self.command_timeout_s = command_timeout_s
        self.files_changed: set[str] = set()
        self.commands: list[dict] = []
        self.change_hook: Callable[[object], None] | None = None  # saves files before each change (checkpoints)

    # ------------------------------------------------------------ steps

    def _stage(self, stage: str, title: str, agent: str | None = None) -> None:
        self.emit(StageStarted(stage=stage, title=title, agent=agent))

    def _agent(self, role: str, task: str, cancel: CancelToken) -> str:
        agent = self.make_agent(role)
        # the pipeline verifies with its own verify/fix stages; a sub-agent must not run them a second time
        agent.settings = agent.settings.model_copy(update={"auto_verify": False})
        agent.change_hook = self.change_hook
        result = agent.run(task, cancel)
        self.files_changed.update(result.files_changed)
        if result.status == "cancelled":
            raise StopBuild("cancelled", "Cancelled by the user.")
        if result.status == "error":
            raise StopBuild("error", f"The {role} agent failed: {result.final_message}")
        if result.status == "blocked" and "declined" in result.final_message:
            raise StopBuild("cancelled", result.final_message)
        return result.final_message

    def _run(self, command: str, cancel: CancelToken) -> ToolOutput:
        """Run a verification command through the normal tool path (permissions, limits, redaction)."""
        call = ToolCall(id=new_call_id(), name="run_command", arguments={"command": command, "timeout_s": self.command_timeout_s})
        self.emit(ToolCallStarted(call_id=call.id, name=call.name, arguments=call.arguments, agent="verify"))
        ctx = ToolContext(workspace=self.workspace, cancel=cancel, policy=self.policy)
        ctx.progress = lambda text: self.emit(ToolOutputDelta(call_id=call.id, text=text, agent="verify"))
        started = time.monotonic()
        out = self.registry.execute(call, ctx)
        summary = next((line for line in out.content.splitlines() if line.strip()), "")[:200]
        self.emit(
            ToolCallFinished(
                call_id=call.id, name=call.name, ok=not out.is_error, summary=summary, output=out.content,
                duration_ms=int((time.monotonic() - started) * 1000), exit_code=out.exit_code, agent="verify",
            )
        )
        if out.declined:
            raise StopBuild("cancelled", f"Stopped: you declined `{command}`.")
        self.commands.append({"command": command, "exit_code": out.exit_code})
        return out

    def _changed(self) -> str:
        return ", ".join(sorted(self.files_changed)) or "(none recorded)"

    # ------------------------------------------------------------ graph nodes

    def _plan(self, s: BuildState) -> dict:
        self._stage("plan", "Plan", "planner")
        return {"plan": self._agent("planner", f"Create an implementation plan for this task:\n\n{s['task']}", self._cancel)}

    def _implement(self, s: BuildState) -> dict:
        self._stage("implement", "Implement", "coder")
        self._agent(
            "coder",
            f"Implement this task completely.\n\nTask:\n{s['task']}\n\nPlan from the planner:\n{s.get('plan', '')}\n\n"
            "When done, list the files you changed and how you checked them.",
            self._cancel,
        )
        return {}

    def _install(self, s: BuildState) -> dict:
        installs = install_commands(self.workspace.root)
        if installs:
            self._stage("install", "Install dependencies")
            for cmd in installs:
                if self._run(cmd, self._cancel).is_error:
                    self.emit(Notice(level="warning", message=f"`{cmd}` failed; the debugger will look at it during verification."))
        return {}

    def _find_verify(self, s: BuildState) -> dict:
        cmds = verify_commands(self.workspace.root)
        if not cmds:
            self._stage("tests", "Write tests", "tester")
            report = self._agent(
                "tester", f"Write and run automated tests for this change.\n\nTask:\n{s['task']}\n\nFiles changed: {self._changed()}", self._cancel
            )
            m = VERIFY_RE.search(report)
            cmds = [m.group(1).strip()] if m else []
            if not cmds:
                self.emit(Notice(level="warning", message="No way to verify this project was found, and the tester did not report a test command."))
        return {"cmds": cmds, "fix_round": 0}

    def _verify(self, s: BuildState) -> dict:
        cmds = s["cmds"]
        if s.get("fix_round", 0) == 0:
            self._stage("verify", ("Verify again: " if s.get("reviewed") else "Verify: ") + " → ".join(cmds))
        for cmd in cmds:
            out = self._run(cmd, self._cancel)
            if out.is_error:  # later steps usually depend on earlier ones (e.g. build before test)
                return {"passed": False, "failure": f"$ {cmd}\n{truncate(out.content, 6000)}"}
        return {"passed": True, "failure": ""}

    def _fix(self, s: BuildState) -> dict:
        n = s.get("fix_round", 0) + 1
        self._stage("fix", f"Fix attempt {n} of {self.max_fix_rounds}", "debugger")
        report = self._agent(
            "debugger",
            f"The task was:\n{s['task']}\n\nFiles changed so far: {self._changed()}\n\nThis verification failed:\n{s.get('failure', '')}\n\n"
            "Find the root cause and fix it, then run the failing command again to confirm.",
            self._cancel,
        )
        cmds = s["cmds"]
        m = VERIFY_RE.search(report)
        if m:  # the debugger found that the verification command itself was wrong
            cmds = [m.group(1).strip()]
            self.emit(Notice(level="info", message=f"Verification command changed to: {cmds[0]}"))
        else:  # the project may have gained a manifest/scripts meanwhile
            cmds = verify_commands(self.workspace.root) or cmds
        return {"fix_round": n, "cmds": cmds}

    def _review(self, s: BuildState) -> dict:
        self._stage("review", "Review", "reviewer")
        passed = s.get("passed")
        review = self._agent(
            "reviewer",
            f"Review this change.\n\nTask:\n{s['task']}\n\nChanged files: {self._changed()}\n\n"
            f"Verification: {'passed' if passed else 'FAILED' if passed is False else 'not run'}.",
            self._cancel,
        )
        return {"review": review, "reviewed": True}

    def _revise(self, s: BuildState) -> dict:
        self._stage("revise", "Address review", "coder")
        self._agent("coder", f"A code review of this task found problems:\n{s['task']}\n\nReview:\n{s.get('review', '')}\n\nFix every valid issue.", self._cancel)
        return {"revised": True, "fix_round": 0}

    # ------------------------------------------------------------ routing

    def _after_find_verify(self, s: BuildState) -> str:
        return "verify" if s.get("cmds") else "review"

    def _after_verify(self, s: BuildState) -> str:
        if not s.get("passed") and s.get("fix_round", 0) < self.max_fix_rounds:
            return "fix"
        return END if s.get("reviewed") else "review"

    def _after_review(self, s: BuildState) -> str:
        if not s.get("revised") and re.search(r"VERDICT:\s*CHANGES_REQUIRED", s.get("review", ""), re.I):
            return "revise"
        return END

    def _after_revise(self, s: BuildState) -> str:
        return "verify" if s.get("cmds") else END

    def graph(self):
        g = StateGraph(BuildState)
        for name in ("plan", "implement", "install", "find_verify", "verify", "fix", "review", "revise"):
            g.add_node(name, getattr(self, f"_{name}"))
        g.add_edge(START, "plan")
        g.add_edge("plan", "implement")
        g.add_edge("implement", "install")
        g.add_edge("install", "find_verify")
        g.add_conditional_edges("find_verify", self._after_find_verify, ["verify", "review"])
        g.add_conditional_edges("verify", self._after_verify, ["fix", "review", END])
        g.add_edge("fix", "verify")
        g.add_conditional_edges("review", self._after_review, ["revise", END])
        g.add_conditional_edges("revise", self._after_revise, ["verify", END])
        return g.compile()

    # ------------------------------------------------------------ pipeline

    def run(self, task: str, cancel: CancelToken) -> BuildFinished:
        started = time.monotonic()
        self._cancel = cancel
        state: dict = {}
        try:
            state = self.graph().invoke({"task": task}, {"recursion_limit": 30 + 4 * self.max_fix_rounds})
            cmds = state.get("cmds") or []
            passed = state.get("passed")
            if passed:
                status, summary = "passed", f"Verified: {' → '.join(cmds)} succeeded."
            elif passed is False:
                status, summary = "failed", f"Still failing after {self.max_fix_rounds} fix attempts:\n{state.get('failure', '')[:1500]}"
            else:
                status, summary = "unverified", "Changes made, but nothing could verify them."
        except StopBuild as e:
            status, summary = e.status, str(e)
        except Cancelled:
            status, summary = "cancelled", "Cancelled by the user."
        result = BuildFinished(
            status=status,  # type: ignore[arg-type]
            summary=summary,
            files_changed=sorted(self.files_changed),
            commands=self.commands,
            review=(state.get("review") or "")[-1500:],
            duration_s=round(time.monotonic() - started, 1),
        )
        self.emit(result)
        return result
