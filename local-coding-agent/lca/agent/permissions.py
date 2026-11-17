"""Permission policy: decides whether a tool call may run, enforced in code (never left to the model).

Order: blocked commands never run → deny rules → read tools run → ask rules force a prompt → read-only commands run →
the mode forbids → risky commands always ask → allow rules, the mode, or "allow all this session" let it run → ask.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from lca.agent.rules import PermissionStore, Rule, RuleSet, suggest_rule

if TYPE_CHECKING:
    from lca.tools.base import FileChange, Tool


class Mode(str, Enum):
    READ_ONLY = "read-only"  # only read tools
    ASK = "ask"  # read freely, ask before writes and commands
    AUTO_EDIT = "auto-edit"  # read and write freely, ask before commands
    FULL = "full"  # everything allowed (dangerous commands still ask; see Milestone 3)
    PLAN = "plan"  # research only, then exit_plan_mode asks the user to approve the plan and switches mode


READ_ONLY_MODES = (Mode.READ_ONLY, Mode.PLAN)


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str = ""
    declined_by_user: bool = False
    feedback: str = ""
    by_mode: bool = False  # refused because of the permission mode (it will be refused again until the mode changes)


@dataclass
class ApprovalRequest:
    tool: str
    risk: str
    description: str
    change: FileChange | None = None
    command: str | None = None
    level: str | None = None  # command assessment: safe / normal / sensitive
    reason: str = ""
    suggested_rule: str | None = None  # what "Always allow" would save, e.g. run_command(npm test:*)
    plan: str | None = None  # exit_plan_mode: the plan to approve


@dataclass
class ApprovalResult:
    approved: bool
    feedback: str = ""
    remember: bool = False  # approve this kind of action for the rest of the session
    always: bool = False  # save the request's suggested_rule as an allow rule for this project
    mode: str | None = None  # plan approval: the mode to continue in (ask, auto-edit, full)


Approver = Callable[[ApprovalRequest], ApprovalResult]

_NEEDS_APPROVAL = {
    Mode.READ_ONLY: set(),
    Mode.PLAN: {"network"},  # research may read the web, with approval
    Mode.ASK: {"write", "execute", "network"},
    Mode.AUTO_EDIT: {"execute", "network"},
    Mode.FULL: set(),
}
_FORBIDDEN = {
    Mode.READ_ONLY: {"write", "execute", "network"},
    Mode.PLAN: {"write", "execute"},
    Mode.ASK: set(),
    Mode.AUTO_EDIT: set(),
    Mode.FULL: set(),
}


def describe(tool: Tool, args: Any, change: FileChange | None) -> str:
    if change is None:
        return tool.describe_call(args)
    verb = {"create": "Create", "modify": "Edit", "delete": "Delete"}[change.action]
    if change.action == "delete":
        return f"{verb} {change.path}{' (folder)' if change.is_dir else ''}"
    from lca.tools.diff import diff_stats

    added, removed = diff_stats(change.diff())
    return f"{verb} {change.path} (+{added} -{removed})"


class PermissionPolicy:
    def __init__(
        self, mode: Mode = Mode.ASK, approver: Approver | None = None, rules: RuleSet | None = None, store: PermissionStore | None = None
    ) -> None:
        self.mode = mode
        self.approver = approver
        self.rules = rules or RuleSet()
        self.store = store  # where "Always allow" rules are saved
        self.session_allowed: set[str] = set()  # risk classes approved for this session

    def check(self, tool: Tool, args: Any, change: FileChange | None = None, assessment: Any = None) -> Decision:
        risk = tool.risk
        level = getattr(assessment, "level", None)
        if level == "blocked":
            return Decision(False, f"Blocked: this command {assessment.reason}, so it is never run.")
        match = self.rules.decide(tool.name, args, change)
        if match is not None and match.kind == "deny":
            return Decision(False, f"Denied by the permission rule {match.rule.raw}. Do not try to work around it; tell the user if you need it.")
        force_ask = match is not None and match.kind == "ask"
        if risk == "read" and not force_ask:
            return Decision(True)
        if level == "safe" and not force_ask:  # read-only commands such as `git status` run without asking, in every mode
            return Decision(True)
        if risk in _FORBIDDEN[self.mode]:
            if self.mode == Mode.PLAN:
                return Decision(False, "You are in plan mode: research only. Present your plan with exit_plan_mode; once the user approves it you can change files and run commands.", by_mode=True)
            return Decision(False, f"{tool.name} is not allowed in {self.mode.value} mode (only read-only commands are).", by_mode=True)
        sensitive = level == "sensitive"  # always asks, even in full mode, after "allow all", or with an allow rule
        if not sensitive and not force_ask:
            if match is not None and match.kind == "allow":
                return Decision(True)
            if risk not in _NEEDS_APPROVAL[self.mode] or risk in self.session_allowed:
                return Decision(True)
        if self.approver is None:
            why = f" (rule {match.rule.raw})" if force_ask and match else ""
            return Decision(False, f"{tool.name} needs user approval{why}, but no one is available to approve it.")
        description = describe(tool, args, change)
        reason = getattr(assessment, "reason", "") if sensitive else ""
        if reason:
            description += f"  ({reason})"
        command = getattr(args, "command", None)
        suggested = suggest_rule(command) if command and not sensitive and self.store is not None else None
        result = self.approver(
            ApprovalRequest(
                tool=tool.name, risk=risk, description=description, change=change,
                command=command, level=level, reason=reason, suggested_rule=suggested,
            )
        )
        if result.approved:
            if result.always and suggested and self.store is not None:
                self.store.add("allow", suggested)
                self.rules.add("allow", Rule.parse(suggested), "saved")
            elif result.remember and not sensitive:
                self.session_allowed.add(risk)
            return Decision(True)
        if result.feedback:
            reason = f"The user declined this action and said: {result.feedback}. Follow that instruction; do not retry the declined action another way."
        else:
            reason = "The user declined this action. Stop and wait for their instructions."
        return Decision(False, reason, declined_by_user=True, feedback=result.feedback)
