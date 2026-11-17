"""Sub-agents (task tool) and plan mode, wired the way the CLI wires them."""

from __future__ import annotations

import pytest

import lca.graph.agent_graph as agent_graph
from lca.agent.events import ModeChanged, PlanUpdated
from lca.agent.permissions import ApprovalResult, Mode
from lca.agent.rules import PermissionStore, RuleSet
from lca.cli import Session, _make_agent
from lca.config import Settings
from lca.memory.checkpoints import CheckpointStore
from lca.providers.base import ServerInfo
from lca.tools import Workspace

from .conftest import ScriptedProvider, call, reply


@pytest.fixture(autouse=True)
def isolated(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("LCA_HOME", str(tmp_path_factory.mktemp("lcahome")))
    monkeypatch.setattr(agent_graph, "verify_commands", lambda root: [])


def session(project, script, **agent):
    provider = ScriptedProvider(script)
    settings = Settings()
    settings.agent = settings.agent.model_copy(update=agent)
    s = Session(
        settings, provider, Workspace(project), ServerInfo(reachable=True, models=["fake"], context_window=8192), 8192, [],
        RuleSet(), PermissionStore(project),
    )
    return s, provider


def agent_for(project, script, mode, approver=None, events=None, require_plan="remind", **kw):
    s, provider = session(project, script, require_plan=require_plan)
    sink = events.append if events is not None else (lambda e: None)
    return _make_agent(s, sink, s.policy(mode, approver), **kw), provider


def tool_names(provider, request):
    return [t.name for t in provider.requests[request][1]]


# ---------------------------------------------------------------- task tool


def test_explore_subagent_returns_only_its_report(project):
    events: list = []
    # one provider serves both agents, in order: parent, sub-agent (2 turns), parent
    agent, provider = agent_for(project, [
        reply("", call("task", description="Find add", prompt="Where is add() defined and what does it return?", agent_type="explore")),
        reply("", call("read_file", path="src/app.py")),
        reply("add() is defined in src/app.py:4 and returns a + b."),
        reply("add is in src/app.py:4."),
    ], Mode.ASK, events=events)
    result = agent.run("where is add?")
    assert result.status == "completed"
    task_out = next(m.content for m in agent.messages if m.role == "tool")
    assert "[explore agent: completed" in task_out and "src/app.py:4 and returns a + b" in task_out
    assert "def add(a, b):" not in task_out  # the file itself stayed in the sub-agent's context
    sub_reads = [e for e in events if e.type == "tool_call_started" and e.name == "read_file"]
    assert sub_reads and sub_reads[0].agent == "explore"
    assert "write_file" not in tool_names(provider, 1) and "task" not in tool_names(provider, 1)  # read-only, no nesting


def test_general_subagent_changes_count_for_verification_and_rewind(project):
    events: list = []
    store = CheckpointStore(project)
    agent, _ = agent_for(project, [
        reply("", call("update_plan", steps=["Delegate creating notes.md"])),
        reply("", call("task", description="Write notes", prompt="Create notes.md containing '# Notes'.", agent_type="general")),
        reply("", call("write_file", path="notes.md", content="# Notes\n")),
        reply("Created notes.md."),
        reply("Done: the sub-agent created notes.md."),
    ], Mode.AUTO_EDIT, events=events, checkpoints=store)
    result = agent.run("add notes")
    assert result.files_changed == ["notes.md"]
    assert result.report.verification.status == "unverified"  # the change was noticed (no verify command in this project)
    cp = next(e for e in events if e.type == "checkpoint")
    assert store.find(cp.checkpoint_id).files == {"notes.md": {"data": None}}
    agent.rewind(cp.checkpoint_id, "code")
    assert not (project / "notes.md").exists()


def test_general_subagent_needs_a_plan_and_is_refused_in_read_only_modes(project):
    agent, _ = agent_for(project, [
        reply("", call("task", description="Write notes", prompt="Create notes.md please.", agent_type="general")),
        reply("ok"),
    ], Mode.AUTO_EDIT, require_plan="enforce")
    agent.run("add notes")
    assert "no plan yet" in next(m.content for m in agent.messages if m.role == "tool")

    agent, _ = agent_for(project, [
        reply("", call("task", description="Write notes", prompt="Create notes.md please.", agent_type="general")),
        reply("ok"),
    ], Mode.PLAN)
    agent.run("add notes")
    assert "Use explore or plan" in next(m.content for m in agent.messages if m.role == "tool")


# ---------------------------------------------------------------- plan mode


PLAN = "## Plan\n1. Create **hello.py** printing hi\n2. Run it to check\n"


def test_plan_mode_researches_then_switches_mode_after_approval(project):
    events: list = []
    seen = []

    def approver(req):
        seen.append(req)
        return ApprovalResult(True, mode="auto-edit")

    agent, provider = agent_for(project, [
        reply("", call("write_file", path="hello.py", content="print('hi')\n")),
        reply("", call("exit_plan_mode", plan=PLAN)),
        reply("", call("write_file", path="hello.py", content="print('hi')\n")),
        reply("Created hello.py."),
    ], Mode.PLAN, approver=approver, events=events)
    first_tools = None
    result = agent.run("add hello.py")
    first_tools = tool_names(provider, 0)
    assert "exit_plan_mode" in first_tools and "write_file" not in first_tools and "update_plan" not in first_tools
    assert "unknown tool 'write_file'" in [m.content for m in agent.messages if m.role == "tool"][0]
    assert seen[0].plan == PLAN and seen[0].tool == "exit_plan_mode"
    assert [e.mode for e in events if isinstance(e, ModeChanged)] == ["auto-edit"]
    assert agent.policy.mode == Mode.AUTO_EDIT
    later_tools = tool_names(provider, 2)
    assert "write_file" in later_tools and "exit_plan_mode" not in later_tools
    assert [s.step for s in next(e for e in events if isinstance(e, PlanUpdated)).steps] == ["Create hello.py printing hi", "Run it to check"]
    assert result.status == "completed" and (project / "hello.py").exists()
    assert "PLAN mode" in provider.requests[0][0][0].content and "PLAN mode" not in provider.requests[2][0][0].content


def test_plan_feedback_keeps_planning_and_rejection_stops(project):
    answers = iter([ApprovalResult(False, feedback="also add a test"), ApprovalResult(False)])
    agent, _ = agent_for(project, [
        reply("", call("exit_plan_mode", plan=PLAN)),
        reply("", call("exit_plan_mode", plan=PLAN + "3. Add test_hello.py\n")),
    ], Mode.PLAN, approver=lambda req: next(answers))
    result = agent.run("add hello.py")
    outputs = [m.content for m in agent.messages if m.role == "tool"]
    assert "also add a test" in outputs[0] and "Stay in plan mode" in outputs[0]
    assert result.status == "blocked" and agent.policy.mode == Mode.PLAN


def test_switching_mode_between_messages_changes_the_tools(project):
    agent, provider = agent_for(project, [reply("hi"), reply("hi again")], Mode.READ_ONLY)
    agent.run("hello")
    agent.policy.mode = Mode.ASK
    agent.run("hello again")
    assert "write_file" not in tool_names(provider, 0) and "write_file" in tool_names(provider, 1)
