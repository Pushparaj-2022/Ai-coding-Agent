"""The agent's working loop: plan before editing, ask the user, editor diagnostics, automatic verification, report."""

from __future__ import annotations

import sys

import pytest

import lca.graph.agent_graph as agent_graph
from lca.agent.events import Diagnostic, PlanUpdated
from lca.agent.loop import AgentLoop
from lca.agent.permissions import ApprovalResult, Mode, PermissionPolicy
from lca.config import AgentSettings
from lca.graph.agent_graph import VERIFY_PREFIX, transcript
from lca.tools import Workspace, default_registry

from .conftest import ScriptedProvider, call, reply

PY = sys.executable.replace("\\", "/")
CHECK = f'"{PY}" check.py'


def make(project, script, *, policy=None, asker=None, diagnostics=None, **settings):
    events: list = []
    provider = ScriptedProvider(script)
    agent = AgentLoop(
        provider, default_registry(interaction=True), Workspace(project), system_prompt="s",
        settings=AgentSettings(**settings), policy=policy or PermissionPolicy(Mode.FULL),
        events=events.append, asker=asker, diagnostics=diagnostics,
    )
    return agent, provider, events


@pytest.fixture
def no_verify(monkeypatch):
    monkeypatch.setattr(agent_graph, "verify_commands", lambda root: [])


@pytest.fixture
def check_script(project, monkeypatch):
    """A verification command that passes once src/app.py contains 'fixed'."""
    (project / "check.py").write_text(
        "import sys\nok = 'fixed' in open('src/app.py').read()\nprint('check ok' if ok else 'FAIL: app.py is not fixed')\nsys.exit(0 if ok else 1)\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(agent_graph, "verify_commands", lambda root: [CHECK])


def tool_outputs(agent):
    return [m.content for m in agent.messages if m.role == "tool"]


# ---------------------------------------------------------------- plan


def test_enforced_plan_refuses_changes_until_there_is_one(project, no_verify):
    agent, _, events = make(project, [
        reply("", call("write_file", path="src/new.py", content="x = 1\n")),
        reply("", call("update_plan", steps=["Create src/new.py", "Check it imports"])),
        reply("", call("write_file", path="src/new.py", content="x = 1\n")),
        reply("Created src/new.py."),
    ], require_plan="enforce")
    result = agent.run("add src/new.py")
    outputs = tool_outputs(agent)
    assert "no plan yet" in outputs[0]
    assert (project / "src" / "new.py").read_text() == "x = 1\n"
    plans = [e for e in events if isinstance(e, PlanUpdated)]
    assert [s.step for s in plans[0].steps] == ["Create src/new.py", "Check it imports"]
    rep = result.report
    assert rep.files_changed == ["src/new.py"] and len(rep.plan) == 2
    assert rep.verification.status == "unverified" and "No build or test command" in rep.verification.note
    assert any("not marked done" in r for r in rep.remaining)


def test_default_plan_reminder_lets_the_change_through_once(project, no_verify):
    agent, _, _ = make(project, [
        reply("", call("write_file", path="a.txt", content="a\n")),
        reply("", call("write_file", path="b.txt", content="b\n")),
        reply("Done."),
    ])
    result = agent.run("create a and b")
    outputs = tool_outputs(agent)
    assert (project / "a.txt").exists() and (project / "b.txt").exists()
    assert "record your plan with update_plan" in outputs[0] and "update_plan" not in outputs[1]
    assert result.status == "completed"


def test_ignored_failing_test_run_sends_the_agent_back(project, check_script):
    agent, provider, _ = make(project, [
        reply("", call("run_command", command=CHECK)),
        reply("Fixed, all tests pass."),  # a false claim: nothing changed and the check failed
        reply("", call("read_file", path="src/app.py")),
        reply("", call("write_file", path="src/app.py", content="# fixed\n")),
        reply("Now it is really fixed."),
    ])
    result = agent.run("make the check pass")
    feedback = provider.requests[2][0][-1]
    assert feedback.content.startswith(VERIFY_PREFIX) and "Do not claim it passes" in feedback.content
    assert result.final_message == "Now it is really fixed." and result.report.verification.status == "passed"


def test_ignored_failure_is_reported_when_rounds_run_out(project, check_script):
    agent, _, _ = make(project, [reply("", call("run_command", command=CHECK)), reply("All good.")], verify_rounds=0)
    rep = agent.run("check it").report
    assert rep.verification.status == "failed" and any("still fails" in r for r in rep.remaining)


def test_questions_need_no_plan_and_no_verification(project):
    agent, _, _ = make(project, [reply("", call("read_file", path="src/app.py")), reply("add() returns a + b.")])
    result = agent.run("what does add do?")
    assert result.status == "completed"
    assert result.report.verification.status == "not_needed" and not result.report.remaining and not result.report.plan


def test_plan_steps_marked_done_are_not_remaining(project, no_verify):
    agent, _, _ = make(project, [
        reply("", call("update_plan", steps=[{"step": "Create a.txt", "status": "in_progress"}])),
        reply("", call("write_file", path="a.txt", content="a\n")),
        reply("", call("update_plan", steps=[{"step": "Create a.txt", "status": "done"}])),
        reply("Done."),
    ])
    rep = agent.run("create a.txt").report
    assert rep.plan[0].status == "done" and not any("Plan step" in r for r in rep.remaining)


def test_plan_not_required_without_the_tool(project):
    agent = AgentLoop(
        ScriptedProvider([reply("", call("write_file", path="a.txt", content="a\n")), reply("ok")]),
        default_registry(), Workspace(project), system_prompt="s", policy=PermissionPolicy(Mode.FULL),
    )
    assert agent.run("create a.txt").status == "completed" and (project / "a.txt").exists()


# ---------------------------------------------------------------- ask the user


def test_ask_user_waits_for_the_answer(project):
    asked = []

    def asker(question, options):
        asked.append((question, options))
        return options[1]

    agent, provider, _ = make(project, [
        reply("", call("ask_user", question="Which storage should I use?", options=["sqlite", "postgres"])),
        reply("Using postgres."),
    ], asker=asker)
    agent.run("add persistence")
    assert asked == [("Which storage should I use?", ["sqlite", "postgres"])]
    second = provider.requests[1][0]
    assert any(m.role == "tool" and "The user answered: postgres" in m.content for m in second)


def test_ask_user_without_a_user_tells_the_model_to_assume(project):
    agent, _, _ = make(project, [reply("", call("ask_user", question="Which one?")), reply("Assumed sqlite.")])
    agent.run("add persistence")
    assert "No user is available" in tool_outputs(agent)[0]


# ---------------------------------------------------------------- editor diagnostics


def test_editor_diagnostics_are_added_to_edit_results_and_report(project, no_verify):
    requested = []

    def diagnostics(paths):
        requested.append(paths)
        return [
            Diagnostic(path="src/app.py", line=5, column=12, message="'b' is not defined", source="pyflakes"),
            Diagnostic(path="src/app.py", line=1, severity="warning", message="'os' imported but unused", source="pyflakes"),
            Diagnostic(path="src/other.py", line=1, message="unrelated"),
        ]

    agent, _, _ = make(project, [
        reply("", call("update_plan", steps=["Edit add"])),
        reply("", call("read_file", path="src/app.py")),
        reply("", call("edit_file", path="src/app.py", old_string="return a + b", new_string="return a + c")),
        reply("Edited."),
    ], diagnostics=diagnostics)
    result = agent.run("change add")
    edit_out = tool_outputs(agent)[2]
    assert "[Editor diagnostics for src/app.py after this change]" in edit_out
    assert "src/app.py:5:12 error: 'b' is not defined [pyflakes]" in edit_out and "unused" in edit_out
    assert "unrelated" not in edit_out
    assert requested[0] == ["src/app.py"]
    rep = result.report
    assert [d.message for d in rep.problems] == ["'b' is not defined"]  # errors in changed files only
    assert any("1 error(s)" in r for r in rep.remaining)


def test_get_diagnostics_without_an_editor(project):
    agent, _, _ = make(project, [reply("", call("get_diagnostics")), reply("No editor.")])
    agent.run("any type errors?")
    assert "not available" in tool_outputs(agent)[0]


def test_get_diagnostics_with_an_editor(project):
    agent, _, _ = make(project, [reply("", call("get_diagnostics", paths=["src/app.py"])), reply("One error.")],
                       diagnostics=lambda paths: [Diagnostic(path="src/app.py", line=3, message="bad")])
    agent.run("any type errors?")
    assert "1 error(s), 0 warning(s)" in tool_outputs(agent)[0]


# ---------------------------------------------------------------- automatic verification


def test_failed_verification_goes_back_to_the_agent(project, check_script):
    agent, provider, events = make(project, [
        reply("", call("update_plan", steps=["Fix app.py"])),
        reply("", call("read_file", path="src/app.py")),
        reply("", call("write_file", path="src/app.py", content="# still broken\n")),
        reply("Done."),
        reply("", call("write_file", path="src/app.py", content="# fixed\n")),
        reply("Fixed for real."),
    ])
    result = agent.run("fix app.py")
    assert result.status == "completed" and result.final_message == "Fixed for real."
    feedback = provider.requests[4][0][-1]
    assert feedback.role == "user" and feedback.content.startswith(VERIFY_PREFIX) and "FAIL: app.py is not fixed" in feedback.content
    rep = result.report
    assert rep.verification.status == "passed" and rep.verification.commands == [{"command": CHECK, "exit_code": 0}]
    verify_events = [e for e in events if getattr(e, "agent", None) == "verify" and e.type == "tool_call_finished"]
    assert [e.exit_code for e in verify_events] == [1, 0]
    notices = [i["text"] for i in transcript(agent.history()) if i["role"] == "notice"]
    assert any(n.startswith("Verification failed.") for n in notices)


def test_verification_failure_is_reported_when_rounds_run_out(project, check_script):
    agent, _, _ = make(project, [
        reply("", call("update_plan", steps=["Fix app.py"])),
        reply("", call("read_file", path="src/app.py")),
        reply("", call("write_file", path="src/app.py", content="# broken\n")),
        reply("Done."),
    ], verify_rounds=0)
    rep = agent.run("fix app.py").report
    assert rep.verification.status == "failed"
    assert any("still fails (exit code 1)" in r for r in rep.remaining)


def test_verification_reuses_the_agents_own_passing_run(project, check_script):
    agent, _, _ = make(project, [
        reply("", call("update_plan", steps=["Fix app.py"])),
        reply("", call("read_file", path="src/app.py")),
        reply("", call("write_file", path="src/app.py", content="# fixed\n")),
        reply("", call("run_command", command=CHECK)),
        reply("Fixed and checked."),
    ])
    result = agent.run("fix app.py")
    assert result.report.verification.status == "passed"
    assert sum(1 for t in result.tool_calls if t.name == "run_command") == 1  # not run a second time


def test_test_runner_finding_no_tests_is_not_a_failure(project, monkeypatch):
    no_tests = f'"{PY}" -c "import sys; sys.exit(5)" pytest'  # exit code 5: pytest/unittest collected no tests
    monkeypatch.setattr(agent_graph, "verify_commands", lambda root: [no_tests])
    agent, provider, _ = make(project, [
        reply("", call("update_plan", steps=["x"])), reply("", call("write_file", path="a.txt", content="a\n")), reply("ok"),
    ])
    result = agent.run("go")
    assert result.status == "completed" and len(provider.requests) == 3  # not sent back to "fix" the tests
    assert result.report.verification.status == "unverified" and "found no tests" in result.report.verification.note


def test_verification_off(project, check_script):
    agent, _, _ = make(project, [
        reply("", call("update_plan", steps=["x"])), reply("", call("write_file", path="a.txt", content="a\n")), reply("ok"),
    ], auto_verify=False)
    assert agent.run("go").report.verification.status == "unverified"


# ---------------------------------------------------------------- report


def test_declined_actions_are_reported(project, no_verify):
    policy = PermissionPolicy(Mode.ASK, lambda req: ApprovalResult(False))
    agent, _, _ = make(project, [
        reply("", call("update_plan", steps=["Create src/new.py"])),
        reply("", call("write_file", path="src/new.py", content="x\n")),
    ], policy=policy)
    result = agent.run("add src/new.py")
    assert result.status == "blocked"
    assert result.report.needs_approval == ["write_file: src/new.py"]
