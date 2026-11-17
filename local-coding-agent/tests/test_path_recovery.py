"""Recovering from invented paths and repeated failing commands (seen with a 2.6B model in real use)."""

from __future__ import annotations

import sys

import pytest

from lca.agent.environment import output_hints
from lca.agent.loop import AgentLoop
from lca.agent.permissions import Mode, PermissionPolicy
from lca.config import AgentSettings
from lca.tools import Workspace, WorkspaceError, default_registry

from .conftest import ScriptedProvider, call, reply

PY = sys.executable.replace("\\", "/")


def test_invented_absolute_prefix_maps_to_the_real_file(project):
    ws = Workspace(project)
    assert ws.rel(ws.resolve("/c/Users/Project/src/app.py")) == "src/app.py"
    assert ws.rel(ws.resolve("/Users/someone/work/src/util.py")) == "src/util.py"
    assert ws.rel(ws.resolve(f"/home/x/{project.name}")) == "."


def test_real_outside_paths_stay_blocked_with_a_helpful_message(project):
    ws = Workspace(project)
    with pytest.raises(WorkspaceError) as e:
        ws.resolve("/Users/Project")
    msg = str(e.value)
    assert "relative to the workspace root" in msg and "src/" in msg and "Never guess absolute paths" in msg
    with pytest.raises(WorkspaceError):
        ws.resolve("../../outside.txt")
    with pytest.raises(WorkspaceError):
        ws.resolve("/c/Users/Project/.env")  # rescue never bypasses the secret check


def test_identical_failing_command_is_refused_the_third_time(project):
    bad = f'"{PY}" -c "import sys; sys.exit(2)"'
    agent = AgentLoop(
        ScriptedProvider([reply("", call("run_command", command=bad)) for _ in range(3)] + [reply("I give up on that command.")]),
        default_registry(), Workspace(project), system_prompt="s", settings=AgentSettings(auto_verify=False, max_stalled_turns=10),
        policy=PermissionPolicy(Mode.FULL),
    )
    agent.run("go")
    outputs = [m.content for m in agent.messages if m.role == "tool"]
    assert outputs[0].startswith("Exit code 2") and outputs[1].startswith("Exit code 2")
    assert "already failed twice and nothing in the project changed" in outputs[2]


def test_absolute_path_hint_for_failing_commands():
    hints = output_hints("ls -la /Users/Project 2>/dev/null", 2, "")
    assert any("Commands already run in the workspace root" in h for h in hints)
    assert not any("absolute path" in h for h in output_hints("ls src", 2, ""))


def test_a_tool_the_mode_keeps_refusing_is_withheld(project):
    from lca.messages import ToolCall

    def cmd(n):
        return ToolCall(id=f"r{n}", name="run_command", arguments={"command": f"py -c \"print({n})\""})

    provider = ScriptedProvider([reply("", cmd(1)), reply("", cmd(2)), reply("Done reading."), reply("ok")])
    agent = AgentLoop(provider, default_registry(), Workspace(project), system_prompt="s",
                      settings=AgentSettings(auto_verify=False), policy=PermissionPolicy(Mode.READ_ONLY))
    agent.run("look around")
    outputs = [m.content for m in agent.messages if m.role == "tool"]
    assert "removed for the rest of this task" in outputs[1]
    assert "run_command" in [t.name for t in provider.requests[0][1]]
    assert "run_command" not in [t.name for t in provider.requests[2][1]]
    agent.run("again")  # a new task gets the tool back
    assert "run_command" in [t.name for t in provider.requests[3][1]]


def test_wrap_up_without_a_plain_answer_falls_back_to_a_summary(project):
    failing = [reply("", call("read_file", path=f"missing{i}.py")) for i in range(3)]
    tool_call_again = reply("", call("read_file", path="src/app.py"))  # the "tool-free" answers still call tools
    agent = AgentLoop(ScriptedProvider([*failing, tool_call_again, tool_call_again]), default_registry(), Workspace(project),
                      system_prompt="s", settings=AgentSettings(auto_verify=False, max_consecutive_failures=3))
    result = agent.run("find the bug")
    assert result.status == "blocked"
    assert result.final_message.startswith("I could not finish this task") and "I did not change any files" in result.final_message


def test_the_model_is_warned_before_the_turn_limit(project):
    script = [reply("", call("read_file", path="src/app.py", start_line=i + 1)) for i in range(7)] + [reply("Here is what I found.")]
    provider = ScriptedProvider(script)
    agent = AgentLoop(provider, default_registry(), Workspace(project), system_prompt="s",
                      settings=AgentSettings(auto_verify=False, max_iterations=8))
    assert agent.run("review").final_message == "Here is what I found."
    warned = [m for m in provider.requests[6][0] if m.role == "tool" and "you have used 6 of 8 turns" in m.content]
    assert len(warned) == 1


def test_claimed_changes_without_any_change_are_reported(project):
    agent = AgentLoop(ScriptedProvider([reply("", call("read_file", path="src/app.py")), reply("Found 2 bugs. All fixes applied.")]),
                      default_registry(), Workspace(project), system_prompt="s", settings=AgentSettings(auto_verify=False))
    rep = agent.run("check app.py").report
    assert "The answer says changes were made, but no file was changed in this task." in rep.remaining


def test_read_only_modes_do_not_offer_memory(project, tmp_path_factory, monkeypatch):
    from lca.agent.rules import PermissionStore, RuleSet
    from lca.cli import Session, _make_agent
    from lca.config import Settings
    from lca.providers.base import ServerInfo

    monkeypatch.setenv("LCA_HOME", str(tmp_path_factory.mktemp("home")))
    provider = ScriptedProvider([reply("a"), reply("b")])
    s = Session(Settings(), provider, Workspace(project), ServerInfo(reachable=True, models=["m"], context_window=8192), 8192, [],
                RuleSet(), PermissionStore(project))
    agent = _make_agent(s, lambda e: None, s.policy(Mode.READ_ONLY))
    agent.run("review")
    agent.policy.mode = Mode.ASK
    agent.run("now change it")
    assert "remember" not in [t.name for t in provider.requests[0][1]] and "remember" in [t.name for t in provider.requests[1][1]]
