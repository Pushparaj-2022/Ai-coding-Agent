from __future__ import annotations

import threading

import pytest

from lca.agent.events import RunFinished, TextDelta, ToolCallFinished, ToolCallStarted
from lca.agent.loop import AgentLoop
from lca.cancel import CancelToken
from lca.config import AgentSettings
from lca.messages import Message, ModelResponse, ToolCall
from lca.providers.base import ContextOverflowError, ProviderConnectionError
from lca.tools import Workspace, default_registry

from .conftest import ScriptedProvider, call, reply


def make_agent(project, script, events=None, window=8192, **agent_settings):
    provider = ScriptedProvider(script)
    agent = AgentLoop(
        provider,
        default_registry(),
        Workspace(project),
        system_prompt="system",
        settings=AgentSettings(**agent_settings),
        events=(events.append if events is not None else None),
        context_window=window,
    )
    return agent, provider


def tool_messages(agent):
    return [m for m in agent.messages if m.role == "tool"]


def test_reads_file_then_answers(project):
    events = []
    agent, provider = make_agent(
        project,
        [reply("Let me read it.", call("read_file", path="src/app.py")), reply("add() returns a + b (src/app.py:4).")],
        events,
    )
    result = agent.run("What does add do?")
    assert result.status == "completed"
    assert result.final_message == "add() returns a + b (src/app.py:4)."
    assert result.iterations == 2 and result.files_read == ["src/app.py"]
    assert [t.ok for t in result.tool_calls] == [True]
    # the model's second request contains the real file content
    second_request = provider.requests[1][0]
    assert any(m.role == "tool" and "def add(a, b):" in m.content for m in second_request)
    types = [e.type for e in events]
    assert types.index("tool_call_started") < types.index("tool_call_finished") < types.index("run_finished")
    assert "".join(e.text for e in events if isinstance(e, TextDelta)).startswith("Let me read it.")


def test_multiple_tool_calls_in_one_turn(project):
    agent, _ = make_agent(project, [reply("", call("read_file", path="src/app.py"), call("read_file", path="src/util.py")), reply("done")])
    result = agent.run("read both")
    assert len(result.tool_calls) == 2 and result.files_read == ["src/app.py", "src/util.py"]


def test_errors_are_fed_back_and_agent_recovers(project):
    bad_args = ToolCall(id="x", name="read_file", raw_arguments='{"path": ', parse_error="arguments are not a valid JSON object")
    agent, provider = make_agent(
        project,
        [
            reply("", call("rm_rf", path="/")),
            reply("", bad_args),
            reply("", call("read_file")),
            reply("", call("read_file", path="../../etc/passwd")),
            reply("", call("read_file", path=".env")),
            reply("Recovered."),
        ],
        max_consecutive_failures=10,
    )
    result = agent.run("go")
    assert result.status == "completed"
    outputs = [m.content for m in tool_messages(agent)]
    assert "unknown tool 'rm_rf'" in outputs[0]
    assert "could not parse" in outputs[1]
    assert "invalid arguments" in outputs[2]
    assert "outside the workspace" in outputs[3]
    assert "secrets" in outputs[4]
    assert not any(t.ok for t in result.tool_calls)


def test_truncated_call_gets_specific_hint(project):
    cut = ToolCall(id="x", name="read_file", raw_arguments='{"pa', parse_error="bad")
    agent, _ = make_agent(project, [reply("", cut, finish="length"), reply("ok")])
    agent.run("go")
    assert "output token limit" in tool_messages(agent)[0].content


def test_truncated_write_file_is_told_to_append_in_parts(project):
    cut = ToolCall(id="x", name="write_file", raw_arguments='{"path": "big.py", "content": "x = 1\\n', parse_error="bad")
    agent, _ = make_agent(project, [reply("", cut, finish="length"), reply("ok")])
    agent.run("go")
    msg = tool_messages(agent)[0].content
    assert "output token limit" in msg and "append=true" in msg
    assert not (project / "big.py").exists()


def test_repeated_read_is_not_run_again(project):
    same = lambda: call("read_file", path="src/app.py")  # noqa: E731
    agent, _ = make_agent(project, [reply("", same()), reply("", same()), reply("fine")])
    result = agent.run("go")
    assert [t.ok for t in result.tool_calls] == [True, False]
    assert "already made this exact read_file call" in tool_messages(agent)[1].content


def test_read_again_after_a_change_is_allowed(project):
    read = lambda: call("read_file", path="src/app.py")  # noqa: E731
    edit = call("edit_file", path="src/app.py", old_string="return a + b", new_string="return b + a")
    from lca.agent.permissions import Mode, PermissionPolicy

    provider = ScriptedProvider([reply("", read()), reply("", edit), reply("", read()), reply("done")])
    agent = AgentLoop(
        provider, default_registry(), Workspace(project), system_prompt="s",
        settings=AgentSettings(auto_verify=False, require_plan="off"), policy=PermissionPolicy(Mode.FULL),
    )
    result = agent.run("go")
    assert [t.ok for t in result.tool_calls] == [True, True, True]


def test_turns_of_only_repeats_stop_the_run(project):
    same = lambda: call("list_files", path="src")  # noqa: E731
    agent, provider = make_agent(project, [reply("", same()) for _ in range(6)], wrap_up=False)
    result = agent.run("go")
    assert result.status == "blocked" and "repeated calls" in result.final_message
    assert len(provider.requests) == 4  # one real call, then three turns of repeats


def test_same_successful_command_is_refused_the_third_time(project):
    import sys

    py = sys.executable.replace("\\", "/")
    from lca.agent.permissions import Mode, PermissionPolicy

    cmd = lambda: call("run_command", command=f'"{py}" -c "print(1)"')  # noqa: E731
    provider = ScriptedProvider([reply("", cmd()), reply("", cmd()), reply("", cmd()), reply("done")])
    agent = AgentLoop(
        provider, default_registry(), Workspace(project), system_prompt="s",
        settings=AgentSettings(auto_verify=False), policy=PermissionPolicy(Mode.FULL),
    )
    result = agent.run("go")
    assert [t.ok for t in result.tool_calls] == [True, True, False]
    assert "already ran 2 times" in tool_messages(agent)[2].content


def test_identical_call_that_keeps_failing_is_refused(project):
    same = lambda: call("read_file", path="src/missing.py")  # noqa: E731
    agent, _ = make_agent(project, [reply("", same()) for _ in range(3)] + [reply("fine")], max_identical_calls=2)
    agent.run("go")
    contents = [m.content for m in tool_messages(agent)]
    assert "not found" in contents[0]
    assert "already failed 2 times" in contents[2]


def test_missing_file_error_suggests_real_paths(project):
    agent, _ = make_agent(project, [reply("", call("read_file", path="app.py")), reply("ok")])
    agent.run("go")
    assert "Did you mean: src/app.py" in tool_messages(agent)[0].content


def test_consecutive_failures_block_the_run(project):
    agent, _ = make_agent(project, [reply("", call("nope")) for _ in range(5)], max_consecutive_failures=3)
    result = agent.run("go")
    assert result.status == "blocked" and result.iterations == 3
    assert "consecutive" in result.final_message


def test_max_iterations(project):
    agent, _ = make_agent(project, [reply("", call("list_files", max_results=i + 1)) for i in range(3)], max_iterations=3, wrap_up=False)
    result = agent.run("loop forever")
    assert result.status == "max_iterations" and result.iterations == 3


def test_model_reports_blocked(project):
    agent, _ = make_agent(project, [reply("BLOCKED: the file you mention does not exist.")])
    assert agent.run("fix foo.py").status == "blocked"


def test_empty_replies_are_nudged_then_error(project):
    agent, provider = make_agent(project, [reply(""), reply(""), reply("")])
    result = agent.run("go")
    assert result.status == "error" and "empty" in result.final_message
    assert provider.requests[1][0][-1].content.startswith("Your reply was empty")


def test_empty_reply_recovers_after_nudge(project):
    agent, _ = make_agent(project, [reply(""), reply("answer")])
    assert agent.run("go").final_message == "answer"


def test_too_many_calls_per_turn_are_capped(project):
    calls = [call("list_files", max_results=i + 1) for i in range(5)]
    agent, _ = make_agent(project, [reply("", *calls), reply("done")], max_tool_calls_per_turn=2)
    result = agent.run("go")
    assert len(result.tool_calls) == 2
    assert any("only the first 2 tool calls" in m.content for m in agent.messages if m.role == "assistant")


def test_provider_error_ends_run_cleanly(project):
    # connection errors are retried (max_retries=2), so three failures exhaust them
    agent, provider = make_agent(project, [ProviderConnectionError("server down")] * 3)
    result = agent.run("go")
    assert result.status == "error" and "server down" in result.final_message
    assert len(provider.requests) == 3


def test_context_overflow_trims_and_retries(project):
    agent, provider = make_agent(project, [ContextOverflowError("too big"), reply("fits now")])
    result = agent.run("go")
    assert result.status == "completed" and len(provider.requests) == 2


def test_cancel_before_model_call(project):
    token = CancelToken()
    token.cancel()
    agent, provider = make_agent(project, [reply("never")])
    result = agent.run("go", token)
    assert result.status == "cancelled" and not provider.requests


def test_cancel_during_tool_round_repairs_history(project):
    token = CancelToken()

    def cancel_then_reply(messages):
        token.cancel()
        return reply("", call("read_file", path="src/app.py"), call("read_file", path="src/util.py"))

    agent, _ = make_agent(project, [cancel_then_reply, reply("second run answer")])
    result = agent.run("go", token)
    assert result.status == "cancelled"
    # every tool call has a result, and the conversation can continue afterwards
    asked = [c.id for m in agent.messages if m.role == "assistant" for c in m.tool_calls]
    answered = [m.tool_call_id for m in agent.messages if m.role == "tool"]
    assert asked == answered
    assert agent.run("continue").final_message == "second run answer"


def test_cancel_from_another_thread(project):
    token = CancelToken()
    started = threading.Event()

    def slow(messages):
        started.set()
        token.wait(5)
        token.raise_if_cancelled()
        return reply("too late")

    agent, _ = make_agent(project, [slow])
    t = threading.Timer(0.1, token.cancel)
    t.start()
    assert agent.run("go", token).status == "cancelled"


def test_multi_turn_conversation_keeps_history(project):
    agent, provider = make_agent(project, [reply("first"), reply("second")])
    agent.run("one")
    agent.run("two")
    roles = [m.role for m in provider.requests[1][0]]
    assert roles == ["system", "user", "assistant", "user"]


def test_secrets_redacted_before_reaching_model(project):
    (project / "settings.py").write_text("token = 'ghp_abcdefghijklmnopqrstuvwxyz0123456789'\n", encoding="utf-8")
    agent, provider = make_agent(project, [reply("", call("read_file", path="settings.py")), reply("ok")])
    agent.run("go")
    sent = " ".join(m.content for m in provider.requests[1][0])
    assert "ghp_abcdef" not in sent and "[REDACTED" in sent


def test_context_budget_trims_old_tool_output(project):
    (project / "big.py").write_text("\n".join(f"value_{i} = {i}" for i in range(900)), encoding="utf-8")
    script = [reply("", call("read_file", path="big.py", start_line=1 + 300 * i, end_line=300 * (i + 1))) for i in range(3)]
    events = []
    agent, provider = make_agent(project, script + [reply("done")], events, window=6000)
    result = agent.run("read all of big.py")
    assert result.status == "completed"
    assert any(e.type == "notice" for e in events)
    assert agent.budget.estimate(provider.requests[-1][0]) <= agent.budget.limit


def test_events_carry_tool_output(project):
    events = []
    agent, _ = make_agent(project, [reply("", call("read_file", path="src/util.py")), reply("ok")], events)
    agent.run("go")
    started = next(e for e in events if isinstance(e, ToolCallStarted))
    finished = next(e for e in events if isinstance(e, ToolCallFinished))
    assert started.arguments == {"path": "src/util.py"} and "def helper" in finished.output
    assert isinstance(events[-1], RunFinished)


def test_stall_guard_stops_endless_failing_commands(project):
    import sys

    py = sys.executable.replace("\\", "/")
    from lca.agent.permissions import Mode, PermissionPolicy

    script = [reply("", call("run_command", command=f'"{py}" -c "import sys; sys.exit({i % 5 + 1})"')) for i in range(10)]
    provider = ScriptedProvider(script)
    agent = AgentLoop(
        provider, default_registry(), Workspace(project), system_prompt="s",
        settings=AgentSettings(max_stalled_turns=4, wrap_up=False), policy=PermissionPolicy(Mode.FULL),
    )
    result = agent.run("go")
    assert result.status == "blocked" and "nothing succeeded in 4 consecutive turns" in result.final_message
    assert len(provider.requests) == 4


def test_one_failing_test_run_does_not_stop_the_agent(project):
    import sys

    py = sys.executable.replace("\\", "/")
    from lca.agent.permissions import Mode, PermissionPolicy

    provider = ScriptedProvider([
        reply("", call("run_command", command=f'"{py}" -c "import sys; sys.exit(1)"')),
        reply("", call("read_file", path="src/app.py")),
        reply("", call("run_command", command=f'"{py}" -c "print(1)"')),
        reply("Fixed and verified."),
    ])
    agent = AgentLoop(
        provider, default_registry(), Workspace(project), system_prompt="s",
        settings=AgentSettings(max_stalled_turns=2), policy=PermissionPolicy(Mode.FULL),
    )
    assert agent.run("go").status == "completed"


def test_stopped_run_still_ends_with_an_answer(project):
    script = [reply("", call("read_file", path=f"nope{i}.py")) for i in range(3)] + [reply("The project has src/app.py; nope*.py do not exist.")]
    agent, provider = make_agent(project, script, max_consecutive_failures=3)
    result = agent.run("summarize the project")
    assert result.status == "completed" and "src/app.py" in result.final_message
    assert provider.requests[-1][1] == []  # the wrap-up call offers no tools
    assert "Stop using tools now" in provider.requests[-1][0][-1].content
