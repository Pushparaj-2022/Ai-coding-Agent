"""Guards against the loops small local models fall into: re-reading after compaction, retrying variations of a
failing command, alternating between the same calls, drifting through unrelated files, and reading agent internals."""

from __future__ import annotations

import sys

import pytest

from lca.agent.loop import AgentLoop
from lca.agent.permissions import Mode, PermissionPolicy
from lca.config import AgentSettings
from lca.graph.agent_graph import _task_text, command_family, find_cycle
from lca.tools import Workspace, default_registry
from lca.tools.workspace import WorkspaceError

from .conftest import ScriptedProvider, call, reply

PY = sys.executable.replace("\\", "/")


def make(project, script, **settings):
    provider = ScriptedProvider(script)
    agent = AgentLoop(
        provider, default_registry(), Workspace(project), system_prompt="s",
        settings=AgentSettings(auto_verify=False, **settings), policy=PermissionPolicy(Mode.FULL),
    )
    return agent, provider


def tool_contents(agent):
    return [m.content for m in agent.messages if m.role == "tool"]


@pytest.mark.parametrize(
    ("command", "family"),
    [
        ('cd damn && py run-forge.py test --isolate --match-path "test/x.t.sol"', "py run-forge.py test"),
        ("py run-forge.py test --isolate 2>&1 | head -40", "py run-forge.py test"),
        ("cd 'a b' && cd c && py   run-forge.py  test", "py run-forge.py test"),
        ("npm run build", "npm run build"),
        ("Set-Location src; pytest -q tests/test_a.py", "pytest tests/test_a.py"),
    ],
)
def test_command_family_ignores_cosmetic_variations(command, family):
    assert command_family(command) == family


def test_find_cycle():
    assert find_cycle(list("ABABAB")) == 2
    assert find_cycle(list("xABCABCABC")) == 3
    assert find_cycle(list("AAA")) == 1
    assert find_cycle(list("ABCDEF")) == 0
    assert find_cycle(list("ABAB")) == 0  # needs three repetitions


def test_task_text_drops_editor_context():
    msg = "check for bugs\n\n[For reference only: the user has src/a.py open in the editor. Ignore this unless...]"
    assert _task_text(msg) == "check for bugs"


def test_reread_is_capped_even_when_repeat_memory_is_cleared(project):
    # a command (risk=execute) clears seen_results the same way compaction does; the read cap must still hold
    read = lambda: call("read_file", path="src/app.py")  # noqa: E731
    echo = lambda i: call("run_command", command=f'"{PY}" -c "print({i})"')  # noqa: E731
    agent, _ = make(project, [reply("", read()), reply("", echo(1)), reply("", read()), reply("", echo(2)), reply("", read()), reply("done")])
    result = agent.run("go")
    assert [t.ok for t in result.tool_calls] == [True, True, True, True, False]
    assert "already made this exact read_file call 2 times" in tool_contents(agent)[4]


def test_failing_command_variations_get_a_warning_then_are_refused(project):
    variations = [f'"{PY}" -c "import sys; sys.exit(2)" --v{i}' for i in range(8)]
    script = []
    for i, v in enumerate(variations):
        # a successful read in between keeps the stall guard from ending the run first
        script.append(reply("", call("run_command", command=v), call("read_file", path="src/app.py", start_line=i + 1, end_line=i + 1)))
    agent, _ = make(project, script + [reply("BLOCKED: it keeps failing")], max_stalled_turns=50)
    agent.run("go")
    commands = [c for c in tool_contents(agent) if "sys.exit" in c or "commands like" in c or "exit code" in c.lower()]
    assert any("have now failed 3 times" in c for c in tool_contents(agent))
    assert any("already failed 6 times in this task" in c for c in tool_contents(agent))
    assert commands


def test_alternating_reads_get_a_cycle_note(project):
    a = lambda: call("read_file", path="src/app.py")  # noqa: E731
    b = lambda: call("read_file", path="src/util.py")  # noqa: E731
    agent, _ = make(project, [reply("", a(), b()), reply("", a(), b()), reply("", a(), b()), reply("done")])
    agent.run("go")
    assert any("going in circles" in c for c in tool_contents(agent))


def test_no_new_information_ends_the_run(project):
    script = [reply("", call("read_file", path="src/app.py"))] + [reply("", call("read_file", path="src/app.py")) for _ in range(12)]
    agent, provider = make(project, script + [reply("summary")], wrap_up=False, max_stalled_turns=50)
    result = agent.run("go")
    assert result.status == "blocked"
    assert len(provider.requests) < 12


def test_focus_note_restates_the_task(project):
    script = [reply("", call("read_file", path="src/app.py", start_line=i, end_line=i)) for i in range(1, 14)] + [reply("done")]
    agent, _ = make(project, script, max_stalled_turns=50)
    agent.run("explain add()\n\n[For reference only: the user has src/util.py open in the editor.]")
    notes = [c for c in tool_contents(agent) if "Focus check" in c]
    assert notes and '"explain add()"' in notes[0]


def test_current_request_is_restated_after_the_first_turn(project):
    agent, provider = make(project, [reply("", call("read_file", path="src/app.py")), reply("done")])
    agent.run("explain add()")
    first_system = provider.requests[0][0][0].content
    second_system = provider.requests[1][0][0].content
    assert "# Current request" not in first_system
    assert "# Current request" in second_system and "explain add()" in second_system


def test_agent_internals_are_not_readable(project):
    (project / ".lca" / "checkpoints" / "abc").mkdir(parents=True)
    (project / ".lca" / "checkpoints" / "abc" / "00001.json").write_text("{}")
    (project / ".lca" / "memory.md").write_text("# memory")
    ws = Workspace(project)
    with pytest.raises(WorkspaceError, match="internal data"):
        ws.resolve(".lca/checkpoints/abc/00001.json")
    with pytest.raises(WorkspaceError):
        ws.resolve(".lca/sessions.db")
    assert ws.resolve(".lca/memory.md").name == "memory.md"


def test_invented_next_turn_is_cut_and_its_tool_calls_never_run():
    from lca.providers.base import ResponseBuilder

    b = ResponseBuilder({"read_file"})
    b.text("The guard is at line 140.<channel|>\n<|turn>tool\nfake output\n<|tool_call>call:read_file{path:<|\"|>x.py<|\"|>}<tool_call|>")
    r = b.build()
    assert r.content == "The guard is at line 140."
    assert r.tool_calls == []


def test_gemma_thought_channel_becomes_reasoning():
    from lca.providers.base import ResponseBuilder

    b = ResponseBuilder(set())
    b.text("<|channel>thought\nlet me think<channel|>Answer.")
    r = b.build()
    assert r.content == "Answer." and "let me think" in r.reasoning


def test_stream_filter_hides_invented_turns():
    from lca.providers.toolcall_parser import StreamFilter

    out = []
    f = StreamFilter(out.append)
    for chunk in ["Done.<chan", "nel|>\n<|tu", "rn>tool\nfake"]:
        f.push(chunk)
    f.end()
    assert "".join(out).strip() == "Done."


def test_no_server_stop_sequences_and_stream_is_closed_at_an_invented_turn():
    import httpx
    import json as _json

    from lca.config import ProviderSettings
    from lca.providers.openai_compatible import OpenAICompatibleProvider

    sent = []

    def handler(request):
        sent.append(_json.loads(request.content))
        chunks = ["The answer is 42.", "<|turn>tool\n", "fake result", "<|tool_call>call:read_file{path:<|\"|>x<|\"|>}<tool_call|>"]
        events = [_json.dumps({"choices": [{"delta": {"content": c}}]}) for c in chunks]
        body = "".join(f"data: {e}\n\n" for e in events) + "data: [DONE]\n\n"
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body.encode())

    p = OpenAICompatibleProvider(ProviderSettings(base_url="http://127.0.0.1:8080/v1", model="m"), transport=httpx.MockTransport(handler))
    from lca.messages import Message
    from lca.tools import default_registry

    r = p.chat([Message.user("q")], default_registry().specs(["read_file"]))
    assert "stop" not in sent[0]
    assert r.content == "The answer is 42." and r.tool_calls == []


@pytest.mark.parametrize(
    ("text", "announces"),
    [
        ("I found REPEAT_GUARDED in agent_graph.py. Let me read the rest of the file to understand it.", True),
        ("Now I'll run the tests.", True),
        ("add() returns a + b (src/app.py:4).", False),
        ("Done. Let me know if you want me to fix anything else.", False),
        ("Hello! What can I do for you today?", False),
        ("I'll search for the constant. Now let me search for the REPEAT_GUARDED constant:", True),
        ("Here is the summary:", True),
    ],
)
def test_announces_next_step(text, announces):
    from lca.graph.agent_graph import _announces_next_step

    assert _announces_next_step(text) is announces


def test_announced_step_without_a_call_is_nudged(project):
    agent, provider = make(project, [
        reply("Let me read the file."),
        reply("", call("read_file", path="src/app.py")),
        reply("add() returns a + b."),
    ])
    result = agent.run("what does add do")
    assert result.status == "completed" and result.final_message == "add() returns a + b."
    assert [t.name for t in result.tool_calls] == ["read_file"]


def test_common_wrong_argument_names_are_accepted(project):
    agent, _ = make(project, [reply("", call("search_code", query="def add")), reply("found")])
    result = agent.run("find add")
    assert [t.ok for t in result.tool_calls] == [True]


def test_endless_announcing_ends_with_a_tool_free_answer(project):
    script = [reply("Let me read the file.") for _ in range(3)] + [reply("add() returns a + b.")]
    agent, provider = make(project, script)
    result = agent.run("what does add do")
    assert result.final_message == "add() returns a + b."
    assert provider.requests[-1][1] == []  # the wrap-up call offers no tools
