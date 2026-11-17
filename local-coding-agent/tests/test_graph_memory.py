"""LangGraph engine: persistent sessions, long-term memory, compaction, the LangChain chat model."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from lca.agent.context import ContextBudget
from lca.agent.loop import AgentLoop, repair, transcript
from lca.agent.permissions import Mode, PermissionPolicy
from lca.config import AgentSettings
from lca.graph.chat_model import LocalChatModel
from lca.graph.compaction import PROGRESS_PREFIX, SUMMARY_PREFIX, Compactor
from lca.memory.sessions import SessionStore
from lca.memory.store import MemoryStore
from lca.tools import Workspace, default_registry

from .conftest import ScriptedProvider, call, reply


def make(project, script, store=None, thread=None, memory=None, **settings):
    provider = ScriptedProvider(script)
    agent = AgentLoop(
        provider, default_registry(memory=True), Workspace(project), system_prompt="base prompt",
        settings=AgentSettings(**settings), policy=PermissionPolicy(Mode.FULL),
        checkpointer=store.checkpointer if store else None, thread_id=thread, memory=memory,
    )
    return agent, provider


# ------------------------------------------------------------------ history / sessions


def test_history_survives_a_restart(project):
    store = SessionStore(project)
    info = store.create()
    agent, _ = make(project, [reply("", call("read_file", path="src/app.py")), reply("add() returns a + b")], store, info.id)
    agent.run("what does add do?")
    store.touch(info.id, "what does add do?")
    store.close()

    # a brand-new process: new store, new agent, same session id
    store2 = SessionStore(project)
    agent2, provider2 = make(project, [reply("It is in src/app.py.")], store2, info.id)
    assert agent2.run("where is it?").final_message == "It is in src/app.py."
    sent = [m.content for m in provider2.requests[0][0]]
    assert "what does add do?" in sent and "add() returns a + b" in sent  # the old conversation was sent to the model
    listed = store2.list()
    assert listed[0].id == info.id and listed[0].title == "what does add do?"
    items = transcript(agent2.history())
    assert [i["role"] for i in items][:3] == ["user", "tool", "assistant"]
    store2.close()


def test_sessions_are_pruned_to_the_latest_checkpoint(project):
    store = SessionStore(project)
    info = store.create()
    agent, _ = make(project, [reply("", call("list_files")), reply("done")], store, info.id)
    agent.run("go")
    store.touch(info.id, "go")
    rows = store._db.execute("SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?", (info.id,)).fetchone()[0]
    assert rows == 1
    agent2, _ = make(project, [reply("still here")], store, info.id)
    assert agent2.run("again").status == "completed" and len(agent2.history()) == 6
    store.delete(info.id)
    assert store.get(info.id) is None
    store.close()


def test_private_dir_is_gitignored(project):
    SessionStore(project).close()
    assert (project / ".lca" / ".gitignore").read_text().strip().endswith("*")


# ------------------------------------------------------------------ long-term memory


def test_memory_store_add_dedupe_remove_and_redact(project, tmp_path):
    mem = MemoryStore(project, home=tmp_path / "home")
    assert "Saved" in mem.add("Tests run with: py -m pytest -q")
    assert "Already" in mem.add("tests run with: py -m pytest -q")
    mem.add("User prefers tabs", scope="user")
    mem.add("api key is sk-abcdefghijklmnopqrstuvwxyz123456")
    assert mem.entries("project")[0] == "Tests run with: py -m pytest -q"
    assert "sk-abcdefghijklmnopqrstuvwxyz123456" not in "\n".join(mem.entries("project"))
    block = mem.prompt_block()
    assert "py -m pytest -q" in block and "User prefers tabs" in block
    assert mem.remove("prefers tabs") == 1 and mem.entries("user") == []


def test_agent_remembers_and_memory_reaches_the_next_session(project, tmp_path):
    mem = MemoryStore(project, home=tmp_path / "home")
    agent, _ = make(project, [reply("", call("remember", fact="Build with: npm run build")), reply("Noted.")], memory=mem)
    agent.run("remember how to build")
    agent2, provider2 = make(project, [reply("npm run build")], memory=mem)
    agent2.run("how do I build?")
    system = provider2.requests[0][0][0].content
    assert system.startswith("base prompt") and "Build with: npm run build" in system


# ------------------------------------------------------------------ compaction and repair


def test_compaction_summarizes_instead_of_dropping():
    budget = ContextBudget(window=1200, reserve_output=100)
    seen = []
    compactor = Compactor(budget, summarizer=lambda t: seen.append(t) or "- user wanted X; app.py was fixed")
    msgs = [HumanMessage("first question " * 40), AIMessage("first answer " * 40), HumanMessage("current task")]
    for i in range(6):
        msgs += [AIMessage("", tool_calls=[{"name": "read_file", "args": {"path": f"f{i}"}, "id": f"t{i}"}]),
                 ToolMessage("x" * 300, tool_call_id=f"t{i}", name="read_file")]
    new, note = compactor.fit("system", msgs)
    assert new is not None and "Context compacted" in note
    assert new[0].content.startswith(SUMMARY_PREFIX) and "app.py was fixed" in new[0].content
    assert any(isinstance(m, HumanMessage) and m.content.startswith("current task") for m in new)  # the task is kept
    assert seen  # the model was asked to summarize


def test_long_single_task_summarizes_its_earlier_steps():
    compactor = Compactor(ContextBudget(window=900, reserve_output=100), summarizer=lambda t: "- read f0..f9, all fine")
    msgs = [HumanMessage("current task")]
    for i in range(10):
        msgs += [AIMessage("", tool_calls=[{"name": "read_file", "args": {"path": f"f{i}"}, "id": f"t{i}"}]),
                 ToolMessage("y" * 400, tool_call_id=f"t{i}", name="read_file")]
    new, _ = compactor.fit("system", msgs)
    assert new[0].content.startswith("current task") and PROGRESS_PREFIX in new[0].content
    assert isinstance(new[1], AIMessage)  # the kept tail starts at a tool call, so every call keeps its result


def test_compaction_falls_back_to_a_digest_when_the_model_fails():
    def broken(_):
        raise RuntimeError("model down")

    compactor = Compactor(ContextBudget(window=900, reserve_output=100), summarizer=broken)
    msgs = [HumanMessage("old request " * 200), AIMessage("old reply " * 200), HumanMessage("now")]
    new, _ = compactor.fit("system", msgs)
    assert new[0].content.startswith(SUMMARY_PREFIX) and "User asked: old request" in new[0].content


def test_repair_answers_dangling_calls_and_separates_user_messages():
    msgs = [HumanMessage("a"), AIMessage("", tool_calls=[{"name": "list_files", "args": {}, "id": "x"}]), HumanMessage("b")]
    fixed, changed = repair(msgs)
    assert changed
    assert isinstance(fixed[2], ToolMessage) and fixed[2].tool_call_id == "x"
    assert isinstance(fixed[3], AIMessage) and isinstance(fixed[4], HumanMessage)


# ------------------------------------------------------------------ LangChain chat model


def test_local_chat_model_is_a_langchain_chat_model():
    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    provider = ScriptedProvider([reply("", call("add", a=1, b=2)), reply("3")])
    model = LocalChatModel(provider=provider).bind_tools([add])
    first = model.invoke([SystemMessage("s"), HumanMessage("1+2?")])
    assert isinstance(first, AIMessage) and first.tool_calls[0]["name"] == "add" and first.tool_calls[0]["args"] == {"a": 1, "b": 2}
    assert provider.requests[0][1][0].name == "add"  # the LangChain tool was converted to a tool spec
    second = model.invoke([HumanMessage("1+2?"), first, ToolMessage("3", tool_call_id=first.tool_calls[0]["id"])])
    assert second.content == "3" and second.usage_metadata["input_tokens"] == 100
