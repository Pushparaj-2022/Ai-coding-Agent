"""Checkpoints / rewind, and allow / ask / deny permission rules."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from lca.agent.events import CheckpointCreated
from lca.agent.loop import AgentLoop
from lca.agent.permissions import ApprovalResult, Mode, PermissionPolicy
from lca.agent.rules import PermissionStore, Rule, RuleSet, build_ruleset, split_command, suggest_rule
from lca.config import AgentSettings, load_settings
from lca.graph.agent_graph import transcript
from lca.memory.checkpoints import CheckpointStore
from lca.tools import Workspace, default_registry

from .conftest import ScriptedProvider, call, reply


@pytest.fixture(autouse=True)
def lca_home(tmp_path_factory, monkeypatch):
    home = tmp_path_factory.mktemp("lcahome")
    monkeypatch.setenv("LCA_HOME", str(home))
    return home


# ---------------------------------------------------------------- checkpoint store


def test_rewind_restores_modified_created_and_deleted_files(project):
    store = CheckpointStore(project)
    original = (project / "src" / "app.py").read_bytes()
    cp1 = store.begin("s1", "first", "m1")
    store.record(cp1, "src/app.py")
    (project / "src" / "app.py").write_text("changed\n")
    store.record(cp1, "new.txt")
    (project / "new.txt").write_text("new\n")
    cp2 = store.begin("s1", "second", "m2")
    store.record(cp2, "src/app.py")  # second change of the same file: cp1 still holds the original
    (project / "src" / "app.py").write_text("changed again\n")
    store.record(cp2, "src", is_dir=True)
    util = (project / "src" / "util.py").read_bytes()
    (project / "src" / "util.py").unlink()

    result = store.rewind(cp1.id)
    assert (project / "src" / "app.py").read_bytes() == original
    assert (project / "src" / "util.py").read_bytes() == util
    assert not (project / "new.txt").exists()
    assert "new.txt" in result.removed and "src/app.py" in result.restored and result.prompt == "first"
    assert store.list("s1") == []  # rewound checkpoints are dropped


def test_rewind_to_a_later_checkpoint_keeps_earlier_changes(project):
    store = CheckpointStore(project)
    cp1 = store.begin("s", "one", "m1")
    store.record(cp1, "a.txt")
    (project / "a.txt").write_text("from one\n")
    cp2 = store.begin("s", "two", "m2")
    store.record(cp2, "a.txt")
    (project / "a.txt").write_text("from two\n")
    store.rewind(cp2.id)
    assert (project / "a.txt").read_text() == "from one\n"
    assert [c.prompt for c in store.list("s")] == ["one"]


# ---------------------------------------------------------------- rewind through the agent


def make_agent(project, script, events=None):
    return AgentLoop(
        ScriptedProvider(script), default_registry(), Workspace(project), system_prompt="s",
        settings=AgentSettings(auto_verify=False), policy=PermissionPolicy(Mode.FULL),
        events=events.append if events is not None else None, checkpoints=CheckpointStore(project),
    )


def test_agent_rewind_code_and_conversation(project):
    events: list = []
    agent = make_agent(project, [
        reply("", call("read_file", path="src/app.py")),
        reply("", call("edit_file", path="src/app.py", old_string="return a + b", new_string="return a - b")),
        reply("Changed add."),
        reply("", call("write_file", path="notes.md", content="# notes\n")),
        reply("Wrote notes."),
    ], events)
    agent.run("make add subtract")
    agent.run("write notes")
    cps = [e for e in events if isinstance(e, CheckpointCreated)]
    assert [c.prompt for c in cps] == ["make add subtract", "write notes"]
    assert (project / "notes.md").exists() and "a - b" in (project / "src" / "app.py").read_text()

    result = agent.rewind(cps[1].checkpoint_id, "both")
    assert result.removed == ["notes.md"] and not (project / "notes.md").exists()
    users = [i["text"] for i in transcript(agent.history()) if i["role"] == "user"]
    assert users == ["make add subtract"]
    assert "a - b" in (project / "src" / "app.py").read_text()  # the first message's change stays

    result = agent.rewind(cps[0].checkpoint_id, "code")
    assert result.restored == ["src/app.py"] and "a + b" in (project / "src" / "app.py").read_text()
    assert [i["text"] for i in transcript(agent.history()) if i["role"] == "user"] == ["make add subtract"]


def test_agent_rewind_conversation_only_keeps_files(project):
    events: list = []
    agent = make_agent(project, [reply("", call("write_file", path="x.txt", content="x\n")), reply("ok")], events)
    agent.run("make x")
    cp = next(e for e in events if isinstance(e, CheckpointCreated))
    agent.rewind(cp.checkpoint_id, "conversation")
    assert (project / "x.txt").exists() and agent.history() == []


def test_rewind_rejects_a_checkpoint_of_another_conversation(project):
    agent = make_agent(project, [])
    other = CheckpointStore(project).begin("someone-else", "x", "m")
    with pytest.raises(ValueError):
        agent.rewind(other.id)


# ---------------------------------------------------------------- rules


def cmd(command):
    return SimpleNamespace(command=command)


def test_rule_parsing_and_aliases():
    assert Rule.parse("Bash(npm test:*)").tools() == ["run_command"]
    assert Rule.parse("Edit(src/**)").tools() == ["write_file", "edit_file", "multi_edit", "delete_file"]
    assert Rule.parse("run_command").pattern is None
    with pytest.raises(ValueError):
        Rule.parse("not a rule!")


@pytest.mark.parametrize(("pattern", "command", "ok"), [
    ("npm test:*", "npm test", True),
    ("npm test:*", "npm test -- --watch", True),
    ("npm test:*", "npm testing", False),
    ("npm test", "npm test -x", False),
    ("git log *", "git log --oneline", True),
])
def test_command_patterns(pattern, command, ok):
    rs = RuleSet()
    rs.add("allow", Rule.parse(f"run_command({pattern})"), "t")
    assert (rs.decide("run_command", cmd(command)) is not None) == ok


def test_compound_commands_need_every_part_allowed():
    rs = RuleSet()
    rs.add("allow", Rule.parse("run_command(npm test:*)"), "t")
    assert rs.decide("run_command", cmd("npm test && rm -rf build")) is None
    assert rs.decide("run_command", cmd("npm test > out.txt")) is None  # redirection is never auto-allowed
    assert rs.decide("run_command", cmd("npm test $(evil)")) is None
    rs.add("allow", Rule.parse("run_command(npm run lint:*)"), "t")
    assert rs.decide("run_command", cmd("npm run lint && npm test")).kind == "allow"


def test_deny_matches_any_part_and_wins():
    rs = RuleSet()
    rs.add("allow", Rule.parse("run_command"), "t")
    rs.add("deny", Rule.parse("Bash(git push:*)"), "t")
    assert rs.decide("run_command", cmd("git add . && git push origin main")).kind == "deny"
    assert rs.decide("run_command", cmd("git add .")).kind == "allow"


def test_path_patterns():
    rs = RuleSet()
    rs.add("allow", Rule.parse("Edit(src/**)"), "t")
    rs.add("deny", Rule.parse("edit_file(src/generated/)"), "t")
    rs.add("ask", Rule.parse("read_file(*.lock)"), "t")
    assert rs.decide("write_file", SimpleNamespace(path="src/a/b.py")).kind == "allow"
    assert rs.decide("edit_file", SimpleNamespace(path="./src/generated/x.py")).kind == "deny"
    assert rs.decide("write_file", SimpleNamespace(path="lib/x.py")) is None
    assert rs.decide("read_file", SimpleNamespace(path="poetry.lock")).kind == "ask"


def test_split_command_respects_quotes():
    assert split_command('echo "a && b" && ls') == (['echo "a && b"', "ls"], False)
    assert split_command("cat x | grep y")[0] == ["cat x", "grep y"]


@pytest.mark.parametrize(("command", "rule"), [
    ("npm test", "run_command(npm test:*)"),
    ("npm run build -- --prod", "run_command(npm run build:*)"),
    ("py -m pytest -q", "run_command(py -m pytest:*)"),
    ("cargo test --all", "run_command(cargo test:*)"),
    ("node script.js", None),  # a bare interpreter would allow anything
    ("npm test && ls", None),
    ("make > log.txt", None),
])
def test_suggested_rules(command, rule):
    assert suggest_rule(command) == rule


# ---------------------------------------------------------------- policy


def run_tool():
    reg = default_registry()
    return reg.get("run_command"), reg.get("write_file")


def normal():
    return SimpleNamespace(level="normal", reason="")


def test_allow_rule_skips_approval_and_deny_rule_blocks_even_in_full_mode():
    run, write = run_tool()
    rs = RuleSet()
    rs.add("allow", Rule.parse("run_command(npm test:*)"), "t")
    rs.add("deny", Rule.parse("Edit(.github/**)"), "t")
    ask = PermissionPolicy(Mode.ASK, approver=None, rules=rs)
    assert ask.check(run, run.Args(command="npm test"), None, normal()).allowed
    assert not ask.check(run, run.Args(command="npm publish"), None, normal()).allowed  # needs approval, nobody to ask
    full = PermissionPolicy(Mode.FULL, rules=rs)
    change = SimpleNamespace(path=".github/workflows/ci.yml", action="create", is_dir=False, diff=lambda: "")
    d = full.check(write, write.Args(path=".github/workflows/ci.yml", content="x"), change)
    assert not d.allowed and "Denied by the permission rule Edit(.github/**)" in d.reason


def test_ask_rule_asks_even_in_full_mode_and_risky_commands_ignore_allow_rules():
    run, _ = run_tool()
    asked = []
    rs = RuleSet()
    rs.add("ask", Rule.parse("run_command(npm publish:*)"), "t")
    rs.add("allow", Rule.parse("run_command(git push:*)"), "t")
    policy = PermissionPolicy(Mode.FULL, lambda req: asked.append(req.command) or ApprovalResult(True), rs)
    assert policy.check(run, run.Args(command="npm publish"), None, normal()).allowed
    sensitive = SimpleNamespace(level="sensitive", reason="pushes to a remote")
    assert policy.check(run, run.Args(command="git push"), None, sensitive).allowed
    assert asked == ["npm publish", "git push"]


def test_always_allow_saves_a_rule_for_the_project(project):
    run, _ = run_tool()
    store = PermissionStore(project)
    seen = []

    def approver(req):
        seen.append(req.suggested_rule)
        return ApprovalResult(True, always=True)

    policy = PermissionPolicy(Mode.ASK, approver, RuleSet(), store)
    assert policy.check(run, run.Args(command="npm test -- -u"), None, normal()).allowed
    assert seen == ["run_command(npm test:*)"]
    assert store.load()["allow"] == ["run_command(npm test:*)"]
    assert policy.check(run, run.Args(command="npm test"), None, normal()).allowed and len(seen) == 1  # no second prompt
    assert project not in store.path.parents  # saved outside the repository


def test_config_combines_lists_and_ignores_project_allow(project, lca_home):
    (lca_home / "config.toml").write_text('[permissions]\nallow = ["run_command(npm test:*)"]\ndeny = ["Bash(git push:*)"]\n')
    (project / ".lca").mkdir(exist_ok=True)
    (project / ".lca" / "config.toml").write_text('[permissions]\nallow = ["run_command"]\ndeny = ["Edit(dist/**)"]\n')
    settings, warnings = load_settings(project)
    assert settings.permissions.allow == ["run_command(npm test:*)"]
    assert settings.permissions.deny == ["Bash(git push:*)", "Edit(dist/**)"]
    assert any("cannot allow actions" in w for w in warnings)
    rs = build_ruleset(settings.permissions.model_dump(), PermissionStore(project), warnings)
    assert [i["rule"] for i in rs.listing()] == ["Bash(git push:*)", "Edit(dist/**)", "run_command(npm test:*)"]
