"""multi_edit, @-mentions, git context in the prompt, and the built-in commands."""

from __future__ import annotations

import shutil
import subprocess

import pytest

from lca.agent.commands import expand_input, load_commands
from lca.agent.mentions import expand_mentions
from lca.agent.prompts import git_context
from lca.agent.rules import Rule, RuleSet
from lca.graph.agent_graph import transcript
from lca.messages import ToolCall
from lca.tools import Workspace, default_registry


@pytest.fixture(autouse=True)
def home(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("LCA_HOME", str(tmp_path_factory.mktemp("lcahome")))


def execute(ctx, name, **args):
    return default_registry().execute(ToolCall(id="t", name=name, arguments=args), ctx)


def test_multi_edit_applies_edits_in_order_as_one_change(project, ctx):
    execute(ctx, "read_file", path="src/app.py")
    out = execute(ctx, "multi_edit", path="src/app.py", edits=[
        {"old_string": "def add(a, b):", "new_string": "def add(a, b, c=0):"},
        {"old_string": "return a + b", "new_string": "return a + b + c"},
        {"old_string": "add(a, b, c=0)", "new_string": "add(a: int, b: int, c: int = 0)"},  # sees the first edit
    ])
    assert not out.is_error and out.content.startswith("Edited src/app.py")
    text = (project / "src" / "app.py").read_text()
    assert "def add(a: int, b: int, c: int = 0):\n    return a + b + c" in text
    assert ctx.files_changed == {"src/app.py"}


def test_write_file_under_an_existing_file_is_a_clear_error(project, ctx):
    out = execute(ctx, "write_file", path="src/app.py/inner.py", content="x = 1\n")
    assert out.is_error and "src/app.py is a file, not a folder" in out.content
    assert (project / "src" / "app.py").is_file()


def test_multi_edit_is_all_or_nothing(project, ctx):
    before = (project / "src" / "app.py").read_text()
    execute(ctx, "read_file", path="src/app.py")
    out = execute(ctx, "multi_edit", path="src/app.py", edits=[
        {"old_string": "return a + b", "new_string": "return a - b"},
        {"old_string": "this text is not there", "new_string": "x"},
    ])
    assert out.is_error and "edits[2].old_string was not found" in out.content
    assert (project / "src" / "app.py").read_text() == before


def test_multi_edit_needs_a_read_and_counts_as_an_edit_for_rules(project, ctx):
    out = execute(ctx, "multi_edit", path="src/app.py", edits=[{"old_string": "a", "new_string": "b"}])
    assert out.is_error and "Read src/app.py with read_file" in out.content
    rs = RuleSet()
    rs.add("deny", Rule.parse("Edit(src/**)"), "t")
    assert rs.decide("multi_edit", type("A", (), {"path": "src/app.py"})()).kind == "deny"


def test_mentions_attach_files_and_folders_but_not_secrets(project):
    ws = Workspace(project)
    text, found, _ = expand_mentions(ws, "Why does @src/app.py print 5? See also @src/ and @.env, mail me@example.com, @missing.py")
    assert found == ["src/app.py", "src"]
    assert "[File @src/app.py:]\n```\nimport os" in text
    assert "[Folder @src:]\napp.py\nutil.py" in text
    assert "supersecretvalue" not in text and "[File @.env" not in text
    assert "me@example.com" in text
    plain, none, _ = expand_mentions(ws, "no mentions here")
    assert plain == "no mentions here" and none == []


def test_transcript_hides_attachments():
    from langchain_core.messages import HumanMessage

    msg = HumanMessage("Explain @src/app.py\n\n[File @src/app.py:]\n```\ncode\n```")
    assert transcript([msg])[0]["text"] == "Explain @src/app.py"


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_git_context_shows_branch_changes_and_commits(tmp_path):
    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-b", "feature/x")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "T")
    (tmp_path / "a.txt").write_text("1\n")
    git("add", ".")
    git("commit", "-m", "first commit")
    (tmp_path / "a.txt").write_text("2\n")
    (tmp_path / "b.txt").write_text("new\n")
    ctx = git_context(Workspace(tmp_path))
    assert "- Branch: feature/x" in ctx
    assert "Uncommitted changes in 2 file(s): a.txt, b.txt" in ctx
    assert "first commit" in ctx


def test_git_context_outside_a_repository(project):
    assert git_context(Workspace(project)) == ""


def test_builtin_commands_and_overrides(project):
    cmds = load_commands(project)
    assert {"init", "commit", "review", "pr"} <= set(cmds) and cmds["init"].source == "builtin"
    assert "AGENTS.md" in expand_input(project, "/init")[0]
    assert "Target: src/" in expand_input(project, "/review src/")[0]
    (project / ".lca" / "commands").mkdir(parents=True, exist_ok=True)
    (project / ".lca" / "commands" / "review.md").write_text("Our own review of $ARGUMENTS.")
    assert expand_input(project, "/review x")[0] == "Our own review of x."
