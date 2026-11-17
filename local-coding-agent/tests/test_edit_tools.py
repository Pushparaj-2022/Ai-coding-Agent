from __future__ import annotations

from pathlib import Path

import pytest

from lca.agent.loop import AgentLoop
from lca.agent.permissions import ApprovalRequest, ApprovalResult, Mode, PermissionPolicy
from lca.cancel import CancelToken
from lca.config import AgentSettings
from lca.messages import ToolCall
from lca.tools import Workspace, default_registry
from lca.tools.base import ToolContext

from .conftest import ScriptedProvider, call, reply


@pytest.fixture
def wctx(project: Path) -> ToolContext:
    return ToolContext(workspace=Workspace(project), cancel=CancelToken(), policy=PermissionPolicy(Mode.AUTO_EDIT))


def run(ctx, name, **args):
    return default_registry().execute(ToolCall(id="t", name=name, arguments=args), ctx)


def read(ctx, path):
    out = run(ctx, "read_file", path=path)
    assert not out.is_error, out.content


# ------------------------------------------------------------------ write_file


def test_create_new_file_with_parent_dirs(wctx, project):
    out = run(wctx, "write_file", path="pkg/new/mod.py", content="x = 1\ny = 2\n")
    assert not out.is_error and out.content.startswith("Created pkg/new/mod.py (+2 -0 lines)")
    assert (project / "pkg/new/mod.py").read_text() == "x = 1\ny = 2\n"
    assert "pkg/new/mod.py" in wctx.files_changed


def test_overwrite_requires_reading_first(wctx, project):
    out = run(wctx, "write_file", path="src/util.py", content="new\n")
    assert out.is_error and "Read src/util.py" in out.content
    assert (project / "src/util.py").read_text() == "def helper():\n    return 'help'\n"
    read(wctx, "src/util.py")
    out = run(wctx, "write_file", path="src/util.py", content="new\n")
    assert not out.is_error and (project / "src/util.py").read_text() == "new\n"


def test_refuses_when_user_changed_file_after_read(wctx, project):
    read(wctx, "src/util.py")
    (project / "src/util.py").write_text("user edit\n")
    out = run(wctx, "edit_file", path="src/util.py", old_string="user edit", new_string="agent edit")
    assert out.is_error and "changed on disk" in out.content
    assert (project / "src/util.py").read_text() == "user edit\n"


def test_write_cannot_touch_secrets_or_escape(wctx):
    assert "secrets" in run(wctx, "write_file", path=".env", content="X=1").content
    assert "outside" in run(wctx, "write_file", path="../evil.py", content="x").content


def test_successive_edits_without_rereading(wctx, project):
    read(wctx, "src/app.py")
    assert not run(wctx, "edit_file", path="src/app.py", old_string="return a + b", new_string="return a + b  # sum").is_error
    assert not run(wctx, "edit_file", path="src/app.py", old_string="print(add(2, 3))", new_string="print(add(4, 5))").is_error
    text = (project / "src/app.py").read_text()
    assert "# sum" in text and "add(4, 5)" in text


# ------------------------------------------------------------------ edit_file


def test_edit_exact_and_diff_in_output(wctx, project):
    read(wctx, "src/app.py")
    out = run(wctx, "edit_file", path="src/app.py", old_string="def add(a, b):\n    return a + b", new_string="def add(a: int, b: int) -> int:\n    return a + b")
    assert not out.is_error
    assert "-def add(a, b):" in out.content and "+def add(a: int, b: int) -> int:" in out.content


def test_edit_tolerates_line_number_prefixes_and_indentation(wctx, project):
    read(wctx, "src/app.py")
    out = run(wctx, "edit_file", path="src/app.py", old_string="     5\t    return a + b", new_string="    return b + a")
    assert not out.is_error and "return b + a" in (project / "src/app.py").read_text()
    out = run(wctx, "edit_file", path="src/app.py", old_string="def main():\nprint(add(2, 3))", new_string="def main():\n    print('hi')")
    assert not out.is_error and "print('hi')" in (project / "src/app.py").read_text()


def test_edit_not_found_and_ambiguous(wctx, project):
    (project / "dup.py").write_text("a = 1\nb = 2\na = 1\n")
    read(wctx, "dup.py")
    assert "was not found" in run(wctx, "edit_file", path="dup.py", old_string="zzz", new_string="y").content
    out = run(wctx, "edit_file", path="dup.py", old_string="a = 1", new_string="a = 9")
    assert out.is_error and "occurs 2 times" in out.content and "lines 1, 3" in out.content
    assert not run(wctx, "edit_file", path="dup.py", old_string="a = 1", new_string="a = 9", replace_all=True).is_error
    assert (project / "dup.py").read_text() == "a = 9\nb = 2\na = 9\n"


def test_edit_preserves_crlf(wctx, project):
    (project / "win.txt").write_bytes(b"one\r\ntwo\r\n")
    read(wctx, "win.txt")
    assert not run(wctx, "edit_file", path="win.txt", old_string="two", new_string="TWO").is_error
    assert (project / "win.txt").read_bytes() == b"one\r\nTWO\r\n"


def test_edit_refuses_binary_and_missing(wctx, project):
    assert "Use write_file" in run(wctx, "edit_file", path="nope.py", old_string="a", new_string="b").content
    (project / "blob.bin").write_bytes(b"\0\1\2")
    from lca.tools.edit import record_snapshot

    record_snapshot(wctx, "blob.bin", project / "blob.bin")
    assert "binary" in run(wctx, "edit_file", path="blob.bin", old_string="a", new_string="b").content


# ------------------------------------------------------------------ delete_file


def test_delete_moves_to_trash(wctx, project):
    out = run(wctx, "delete_file", path="src/util.py")
    assert not out.is_error and not (project / "src/util.py").exists()
    trashed = list((project / ".lca" / "trash").rglob("util.py"))
    assert len(trashed) == 1 and "def helper" in trashed[0].read_text()


def test_delete_folder_and_root_protection(wctx, project):
    assert "workspace root" in run(wctx, "delete_file", path=".").content
    assert not run(wctx, "delete_file", path="src").is_error
    assert not (project / "src").exists()


# ------------------------------------------------------------------ approvals


def test_ask_mode_shows_diff_and_respects_decline(project):
    requests: list[ApprovalRequest] = []

    def approver(req):
        requests.append(req)
        return ApprovalResult(False, feedback="use snake_case")

    ctx = ToolContext(workspace=Workspace(project), cancel=CancelToken(), policy=PermissionPolicy(Mode.ASK, approver))
    read(ctx, "src/util.py")
    out = run(ctx, "edit_file", path="src/util.py", old_string="def helper", new_string="def Helper")
    assert out.is_error and "use snake_case" in out.content
    assert (project / "src/util.py").read_text().startswith("def helper")
    req = requests[0]
    assert req.description == "Edit src/util.py (+1 -1)"
    assert "-def helper():" in req.change.diff() and "+def Helper():" in req.change.diff()


def test_remember_approval_for_session(project):
    calls = []

    def approver(req):
        calls.append(req)
        return ApprovalResult(True, remember=True)

    ctx = ToolContext(workspace=Workspace(project), cancel=CancelToken(), policy=PermissionPolicy(Mode.ASK, approver))
    run(ctx, "write_file", path="a.txt", content="1")
    run(ctx, "write_file", path="b.txt", content="2")
    assert len(calls) == 1 and (project / "b.txt").exists()


def test_file_changed_while_waiting_for_approval(project):
    def approver(req):
        (project / "src/util.py").write_text("changed by user\n")
        return ApprovalResult(True)

    ctx = ToolContext(workspace=Workspace(project), cancel=CancelToken(), policy=PermissionPolicy(Mode.ASK, approver))
    read(ctx, "src/util.py")
    out = run(ctx, "edit_file", path="src/util.py", old_string="def helper", new_string="def other")
    assert out.is_error and "while waiting for approval" in out.content
    assert (project / "src/util.py").read_text() == "changed by user\n"


def test_read_only_mode_blocks_writes(project):
    ctx = ToolContext(workspace=Workspace(project), cancel=CancelToken(), policy=PermissionPolicy(Mode.READ_ONLY))
    out = run(ctx, "write_file", path="x.py", content="1")
    assert out.is_error and "not allowed in read-only mode" in out.content and not (project / "x.py").exists()


def test_agent_creates_and_edits_file_end_to_end(project):
    provider = ScriptedProvider([
        reply("", call("write_file", path="calc.py", content="def mul(a, b):\n    return a * b\n")),
        reply("", call("edit_file", path="calc.py", old_string="return a * b", new_string="return a * b  # product")),
        reply("Created calc.py with mul()."),
    ])
    agent = AgentLoop(
        provider, default_registry(), Workspace(project), system_prompt="s",
        settings=AgentSettings(), policy=PermissionPolicy(Mode.AUTO_EDIT),
    )
    result = agent.run("create calc.py")
    assert result.status == "completed" and result.files_changed == ["calc.py"]
    assert (project / "calc.py").read_text() == "def mul(a, b):\n    return a * b  # product\n"


def _agent_with_approver(project, script, approver):
    provider = ScriptedProvider(script)
    agent = AgentLoop(
        provider, default_registry(), Workspace(project), system_prompt="s",
        settings=AgentSettings(), policy=PermissionPolicy(Mode.ASK, approver),
    )
    return agent, provider


def test_plain_decline_stops_the_run(project):
    agent, provider = _agent_with_approver(
        project,
        [reply("", call("delete_file", path="README.md"), call("write_file", path="x.txt", content="1")), reply("should not be asked")],
        lambda req: ApprovalResult(False),
    )
    result = agent.run("delete the readme")
    assert result.status == "blocked" and "declined delete_file on README.md" in result.final_message
    assert len(provider.requests) == 1  # the model is not called again
    assert (project / "README.md").exists() and not (project / "x.txt").exists()
    assert "Skipped" in agent.messages[-2].content  # the second call in the turn did not run


def test_decline_with_feedback_continues_once_then_stops(project):
    answers = iter([ApprovalResult(False, "keep it, add a note instead"), ApprovalResult(False, "no")])
    agent, provider = _agent_with_approver(
        project,
        [
            reply("", call("delete_file", path="README.md")),
            reply("", call("write_file", path="NOTE.md", content="note")),
            reply("unreachable"),
        ],
        lambda req: next(answers),
    )
    result = agent.run("delete the readme")
    assert result.status == "blocked" and "declined two changes" in result.final_message
    assert "keep it, add a note instead" in provider.requests[1][0][-1].content
    assert len(provider.requests) == 2
