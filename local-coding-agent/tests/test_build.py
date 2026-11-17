"""The /build pipeline: specialist agents plus real verification commands."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from lca.agent.build import BuildPipeline
from lca.agent.loop import AgentLoop
from lca.agent.permissions import ApprovalResult, Mode, PermissionPolicy
from lca.agent.project import install_commands, verify_commands
from lca.cancel import CancelToken
from lca.config import AgentSettings
from lca.tools import Workspace, default_registry

from .conftest import ScriptedProvider, call, reply

PY = sys.executable.replace("\\", "/")


def make_pipeline(project: Path, scripts: dict[str, list], policy=None, events=None):
    """scripts: role -> list of scripted replies for each agent created with that role (in order)."""
    registry = default_registry()
    created: list[str] = []
    policy = policy or PermissionPolicy(Mode.FULL)
    sink = events.append if events is not None else (lambda e: None)

    def factory(role: str) -> AgentLoop:
        created.append(role)
        provider = ScriptedProvider(scripts[role].pop(0))
        return AgentLoop(
            provider, registry, Workspace(project), system_prompt=role, settings=AgentSettings(),
            policy=PermissionPolicy(Mode.READ_ONLY) if role in ("planner", "reviewer") else policy, label=role,
            events=sink,
        )

    return BuildPipeline(factory, registry, Workspace(project), policy, sink, max_fix_rounds=2), created


# ------------------------------------------------------------------ project detection


def test_detects_node_install_and_scripts(tmp_path):
    (tmp_path / "package.json").write_text(json.dumps({"scripts": {"build": "tsc", "test": "vitest run", "lint": "eslint ."}}))
    assert install_commands(tmp_path) == ["npm install"]
    assert verify_commands(tmp_path) == ["npm run build", "npm test"]
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "pnpm-lock.yaml").write_text("")
    assert install_commands(tmp_path) == [] and verify_commands(tmp_path) == ["pnpm run build", "pnpm test"]


def test_ignores_npm_placeholder_test_script(tmp_path):
    (tmp_path / "package.json").write_text(json.dumps({"scripts": {"test": 'echo "Error: no test specified" && exit 1'}}))
    assert verify_commands(tmp_path) == []


def test_detects_python_tests(tmp_path, monkeypatch):
    monkeypatch.setattr("lca.agent.project.python_command", lambda: ("py", ""))
    (tmp_path / "test_calc.py").write_text("import unittest\n\nclass T(unittest.TestCase):\n    pass\n")
    assert verify_commands(tmp_path) == ["py -m unittest discover -v"]
    (tmp_path / "requirements.txt").write_text("pytest\n")
    assert verify_commands(tmp_path) == ["py -m pytest -q"]


def test_plain_test_functions_mean_pytest(tmp_path, monkeypatch):
    monkeypatch.setattr("lca.agent.project.python_command", lambda: ("py", ""))
    (tmp_path / "test_calc.py").write_text("def test_add():\n    assert 1 + 1 == 2\n")
    assert verify_commands(tmp_path) == ["py -m pytest -q"]


def test_detects_other_ecosystems(tmp_path):
    (tmp_path / "go.mod").write_text("module x")
    assert verify_commands(tmp_path) == ["go build ./...", "go test ./..."]


# ------------------------------------------------------------------ pipeline


def write_failing_project(project: Path) -> None:
    (project / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    (project / "check.py").write_text("from calc import add\nassert add(2, 3) == 5, 'add is wrong'\nprint('ok')\n")


def test_full_pipeline_fixes_failure_and_verifies(project, monkeypatch):
    write_failing_project(project)
    verify = f'"{PY}" check.py'
    monkeypatch.setattr("lca.agent.build.verify_commands", lambda root: [verify])
    events = []
    pipeline, created = make_pipeline(project, {
        "planner": [[reply("Plan: fix add() in calc.py.")]],
        "coder": [[reply("Nothing to change yet.")]],
        "debugger": [[
            reply("", call("read_file", path="calc.py")),
            reply("", call("edit_file", path="calc.py", old_string="return a - b", new_string="return a + b")),
            reply("Fixed the sign."),
        ]],
        "reviewer": [[reply("Looks right.\nVERDICT: APPROVE")]],
    }, events=events)
    result = pipeline.run("make add() correct", CancelToken())
    assert result.status == "passed", result.summary
    assert created == ["planner", "coder", "debugger", "reviewer"]
    assert [c["exit_code"] for c in result.commands] == [1, 0]  # failed, then passed after the fix
    assert result.files_changed == ["calc.py"]
    stages = [e.stage for e in events if e.type == "stage_started"]
    assert stages == ["plan", "implement", "verify", "fix", "review"]
    assert any(e.type == "tool_call_started" and e.agent == "debugger" for e in events)
    assert events[-1].type == "build_finished"


def test_pipeline_reports_failure_after_max_rounds(project, monkeypatch):
    write_failing_project(project)
    monkeypatch.setattr("lca.agent.build.verify_commands", lambda root: [f'"{PY}" check.py'])
    pipeline, _ = make_pipeline(project, {
        "planner": [[reply("plan")]],
        "coder": [[reply("done")]],
        "debugger": [[reply("I could not fix it.")], [reply("Still stuck.")]],
        "reviewer": [[reply("VERDICT: APPROVE")]],
    })
    result = pipeline.run("x", CancelToken())
    assert result.status == "failed" and "add is wrong" in result.summary
    assert len(result.commands) == 3  # initial run + 2 fix rounds


def test_pipeline_uses_tester_when_nothing_to_verify(project, monkeypatch):
    monkeypatch.setattr("lca.agent.build.verify_commands", lambda root: [])
    (project / "ok.py").write_text("print('fine')\n")
    pipeline, created = make_pipeline(project, {
        "planner": [[reply("plan")]],
        "coder": [[reply("done")]],
        "tester": [[reply(f'Wrote tests.\nVERIFY_COMMAND: "{PY}" ok.py')]],
        "reviewer": [[reply("VERDICT: APPROVE")]],
    })
    result = pipeline.run("x", CancelToken())
    assert result.status == "passed" and "tester" in created


def test_review_changes_trigger_revision_and_reverify(project, monkeypatch):
    (project / "ok.py").write_text("print('fine')\n")
    monkeypatch.setattr("lca.agent.build.verify_commands", lambda root: [f'"{PY}" ok.py'])
    pipeline, created = make_pipeline(project, {
        "planner": [[reply("plan")]],
        "coder": [[reply("done")], [reply("Addressed the review.")]],
        "reviewer": [[reply("calc.py:1 misses input validation.\nVERDICT: CHANGES_REQUIRED")]],
    })
    result = pipeline.run("x", CancelToken())
    assert created == ["planner", "coder", "reviewer", "coder"]
    assert result.status == "passed" and len(result.commands) == 2


def test_declining_a_verification_command_stops_the_build(project, monkeypatch):
    monkeypatch.setattr("lca.agent.build.verify_commands", lambda root: [f'"{PY}" -c "print(1)"'])
    policy = PermissionPolicy(Mode.ASK, lambda req: ApprovalResult(False))
    pipeline, _ = make_pipeline(project, {"planner": [[reply("plan")]], "coder": [[reply("done")]]}, policy=policy)
    result = pipeline.run("x", CancelToken())
    assert result.status == "cancelled" and "declined" in result.summary


def test_cancel_stops_the_build(project):
    token = CancelToken()
    token.cancel()
    pipeline, _ = make_pipeline(project, {"planner": [[reply("plan")]]})
    assert pipeline.run("x", token).status == "cancelled"


def test_debugger_can_correct_the_verification_command(project, monkeypatch):
    monkeypatch.setattr("lca.agent.build.verify_commands", lambda root: [f'"{PY}" -c "import sys; sys.exit(5)"'])
    pipeline, _ = make_pipeline(project, {
        "planner": [[reply("plan")]],
        "coder": [[reply("done")]],
        "debugger": [[reply(f'The command was wrong.\nVERIFY_COMMAND: "{PY}" -c "print(1)"')]],
        "reviewer": [[reply("VERDICT: APPROVE")]],
    })
    result = pipeline.run("x", CancelToken())
    assert result.status == "passed" and [c["exit_code"] for c in result.commands] == [5, 0]
