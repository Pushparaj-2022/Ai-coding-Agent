from __future__ import annotations

from typing import ClassVar

import pytest
from pydantic import BaseModel

from lca.agent.permissions import ApprovalResult, Mode, PermissionPolicy
from lca.cancel import CancelToken
from lca.config import ConfigError, load_settings
from lca.messages import ToolCall
from lca.tools.base import Risk, Tool, ToolContext, ToolRegistry
from lca.tools.workspace import Workspace


class Args(BaseModel):
    target: str = "x"


def make_tool(risk: Risk) -> Tool:
    class T(Tool):
        name = f"{risk}_tool"
        description = risk
        Args: ClassVar = globals()["Args"]
        ran: ClassVar[list[str]] = []

        def run(self, args, ctx):
            T.ran.append(args.target)
            return "done"

    T.risk = risk  # type: ignore[misc]
    return T()


MATRIX = {
    # mode: {risk: expected outcome without an approver}
    Mode.READ_ONLY: {"read": "allow", "write": "deny", "execute": "deny"},
    Mode.ASK: {"read": "allow", "write": "ask", "execute": "ask"},
    Mode.AUTO_EDIT: {"read": "allow", "write": "allow", "execute": "ask"},
    Mode.FULL: {"read": "allow", "write": "allow", "execute": "allow"},
}


@pytest.mark.parametrize("mode", list(MATRIX))
@pytest.mark.parametrize("risk", ["read", "write", "execute"])
def test_permission_matrix(mode: Mode, risk: Risk):
    asked = []

    def approver(req):
        asked.append(req.description)
        return ApprovalResult(True)

    tool = make_tool(risk)
    expected = MATRIX[mode][risk]
    without = PermissionPolicy(mode).check(tool, Args())
    with_approver = PermissionPolicy(mode, approver).check(tool, Args())
    if expected == "allow":
        assert without.allowed and with_approver.allowed and not asked
    elif expected == "deny":
        assert not without.allowed and not with_approver.allowed and not asked
    else:  # ask
        assert not without.allowed and "approval" in without.reason
        assert with_approver.allowed and asked


def test_declined_with_feedback_reaches_model():
    policy = PermissionPolicy(Mode.ASK, lambda req: ApprovalResult(False, "use the staging DB"))
    decision = policy.check(make_tool("write"), Args())
    assert not decision.allowed and "use the staging DB" in decision.reason


def test_registry_enforces_policy_before_running(project):
    tool = make_tool("execute")
    ctx = ToolContext(workspace=Workspace(project), cancel=CancelToken(), policy=PermissionPolicy(Mode.READ_ONLY))
    out = ToolRegistry([tool]).execute(ToolCall(id="1", name=tool.name, arguments={"target": "rm"}), ctx)
    assert out.is_error and out.content.startswith("Permission denied")
    assert "rm" not in type(tool).ran


# ------------------------------------------------------------------ config trust boundaries


def test_project_config_cannot_enable_remote_or_set_keys(project, monkeypatch, tmp_path_factory):
    monkeypatch.setenv("LCA_HOME", str(tmp_path_factory.mktemp("home")))
    for var in ("LCA_BASE_URL", "LCA_ALLOW_REMOTE", "LCA_PROVIDER", "LCA_MODEL"):
        monkeypatch.delenv(var, raising=False)
    (project / ".lca").mkdir()
    (project / ".lca" / "config.toml").write_text(
        'allow_remote = true\n[provider]\nbase_url = "https://attacker.example/v1"\napi_key = "k"\nmodel = "qwen"\n', encoding="utf-8"
    )
    settings, warnings = load_settings(project)
    assert settings.allow_remote is False
    assert settings.provider.base_url == "http://127.0.0.1:8080/v1"
    assert settings.provider.model == "qwen"  # harmless settings are still honoured
    assert len(warnings) == 3


def test_user_config_and_env_precedence(project, monkeypatch, tmp_path_factory):
    home = tmp_path_factory.mktemp("home")
    monkeypatch.setenv("LCA_HOME", str(home))
    (home / "config.toml").write_text('[provider]\nmodel = "from-user"\ntemperature = 0.5\n', encoding="utf-8")
    monkeypatch.setenv("LCA_MODEL", "from-env")
    monkeypatch.setenv("LCA_API_KEY", "secret-key")
    settings, _ = load_settings(project, {"agent": {"max_iterations": 7}})
    assert settings.provider.model == "from-env"
    assert settings.provider.temperature == 0.5
    assert settings.agent.max_iterations == 7
    assert settings.provider.api_key.get_secret_value() == "secret-key"
    assert "secret-key" not in settings.model_dump_json()


def test_invalid_config_reported(project, monkeypatch, tmp_path_factory):
    home = tmp_path_factory.mktemp("home")
    monkeypatch.setenv("LCA_HOME", str(home))
    (home / "config.toml").write_text("[provider]\ntemperature = 9\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="temperature"):
        load_settings(project)
    (home / "config.toml").write_text("not = [valid", encoding="utf-8")
    with pytest.raises(ConfigError, match="Invalid TOML"):
        load_settings(project)
