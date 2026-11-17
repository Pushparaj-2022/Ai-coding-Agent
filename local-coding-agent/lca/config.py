"""Settings: defaults < ~/.lca/config.toml < <project>/.lca/config.toml < environment < CLI flags.

The project config comes from the repository, which may not be trustworthy, so it cannot enable
remote inference or point the agent at a non-local server.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, SecretStr, ValidationError, model_validator

from lca.providers.base import is_local_url

DEFAULT_URLS = {"openai": "http://127.0.0.1:8080/v1", "ollama": "http://127.0.0.1:11434"}


class ConfigError(Exception):
    pass


class ProviderSettings(BaseModel):
    kind: Literal["openai", "ollama"] = "openai"
    base_url: str = ""
    model: str = ""  # empty = first model the server reports
    api_key: SecretStr | None = Field(default=None, exclude=True)  # only ever read from the environment
    temperature: float = Field(default=0.2, ge=0, le=2)
    max_tokens: int = Field(default=4096, ge=64)
    timeout_s: float = Field(default=300, gt=0)
    connect_timeout_s: float = Field(default=10, gt=0)
    max_retries: int = Field(default=2, ge=0, le=10)
    retry_backoff_s: float = Field(default=1.0, ge=0)
    tool_mode: Literal["auto", "native", "text"] = "auto"
    context_window: int | None = Field(default=None, ge=1024)
    vision: bool | None = None  # the model reads images; None = ask the server (llama.cpp /props, Ollama capabilities)

    @model_validator(mode="after")
    def _default_url(self) -> ProviderSettings:
        if not self.base_url:
            self.base_url = DEFAULT_URLS[self.kind]
        self.base_url = self.base_url.rstrip("/")
        return self


class AgentSettings(BaseModel):
    max_iterations: int = Field(default=60, ge=1, le=500)
    max_tool_calls_per_turn: int = Field(default=10, ge=1, le=64)
    max_consecutive_failures: int = Field(default=6, ge=1, le=50)  # turns in a row where every tool call errored
    max_stalled_turns: int = Field(default=10, ge=2, le=100)  # turns in a row where nothing succeeded, commands included
    max_identical_calls: int = Field(default=3, ge=1, le=20)  # identical calls that already FAILED this many times are refused
    summarize_context: bool = True  # summarize old history with the model instead of dropping it
    wrap_up: bool = True  # when a run must stop, ask the model for a final answer from what it found
    # update_plan before the first file change: "remind" lets the change through with a reminder (small models often
    # never plan and a refusal stalls them), "enforce" refuses changes until there is a plan, "off" says nothing
    require_plan: Literal["off", "remind", "enforce"] = "remind"
    auto_verify: bool = True  # after a run that changed files, run the project's build/tests and feed failures back
    verify_rounds: int = Field(default=2, ge=0, le=10)  # how often a failed automatic verification goes back to the agent
    verify_timeout_s: int = Field(default=600, ge=10, le=3600)
    max_parallel_agents: int = Field(default=3, ge=1, le=8)  # task calls in one reply run at the same time, up to this many
    tool_output_chars: int = Field(default=12000, ge=1000)
    shell: Literal["auto", "bash", "powershell", "cmd"] = "auto"
    passthrough_env: list[str] = Field(default_factory=list)  # secret-looking env vars commands may still see


class PermissionSettings(BaseModel):
    """Permission rules (see lca.agent.rules). The lists of the user and project configs are combined; a project
    config cannot add allow rules."""

    allow: list[str] = Field(default_factory=list)
    ask: list[str] = Field(default_factory=list)
    deny: list[str] = Field(default_factory=list)


class WebSettings(BaseModel):
    enabled: bool = True  # web_fetch and web_search (they always need approval unless the mode is full)
    search: Literal["duckduckgo", "searxng", "brave"] = "duckduckgo"
    searxng_url: str = ""
    allow_private: bool = False  # let web_fetch reach localhost / the local network


class StatusLineSettings(BaseModel):
    command: str = ""  # see lca.agent.statusline; empty = no status line
    timeout_s: float = Field(default=5, gt=0, le=30)


class Settings(BaseModel):
    provider: ProviderSettings = Field(default_factory=ProviderSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    permissions: PermissionSettings = Field(default_factory=PermissionSettings)
    hooks: dict[str, Any] = Field(default_factory=dict)  # from the user config only; see lca.agent.hooks
    mcp: dict[str, Any] = Field(default_factory=dict)  # from the user config only; see lca.mcp
    web: WebSettings = Field(default_factory=WebSettings)
    status_line: StatusLineSettings = Field(default_factory=StatusLineSettings)
    allow_remote: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


def lca_home() -> Path:
    return Path(os.environ.get("LCA_HOME") or Path.home() / ".lca")


def _deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in extra.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as f:
            return tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"Invalid TOML in {path}: {e}") from e
    except OSError as e:
        raise ConfigError(f"Cannot read {path}: {e}") from e


def _sanitize_project_config(data: dict[str, Any], warnings: list[str]) -> dict[str, Any]:
    data = dict(data)
    data.pop("hooks", None)  # project hooks and MCP servers run only once trusted; lca.agent.trust reads them itself
    data.pop("mcp", None)
    if data.pop("status_line", None) is not None:
        warnings.append("Ignored [status_line] in the project config: it runs a command, so only your user config can set it.")
    web = data.get("web")
    if isinstance(web, dict) and web.get("allow_private"):
        data["web"] = {k: v for k, v in web.items() if k != "allow_private"}
        warnings.append("Ignored web.allow_private in the project config: only your user config can let the agent reach local addresses.")
    if data.pop("allow_remote", None) is not None:
        warnings.append("Ignored 'allow_remote' in the project config: only your user config, env or CLI can enable it.")
    perms = data.get("permissions")
    if isinstance(perms, dict) and perms.get("allow"):
        data["permissions"] = {k: v for k, v in perms.items() if k != "allow"}
        warnings.append(
            "Ignored permissions.allow in the project config: a repository cannot allow actions. "
            "Use `lca permissions --allow` or \"Always allow\" instead."
        )
    provider = dict(data.get("provider") or {})
    for key in [k for k in provider if k.startswith("api_key")]:
        provider.pop(key)
        warnings.append(f"Ignored provider.{key} in the project config: API keys come from the LCA_API_KEY environment variable.")
    url = provider.get("base_url")
    if url and not is_local_url(str(url)):
        provider.pop("base_url")
        warnings.append(f"Ignored non-local provider.base_url '{url}' from the project config.")
    if provider:
        data["provider"] = provider
    else:
        data.pop("provider", None)
    return data


def _env_overrides() -> dict[str, Any]:
    env = os.environ
    provider: dict[str, Any] = {}
    data: dict[str, Any] = {}
    if env.get("LCA_PROVIDER"):
        provider["kind"] = env["LCA_PROVIDER"]
    if env.get("LCA_BASE_URL"):
        provider["base_url"] = env["LCA_BASE_URL"]
    if env.get("LCA_MODEL"):
        provider["model"] = env["LCA_MODEL"]
    if env.get("LCA_TOOL_MODE"):
        provider["tool_mode"] = env["LCA_TOOL_MODE"]
    if env.get("LCA_ALLOW_REMOTE"):
        data["allow_remote"] = env["LCA_ALLOW_REMOTE"].strip().lower() in ("1", "true", "yes")
    if env.get("LCA_LOG_LEVEL"):
        data["log_level"] = env["LCA_LOG_LEVEL"].upper()
    if provider:
        data["provider"] = provider
    return data


def load_settings(root: Path, overrides: dict[str, Any] | None = None) -> tuple[Settings, list[str]]:
    """Returns the merged settings and warnings to show the user."""
    warnings: list[str] = []
    data: dict[str, Any] = {}
    permissions: dict[str, list[Any]] = {"allow": [], "ask": [], "deny": []}

    def take_permissions(source: dict[str, Any]) -> dict[str, Any]:
        """Permission lists add up across files instead of the later file replacing the earlier one."""
        source = dict(source)
        perms = source.pop("permissions", None)
        if isinstance(perms, dict):
            for kind in permissions:
                value = perms.get(kind)
                if isinstance(value, list):
                    permissions[kind] += [v for v in value if v not in permissions[kind]]
                elif value is not None:
                    warnings.append(f"permissions.{kind} must be a list of rules, e.g. [\"run_command(npm test:*)\"]")
        return source

    user_file = lca_home() / "config.toml"
    if user_file.is_file():
        data = _deep_merge(data, take_permissions(_read_toml(user_file)))
    project_file = root / ".lca" / "config.toml"
    if project_file.is_file():
        data = _deep_merge(data, take_permissions(_sanitize_project_config(_read_toml(project_file), warnings)))
    data["permissions"] = permissions
    data = _deep_merge(data, _env_overrides())
    if overrides:
        data = _deep_merge(data, overrides)
    # a different provider kind implies its default URL unless one was given explicitly
    try:
        settings = Settings.model_validate(data)
    except ValidationError as e:
        problems = "; ".join(f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}" for err in e.errors())
        raise ConfigError(f"Invalid settings: {problems}") from e
    key = os.environ.get("LCA_API_KEY")
    if key:
        settings.provider.api_key = SecretStr(key)
    return settings, warnings
