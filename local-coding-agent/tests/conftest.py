from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from lca.cancel import CancelToken
from lca.config import ProviderSettings
from lca.messages import Message, ModelResponse, ToolCall, Usage
from lca.providers.base import Provider, ServerInfo, TextCallback, ToolSpec
from lca.tools.base import ToolContext
from lca.tools.workspace import Workspace

Step = ModelResponse | Exception | Callable[[list[Message]], ModelResponse]


class ScriptedProvider(Provider):
    """Replays a fixed list of model replies; records every request it receives."""

    kind = "fake"

    def __init__(self, script: list[Step], **settings: Any) -> None:
        super().__init__(ProviderSettings(base_url="http://127.0.0.1:9", model="fake", retry_backoff_s=0, **settings))
        self.script = list(script)
        self.requests: list[tuple[list[Message], list[ToolSpec]]] = []

    def _chat_once(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        on_text: TextCallback,
        on_reasoning: TextCallback | None,
        cancel: CancelToken | None,
    ) -> ModelResponse:
        self.requests.append(([m.model_copy(deep=True) for m in messages], tools))
        if not self.script:
            raise AssertionError("ScriptedProvider ran out of replies")
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        if not isinstance(step, ModelResponse):
            step = step(messages)
        if step.content:
            on_text(step.content)
        return step

    def server_info(self) -> ServerInfo:
        return ServerInfo(reachable=True, models=["fake"], context_window=8192)


_n = 0


def call(name: str, **arguments: Any) -> ToolCall:
    global _n
    _n += 1
    return ToolCall(id=f"c{_n}", name=name, arguments=arguments)


def reply(content: str = "", *calls: ToolCall, finish: str = "stop") -> ModelResponse:
    return ModelResponse(
        content=content,
        tool_calls=list(calls),
        finish_reason=finish,
        usage=Usage(prompt_tokens=100, completion_tokens=10),
        tool_call_source="native" if calls else "none",
    )


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A small sample project."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text(
        "import os\n\n\ndef add(a, b):\n    return a + b\n\n\ndef main():\n    print(add(2, 3))\n", encoding="utf-8"
    )
    (tmp_path / "src" / "util.py").write_text("def helper():\n    return 'help'\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# Sample\nA sample project.\n", encoding="utf-8")
    (tmp_path / ".env").write_text("DATABASE_PASSWORD=supersecretvalue123\n", encoding="utf-8")
    (tmp_path / ".env.example").write_text("DATABASE_PASSWORD=\n", encoding="utf-8")
    (tmp_path / "node_modules" / "pkg").mkdir(parents=True)
    (tmp_path / "node_modules" / "pkg" / "index.js").write_text("module.exports = 1\n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def ctx(project: Path) -> ToolContext:
    return ToolContext(workspace=Workspace(project), cancel=CancelToken(), policy=None)
