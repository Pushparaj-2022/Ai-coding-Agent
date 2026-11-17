"""Ollama's native /api/chat endpoint (newline-delimited JSON streaming)."""

from __future__ import annotations

import json
from typing import Any

import httpx

from lca.cancel import Cancelled, CancelToken
from lca.config import ProviderSettings
from lca.messages import Message, ModelResponse, Usage
from lca.providers.base import (
    MalformedResponseError,
    Provider,
    ProviderConnectionError,
    ProviderError,
    ProviderTimeoutError,
    ResponseBuilder,
    ServerInfo,
    TextCallback,
    ToolSpec,
    classify_http_error,
)
from lca.providers.text_tools import render_text_mode


def _to_wire(m: Message) -> dict[str, Any]:
    if m.role == "assistant" and m.tool_calls:
        return {
            "role": "assistant",
            "content": m.content or "",
            "tool_calls": [{"function": {"name": c.name, "arguments": c.arguments}} for c in m.tool_calls],
        }
    if m.role == "tool":
        return {"role": "tool", "content": m.content, "tool_name": m.name}
    if m.images:  # Ollama wants bare base64 in "images"
        return {"role": m.role, "content": m.content, "images": [u.split(",", 1)[-1] for u in m.images]}
    return {"role": m.role, "content": m.content}


class OllamaProvider(Provider):
    kind = "ollama"

    def __init__(self, settings: ProviderSettings, transport: httpx.BaseTransport | None = None) -> None:
        super().__init__(settings)
        headers = {}
        if settings.api_key:
            headers["Authorization"] = f"Bearer {settings.api_key.get_secret_value()}"
        self._client = httpx.Client(
            headers=headers,
            timeout=httpx.Timeout(settings.timeout_s, connect=settings.connect_timeout_s),
            transport=transport,
        )

    @property
    def base_url(self) -> str:
        return self.settings.base_url

    def close(self) -> None:
        self._client.close()

    def _payload(self, messages: list[Message], tools: list[ToolSpec]) -> dict[str, Any]:
        text_mode = bool(tools) and self.effective_tool_mode == "text"
        wire = render_text_mode(messages, tools) if text_mode else messages
        options: dict[str, Any] = {"temperature": self.settings.temperature, "num_predict": self.settings.max_tokens}
        if self.settings.context_window:
            options["num_ctx"] = self.settings.context_window
        body: dict[str, Any] = {"model": self.settings.model, "messages": [_to_wire(m) for m in wire], "stream": True, "options": options}
        if tools and not text_mode:
            body["tools"] = [{"type": "function", "function": t.model_dump()} for t in tools]
        return body

    def _chat_once(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        on_text: TextCallback,
        on_reasoning: TextCallback | None,
        cancel: CancelToken | None,
    ) -> ModelResponse:
        if not self.settings.model:
            raise ProviderError("Ollama needs a model name: set --model or LCA_MODEL (see `ollama list`).")
        body = self._payload(messages, tools)
        b = ResponseBuilder({t.name for t in tools})
        try:
            with self._client.stream("POST", f"{self.base_url}/api/chat", json=body) as resp:
                unregister = cancel.on_cancel(resp.close) if cancel else (lambda: None)
                try:
                    if resp.status_code >= 400:
                        raise classify_http_error(resp.status_code, resp.read().decode("utf-8", "replace"), "tools" in body, any(m.images for m in messages))
                    lines = 0
                    for line in resp.iter_lines():
                        if cancel and cancel.cancelled:
                            raise Cancelled()
                        if not line.strip():
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        lines += 1
                        if obj.get("error"):
                            raise classify_http_error(400, str(obj["error"]), "tools" in body, any(m.images for m in messages))
                        msg = obj.get("message") or {}
                        if msg.get("thinking"):
                            b.reasoning(msg["thinking"])
                            if on_reasoning:
                                on_reasoning(msg["thinking"])
                        if msg.get("content"):
                            b.text(msg["content"])
                            on_text(msg["content"])
                        for tc in msg.get("tool_calls") or []:
                            fn = tc.get("function") or {}
                            if fn.get("name"):
                                b.native_complete(tc.get("id"), fn["name"], fn.get("arguments", {}))
                        if obj.get("done"):
                            b.finish_reason = obj.get("done_reason") or "stop"
                            b.usage = Usage(prompt_tokens=obj.get("prompt_eval_count") or 0, completion_tokens=obj.get("eval_count") or 0)
                    if not lines:
                        raise MalformedResponseError("Ollama returned an empty response.")
                finally:
                    unregister()
        except ProviderError:
            raise
        except httpx.ConnectError as e:
            raise ProviderConnectionError(f"Cannot connect to Ollama at {self.base_url}: {e}. Is `ollama serve` running?") from e
        except httpx.TimeoutException as e:
            raise ProviderTimeoutError(f"Ollama timed out ({type(e).__name__}).") from e
        except (httpx.HTTPError, httpx.StreamError) as e:
            if cancel and cancel.cancelled:
                raise Cancelled() from e
            raise ProviderConnectionError(f"Connection to Ollama failed: {e}") from e
        if cancel and cancel.cancelled:
            raise Cancelled()
        return b.build()

    def server_info(self) -> ServerInfo:
        try:
            r = self._client.get(f"{self.base_url}/api/tags", timeout=5)
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", []) if isinstance(m, dict) and m.get("name")]
        except httpx.HTTPError as e:
            return ServerInfo(reachable=False, detail=str(e) or type(e).__name__)
        except ValueError:
            return ServerInfo(reachable=True, detail="/api/tags did not return JSON")
        ctx = self.settings.context_window
        vision = None
        model = self.settings.model or (models[0] if models else "")
        if model:
            try:
                info = self._client.post(f"{self.base_url}/api/show", json={"model": model}, timeout=5).json()
                ctx = ctx or next((v for k, v in (info.get("model_info") or {}).items() if k.endswith(".context_length")), None)
                if isinstance(info.get("capabilities"), list):  # newer Ollama: ["completion", "tools", "vision", ...]
                    vision = "vision" in info["capabilities"]
            except (httpx.HTTPError, ValueError, AttributeError):
                pass
        return ServerInfo(reachable=True, models=models, context_window=int(ctx) if ctx else None, vision=vision)
