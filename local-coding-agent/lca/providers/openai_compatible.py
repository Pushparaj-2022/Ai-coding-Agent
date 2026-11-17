"""OpenAI-compatible chat completions: llama.cpp llama-server, LM Studio, vLLM, Ollama's /v1, and hosted APIs."""

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
from lca.providers.toolcall_parser import TURN_MARKERS


def _ran_past_turn(parts: list[str]) -> bool:
    """Whether the newest streamed text contains a turn marker (checked on a short tail, so it stays cheap)."""
    tail = "".join(parts[-8:])[-80:]
    return any(m in tail for m in TURN_MARKERS)


def _to_wire(m: Message) -> dict[str, Any]:
    if m.role == "assistant" and m.tool_calls:
        return {
            "role": "assistant",
            "content": m.content or "",
            "tool_calls": [
                {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
                for c in m.tool_calls
            ],
        }
    if m.role == "tool":
        return {"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content}
    if m.images:  # llama.cpp (with --mmproj), LM Studio and vLLM take OpenAI-style image blocks
        return {"role": m.role, "content": [{"type": "text", "text": m.content}, *({"type": "image_url", "image_url": {"url": u}} for u in m.images)]}
    return {"role": m.role, "content": m.content}


class OpenAICompatibleProvider(Provider):
    kind = "openai"

    def __init__(self, settings: ProviderSettings, transport: httpx.BaseTransport | None = None) -> None:
        super().__init__(settings)
        headers = {"Content-Type": "application/json"}
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
        body: dict[str, Any] = {
            "model": self.settings.model or "local",
            "messages": [_to_wire(m) for m in wire],
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        # No `stop` sequences here: with them, llama.cpp's gemma-4 tool-call handling often drops the tool call. A reply
        # that runs on into an invented next turn is cut on our side instead (_consume_sse, ResponseBuilder.build).
        if tools and not text_mode:
            body["tools"] = [{"type": "function", "function": t.model_dump()} for t in tools]
            body["tool_choice"] = "auto"
        return body

    def _chat_once(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        on_text: TextCallback,
        on_reasoning: TextCallback | None,
        cancel: CancelToken | None,
    ) -> ModelResponse:
        body = self._payload(messages, tools)
        builder = ResponseBuilder({t.name for t in tools})
        try:
            with self._client.stream("POST", f"{self.base_url}/chat/completions", json=body) as resp:
                unregister = cancel.on_cancel(resp.close) if cancel else (lambda: None)
                try:
                    if resp.status_code >= 400:
                        text = resp.read().decode("utf-8", "replace")
                        raise classify_http_error(resp.status_code, text, "tools" in body, any(m.images for m in messages))
                    if "text/event-stream" in resp.headers.get("content-type", ""):
                        self._consume_sse(resp, builder, on_text, on_reasoning, cancel, "tools" in body)
                    else:
                        self._consume_json(resp.read(), builder, on_text, "tools" in body)
                finally:
                    unregister()
        except ProviderError:
            raise
        except httpx.ConnectError as e:
            raise ProviderConnectionError(f"Cannot connect to {self.base_url}: {e}. Is the model server running?") from e
        except httpx.TimeoutException as e:
            raise ProviderTimeoutError(f"Model server timed out ({type(e).__name__}).") from e
        except (httpx.HTTPError, httpx.StreamError) as e:
            if cancel and cancel.cancelled:
                raise Cancelled() from e
            raise ProviderConnectionError(f"Connection to the model server failed: {e}") from e
        if cancel and cancel.cancelled:
            raise Cancelled()
        return builder.build()

    @staticmethod
    def _apply(obj: dict[str, Any], b: ResponseBuilder, on_text: TextCallback, on_reasoning: TextCallback | None) -> None:
        usage = obj.get("usage")
        if isinstance(usage, dict):
            b.usage = Usage(prompt_tokens=usage.get("prompt_tokens") or 0, completion_tokens=usage.get("completion_tokens") or 0)
        choices = obj.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            return
        choice = choices[0]
        delta = choice.get("delta") or choice.get("message") or {}
        thinking = delta.get("reasoning_content") or delta.get("reasoning")
        if thinking:
            b.reasoning(thinking)
            if on_reasoning:
                on_reasoning(thinking)
        content = delta.get("content")
        if isinstance(content, str) and content:
            b.text(content)
            on_text(content)
        for i, tc in enumerate(delta.get("tool_calls") or []):
            fn = tc.get("function") or {}
            b.native_delta(tc.get("index", i), tc.get("id"), fn.get("name"), fn.get("arguments"))
        if choice.get("finish_reason"):
            b.finish_reason = choice["finish_reason"]

    def _consume_sse(
        self,
        resp: httpx.Response,
        b: ResponseBuilder,
        on_text: TextCallback,
        on_reasoning: TextCallback | None,
        cancel: CancelToken | None,
        tools_sent: bool,
    ) -> None:
        data_lines = bad = 0
        for line in resp.iter_lines():
            if cancel and cancel.cancelled:
                raise Cancelled()
            line = line.strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            data_lines += 1
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                bad += 1
                continue
            if not isinstance(obj, dict):
                bad += 1
                continue
            if obj.get("error"):
                err = obj["error"]
                message = err.get("message", str(err)) if isinstance(err, dict) else str(err)
                raise classify_http_error(400, message, tools_sent)
            self._apply(obj, b, on_text, on_reasoning)
            if b.text_parts and _ran_past_turn(b.text_parts):
                break  # the model is inventing the next turn: closing the stream stops generation on the server
        if data_lines == 0:
            raise MalformedResponseError("The model server returned an empty stream.")
        if bad == data_lines:
            raise MalformedResponseError("The model server sent a stream with no valid JSON events.")

    def _consume_json(self, raw: bytes, b: ResponseBuilder, on_text: TextCallback, tools_sent: bool) -> None:
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as e:
            raise MalformedResponseError(f"Unexpected non-JSON reply: {raw[:200]!r}") from e
        if not isinstance(obj, dict):
            raise MalformedResponseError("Unexpected reply shape from the model server.")
        if obj.get("error"):
            err = obj["error"]
            raise classify_http_error(400, err.get("message", str(err)) if isinstance(err, dict) else str(err), tools_sent)
        if not obj.get("choices"):
            raise MalformedResponseError("Reply has no 'choices'.")
        self._apply(obj, b, on_text, None)

    def server_info(self) -> ServerInfo:
        try:
            r = self._client.get(f"{self.base_url}/models", timeout=5)
            r.raise_for_status()
            data = r.json()
        except httpx.HTTPError as e:
            return ServerInfo(reachable=False, detail=str(e) or type(e).__name__)
        except ValueError:
            return ServerInfo(reachable=True, detail="/models did not return JSON")
        models = [m["id"] for m in data.get("data", []) if isinstance(m, dict) and m.get("id")]
        models += [m["name"] for m in data.get("models", []) if isinstance(m, dict) and m.get("name") and m["name"] not in models]
        ctx = vision = None
        root = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url
        try:  # llama.cpp reports the per-slot context size here, and whether an --mmproj (vision) is loaded
            props = self._client.get(f"{root}/props", timeout=3).json()
            ctx = (props.get("default_generation_settings") or {}).get("n_ctx") or props.get("n_ctx")
            modalities = props.get("modalities")
            if isinstance(modalities, dict) and "vision" in modalities:
                vision = bool(modalities["vision"])
        except (httpx.HTTPError, ValueError, AttributeError):
            pass
        return ServerInfo(reachable=True, models=models, context_window=int(ctx) if ctx else None, vision=vision)
