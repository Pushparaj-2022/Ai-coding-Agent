"""Provider interface, errors and the response builder shared by all model backends."""

from __future__ import annotations

import ipaddress
import itertools
import logging
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from lca.cancel import Cancelled, CancelToken
from lca.messages import Message, ModelResponse, ToolCall, Usage
from lca.providers.toolcall_parser import THINK, cut_at_turn_marker, extract_tool_calls, is_parse_failure, parse_json_loose

if TYPE_CHECKING:
    from lca.config import ProviderSettings

log = logging.getLogger("lca.provider")

TextCallback = Callable[[str], None]


class ProviderError(Exception):
    retryable = False


class ProviderConnectionError(ProviderError):
    retryable = True


class ProviderTimeoutError(ProviderError):
    retryable = True


class ProviderHTTPError(ProviderError):
    def __init__(self, message: str, status: int) -> None:
        super().__init__(message)
        self.status = status
        self.retryable = status in (429, 502, 503, 504)


class MalformedResponseError(ProviderError):
    pass


class ContextOverflowError(ProviderError):
    pass


class ToolsUnsupportedError(ProviderError):
    pass


class ToolSpec(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]


class ServerInfo(BaseModel):
    reachable: bool
    models: list[str] = Field(default_factory=list)
    context_window: int | None = None
    vision: bool | None = None  # the model reads images (None: the server does not say)
    detail: str = ""


def is_local_url(url: str) -> bool:
    """True only if the URL points at this machine (loopback). LAN addresses count as remote."""
    host = (urlparse(url).hostname or "").lower()
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


_CONTEXT_RE = re.compile(r"context|exceed|too long|n_ctx|maximum context|token limit", re.I)
_TOOLS_RE = re.compile(r"tool|jinja|function", re.I)


def classify_http_error(status: int, body: str, tools_sent: bool, images_sent: bool = False) -> ProviderError:
    snippet = body.strip()[:500]
    if status in (400, 413, 500) and _CONTEXT_RE.search(body):
        return ContextOverflowError(f"Request exceeds the model's context window: {snippet}")
    if images_sent and status in (400, 415, 422, 500, 501) and re.search(r"image|multimodal|mmproj|vision|modalit", body, re.I):
        return ProviderHTTPError(
            f"The model server cannot read images (HTTP {status}). Use a vision model: with llama.cpp start llama-server "
            f"with the model's --mmproj file; with Ollama pick a vision model (e.g. qwen2.5vl, gemma3). {snippet}", status
        )
    if tools_sent and status in (400, 422, 500, 501) and _TOOLS_RE.search(body):
        return ToolsUnsupportedError(f"Server rejected native tool calling: {snippet}")
    if status == 404:
        return ProviderHTTPError(f"Not found (HTTP 404) - check the base URL and model name: {snippet}", status)
    if status in (401, 403):
        return ProviderHTTPError(f"Authentication failed (HTTP {status}) - check LCA_API_KEY: {snippet}", status)
    return ProviderHTTPError(f"Model server returned HTTP {status}: {snippet}", status)


_ids = itertools.count(1)


def new_call_id() -> str:
    return f"call_{int(time.time() * 1000)}_{next(_ids)}"


def make_tool_call(call_id: str | None, name: str, arguments: Any) -> ToolCall:
    call_id = call_id or new_call_id()
    if isinstance(arguments, dict):
        return ToolCall(id=call_id, name=name, arguments=arguments)
    raw = "" if arguments is None else str(arguments)
    value = parse_json_loose(raw) if raw.strip() else {}
    if not is_parse_failure(value) and isinstance(value, dict):
        return ToolCall(id=call_id, name=name, arguments=value)
    return ToolCall(id=call_id, name=name, raw_arguments=raw, parse_error="arguments are not a valid JSON object")


class ResponseBuilder:
    """Accumulates streamed pieces into a normalized ModelResponse."""

    def __init__(self, known_tools: set[str]) -> None:
        self.known_tools = known_tools
        self.text_parts: list[str] = []
        self.reasoning_parts: list[str] = []
        self.native: dict[int, dict[str, Any]] = {}
        self.finish_reason: str | None = None
        self.usage: Usage | None = None

    def text(self, s: str) -> None:
        self.text_parts.append(s)

    def reasoning(self, s: str) -> None:
        self.reasoning_parts.append(s)

    def native_delta(self, index: int, call_id: str | None, name: str | None, args_fragment: Any) -> None:
        slot = self.native.setdefault(index, {"id": None, "name": "", "args": ""})
        if call_id:
            slot["id"] = call_id
        if name:
            slot["name"] += name
        if isinstance(args_fragment, dict):
            slot["args"] = args_fragment
        elif args_fragment:
            slot["args"] = (slot["args"] if isinstance(slot["args"], str) else "") + str(args_fragment)

    def native_complete(self, call_id: str | None, name: str, arguments: Any) -> None:
        self.native[len(self.native)] = {"id": call_id, "name": name, "args": arguments}

    def build(self) -> ModelResponse:
        content = "".join(self.text_parts)
        reasoning = list(self.reasoning_parts)

        def keep_thinking(m: re.Match[str]) -> str:
            reasoning.append(m.group(1) if m.group(1) is not None else (m.group(2) or ""))
            return ""

        content = THINK.sub(keep_thinking, content)
        # before parsing text tool calls: calls inside an invented next turn must never run
        content = cut_at_turn_marker(content)
        calls =[make_tool_call(s["id"], s["name"], s["args"]) for _, s in sorted(self.native.items()) if s["name"]]
        source = "native" if calls else "none"
        if not calls:
            parsed, cleaned = extract_tool_calls(content, self.known_tools)
            if parsed:
                content = cleaned
                source = "text"
                calls = [
                    ToolCall(id=new_call_id(), name=p.name, arguments=p.arguments, parse_error=p.error, raw_arguments=p.raw or None)
                    for p in parsed
                ]
        return ModelResponse(
            content=content.strip(),
            reasoning="".join(reasoning),
            tool_calls=calls,
            finish_reason=self.finish_reason,
            usage=self.usage,
            tool_call_source=source,  # type: ignore[arg-type]
        )


class Provider(ABC):
    kind = "base"

    def __init__(self, settings: ProviderSettings) -> None:
        self.settings = settings
        # "auto" starts with native tool calls and falls back to the text format if the server rejects them
        self.effective_tool_mode = "text" if settings.tool_mode == "text" else "native"

    @property
    def is_local(self) -> bool:
        return is_local_url(self.settings.base_url)

    @property
    def model(self) -> str:
        return self.settings.model

    def chat(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        *,
        on_text: TextCallback | None = None,
        on_reasoning: TextCallback | None = None,
        on_notice: TextCallback | None = None,
        cancel: CancelToken | None = None,
    ) -> ModelResponse:
        """Send the conversation and return the normalized reply, with retries and tool-mode fallback."""
        attempt = 0
        while True:
            attempt += 1
            streamed = False

            def text_cb(t: str) -> None:
                nonlocal streamed
                streamed = True
                if on_text:
                    on_text(t)

            started = time.monotonic()
            try:
                response = self._chat_once(messages, tools or [], text_cb, on_reasoning, cancel)
                log.info(
                    "model reply",
                    extra={"fields": {
                        "provider": self.kind,
                        "model": self.model,
                        "seconds": round(time.monotonic() - started, 2),
                        "tool_calls": len(response.tool_calls),
                        "source": response.tool_call_source,
                        "finish": response.finish_reason,
                    }},
                )
                return response
            except ToolsUnsupportedError:
                if self.settings.tool_mode == "auto" and self.effective_tool_mode == "native":
                    self.effective_tool_mode = "text"
                    if on_notice:
                        on_notice("The server rejected native tool calling; switched to the text tool-call format.")
                    continue
                raise
            except ProviderError as e:
                if not e.retryable or streamed or attempt > self.settings.max_retries:
                    raise
                delay = min(self.settings.retry_backoff_s * (2 ** (attempt - 1)), 10)
                log.warning("retrying model request", extra={"fields": {"attempt": attempt, "error": str(e)}})
                if cancel and cancel.wait(delay):
                    raise Cancelled() from e
                if not cancel:
                    time.sleep(delay)

    @abstractmethod
    def _chat_once(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        on_text: TextCallback,
        on_reasoning: TextCallback | None,
        cancel: CancelToken | None,
    ) -> ModelResponse: ...

    @abstractmethod
    def server_info(self) -> ServerInfo: ...

    def close(self) -> None:  # pragma: no cover - default no-op
        pass
