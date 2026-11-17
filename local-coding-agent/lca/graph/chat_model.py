"""LangChain integration: a BaseChatModel backed by lca's local-model providers.

Using our provider underneath (instead of langchain-openai) keeps what small local models need: the text tool-call
fallback, LFM/pythonic tool-call parsing, retries, and context-overflow detection. On top, the model behaves like
any LangChain chat model: it accepts LangChain messages, supports bind_tools(), streams tokens through callbacks,
and returns AIMessages with tool_calls / invalid_tool_calls and usage metadata.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ConfigDict, Field

from lca.messages import Message, ModelResponse, ToolCall
from lca.providers.base import Provider, ToolSpec

# ---------------------------------------------------------------- message conversion


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # content blocks
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content or "")


def images_of(m: BaseMessage) -> list[str]:
    """Data URLs of the images in a message's content blocks."""
    if not isinstance(m.content, list):
        return []
    out = []
    for b in m.content:
        if isinstance(b, dict) and b.get("type") == "image_url":
            url = b.get("image_url")
            out.append(url.get("url", "") if isinstance(url, dict) else str(url))
    return [u for u in out if u]


def human_message(text: str, images: Sequence[str] = (), **kwargs: Any) -> HumanMessage:
    """A user message, with images as OpenAI-style content blocks when there are any."""
    if not images:
        return HumanMessage(text, **kwargs)
    return HumanMessage([{"type": "text", "text": text}, *({"type": "image_url", "image_url": {"url": u}} for u in images)], **kwargs)


def to_lca(messages: Sequence[BaseMessage]) -> list[Message]:
    """LangChain messages → lca Messages (what providers send to the model)."""
    out: list[Message] = []
    for m in messages:
        if isinstance(m, SystemMessage):
            out.append(Message.system(_text(m.content)))
        elif isinstance(m, HumanMessage):
            out.append(Message(role="user", content=_text(m.content), images=images_of(m)))
        elif isinstance(m, ToolMessage):
            out.append(Message.tool(m.tool_call_id, m.name or "", _text(m.content)))
        elif isinstance(m, AIMessage):
            calls = [ToolCall(id=c["id"] or "", name=c["name"], arguments=c.get("args") or {}) for c in m.tool_calls]
            calls += [
                ToolCall(id=c.get("id") or "", name=c.get("name") or "unknown", raw_arguments=c.get("args"), parse_error=c.get("error") or "invalid")
                for c in m.invalid_tool_calls
            ]
            out.append(Message.assistant(_text(m.content), calls))
    return out


def from_lca(messages: Sequence[Message]) -> list[BaseMessage]:
    out: list[BaseMessage] = []
    for m in messages:
        if m.role == "system":
            out.append(SystemMessage(m.content))
        elif m.role == "user":
            out.append(human_message(m.content, m.images))
        elif m.role == "tool":
            out.append(ToolMessage(m.content, tool_call_id=m.tool_call_id or "", name=m.name))
        else:
            out.append(ai_message(ModelResponse(content=m.content, tool_calls=m.tool_calls)))
    return out


def ai_message(resp: ModelResponse) -> AIMessage:
    valid = [c for c in resp.tool_calls if not c.parse_error]
    invalid = [c for c in resp.tool_calls if c.parse_error]
    usage = None
    if resp.usage:
        usage = {
            "input_tokens": resp.usage.prompt_tokens,
            "output_tokens": resp.usage.completion_tokens,
            "total_tokens": resp.usage.prompt_tokens + resp.usage.completion_tokens,
        }
    return AIMessage(
        content=resp.content,
        tool_calls=[{"name": c.name, "args": c.arguments, "id": c.id, "type": "tool_call"} for c in valid],
        invalid_tool_calls=[
            {"name": c.name, "args": c.raw_arguments or "", "id": c.id, "error": c.parse_error, "type": "invalid_tool_call"}
            for c in invalid
        ],
        usage_metadata=usage,
        response_metadata={"finish_reason": resp.finish_reason, "tool_call_source": resp.tool_call_source},
        additional_kwargs={"reasoning_content": resp.reasoning} if resp.reasoning else {},
    )


def tool_calls_of(msg: AIMessage) -> list[ToolCall]:
    """All tool calls of an AIMessage (valid first, then unparseable ones), as lca ToolCalls."""
    return to_lca([msg])[0].tool_calls


def _to_spec(tool: Any) -> ToolSpec:
    if isinstance(tool, ToolSpec):
        return tool
    fn = convert_to_openai_tool(tool)["function"]
    return ToolSpec(name=fn["name"], description=fn.get("description", ""), parameters=fn.get("parameters", {"type": "object"}))


# ---------------------------------------------------------------- the chat model


class LocalChatModel(BaseChatModel):
    """LangChain chat model for local OpenAI-compatible servers (llama.cpp, LM Studio, vLLM) and Ollama."""

    provider: Provider
    tools: list[ToolSpec] = Field(default_factory=list)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def _llm_type(self) -> str:
        return f"lca-{self.provider.kind}"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"provider": self.provider.kind, "model": self.provider.model, "base_url": self.provider.settings.base_url}

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> LocalChatModel:  # type: ignore[override]
        """Accepts lca ToolSpecs, LangChain tools, pydantic models or OpenAI tool dicts."""
        return self.model_copy(update={"tools": [_to_spec(t) for t in tools]})

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        on_text: Callable[[str], None] | None = kwargs.get("on_text")

        def text(t: str) -> None:
            if on_text:
                on_text(t)
            if run_manager:
                run_manager.on_llm_new_token(t)

        resp = self.provider.chat(
            to_lca(messages),
            kwargs.get("tools", self.tools),
            on_text=text,
            on_reasoning=kwargs.get("on_reasoning"),
            on_notice=kwargs.get("on_notice"),
            cancel=kwargs.get("cancel"),
        )
        msg = ai_message(resp)
        return ChatResult(generations=[ChatGeneration(message=msg)], llm_output={"usage": resp.usage.model_dump() if resp.usage else {}})


def describe_args(arguments: dict[str, Any]) -> str:
    return json.dumps(arguments, sort_keys=True, default=str)
