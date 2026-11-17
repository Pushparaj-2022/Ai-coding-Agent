"""Provider-neutral conversation types shared by the agent loop, providers and tools."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant", "tool"]
ToolCallSource = Literal["native", "text", "none"]


class ToolCall(BaseModel):
    """A tool invocation requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    raw_arguments: str | None = None  # original text when the arguments could not be parsed
    parse_error: str | None = None

    def signature(self) -> str:
        return f"{self.name}:{json.dumps(self.arguments, sort_keys=True, default=str)}"


class Message(BaseModel):
    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None  # for role == "tool"
    name: str | None = None  # tool name for role == "tool"
    images: list[str] = Field(default_factory=list)  # user messages: images as data URLs (data:image/png;base64,...)

    @classmethod
    def system(cls, content: str) -> Message:
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> Message:
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: str, tool_calls: list[ToolCall] | None = None) -> Message:
        return cls(role="assistant", content=content, tool_calls=tool_calls or [])

    @classmethod
    def tool(cls, call_id: str, name: str, content: str) -> Message:
        return cls(role="tool", content=content, tool_call_id=call_id, name=name)


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def add(self, other: Usage | None) -> None:
        if other:
            self.prompt_tokens += other.prompt_tokens
            self.completion_tokens += other.completion_tokens


class ModelResponse(BaseModel):
    """A complete model reply, normalized across providers and tool-call formats."""

    content: str = ""
    reasoning: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: str | None = None
    usage: Usage | None = None
    tool_call_source: ToolCallSource = "none"
