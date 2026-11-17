"""Constrained text format for models without native tool calling."""

from __future__ import annotations

import json

from lca.messages import Message
from lca.providers.base import ToolSpec


def tool_instructions(tools: list[ToolSpec]) -> str:
    listing = "\n\n".join(
        f"### {t.name}\n{t.description}\nArguments (JSON schema): {json.dumps(t.parameters, separators=(',', ':'))}"
        for t in tools
    )
    return f"""

# Tool use
To call a tool, write exactly this, with valid JSON (double quotes, newlines escaped as \\n):
<tool_call>{{"name": "TOOL_NAME", "arguments": {{...}}}}</tool_call>
You may call several tools in one reply. After your calls, stop and wait: results arrive in <tool_result> blocks.
When you are finished, reply normally without any <tool_call>.

## Available tools

{listing}"""


def render_text_mode(messages: list[Message], tools: list[ToolSpec]) -> list[Message]:
    """Rewrite a native-style conversation into plain user/assistant messages with tool-call markup."""
    out: list[Message] = []
    has_system = False
    for m in messages:
        if m.role == "system":
            has_system = True
            out.append(Message.system(m.content + tool_instructions(tools)))
        elif m.role == "assistant" and m.tool_calls:
            calls = "\n".join(
                f"<tool_call>{json.dumps({'name': c.name, 'arguments': c.arguments})}</tool_call>" for c in m.tool_calls
            )
            out.append(Message.assistant(f"{m.content}\n{calls}".strip()))
        elif m.role == "tool":
            out.append(Message.user(f'<tool_result name="{m.name}">\n{m.content}\n</tool_result>'))
        else:
            out.append(Message(role=m.role, content=m.content))
    if not has_system:
        out.insert(0, Message.system(tool_instructions(tools).strip()))
    merged: list[Message] = []
    for m in out:  # chat templates expect user/assistant to alternate
        if merged and m.role == "user" and merged[-1].role == "user":
            merged[-1] = Message.user(f"{merged[-1].content}\n\n{m.content}")
        else:
            merged.append(m)
    return merged
