"""Context compaction for the graph: keep the conversation inside the model's window without losing what matters.

Steps, cheapest first, each applied only if the previous one was not enough:
1. shorten old tool outputs (files can be re-read),
2. summarize earlier conversation turns into one message (decisions, files, errors, open items),
3. summarize the completed steps of the current task into the task message,
4. hard-trim whatever tool output is left.
Summaries come from the model when available, otherwise from a mechanical digest, so compaction never fails.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from lca.agent.context import TRIM_NOTE, ContextBudget
from lca.graph.chat_model import _text, to_lca

log = logging.getLogger("lca.graph")

SUMMARY_PREFIX = "[Summary of the earlier conversation]"
PROGRESS_PREFIX = "[Progress on this task so far, summarized to save context]"
Summarizer = Callable[[str], str]  # transcript -> summary

SUMMARY_INSTRUCTIONS = (
    "Summarize this part of a coding session so the work can continue without it. Keep: what the user asked for, "
    "decisions made, files created/changed/read and what they contain that matters, commands run and their results, "
    "errors and whether they were fixed, and what is still left to do. Use short bullet points. Be factual."
)


def _is_real_user(m: BaseMessage) -> bool:
    return isinstance(m, HumanMessage) and not _text(m.content).startswith(SUMMARY_PREFIX)


def render_transcript(messages: Sequence[BaseMessage], limit_chars: int = 24000) -> str:
    lines: list[str] = []
    for m in to_lca(messages):
        if m.role == "tool":
            lines.append(f"TOOL RESULT ({m.name}): {m.content[:1200]}")
        elif m.role == "assistant":
            if m.content:
                lines.append(f"ASSISTANT: {m.content[:1500]}")
            for c in m.tool_calls:
                lines.append(f"ASSISTANT CALLED {c.name}({json.dumps(c.arguments, default=str)[:300]})")
        else:
            lines.append(f"{m.role.upper()}: {m.content[:2000]}")
    text = "\n".join(lines)
    return text if len(text) <= limit_chars else "...\n" + text[-limit_chars:]


def digest(messages: Sequence[BaseMessage]) -> str:
    """Mechanical summary used when the model cannot summarize (or summarizing is disabled)."""
    out: list[str] = []
    pending: dict[str, str] = {}
    for m in to_lca(messages):
        if m.role == "user":
            out.append(f"- User asked: {m.content[:300]}")
        elif m.role == "assistant":
            for c in m.tool_calls:
                pending[c.id] = f"{c.name}({json.dumps(c.arguments, default=str)[:120]})"
            if m.content and not m.tool_calls:
                out.append(f"- Assistant answered: {m.content[:300]}")
        elif m.role == "tool":
            first = next((line for line in m.content.splitlines() if line.strip()), "")[:160]
            out.append(f"- {pending.get(m.tool_call_id or '', m.name)} → {first}")
    return "\n".join(out[-60:]) or "- (nothing important)"


class Compactor:
    def __init__(self, budget: ContextBudget, summarizer: Summarizer | None = None) -> None:
        self.budget = budget
        self.summarizer = summarizer

    def estimate(self, system: str, messages: Sequence[BaseMessage]) -> int:
        return self.budget.estimate(to_lca([SystemMessage(system), *messages]))

    def _summary(self, messages: Sequence[BaseMessage]) -> str:
        if self.summarizer:
            try:
                text = self.summarizer(render_transcript(messages)).strip()
                if text:
                    return text
            except Exception:  # summarizing is an optimization; fall back to the digest
                log.warning("model summary failed; using digest", exc_info=True)
        return digest(messages)

    def fit(self, system: str, messages: list[BaseMessage], aggressive: bool = False) -> tuple[list[BaseMessage] | None, str | None]:
        """Returns (new message list, notice) when something had to change, else (None, None)."""
        limit = int(self.budget.limit * (0.6 if aggressive else 0.85))
        self.budget.last_estimate = self.estimate(system, messages)
        if self.budget.last_estimate <= limit:
            return None, None
        msgs = list(messages)

        def fits() -> bool:
            self.budget.last_estimate = self.estimate(system, msgs)
            return self.budget.last_estimate <= limit

        # 1. shorten old tool outputs
        for i in range(max(0, len(msgs) - 4)):
            m = msgs[i]
            content = _text(m.content)
            if isinstance(m, ToolMessage) and len(content) > 900 and TRIM_NOTE not in content:
                msgs[i] = m.model_copy(update={"content": content[:500] + TRIM_NOTE + content[-250:]})
        if fits():
            return msgs, "Shortened older tool outputs to stay within the context window."

        users = [i for i, m in enumerate(msgs) if _is_real_user(m)]
        last_user = users[-1] if users else 0

        # 2. summarize earlier turns
        notes: list[str] = []
        if last_user > 0:
            head = msgs[:last_user]
            summary = self._summary(head)
            msgs = [HumanMessage(f"{SUMMARY_PREFIX}\n{summary}"), AIMessage("Understood, I have that context."), *msgs[last_user:]]
            last_user = 2
            notes.append(f"summarized {len(head)} earlier message(s)")
            if fits():
                return msgs, "Context compacted: " + ", ".join(notes) + "."

        # 3. summarize completed steps of the current task, keeping the most recent exchange
        tail_start = next((i for i in range(max(last_user + 1, len(msgs) - 6), len(msgs)) if isinstance(msgs[i], AIMessage)), None)
        if tail_start is not None and tail_start > last_user + 1:
            middle = msgs[last_user + 1 : tail_start]
            summary = self._summary(middle)
            task = _text(msgs[last_user].content).split(f"\n\n{PROGRESS_PREFIX}")[0]
            msgs = [*msgs[:last_user], HumanMessage(f"{task}\n\n{PROGRESS_PREFIX}\n{summary}"), *msgs[tail_start:]]
            notes.append(f"summarized {len(middle)} earlier step(s) of this task")
            if fits():
                return msgs, "Context compacted: " + ", ".join(notes) + "."

        # 4. last resort: trim every remaining tool output
        for i, m in enumerate(msgs):
            content = _text(m.content)
            if isinstance(m, ToolMessage) and len(content) > 600:
                msgs[i] = m.model_copy(update={"content": content[:350] + TRIM_NOTE + content[-150:]})
        fits()
        notes.append("trimmed recent tool outputs")
        return msgs, "Context compacted: " + ", ".join(notes) + "."
