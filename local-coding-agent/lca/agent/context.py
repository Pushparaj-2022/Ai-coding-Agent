"""Context-window budgeting: estimate prompt size and shed old tool output before the window overflows.

Full summarizing compaction arrives in Milestone 4; this keeps long tool chains within the window now.
"""

from __future__ import annotations

import json

from lca.messages import Message

IMAGE_CHARS = 3500  # an image costs roughly 1000 tokens (varies by model and resolution)
TRIM_NOTE = "\n…[older tool output trimmed to save context; read the file again if you need it]…\n"


class ContextBudget:
    def __init__(self, window: int, reserve_output: int) -> None:
        self.window = window
        self.reserve = min(reserve_output, window // 2)
        self.ratio = 1.0  # real tokens per estimated token, learned from server usage reports
        self.last_estimate = 0

    @property
    def limit(self) -> int:
        return self.window - self.reserve

    def estimate(self, messages: list[Message]) -> int:
        chars = 0
        for m in messages:
            chars += len(m.content) + 12 + IMAGE_CHARS * len(m.images)
            for c in m.tool_calls:
                chars += len(c.name) + len(json.dumps(c.arguments)) + 20
        return int(chars / 3.5 * self.ratio)

    def calibrate(self, prompt_tokens: int) -> None:
        """Use the server's real prompt-token count to correct the character-based estimate."""
        if prompt_tokens <= 0 or self.last_estimate <= 0:
            return
        observed = prompt_tokens / (self.last_estimate / self.ratio)
        self.ratio = max(0.5, min(3.0, 0.5 * self.ratio + 0.5 * observed))

    def fit(self, messages: list[Message], aggressive: bool = False) -> str | None:
        """Shrink `messages` in place if needed. Returns a notice describing what was removed, or None."""
        limit = int(self.limit * (0.6 if aggressive else 0.85))
        self.last_estimate = self.estimate(messages)
        if self.last_estimate <= limit:
            return None

        protected_from = max(1, len(messages) - 4)
        for m in messages[1:protected_from]:
            if m.role == "tool" and len(m.content) > 900 and TRIM_NOTE not in m.content:
                m.content = m.content[:500] + TRIM_NOTE + m.content[-250:]
        self.last_estimate = self.estimate(messages)
        if self.last_estimate <= limit:
            return "Trimmed older tool outputs to stay within the context window."

        dropped = 0
        while self.last_estimate > limit and _drop_oldest(messages, max(1, len(messages) - 4)):
            dropped += 1
            self.last_estimate = self.estimate(messages)
        if dropped:
            return f"Removed {dropped} older exchange(s) to stay within the context window."
        return "Context is nearly full and nothing more can be trimmed."


def _drop_oldest(messages: list[Message], keep_from: int) -> bool:
    user_idx = [i for i, m in enumerate(messages) if m.role == "user"]
    last_user = user_idx[-1] if user_idx else -1
    i = 1
    while i < min(len(messages), keep_from):
        m = messages[i]
        if i == last_user:
            i += 1
            continue
        if m.role == "assistant" and m.tool_calls:
            j = i + 1
            while j < len(messages) and messages[j].role == "tool":
                j += 1
            if j > keep_from:
                return False
            del messages[i:j]
            return True
        if m.role in ("user", "assistant") and i < last_user:
            del messages[i]
            return True
        i += 1
    return False
