"""Parse tool calls that a model wrote as text instead of using native tool calling.

Supported formats:
  <tool_call>{"name": "read_file", "arguments": {"path": "a.py"}}</tool_call>
  <|tool_call_start|>[read_file(path="a.py")]<|tool_call_end|>     (LFM2 / pythonic style)
  <tools>{...}</tools> or <json>{...}</json>   (wrong tag, seen from fine-tuned Qwen models; JSON calls only)
  <|tool_call>call:read_file{"path": "a.py"}<tool_call|>   (gemma-4 native; args may also be {path:<|"|>a.py<|"|>})
  a bare JSON object {"name": ..., "arguments": ...} when the name is a known tool
Everything parsed here is validated again against the tool's schema before execution.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

TAG_JSON = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
TAG_LFM = re.compile(r"<\|tool_call_start\|>\s*(.*?)\s*<\|tool_call_end\|>", re.S)
TAG_ALIAS = re.compile(r"<(tools|json)>\s*(.*?)\s*</\1>", re.S)
OPEN_TAG = re.compile(r"<tool_call>|<\|tool_call_start\|>")
FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")
THINK = re.compile(r"<think>(.*?)(?:</think>|$)|<\|channel>(?:thought\n?)?(.*?)(?:<channel\|>|$)", re.S)
GEMMA_OPEN = "<|tool_call>"
GEMMA_CLOSE = "<tool_call|>"
GEMMA_QUOTE = '<|"|>'

# Chat-template markers that start or end a turn. A reply never legitimately contains them: when a model emits one it
# has run past the end of its turn and is inventing the next message (typically a fake tool result), so the reply is
# cut there. Also sent to the server as stop sequences.
TURN_MARKERS = (
    "<|turn>", "<turn|>",  # gemma-4
    "<start_of_turn>", "<end_of_turn>",  # gemma-2/3
    "<|im_start|>", "<|im_end|>",  # chatml (qwen, ...)
    "<|eot_id|>", "<|start_header_id|>",  # llama-3
    "<|endoftext|>",
)


def cut_at_turn_marker(text: str) -> str:
    """The reply up to the first turn marker, with gemma-4 thought-channel tags removed."""
    cut = min((i for i in (text.find(m) for m in TURN_MARKERS) if i >= 0), default=-1)
    if cut >= 0:
        text = text[:cut]
    return text.replace("<channel|>", "").replace("<|channel>", "")
GEMMA_HEAD = re.compile(r"\s*(?:call:)?([A-Za-z_][\w.\-]*)\s*")

_FAIL = object()


@dataclass
class ParsedCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    raw: str = ""


def parse_json_loose(text: str) -> Any:
    """json.loads that tolerates code fences and trailing commas. Returns _FAIL on failure."""
    s = FENCE.sub("", text.strip())
    if not s:
        return _FAIL
    for candidate in (s, re.sub(r",\s*([}\]])", r"\1", s)):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return _FAIL


def is_parse_failure(value: Any) -> bool:
    return value is _FAIL


def _normalize(obj: Any, raw: str) -> ParsedCall | None:
    if not isinstance(obj, dict):
        return None
    fn = obj.get("function") if isinstance(obj.get("function"), dict) else {}
    name = obj.get("name") or obj.get("tool") or fn.get("name")
    if not isinstance(name, str) or not name:
        return None
    args: Any = {}
    for key in ("arguments", "parameters", "args", "input"):
        if key in obj:
            args = obj[key]
            break
    else:
        args = fn.get("arguments", {})
    if isinstance(args, str):
        parsed = parse_json_loose(args) if args.strip() else {}
        if parsed is _FAIL:
            return ParsedCall(name=name, error="arguments are not valid JSON", raw=args)
        args = parsed
    if args is None:
        args = {}
    if not isinstance(args, dict):
        return ParsedCall(name=name, error="arguments must be a JSON object", raw=raw)
    return ParsedCall(name=name, arguments=args, raw=raw)


def _parse_body(raw: str) -> list[ParsedCall]:
    value = parse_json_loose(raw)
    if value is not _FAIL:
        items = value if isinstance(value, list) else [value]
        calls = [c for c in (_normalize(item, raw) for item in items) if c]
        if calls:
            return calls
    try:
        return parse_pythonic_calls(raw)
    except ValueError as e:
        return [ParsedCall(name="__invalid__", error=f"could not parse tool call: {e}", raw=raw)]


def _gemma_args_to_json(s: str, i: int) -> tuple[str, int]:
    """Read the {...} block that starts at s[i]; returns (JSON text, index after the block).

    Accepts plain JSON and gemma-4's own syntax: bare keys and <|"|>string<|"|> quoting.
    Raises ValueError when the braces never close."""
    out: list[str] = []
    depth = 0
    n = len(s)
    while i < n:
        if s.startswith(GEMMA_QUOTE, i):
            end = s.find(GEMMA_QUOTE, i + len(GEMMA_QUOTE))
            if end < 0:
                raise ValueError("unterminated <|\"|> string")
            out.append(json.dumps(s[i + len(GEMMA_QUOTE) : end]))
            i = end + len(GEMMA_QUOTE)
            continue
        c = s[i]
        if c == '"':
            j = i + 1
            while j < n and s[j] != '"':
                j += 2 if s[j] == "\\" else 1
            if j >= n:
                raise ValueError("unterminated string")
            out.append(s[i : j + 1])
            i = j + 1
            continue
        if c.isalpha() or c == "_":
            m = _IDENT.match(s, i)
            assert m
            word = m.group()
            rest = s[m.end() :].lstrip()
            if rest.startswith(":") and word not in ("true", "false", "null"):
                out.append(json.dumps(word))
            else:
                out.append(word)
            i = m.end()
            continue
        if c in "{[":
            depth += 1
        elif c in "}]":
            depth -= 1
        out.append(c)
        i += 1
        if depth == 0:
            return "".join(out), i
    raise ValueError("unterminated arguments")


def _extract_gemma_calls(content: str) -> tuple[list[ParsedCall], str]:
    """gemma-4 native markup: <|tool_call>call:NAME{args}<tool_call|> (closing tag optional)."""
    if GEMMA_OPEN not in content:
        return [], content
    calls: list[ParsedCall] = []
    kept: list[str] = []
    pos = 0
    while True:
        start = content.find(GEMMA_OPEN, pos)
        if start < 0:
            kept.append(content[pos:])
            break
        kept.append(content[pos:start])
        i = start + len(GEMMA_OPEN)
        head = GEMMA_HEAD.match(content, i)
        if not head:
            calls.append(ParsedCall(name="__invalid__", error="tool call has no name", raw=content[start:]))
            break
        name, i = head.group(1), head.end()
        if i < len(content) and content[i] == "{":
            try:
                text, i = _gemma_args_to_json(content, i)
            except ValueError as e:
                calls.append(ParsedCall(name=name, error=f"could not parse tool call: {e}", raw=content[start:]))
                break
            value = parse_json_loose(text)
            if is_parse_failure(value) or not isinstance(value, dict):
                calls.append(ParsedCall(name=name, error="arguments are not a valid JSON object", raw=text))
            else:
                calls.append(ParsedCall(name=name, arguments=value, raw=content[start:i]))
        else:
            calls.append(ParsedCall(name=name, raw=content[start:i]))
        rest = content[i:]
        stripped = rest.lstrip()
        if stripped.startswith(GEMMA_CLOSE):
            i += len(rest) - len(stripped) + len(GEMMA_CLOSE)
        pos = i
    return calls, "".join(kept)


def extract_tool_calls(content: str, known_tools: set[str] | None = None) -> tuple[list[ParsedCall], str]:
    """Returns (calls, content with the tool-call markup removed)."""
    calls, content = _extract_gemma_calls(content)
    for m in TAG_JSON.finditer(content):
        calls.extend(_parse_body(m.group(1)))
    for m in TAG_LFM.finditer(content):
        calls.extend(_parse_body(m.group(1)))
    cleaned = TAG_LFM.sub("", TAG_JSON.sub("", content))

    def alias_call(m: re.Match[str]) -> str:
        """These tags also occur in ordinary text, so they count only when the body is a well-formed JSON call."""
        value = parse_json_loose(m.group(2))
        items = value if isinstance(value, list) else [value]
        found = [_normalize(item, m.group(2)) for item in items] if value is not _FAIL else []
        if not found or not all(c and not c.error for c in found):
            return m.group(0)
        calls.extend(c for c in found if c)
        return ""

    cleaned = TAG_ALIAS.sub(alias_call, cleaned)

    # An opening tag without a closing one (model forgot it or hit the token limit).
    m = OPEN_TAG.search(cleaned)
    if m:
        calls.extend(_parse_body(cleaned[m.end():]))
        cleaned = cleaned[: m.start()]

    if not calls and known_tools:
        value = parse_json_loose(cleaned)
        if value is not _FAIL:
            call = _normalize(value, cleaned)
            if call and call.name in known_tools and ("arguments" in value or "parameters" in value):
                return [call], ""
    return calls, cleaned.strip()


# ---------------------------------------------------------------- pythonic calls: [fn(a="x", b=1)]

_IDENT = re.compile(r"[A-Za-z_][\w.\-]*")
_LITERAL = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|True|False|None|true|false|null")
_ESC = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\", '"': '"', "'": "'", "0": "\0"}


class _PyParser:
    def __init__(self, s: str) -> None:
        self.s = s
        self.i = 0

    def fail(self, msg: str) -> None:
        raise ValueError(f"{msg} at position {self.i}")

    def peek(self) -> str:
        return self.s[self.i] if self.i < len(self.s) else ""

    def ws(self) -> None:
        while self.i < len(self.s) and self.s[self.i].isspace():
            self.i += 1

    def ident(self) -> str:
        m = _IDENT.match(self.s, self.i)
        if not m:
            self.fail("expected identifier")
        self.i = m.end()
        return m.group()

    def string(self, quote_char: str) -> str:
        q = quote_char * 3 if self.s.startswith(quote_char * 3, self.i) else quote_char
        self.i += len(q)
        out: list[str] = []
        while not self.s.startswith(q, self.i):
            if self.i >= len(self.s):
                self.fail("unterminated string")
            c = self.s[self.i]
            if c == "\\" and self.i + 1 < len(self.s):
                n = self.s[self.i + 1]
                if n == "u":
                    try:
                        out.append(chr(int(self.s[self.i + 2 : self.i + 6], 16)))
                    except ValueError:
                        self.fail("bad unicode escape")
                    self.i += 6
                    continue
                out.append(_ESC.get(n, "\\" + n))
                self.i += 2
            else:
                out.append(c)
                self.i += 1
        self.i += len(q)
        return "".join(out)

    def sequence(self, close: str) -> list[Any]:
        self.i += 1
        items: list[Any] = []
        while True:
            self.ws()
            if self.peek() == close:
                self.i += 1
                return items
            if not self.peek():
                self.fail("unterminated list")
            items.append(self.value())
            self.ws()
            if self.peek() == ",":
                self.i += 1
            elif self.peek() != close:
                self.fail("expected ',' in list")

    def mapping(self) -> dict[Any, Any]:
        self.i += 1
        obj: dict[Any, Any] = {}
        while True:
            self.ws()
            if self.peek() == "}":
                self.i += 1
                return obj
            if not self.peek():
                self.fail("unterminated dict")
            key = self.value()
            self.ws()
            if self.peek() != ":":
                self.fail("expected ':'")
            self.i += 1
            obj[key] = self.value()
            self.ws()
            if self.peek() == ",":
                self.i += 1
            elif self.peek() != "}":
                self.fail("expected ',' in dict")

    def value(self) -> Any:
        self.ws()
        ch = self.peek()
        if not ch:
            self.fail("expected value")
        if ch in "\"'":
            return self.string(ch)
        if ch == "[":
            return self.sequence("]")
        if ch == "(":
            return self.sequence(")")
        if ch == "{":
            return self.mapping()
        m = _LITERAL.match(self.s, self.i)
        if not m:
            self.fail("expected value")
        self.i = m.end()
        v = m.group()
        if v in ("True", "true"):
            return True
        if v in ("False", "false"):
            return False
        if v in ("None", "null"):
            return None
        return float(v) if any(c in v for c in ".eE") else int(v)


def parse_pythonic_calls(src: str) -> list[ParsedCall]:
    p = _PyParser(src.strip())
    calls: list[ParsedCall] = []
    p.ws()
    is_list = p.peek() == "["
    if is_list:
        p.i += 1
    while True:
        p.ws()
        if not p.peek() or (is_list and p.peek() == "]"):
            break
        name = p.ident()
        p.ws()
        if p.peek() != "(":
            p.fail("expected '('")
        p.i += 1
        args: dict[str, Any] = {}
        while True:
            p.ws()
            if p.peek() == ")":
                p.i += 1
                break
            if not p.peek():
                p.fail("unterminated call")
            key = p.ident()
            p.ws()
            if p.peek() != "=":
                p.fail("expected '='")
            p.i += 1
            args[key] = p.value()
            p.ws()
            if p.peek() == ",":
                p.i += 1
            elif p.peek() != ")":
                p.fail("expected ',' or ')'")
        calls.append(ParsedCall(name=name, arguments=args, raw=src))
        p.ws()
        if p.peek() == ",":
            p.i += 1
        elif not is_list:
            break
    if not calls:
        raise ValueError("no calls found")
    return calls


# ---------------------------------------------------------------- streaming display filter


class StreamFilter:
    """Removes tool-call and <think> markup from streamed text, even when tags are split across chunks."""

    TAGS = [
        ("<tool_call>", "</tool_call>"),
        ("<|tool_call_start|>", "<|tool_call_end|>"),
        (GEMMA_OPEN, GEMMA_CLOSE),
        ("<think>", "</think>"),
        ("<|channel>", "<channel|>"),  # gemma-4 thought channel
        ("<channel|>", ""),  # a stray closing marker: drop just the tag
        # the model started writing the next turn itself (an invented tool result or user message): drop the rest
        *((m, "\x00never") for m in TURN_MARKERS),
    ]

    def __init__(self, out: Callable[[str], None]) -> None:
        self.out = out
        self.buf = ""
        self.close: str | None = None

    def push(self, text: str) -> None:
        self.buf += text
        self._drain(final=False)

    def end(self) -> None:
        self._drain(final=True)

    def _emit(self, text: str) -> None:
        if text:
            self.out(text)

    def _drain(self, final: bool) -> None:
        while True:
            if self.close:
                idx = self.buf.find(self.close)
                if idx < 0:
                    self.buf = "" if final else self.buf[-len(self.close) :]
                    return
                self.buf = self.buf[idx + len(self.close) :]
                self.close = None
                continue
            best, pair = -1, None
            for open_tag, close_tag in self.TAGS:
                idx = self.buf.find(open_tag)
                if idx >= 0 and (best < 0 or idx < best):
                    best, pair = idx, (open_tag, close_tag)
            if pair:
                self._emit(self.buf[:best])
                self.buf = self.buf[best + len(pair[0]) :]
                self.close = pair[1]
                continue
            lt = self.buf.rfind("<")
            if not final and lt >= 0 and any(t[0].startswith(self.buf[lt:]) for t in self.TAGS):
                self._emit(self.buf[:lt])
                self.buf = self.buf[lt:]
            else:
                self._emit(self.buf)
                self.buf = ""
            return
