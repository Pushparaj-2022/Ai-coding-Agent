from __future__ import annotations

import json
import threading

import httpx
import pytest

from lca.cancel import Cancelled, CancelToken
from lca.config import ProviderSettings
from lca.messages import Message
from lca.providers.base import (
    ContextOverflowError,
    MalformedResponseError,
    ProviderConnectionError,
    ProviderHTTPError,
    ProviderTimeoutError,
    ToolSpec,
    is_local_url,
)
from lca.providers.ollama import OllamaProvider
from lca.providers.openai_compatible import OpenAICompatibleProvider

TOOLS = [ToolSpec(name="read_file", description="Read", parameters={"type": "object", "properties": {"path": {"type": "string"}}})]
MSGS = [Message.system("sys"), Message.user("hi")]


def sse(*events: dict | str) -> bytes:
    lines = [f"data: {e if isinstance(e, str) else json.dumps(e)}\n\n" for e in events]
    return ("".join(lines) + "data: [DONE]\n\n").encode()


def delta(**d) -> dict:
    return {"choices": [{"delta": d}]}


def make(handler, kind="openai", **settings) -> OpenAICompatibleProvider | OllamaProvider:
    s = ProviderSettings(kind=kind, model="m", retry_backoff_s=0, **settings)
    cls = OllamaProvider if kind == "ollama" else OpenAICompatibleProvider
    return cls(s, transport=httpx.MockTransport(handler))


def stream_response(body: bytes, status: int = 200) -> httpx.Response:
    return httpx.Response(status, headers={"content-type": "text/event-stream"}, content=body)


def test_streaming_text_and_fragmented_native_tool_call():
    body = sse(
        delta(content="Let me "),
        delta(content="look."),
        delta(tool_calls=[{"index": 0, "id": "x1", "function": {"name": "read_file", "arguments": '{"pa'}}]),
        delta(tool_calls=[{"index": 0, "function": {"arguments": 'th": "a.py"}'}}]),
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 50, "completion_tokens": 7}},
    )
    seen = []
    p = make(lambda req: stream_response(body))
    r = p.chat(MSGS, TOOLS, on_text=seen.append)
    assert "".join(seen) == "Let me look."
    assert r.content == "Let me look."
    assert r.tool_call_source == "native"
    assert [(c.id, c.name, c.arguments) for c in r.tool_calls] == [("x1", "read_file", {"path": "a.py"})]
    assert r.usage.prompt_tokens == 50 and r.finish_reason == "tool_calls"


def test_request_contains_tools_and_wire_format():
    captured = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured.update(json.loads(req.content))
        return stream_response(sse(delta(content="ok")))

    history = MSGS + [
        Message.assistant("", [__import__("lca.messages", fromlist=["ToolCall"]).ToolCall(id="t1", name="read_file", arguments={"path": "a"})]),
        Message.tool("t1", "read_file", "content"),
    ]
    make(handler).chat(history, TOOLS)
    assert captured["tools"][0]["function"]["name"] == "read_file"
    assert captured["messages"][2]["tool_calls"][0]["function"]["arguments"] == '{"path": "a"}'
    assert captured["messages"][3] == {"role": "tool", "tool_call_id": "t1", "content": "content"}


def test_text_tool_calls_in_content_are_parsed():
    body = sse(delta(content='Reading.<tool_call>{"name": "read_file", "arguments": {"path": "b.py"}}</tool_call>'))
    r = make(lambda req: stream_response(body)).chat(MSGS, TOOLS)
    assert r.tool_call_source == "text"
    assert r.content == "Reading."
    assert r.tool_calls[0].arguments == {"path": "b.py"}


def test_invalid_native_arguments_flagged_not_raised():
    body = sse(delta(tool_calls=[{"index": 0, "id": "x", "function": {"name": "read_file", "arguments": '{"path": "a'}}]))
    r = make(lambda req: stream_response(body)).chat(MSGS, TOOLS)
    assert r.tool_calls[0].parse_error and r.tool_calls[0].raw_arguments == '{"path": "a'


def test_reasoning_and_think_tags_separated():
    body = sse(delta(reasoning_content="hmm"), delta(content="<think>inner</think>Answer"))
    r = make(lambda req: stream_response(body)).chat(MSGS)
    assert r.content == "Answer"
    assert "hmm" in r.reasoning and "inner" in r.reasoning


def test_non_streaming_json_reply():
    obj = {"choices": [{"message": {"content": "plain"}, "finish_reason": "stop"}]}
    r = make(lambda req: httpx.Response(200, json=obj)).chat(MSGS)
    assert r.content == "plain"


@pytest.mark.parametrize(
    "body",
    [b"data: not json\n\n", b"", b"event: ping\n\n"],
)
def test_malformed_streams(body):
    with pytest.raises(MalformedResponseError):
        make(lambda req: stream_response(body), max_retries=0).chat(MSGS)


def test_error_event_inside_stream():
    body = sse({"error": {"message": "the request exceeds the available context size"}})
    with pytest.raises(ContextOverflowError):
        make(lambda req: stream_response(body)).chat(MSGS)


def test_context_overflow_http_error():
    with pytest.raises(ContextOverflowError):
        make(lambda req: httpx.Response(400, text="request (9000 tokens) exceeds the available context size")).chat(MSGS)


def test_auto_mode_falls_back_to_text_tools():
    bodies = []

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        bodies.append(body)
        if "tools" in body:
            return httpx.Response(500, text="tools param requires --jinja flag")
        return stream_response(sse(delta(content='<tool_call>{"name":"read_file","arguments":{"path":"x"}}</tool_call>')))

    notices = []
    p = make(handler)
    r = p.chat(MSGS, TOOLS, on_notice=notices.append)
    assert p.effective_tool_mode == "text" and notices
    assert "tools" not in bodies[1]
    assert "<tool_call>" in bodies[1]["messages"][0]["content"]  # instructions added to the system prompt
    assert r.tool_calls[0].name == "read_file"


def test_native_mode_does_not_fall_back():
    p = make(lambda req: httpx.Response(500, text="tools param requires --jinja flag"), tool_mode="native")
    with pytest.raises(Exception, match="native tool calling"):
        p.chat(MSGS, TOOLS)


def test_text_mode_renders_history_as_alternating_messages():
    captured = {}

    def handler(req):
        captured.update(json.loads(req.content))
        return stream_response(sse(delta(content="done")))

    from lca.messages import ToolCall

    history = MSGS + [Message.assistant("", [ToolCall(id="1", name="read_file", arguments={"path": "a"})]), Message.tool("1", "read_file", "DATA")]
    make(handler, tool_mode="text").chat(history, TOOLS)
    roles = [m["role"] for m in captured["messages"]]
    assert roles == ["system", "user", "assistant", "user"]
    assert '<tool_result name="read_file">' in captured["messages"][3]["content"]


def test_connection_error_retries_then_raises():
    attempts = []

    def handler(req):
        attempts.append(1)
        raise httpx.ConnectError("refused")

    with pytest.raises(ProviderConnectionError):
        make(handler, max_retries=2).chat(MSGS)
    assert len(attempts) == 3


def test_retry_recovers_from_503():
    replies = [httpx.Response(503, text="loading model"), stream_response(sse(delta(content="ok")))]
    r = make(lambda req: replies.pop(0)).chat(MSGS)
    assert r.content == "ok"


def test_timeout_error():
    def handler(req):
        raise httpx.ReadTimeout("slow")

    with pytest.raises(ProviderTimeoutError):
        make(handler, max_retries=0).chat(MSGS)


def test_auth_error_not_retried():
    attempts = []

    def handler(req):
        attempts.append(1)
        return httpx.Response(401, text="bad key")

    with pytest.raises(ProviderHTTPError, match="LCA_API_KEY"):
        make(handler).chat(MSGS)
    assert len(attempts) == 1


def test_cancellation_mid_stream():
    token = CancelToken()

    def chunks():
        yield b'data: {"choices":[{"delta":{"content":"a"}}]}\n\n'
        token.cancel()
        yield b'data: {"choices":[{"delta":{"content":"b"}}]}\n\n'

    p = make(lambda req: httpx.Response(200, headers={"content-type": "text/event-stream"}, content=chunks()))
    with pytest.raises(Cancelled):
        p.chat(MSGS, cancel=token)


def test_server_info_reads_models_and_context():
    def handler(req):
        if req.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "lfm"}]})
        if req.url.path.endswith("/props"):
            return httpx.Response(200, json={"default_generation_settings": {"n_ctx": 16384}})
        return httpx.Response(404)

    info = make(handler).server_info()
    assert info.reachable and info.models == ["lfm"] and info.context_window == 16384


def test_server_info_unreachable():
    def handler(req):
        raise httpx.ConnectError("refused")

    assert make(handler).server_info().reachable is False


# ------------------------------------------------------------------ ollama


def ndjson(*objs: dict) -> bytes:
    return "".join(json.dumps(o) + "\n" for o in objs).encode()


def test_ollama_stream_with_tool_calls():
    body = ndjson(
        {"message": {"role": "assistant", "content": "Checking."}, "done": False},
        {"message": {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "read_file", "arguments": {"path": "z"}}}]}, "done": False},
        {"message": {"content": ""}, "done": True, "done_reason": "stop", "prompt_eval_count": 30, "eval_count": 5},
    )
    captured = {}

    def handler(req):
        captured.update(json.loads(req.content))
        return httpx.Response(200, content=body)

    r = make(handler, kind="ollama").chat(MSGS, TOOLS)
    assert captured["tools"][0]["function"]["name"] == "read_file"
    assert r.content == "Checking." and r.tool_calls[0].arguments == {"path": "z"}
    assert r.usage.prompt_tokens == 30


def test_ollama_model_without_tool_support_falls_back():
    def handler(req):
        body = json.loads(req.content)
        if "tools" in body:
            return httpx.Response(400, json={"error": "registry.ollama.ai/library/x does not support tools"})
        return httpx.Response(200, content=ndjson({"message": {"content": "hi"}, "done": True}))

    p = make(handler, kind="ollama")
    assert p.chat(MSGS, TOOLS).content == "hi"
    assert p.effective_tool_mode == "text"


@pytest.mark.parametrize(
    "url,local",
    [
        ("http://127.0.0.1:8080/v1", True),
        ("http://localhost:11434", True),
        ("http://[::1]:8080", True),
        ("http://192.168.1.20:8080", False),
        ("https://api.example.com/v1", False),
        ("http://localhost.evil.com", False),
    ],
)
def test_is_local_url(url, local):
    assert is_local_url(url) is local
