"""End-to-end: the real CLI, provider and tools against a scripted local HTTP model server."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from typer.testing import CliRunner

from lca import __version__
from lca.cli import app


class FakeModelServer:
    def __init__(self, replies: list[dict]) -> None:
        self.replies = list(replies)
        self.requests: list[dict] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _json(self, obj, status=200):
                body = json.dumps(obj).encode()
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path.endswith("/models"):
                    return self._json({"data": [{"id": "fake-model"}]})
                if self.path.endswith("/props"):
                    return self._json({"default_generation_settings": {"n_ctx": 16384}})
                self._json({"error": "not found"}, 404)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["content-length"])))
                outer.requests.append(body)
                delta = outer.replies.pop(0)
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.end_headers()
                for ev in (delta, {"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 42, "completion_tokens": 3}}):
                    self.wfile.write(f"data: {json.dumps({'choices': [{'delta': ev}]} if 'choices' not in ev else ev)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()


@pytest.fixture
def isolated_env(monkeypatch, tmp_path_factory):
    monkeypatch.setenv("LCA_HOME", str(tmp_path_factory.mktemp("lca_home")))
    for var in ("LCA_BASE_URL", "LCA_MODEL", "LCA_PROVIDER", "LCA_ALLOW_REMOTE", "LCA_API_KEY", "LCA_TOOL_MODE"):
        monkeypatch.delenv(var, raising=False)


def test_run_end_to_end_text_output(project, isolated_env):
    server = FakeModelServer([
        {"content": "Reading.", "tool_calls": [{"index": 0, "id": "a", "function": {"name": "read_file", "arguments": '{"path": "src/app.py"}'}}]},
        {"content": "add returns a + b."},
    ])
    try:
        res = CliRunner().invoke(app, ["run", "what does add do", "--root", str(project), "--base-url", server.url])
    finally:
        server.close()
    assert res.exit_code == 0, res.output
    assert "read_file" in res.output and "add returns a + b." in res.output
    assert "completed after 2 iteration(s)" in res.output
    assert server.requests[0]["model"] == "fake-model"
    assert "def add(a, b):" in server.requests[1]["messages"][-1]["content"]


def test_run_json_event_stream(project, isolated_env):
    server = FakeModelServer([{"content": '<tool_call>{"name": "list_files", "arguments": {"glob": "*.py"}}</tool_call>'}, {"content": "Two files."}])
    try:
        res = CliRunner().invoke(app, ["run", "list python files", "--root", str(project), "--base-url", server.url, "-o", "json"])
    finally:
        server.close()
    assert res.exit_code == 0, res.output
    events = [json.loads(line) for line in res.stdout.splitlines() if line.startswith("{")]
    types = [e["type"] for e in events]
    assert types[0] == "turn_started" and types[-1] == "run_finished"
    finished = next(e for e in events if e["type"] == "tool_call_finished")
    assert finished["name"] == "list_files" and "src/app.py" in finished["output"]
    assert events[-1]["result"]["status"] == "completed"


def test_remote_server_refused_without_opt_in(project, isolated_env):
    res = CliRunner().invoke(app, ["run", "x", "--root", str(project), "--base-url", "https://api.example.com/v1"])
    assert res.exit_code == 2
    assert "Refusing" in res.output


def test_unreachable_server_reported(project, isolated_env):
    res = CliRunner().invoke(app, ["run", "x", "--root", str(project), "--base-url", "http://127.0.0.1:9/v1"])
    assert res.exit_code == 2 and "Cannot reach the model server" in res.output


def test_blocked_exit_code(project, isolated_env):
    server = FakeModelServer([{"content": "BLOCKED: need the database schema."}])
    try:
        res = CliRunner().invoke(app, ["run", "x", "--root", str(project), "--base-url", server.url])
    finally:
        server.close()
    assert res.exit_code == 3


def test_tools_and_version_commands():
    runner = CliRunner()
    assert "read_file" in runner.invoke(app, ["tools"]).output
    assert runner.invoke(app, ["--version"]).output.strip() == __version__
