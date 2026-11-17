"""MCP servers (a real stdio subprocess and an HTTP server) and the web tools."""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

import lca.graph.agent_graph as agent_graph
import lca.tools.web as web
from lca.agent.permissions import ApprovalResult, Mode
from lca.agent.rules import PermissionStore, Rule, RuleSet
from lca.agent.trust import trust_section
from lca.cancel import CancelToken
from lca.cli import Session, _make_agent
from lca.config import Settings, WebSettings
from lca.mcp import McpManager, ServerConfig, load_mcp_configs
from lca.messages import ToolCall
from lca.providers.base import ServerInfo
from lca.tools import Workspace, default_registry
from lca.tools.base import ToolContext
from lca.tools.web import WebFetch, WebSearch, check_public_url, html_to_text

from .conftest import ScriptedProvider, call, reply

FAKE = str(Path(__file__).with_name("fake_mcp_server.py"))


@pytest.fixture(autouse=True)
def isolated(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("LCA_HOME", str(tmp_path_factory.mktemp("lcahome")))
    monkeypatch.setattr(agent_graph, "verify_commands", lambda root: [])


@pytest.fixture
def fake_mcp(project):
    manager = McpManager([ServerConfig("fake", command=sys.executable, args=[FAKE])], project)
    manager.start()
    manager.wait(30)
    yield manager
    manager.stop()


def run_tool(tool, ctx, **arguments):
    reg = default_registry()
    reg.register(tool)
    return reg.execute(ToolCall(id="t", name=tool.name, arguments=arguments), ctx)


# ---------------------------------------------------------------- MCP


def test_stdio_server_tools_calls_errors_and_roots(project, fake_mcp, ctx):
    assert fake_mcp.status()[0]["status"] == "ready"
    tools = {t.name: t for t in fake_mcp.tools()}
    assert set(tools) == {"mcp__fake__echo", "mcp__fake__fail", "mcp__fake__roots"}
    assert tools["mcp__fake__echo"].spec().parameters["required"] == ["text"]
    assert tools["mcp__fake__echo"].risk == "execute"
    assert run_tool(tools["mcp__fake__echo"], ctx, text="hi").content == "echo: hi"
    failed = run_tool(tools["mcp__fake__fail"], ctx)
    assert failed.is_error and "it broke" in failed.content
    assert project.resolve().as_uri() in run_tool(tools["mcp__fake__roots"], ctx).content  # the server asked us for roots


def test_tool_allowlist_and_a_server_that_does_not_start(project):
    manager = McpManager([
        ServerConfig("fake", command=sys.executable, args=[FAKE], tools=["echo"]),
        ServerConfig("broken", command="definitely-not-a-real-program-xyz"),
    ], project)
    manager.start()
    manager.wait(30)
    status = {s["name"]: s for s in manager.status()}
    assert status["fake"]["tools"] == ["echo"] and status["broken"]["status"] == "failed" and status["broken"]["error"]
    manager.stop()


def test_agent_uses_mcp_tools_with_approval_and_rules(project, fake_mcp):
    provider = ScriptedProvider([
        reply("", call("mcp__fake__echo", text="one")),
        reply("", call("mcp__fake__echo", text="two")),
        reply("done"),
    ])
    rules = RuleSet()
    s = Session(Settings(), provider, Workspace(project), ServerInfo(reachable=True, models=["m"], context_window=8192), 8192, [], rules,
                PermissionStore(project), mcp=fake_mcp)
    asked = []

    def approver(req):
        asked.append(req.description)
        rules.add("allow", Rule.parse("mcp__fake__echo"), "test")  # as if the user then allowed it for good
        return ApprovalResult(True)

    agent = _make_agent(s, lambda e: None, s.policy(Mode.ASK, approver))
    agent.run("echo twice")
    assert "mcp__fake__echo" in [t.name for t in provider.requests[0][1]]
    assert asked == ['fake: echo {"text": "one"}']  # second call allowed by the rule
    assert [m.content for m in agent.messages if m.role == "tool"] == ["echo: one", "echo: two"]
    assert "mcp__<server>__<tool>" in provider.requests[0][0][0].content


class _McpHttp(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        msg = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        method, rid = msg.get("method"), msg.get("id")
        if rid is None:
            self.send_response(202)
            self.end_headers()
            return
        if method == "initialize":
            body, sse = {"protocolVersion": "2025-06-18", "capabilities": {}, "serverInfo": {"name": "h", "version": "1"}}, False
        elif method == "tools/list":
            assert self.headers.get("Mcp-Session-Id") == "sess-1" and self.headers.get("Authorization") == "Bearer s3cret"
            body, sse = {"tools": [{"name": "time", "description": "now", "inputSchema": {"type": "object"}}]}, False
        else:
            body, sse = {"content": [{"type": "text", "text": "12:00"}]}, True
        data = json.dumps({"jsonrpc": "2.0", "id": rid, "result": body})
        self.send_response(200)
        self.send_header("Mcp-Session-Id", "sess-1")
        if sse:
            self.send_header("Content-Type", "text/event-stream")
            payload = f"event: message\ndata: {data}\n\n".encode()
        else:
            self.send_header("Content-Type", "application/json")
            payload = data.encode()
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def test_http_server_with_session_header_and_event_stream(project, ctx, monkeypatch):
    monkeypatch.setenv("TEST_TOKEN", "s3cret")
    server = ThreadingHTTPServer(("127.0.0.1", 0), _McpHttp)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        cfg = ServerConfig("clock", url=f"http://127.0.0.1:{server.server_port}/mcp", headers={"Authorization": "Bearer ${TEST_TOKEN}"})
        manager = McpManager([cfg], project)
        manager.start()
        manager.wait(20)
        assert manager.status()[0]["status"] == "ready", manager.status()
        assert run_tool(manager.tools()[0], ctx).content == "12:00"
        manager.stop()
    finally:
        server.shutdown()


def test_project_mcp_servers_need_trust(project):
    (project / ".lca").mkdir(exist_ok=True)
    (project / ".lca" / "config.toml").write_text('[mcp.servers.x]\ncommand = "evil"\n')
    warnings: list[str] = []
    assert load_mcp_configs(project, {}, warnings) == [] and "not trusted" in warnings[0]
    trust_section(project, "mcp")
    assert [c.name for c in load_mcp_configs(project, {}, [])] == ["x"]
    user = {"servers": {"y": {"url": "https://example.com/mcp"}, "off": {"command": "x", "enabled": False}, "bad name!": {"command": "x"}}}
    warnings = []
    assert sorted(c.name for c in load_mcp_configs(project, user, warnings)) == ["x", "y"] and len(warnings) == 1


# ---------------------------------------------------------------- web


def test_html_to_text_keeps_structure_and_drops_noise():
    title, text = html_to_text(
        "<html><head><title>Docs</title><script>evil()</script></head><body><nav>menu</nav><h2>Install</h2>"
        "<p>Run <code>pip install x</code>.</p><pre>a = 1\n  b = 2</pre><ul><li>one</li><li>two</li></ul>"
        '<a href="/api">API</a></body></html>', "https://x.dev/start")
    assert title == "Docs" and "evil" not in text and "menu" not in text
    assert "## Install" in text and "- one" in text and "a = 1\n  b = 2" in text and "(https://x.dev/api)" in text


@pytest.mark.parametrize("url", ["http://127.0.0.1:8080/", "http://localhost/", "http://10.0.0.5/", "file:///etc/passwd", "http://169.254.169.254/latest/"])
def test_local_and_private_addresses_are_refused(url):
    with pytest.raises(Exception, match="local network|this machine|http"):
        check_public_url(url)


class _Page(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        body = ("<html><title>Big</title><body>" + "<p>filler text</p>" * 2000 + "<p>The answer is 42.</p></body></html>").encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_web_fetch_condenses_long_pages_with_the_prompt(project):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Page)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    seen = []
    ctx = ToolContext(workspace=Workspace(project), cancel=CancelToken(), policy=None,
                      extract=lambda text, prompt: seen.append((len(text), prompt)) or "The answer is 42.")
    try:
        url = f"http://127.0.0.1:{server.server_port}/page"
        assert "local network" in run_tool(WebFetch(), ctx, url=url).content  # refused by default
        out = run_tool(WebFetch(allow_private=True), ctx, url=url, prompt="what is the answer").content
        assert out.startswith("Big\n") and "[Condensed for: what is the answer]" in out and out.endswith("The answer is 42.")
        assert seen[0][1] == "what is the answer" and seen[0][0] > web.EXTRACT_ABOVE
    finally:
        server.shutdown()


DDG = """<div class="result results_links results_links_deep web-result "><div class="links_main links_deep result__body">
<h2 class="result__title"><a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fdocs.python.org%2F3%2F&amp;rut=x">Python <b>docs</b></a></h2>
<a class="result__snippet" href="x">The official <b>Python</b> documentation.</a></div></div>"""


def test_web_search_parses_duckduckgo(project, ctx, monkeypatch):
    monkeypatch.setattr(web, "_client", lambda check: httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(200, text=DDG))))
    out = run_tool(WebSearch(), ctx, query="python docs").content
    assert out == "1. Python docs\n   https://docs.python.org/3/\n   The official Python documentation."


def test_web_tools_follow_settings_and_mode(project):
    assert "web_fetch" not in default_registry().names
    assert "web_fetch" in default_registry(web=WebSettings()).names and "web_fetch" not in default_registry(web=WebSettings(enabled=False)).names
    provider = ScriptedProvider([reply("a"), reply("b")])
    s = Session(Settings(), provider, Workspace(project), ServerInfo(reachable=True, models=["m"], context_window=8192), 8192, [], RuleSet(), PermissionStore(project))
    agent = _make_agent(s, lambda e: None, s.policy(Mode.PLAN))
    agent.run("x")
    agent.policy.mode = Mode.READ_ONLY
    agent.run("y")
    assert "web_fetch" in [t.name for t in provider.requests[0][1]] and "web_fetch" not in [t.name for t in provider.requests[1][1]]
