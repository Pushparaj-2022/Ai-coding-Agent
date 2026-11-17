"""Background commands, parallel sub-agents, the status line, images, and OAuth sign-in for MCP servers."""

from __future__ import annotations

import base64
import hashlib
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlencode, urlparse

import httpx
import pytest

import lca.graph.agent_graph as agent_graph
from lca.agent.events import ToolCallStarted
from lca.agent.loop import AgentLoop
from lca.agent.mentions import expand_mentions
from lca.agent.permissions import Mode, PermissionPolicy
from lca.agent.statusline import StatusLine
from lca.cancel import CancelToken
from lca.config import AgentSettings
from lca.graph.agent_graph import VIEW_IMAGE_PREFIX, transcript
from lca.images import valid_data_urls
from lca.mcp import McpManager, ServerConfig
from lca.mcp_auth import load_tokens
from lca.messages import Message, ModelResponse
from lca.providers import ollama, openai_compatible
from lca.providers.base import ProviderHTTPError, classify_http_error
from lca.tools import Workspace, default_registry
from lca.tools.base import ToolContext
from lca.tools.terminal import ProcessManager, RunCommand, detect_shell

from .conftest import ScriptedProvider, call, reply

PY = sys.executable.replace("\\", "/")
PNG = "data:image/png;base64," + base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\0" * 32).decode()


@pytest.fixture(autouse=True)
def isolated(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("LCA_HOME", str(tmp_path_factory.mktemp("lcahome")))
    monkeypatch.setattr(agent_graph, "verify_commands", lambda root: [])


# ---------------------------------------------------------------- background commands


def test_running_command_moves_to_the_background(project):
    (project / "slow.py").write_text("import time\nfor i in range(100):\n    print('tick', i, flush=True)\n    time.sleep(0.05)\n")
    manager = ProcessManager()
    tool = RunCommand(detect_shell(), manager)
    event = threading.Event()
    event.set()  # pressed before the command started: must not apply to it
    ctx = ToolContext(workspace=Workspace(project), cancel=CancelToken(), policy=None, background=event)
    out = tool.run(tool.Args(command=f'"{PY}" -c "print(1)"'), ctx)
    assert out.exit_code == 0 and not event.is_set()

    box = {}
    worker = threading.Thread(target=lambda: box.update(out=tool.run(tool.Args(command=f'"{PY}" -u slow.py', timeout_s=60), ctx)))
    started = time.monotonic()
    worker.start()
    time.sleep(1.0)
    event.set()
    worker.join(10)
    assert time.monotonic() - started < 4  # returned long before the command's 5 seconds
    out = box["out"]
    assert "moved this command to the background as bg" in out.content and "tick" in out.content and out.exit_code is None
    bp = next(iter(manager.processes.values()))
    assert bp.status == "running"
    time.sleep(0.5)
    assert "tick" in bp.buffer.take_unread()  # output keeps arriving in the background buffer
    manager.stop_all()


# ---------------------------------------------------------------- parallel sub-agents


def test_task_calls_in_one_reply_run_in_parallel(project):
    windows: list[tuple[float, float]] = []

    def slow_answer(messages):
        start = time.perf_counter()  # monotonic() ticks only every ~15ms on Windows
        time.sleep(0.6)
        windows.append((start, time.perf_counter()))
        return reply("found it")

    events: list = []

    def spawner(kind):
        return AgentLoop(ScriptedProvider([slow_answer]), default_registry(), Workspace(project), system_prompt="sub",
                         settings=AgentSettings(auto_verify=False), policy=PermissionPolicy(Mode.READ_ONLY), label="explore", events=events.append)

    parent = AgentLoop(
        ScriptedProvider([
            reply("", call("task", description="Look at A", prompt="Find where A is defined."),
                  call("task", description="Look at B", prompt="Find where B is defined."),
                  call("task", description="Look at C", prompt="Find where C is defined.")),
            reply("A, B and C found."),
        ]),
        default_registry(interaction=True), Workspace(project), system_prompt="s", settings=AgentSettings(auto_verify=False, max_parallel_agents=2),
        policy=PermissionPolicy(Mode.ASK), spawner=spawner, events=events.append,
    )
    started = time.monotonic()
    result = parent.run("find A, B and C")
    assert result.status == "completed"
    assert [m.content.count("found it") for m in parent.messages if m.role == "tool"] == [1, 1, 1]
    assert time.monotonic() - started < 1.7  # 3 x 0.6s, two at a time: ~1.2s, not ~1.8s
    overlaps = sum(1 for i, (a1, b1) in enumerate(windows) for (a2, b2) in windows[i + 1:] if a1 < b2 and a2 < b1)
    assert overlaps >= 1
    labels = {e.agent for e in events if e.type == "turn_started" and e.agent}
    assert labels == {"explore 1", "explore 2", "explore 3"}


# ---------------------------------------------------------------- status line


def test_status_line_command_gets_the_session_as_json(project):
    (project / "sl.py").write_text(
        "import json, sys\nd = json.load(sys.stdin)\n"
        "print('\\x1b[32m' + d['model'] + '\\x1b[0m | ' + d['mode'] + ' | ctx ' + str(d['context']['percent']) + '%')\nprint('line 2')\nprint('line 3')\n"
    )
    line = StatusLine(f'"{PY}" sl.py', project, detect_shell())
    assert line.render({"model": "qwen", "mode": "ask", "context": {"percent": 42}}) == "qwen | ask | ctx 42%\nline 2"
    assert StatusLine(f'"{PY}" -c "import sys; sys.exit(3)"', project, detect_shell()).render({}) == ""


# ---------------------------------------------------------------- images


def test_images_on_the_wire():
    m = Message(role="user", content="what is wrong here?", images=[PNG])
    wire = openai_compatible._to_wire(m)
    assert wire["content"][0] == {"type": "text", "text": "what is wrong here?"}
    assert wire["content"][1] == {"type": "image_url", "image_url": {"url": PNG}}
    assert ollama._to_wire(m)["images"] == [PNG.split(",", 1)[1]]
    assert openai_compatible._to_wire(Message.user("plain")) == {"role": "user", "content": "plain"}
    err = classify_http_error(400, "image input is not supported (no mmproj)", False, True)
    assert isinstance(err, ProviderHTTPError) and "--mmproj" in str(err)


def test_pasted_images_and_view_image_reach_the_model(project):
    (project / "shot.png").write_bytes(base64.b64decode(PNG.split(",")[1]))
    provider = ScriptedProvider([
        reply("", call("view_image", path="shot.png")),
        reply("The button is misaligned."),
    ])
    agent = AgentLoop(provider, default_registry(interaction=True), Workspace(project), system_prompt="s",
                      settings=AgentSettings(auto_verify=False), policy=PermissionPolicy(Mode.ASK))
    agent.run("fix the layout", images=[PNG])
    first = provider.requests[0][0][-1]
    assert first.role == "user" and first.images == [PNG]
    second = provider.requests[1][0]
    assert [m.role for m in second[-3:]] == ["assistant", "tool", "user"]  # no "(interrupted)" between them
    assert second[-1].content.startswith(VIEW_IMAGE_PREFIX) and second[-1].images == [PNG]
    items = transcript(agent.history())
    assert items[0]["images"] == 1 and any(i["role"] == "notice" and "shot.png" in i["text"] for i in items)


def test_only_the_newest_images_are_sent(project):
    provider = ScriptedProvider([reply("a"), reply("b"), reply("c")])
    agent = AgentLoop(provider, default_registry(), Workspace(project), system_prompt="s", settings=AgentSettings(auto_verify=False))
    for text in ("one", "two", "three"):
        agent.run(text, images=[PNG])
    last = [m for m in provider.requests[2][0] if m.role == "user"]
    assert [len(m.images) for m in last] == [0, 1, 1]
    assert "image(s) omitted" in last[0].content


def test_image_mentions_and_pasted_image_validation(project):
    (project / "ui.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    text, found, images = expand_mentions(Workspace(project), "compare @ui.png with @src/util.py")
    assert found == ["ui.png", "src/util.py"] and len(images) == 1 and images[0].startswith("data:image/png;base64,")
    assert "[File @src/util.py:]" in text and "ui.png:]" not in text
    _, _, none = expand_mentions(Workspace(project), "@ui.png", images=False)
    assert none == []
    assert valid_data_urls([PNG, "data:text/html;base64,PGh0bWw+", "javascript:alert(1)", 5]) == [PNG]


# ---------------------------------------------------------------- OAuth for MCP


class _OAuthMcp(BaseHTTPRequestHandler):
    """Authorization server + MCP server in one: dynamic registration, PKCE, refresh tokens."""

    valid = {"tok-1"}
    codes: dict[str, str] = {}
    refreshed = 0

    def log_message(self, *a):
        pass

    def _json(self, status, body, headers=()):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        for k, v in headers:
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    @property
    def base(self):
        return f"http://127.0.0.1:{self.server.server_port}"

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/.well-known/oauth-protected-resource/mcp":
            return self._json(200, {"resource": self.base + "/mcp", "authorization_servers": [self.base]})
        if u.path == "/.well-known/oauth-authorization-server":
            return self._json(200, {"issuer": self.base, "authorization_endpoint": self.base + "/authorize",
                                    "token_endpoint": self.base + "/token", "registration_endpoint": self.base + "/register"})
        if u.path == "/authorize":  # the user "approves" at once
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            assert q["code_challenge_method"] == "S256" and q["resource"] == self.base + "/mcp" and q["client_id"] == "client-1"
            type(self).codes["code-1"] = q["code_challenge"]
            self.send_response(302)
            self.send_header("Location", q["redirect_uri"] + "?" + urlencode({"code": "code-1", "state": q["state"]}))
            self.end_headers()
            return
        self._json(404, {})

    def do_POST(self):
        u = urlparse(self.path)
        raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if u.path == "/register":
            return self._json(201, {"client_id": "client-1", **json.loads(raw)})
        if u.path == "/token":
            form = {k: v[0] for k, v in parse_qs(raw.decode()).items()}
            if form["grant_type"] == "authorization_code":
                challenge = base64.urlsafe_b64encode(hashlib.sha256(form["code_verifier"].encode()).digest()).rstrip(b"=").decode()
                if type(self).codes.get(form["code"]) != challenge:
                    return self._json(400, {"error": "invalid_grant"})
                return self._json(200, {"access_token": "tok-1", "refresh_token": "ref-1", "expires_in": 3600, "token_type": "Bearer"})
            if form["grant_type"] == "refresh_token" and form["refresh_token"] == "ref-1":
                type(self).refreshed += 1
                type(self).valid = {"tok-2"}
                return self._json(200, {"access_token": "tok-2", "expires_in": 3600, "token_type": "Bearer"})
            return self._json(400, {"error": "invalid_grant"})
        # the MCP endpoint
        auth = self.headers.get("Authorization", "")
        if auth.removeprefix("Bearer ") not in type(self).valid:
            return self._json(401, {"error": "unauthorized"}, [("WWW-Authenticate", f'Bearer resource_metadata="{self.base}/.well-known/oauth-protected-resource/mcp"')])
        msg = json.loads(raw)
        if "id" not in msg:
            self.send_response(202)
            self.end_headers()
            return
        result = {"initialize": {"protocolVersion": "2025-06-18", "capabilities": {}, "serverInfo": {"name": "o", "version": "1"}},
                  "tools/list": {"tools": [{"name": "whoami", "inputSchema": {"type": "object"}}]}}.get(msg["method"], {"content": [{"type": "text", "text": "you"}]})
        self._json(200, {"jsonrpc": "2.0", "id": msg["id"], "result": result})


def test_mcp_oauth_sign_in_and_refresh(project):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _OAuthMcp)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}/mcp"
    try:
        manager = McpManager([ServerConfig("secure", url=url)], project)
        manager.start()
        manager.wait(20)
        status = manager.status()[0]
        assert status["status"] == "needs_login" and "lca mcp --login secure" in status["error"]

        opened = []

        def browser(link):  # follows the authorization redirect back to the agent's listener, like a browser
            opened.append(link)
            threading.Thread(target=lambda: httpx.get(link, follow_redirects=True, timeout=10), daemon=True).start()

        status = manager.login("secure", browser)
        assert status["status"] == "ready" and status["tools"] == ["whoami"]
        assert "code_challenge=" in opened[0] and "redirect_uri=http%3A%2F%2F127.0.0.1" in opened[0]
        tokens = load_tokens(url)
        assert tokens.access_token == "tok-1" and tokens.refresh_token == "ref-1" and tokens.client_id == "client-1"

        _OAuthMcp.valid = {"tok-2"}  # the server revoked tok-1: the client refreshes once and carries on
        tool = manager.tools()[0]
        ctx = ToolContext(workspace=Workspace(project), cancel=CancelToken(), policy=None)
        assert tool.run(tool.Args(), ctx).content == "you"
        assert _OAuthMcp.refreshed == 1 and load_tokens(url).access_token == "tok-2"
        manager.stop()
    finally:
        server.shutdown()


# ---------------------------------------------------------------- vision detection and image shrinking


def test_vision_support_is_read_from_the_server(monkeypatch):
    from lca.config import ProviderSettings

    def fake(props):
        def handler(req):
            if req.url.path.endswith("/models"):
                return httpx.Response(200, json={"data": [{"id": "m"}]})
            return httpx.Response(200, json=props)
        p = openai_compatible.OpenAICompatibleProvider(ProviderSettings())
        p._client = httpx.Client(transport=httpx.MockTransport(handler))
        return p.server_info()

    assert fake({"modalities": {"vision": False}, "n_ctx": 4096}).vision is False
    assert fake({"modalities": {"vision": True}}).vision is True
    assert fake({}).vision is None  # not reported: the setting decides (default: allow)


def test_large_images_are_shrunk(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    import io

    from lca.images import MAX_SIDE, image_data_url

    big = tmp_path / "big.png"
    Image.new("RGB", (4000, 3000), (200, 30, 30)).save(big)
    url = image_data_url(big)
    data = base64.b64decode(url.split(",", 1)[1])
    with Image.open(io.BytesIO(data)) as im:
        assert max(im.size) == MAX_SIDE and im.size[0] / im.size[1] == pytest.approx(4 / 3, rel=0.01)
    small = tmp_path / "small.png"
    Image.new("RGB", (200, 100)).save(small)
    assert base64.b64decode(image_data_url(small).split(",", 1)[1]) == small.read_bytes()  # untouched
