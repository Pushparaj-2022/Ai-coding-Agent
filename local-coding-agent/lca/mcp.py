"""MCP (Model Context Protocol) client: tools from MCP servers become agent tools named mcp__<server>__<tool>.

    [mcp.servers.github]                       # ~/.lca/config.toml (a project's own servers need `lca mcp --trust`)
    command = "npx"
    args = ["-y", "@modelcontextprotocol/server-github"]
    env = { GITHUB_PERSONAL_ACCESS_TOKEN = "${GITHUB_TOKEN}" }   # ${VAR} is read from your environment
    tools = ["get_issue", "list_issues"]       # optional: only these tools (small models do better with fewer)

    [mcp.servers.docs]
    url = "https://example.com/mcp"            # streamable HTTP
    headers = { Authorization = "Bearer ${DOCS_TOKEN}" }

Servers start in the background when the agent starts; their tools appear once they are ready. MCP tools can do
anything the server allows, so they need approval like commands (risk "execute"), and permission rules can match
them by name, e.g. allow = ["mcp__github__get_issue"].
"""

from __future__ import annotations

import json
import logging
import os
import queue
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import httpx
from pydantic import BaseModel, ConfigDict

from lca import __version__
from lca.cancel import Cancelled, CancelToken
from lca.providers.base import ToolSpec
from lca.tools.base import Tool, ToolContext, ToolError

log = logging.getLogger("lca.mcp")

PROTOCOL_VERSION = "2025-06-18"
MAX_RESULT_CHARS = 30_000


class McpError(Exception):
    pass


class NeedsLogin(McpError):
    """An HTTP server wants an OAuth sign-in (see lca.mcp_auth)."""


def _expand_env(value: str) -> str:
    return re.sub(r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), ""), value)


@dataclass
class ServerConfig:
    name: str
    command: str = ""
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    url: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    tools: list[str] | None = None  # allowlist
    timeout: float = 120
    source: str = "user"
    oauth: dict[str, Any] = field(default_factory=dict)  # optional client_id / client_secret / scopes

    @classmethod
    def parse(cls, name: str, data: dict[str, Any], source: str) -> ServerConfig:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError(f"MCP server name {name!r} may only use letters, digits, - and _.")
        if not data.get("command") and not data.get("url"):
            raise ValueError(f"MCP server {name!r} needs a command (stdio) or a url (HTTP).")
        return cls(
            name=name, command=str(data.get("command", "")), args=[str(a) for a in data.get("args", []) or []],
            env={str(k): str(v) for k, v in (data.get("env") or {}).items()}, url=str(data.get("url", "")),
            headers={str(k): str(v) for k, v in (data.get("headers") or {}).items()},
            tools=[str(t) for t in data["tools"]] if isinstance(data.get("tools"), list) else None,
            timeout=float(data.get("timeout", 120) or 120), source=source,
            oauth=dict(data.get("oauth") or {}),
        )


# ---------------------------------------------------------------- transports (JSON-RPC 2.0)


class _Stdio:
    def __init__(self, cfg: ServerConfig, cwd: Path) -> None:
        exe = shutil.which(cfg.command) or cfg.command
        env = {**os.environ, **{k: _expand_env(v) for k, v in cfg.env.items()}}
        self.proc = subprocess.Popen(
            [exe, *cfg.args], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd, env=env,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        self.incoming: queue.Queue[dict] = queue.Queue()
        self.stderr_tail: list[str] = []
        threading.Thread(target=self._read, daemon=True).start()
        threading.Thread(target=self._drain, daemon=True).start()
        self._lock = threading.Lock()

    def _read(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                self.incoming.put(json.loads(line))
            except ValueError:
                log.debug("mcp non-JSON line: %s", line[:200])
        self.incoming.put({"_closed": True})

    def _drain(self) -> None:
        assert self.proc.stderr is not None
        for line in self.proc.stderr:
            self.stderr_tail = (self.stderr_tail + [line.rstrip()])[-20:]

    def send(self, msg: dict) -> None:
        if self.proc.poll() is not None:
            raise McpError(f"the server exited (code {self.proc.returncode}). {' '.join(self.stderr_tail[-3:])}".strip())
        with self._lock:
            assert self.proc.stdin is not None
            self.proc.stdin.write(json.dumps(msg) + "\n")
            self.proc.stdin.flush()

    def close(self) -> None:
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
            self.proc.wait(3)
        except (OSError, subprocess.TimeoutExpired):
            self.proc.kill()


class _Http:
    """Streamable HTTP: every message is a POST; the reply is JSON or a server-sent event stream."""

    def __init__(self, cfg: ServerConfig) -> None:
        self.cfg = cfg
        self.url = cfg.url
        self.headers = {k: _expand_env(v) for k, v in cfg.headers.items()}
        self.client = httpx.Client(timeout=httpx.Timeout(cfg.timeout, connect=15))
        self.incoming: queue.Queue[dict] = queue.Queue()
        self.session_id: str | None = None
        self.stderr_tail: list[str] = []

    def _post(self, msg: dict, token: str | None) -> httpx.Response:
        headers = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json", **self.headers}
        if token and not any(k.lower() == "authorization" for k in self.headers):
            headers["Authorization"] = f"Bearer {token}"
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        try:
            return self.client.post(self.url, json=msg, headers=headers)
        except httpx.HTTPError as e:
            raise McpError(f"HTTP request failed: {e}") from e

    def send(self, msg: dict) -> None:
        from lca.mcp_auth import access_token

        r = self._post(msg, access_token(self.url))
        if r.status_code == 401:  # expired or revoked: refresh once, then ask for a sign-in
            fresh = access_token(self.url, force_refresh=True)
            if fresh:
                r = self._post(msg, fresh)
            if r.status_code == 401:
                raise NeedsLogin(f"{self.cfg.name} needs you to sign in: run `lca mcp --login {self.cfg.name}`, or use Sign in under /mcp.")
        if r.headers.get("mcp-session-id"):
            self.session_id = r.headers["mcp-session-id"]
        if r.status_code == 202 or not r.content:
            return
        if r.status_code >= 400:
            raise McpError(f"HTTP {r.status_code}: {r.text[:300]}")
        if "text/event-stream" in r.headers.get("content-type", ""):
            for block in r.text.split("\n\n"):
                data = "\n".join(ln[5:].lstrip() for ln in block.splitlines() if ln.startswith("data:"))
                if data:
                    try:
                        self.incoming.put(json.loads(data))
                    except ValueError:
                        pass
        else:
            body = r.json()
            for item in body if isinstance(body, list) else [body]:
                self.incoming.put(item)

    def close(self) -> None:
        self.client.close()


# ---------------------------------------------------------------- a server connection


class McpServer:
    def __init__(self, cfg: ServerConfig, workspace: Path) -> None:
        self.cfg = cfg
        self.workspace = workspace
        self.status = "starting"  # starting | ready | failed | stopped
        self.error = ""
        self.tools: list[dict] = []
        self._transport: _Stdio | _Http | None = None
        self._next_id = 0
        self._pending: dict[int, queue.Queue] = {}
        self._lock = threading.Lock()

    # JSON-RPC plumbing: one dispatcher thread routes replies to waiting callers and answers server requests
    def _dispatch(self) -> None:
        t = self._transport
        assert t is not None
        while True:
            msg = t.incoming.get()
            if msg.get("_closed"):
                for q in list(self._pending.values()):
                    q.put({"error": {"message": "the server closed the connection"}})
                if self.status == "ready":
                    self.status, self.error = "failed", "the server exited"
                return
            if "id" in msg and ("result" in msg or "error" in msg):
                q = self._pending.pop(msg["id"], None)
                if q:
                    q.put(msg)
            elif "method" in msg and "id" in msg:  # a request from the server
                if msg["method"] == "roots/list":
                    self._reply(msg["id"], {"roots": [{"uri": self.workspace.resolve().as_uri(), "name": self.workspace.name}]})
                elif msg["method"] == "ping":
                    self._reply(msg["id"], {})
                else:
                    self._send({"jsonrpc": "2.0", "id": msg["id"], "error": {"code": -32601, "message": "Method not supported by this client"}})

    def _send(self, msg: dict) -> None:
        assert self._transport is not None
        self._transport.send(msg)

    def _reply(self, rid: Any, result: dict) -> None:
        try:
            self._send({"jsonrpc": "2.0", "id": rid, "result": result})
        except McpError:
            pass

    def request(self, method: str, params: dict | None = None, timeout: float | None = None, cancel: CancelToken | None = None) -> dict:
        with self._lock:
            self._next_id += 1
            rid = self._next_id
        q: queue.Queue = queue.Queue()
        self._pending[rid] = q
        self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}})
        deadline = time.monotonic() + (timeout or self.cfg.timeout)
        while True:
            try:
                msg = q.get(timeout=0.2)
                break
            except queue.Empty:
                if cancel is not None and cancel.cancelled:
                    self._pending.pop(rid, None)
                    self._notify("notifications/cancelled", {"requestId": rid, "reason": "cancelled by the user"})
                    raise Cancelled() from None
                if time.monotonic() > deadline:
                    self._pending.pop(rid, None)
                    raise McpError(f"{method} timed out after {int(timeout or self.cfg.timeout)}s")
        if "error" in msg:
            raise McpError(str(msg["error"].get("message", msg["error"])))
        return msg.get("result") or {}

    def _notify(self, method: str, params: dict | None = None) -> None:
        try:
            self._send({"jsonrpc": "2.0", "method": method, **({"params": params} if params else {})})
        except McpError:
            pass

    def start(self) -> None:
        try:
            self._transport = _Http(self.cfg) if self.cfg.url else _Stdio(self.cfg, self.workspace)
            threading.Thread(target=self._dispatch, daemon=True).start()
            self.request("initialize", {
                "protocolVersion": PROTOCOL_VERSION, "capabilities": {"roots": {"listChanged": False}},
                "clientInfo": {"name": "local-coding-agent", "version": __version__},
            }, timeout=60)
            self._notify("notifications/initialized")
            tools: list[dict] = []
            cursor = None
            while True:
                page = self.request("tools/list", {"cursor": cursor} if cursor else {}, timeout=60)
                tools += page.get("tools", [])
                cursor = page.get("nextCursor")
                if not cursor:
                    break
            if self.cfg.tools is not None:
                tools = [t for t in tools if t.get("name") in self.cfg.tools]
            self.tools = tools
            self.status = "ready"
            log.info("mcp server ready", extra={"fields": {"server": self.cfg.name, "tools": len(tools)}})
        except NeedsLogin as e:
            self.status, self.error = "needs_login", str(e)
            log.info("mcp server needs login", extra={"fields": {"server": self.cfg.name}})
        except (McpError, OSError, Cancelled) as e:
            self.status = "failed"
            tail = " ".join(getattr(self._transport, "stderr_tail", [])[-3:])
            self.error = f"{e}{' | ' + tail if tail else ''}"[:500]
            log.warning("mcp server failed", extra={"fields": {"server": self.cfg.name, "error": self.error}})

    def call(self, tool: str, arguments: dict, cancel: CancelToken | None) -> tuple[str, bool]:
        if self.status != "ready":
            raise McpError(f"the {self.cfg.name} server is {self.status}{': ' + self.error if self.error else ''}")
        result = self.request("tools/call", {"name": tool, "arguments": arguments}, cancel=cancel)
        parts = []
        for item in result.get("content", []) or []:
            kind = item.get("type")
            if kind == "text":
                parts.append(str(item.get("text", "")))
            elif kind == "resource":
                res = item.get("resource") or {}
                parts.append(str(res.get("text") or f"[resource {res.get('uri', '')}]"))
            else:
                parts.append(f"[{kind} content not shown]")
        if result.get("structuredContent") is not None and not parts:
            parts.append(json.dumps(result["structuredContent"], indent=2)[:MAX_RESULT_CHARS])
        return "\n".join(parts)[:MAX_RESULT_CHARS] or "(no output)", bool(result.get("isError"))

    def stop(self) -> None:
        if self._transport is not None:
            self._transport.close()
        self.status = "stopped"


# ---------------------------------------------------------------- tools and the manager


class _AnyArgs(BaseModel):
    model_config = ConfigDict(extra="allow")


class McpTool(Tool):
    """One tool of an MCP server. The server validates the arguments against its own schema."""

    risk: ClassVar = "execute"
    Args = _AnyArgs

    def __init__(self, server: McpServer, spec: dict) -> None:
        self.server = server
        self.tool = str(spec.get("name"))
        self.name = f"mcp__{server.cfg.name}__{self.tool}"  # type: ignore[misc]
        self.description = f"[MCP {server.cfg.name}] {str(spec.get('description') or '').strip()}"[:1500]  # type: ignore[misc]
        self.schema = spec.get("inputSchema") or {"type": "object", "properties": {}}

    def spec(self) -> ToolSpec:
        return ToolSpec(name=self.name, description=self.description, parameters=self.schema)

    def describe_call(self, args: BaseModel) -> str:
        text = json.dumps(args.model_dump(), ensure_ascii=False)
        return f"{self.server.cfg.name}: {self.tool} {text if len(text) < 300 else text[:297] + '...'}"

    def run(self, args: _AnyArgs, ctx: ToolContext) -> Any:
        from lca.tools.base import ToolOutput

        try:
            text, is_error = self.server.call(self.tool, args.model_dump(), ctx.cancel)
        except McpError as e:
            raise ToolError(f"MCP {self.server.cfg.name}: {e}") from e
        return ToolOutput(content=text, is_error=is_error)


class McpManager:
    def __init__(self, configs: list[ServerConfig], workspace: Path) -> None:
        self.servers = [McpServer(c, workspace) for c in configs]
        self._tools: dict[str, McpTool] = {}

    def start(self) -> None:
        """Start every server in the background; tools become available as servers get ready."""
        for s in self.servers:
            threading.Thread(target=s.start, daemon=True, name=f"mcp-{s.cfg.name}").start()

    def wait(self, timeout: float = 30) -> None:
        deadline = time.monotonic() + timeout
        while any(s.status == "starting" for s in self.servers) and time.monotonic() < deadline:
            time.sleep(0.1)

    def tools(self) -> list[McpTool]:
        out = []
        for s in self.servers:
            if s.status != "ready":
                continue
            for spec in s.tools:
                name = f"mcp__{s.cfg.name}__{spec.get('name')}"
                if name not in self._tools:
                    self._tools[name] = McpTool(s, spec)
                out.append(self._tools[name])
        return out

    def status(self) -> list[dict]:
        return [
            {"name": s.cfg.name, "status": s.status, "error": s.error, "tools": [t.get("name") for t in s.tools],
             "transport": "http" if s.cfg.url else "stdio", "source": s.cfg.source}
            for s in self.servers
        ]

    def stop(self) -> None:
        for s in self.servers:
            s.stop()

    def login(self, name: str, open_url: Any, cancelled: Any = lambda: False) -> dict:
        """OAuth sign-in for one HTTP server, then reconnect it. Returns its new status."""
        from lca.mcp_auth import AuthError, login

        idx = next((i for i, srv in enumerate(self.servers) if srv.cfg.name == name), None)
        if idx is None:
            raise McpError(f"no MCP server named {name!r}.")
        old = self.servers[idx]
        if not old.cfg.url:
            raise McpError(f"{name} is a local (stdio) server; sign-in is only for HTTP servers.")
        try:
            login(name, old.cfg.url, open_url, old.cfg.oauth, cancelled=cancelled)
        except AuthError as e:
            raise McpError(str(e)) from e
        old.stop()
        fresh = McpServer(old.cfg, old.workspace)
        self.servers[idx] = fresh
        self._tools = {k: t for k, t in self._tools.items() if t.server is not old}
        fresh.start()
        return next(st for st in self.status() if st["name"] == name)


def load_mcp_configs(root: Path, user_data: Any, warnings: list[str]) -> list[ServerConfig]:
    from lca.agent.trust import is_trusted, read_project_section

    configs: dict[str, ServerConfig] = {}

    def add(data: Any, source: str) -> None:
        servers = (data or {}).get("servers") if isinstance(data, dict) else None
        for name, entry in (servers or {}).items():
            if not isinstance(entry, dict) or entry.get("enabled", True) is False:
                continue
            try:
                configs[name] = ServerConfig.parse(name, entry, source)
            except ValueError as e:
                warnings.append(str(e))

    add(user_data, "user")
    project = read_project_section(root, "mcp")
    if project:
        if is_trusted(root, "mcp", project):
            add(project, "project")
        else:
            warnings.append("This project's .lca/config.toml defines MCP servers that are not trusted, so they do not start. "
                            "Review them, then run `lca mcp --trust` to enable them.")
    return list(configs.values())
