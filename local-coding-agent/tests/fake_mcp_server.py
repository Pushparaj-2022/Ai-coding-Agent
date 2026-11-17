"""A minimal MCP server over stdio for tests: tools echo, fail and roots (which asks the client for its roots)."""

import json
import sys

TOOLS = [
    {"name": "echo", "description": "Echo the text back", "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}},
    {"name": "fail", "description": "Always fails", "inputSchema": {"type": "object", "properties": {}}},
    {"name": "roots", "description": "Report the client's roots", "inputSchema": {"type": "object", "properties": {}}},
]


def send(msg):
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def read():
    line = sys.stdin.readline()
    return json.loads(line) if line else None


pending_roots = None
while True:
    msg = read()
    if msg is None:
        break
    method, rid = msg.get("method"), msg.get("id")
    if method is None and rid == 9000 and pending_roots is not None:  # the client's answer to roots/list
        roots = [r["uri"] for r in msg["result"]["roots"]]
        send({"jsonrpc": "2.0", "id": pending_roots, "result": {"content": [{"type": "text", "text": "roots: " + ", ".join(roots)}]}})
        pending_roots = None
    elif method == "initialize":
        send({"jsonrpc": "2.0", "id": rid, "result": {"protocolVersion": msg["params"]["protocolVersion"], "capabilities": {"tools": {}}, "serverInfo": {"name": "fake", "version": "1"}}})
    elif method == "tools/list":
        send({"jsonrpc": "2.0", "id": rid, "result": {"tools": TOOLS}})
    elif method == "tools/call":
        name, args = msg["params"]["name"], msg["params"].get("arguments", {})
        if name == "echo":
            send({"jsonrpc": "2.0", "id": rid, "result": {"content": [{"type": "text", "text": "echo: " + args.get("text", "")}]}})
        elif name == "fail":
            send({"jsonrpc": "2.0", "id": rid, "result": {"content": [{"type": "text", "text": "it broke"}], "isError": True}})
        elif name == "roots":
            pending_roots = rid
            send({"jsonrpc": "2.0", "id": 9000, "method": "roots/list", "params": {}})
    elif rid is not None:
        send({"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": "unknown method"}})
