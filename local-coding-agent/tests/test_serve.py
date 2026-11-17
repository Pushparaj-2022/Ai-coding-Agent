"""The `lca serve` JSON-lines protocol, driven like the VS Code extension drives it (real subprocess)."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading

import pytest

from .test_cli import FakeModelServer


class ServeProcess:
    def __init__(self, root, url, home, *extra):
        env = {**os.environ, "LCA_HOME": str(home), "PYTHONUTF8": "1"}
        for var in ("LCA_BASE_URL", "LCA_MODEL", "LCA_PROVIDER", "LCA_ALLOW_REMOTE"):
            env.pop(var, None)
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "lca", "serve", "--root", str(root), "--base-url", url, *extra],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", env=env,
        )
        self.events: queue.Queue = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.proc.stdout:
            self.events.put(json.loads(line))

    def send(self, obj):
        self.proc.stdin.write(json.dumps(obj) + "\n")
        self.proc.stdin.flush()

    def wait_for(self, type_, timeout=20):
        seen = []
        while True:
            ev = self.events.get(timeout=timeout)
            seen.append(ev)
            if ev["type"] == type_:
                return ev, seen

    def wait_for_any(self, types, timeout=20):
        while True:
            ev = self.events.get(timeout=timeout)
            if ev["type"] in types:
                return ev, None

    def close(self):
        try:
            self.send({"type": "shutdown"})
            self.proc.wait(10)
        except Exception:
            self.proc.kill()


@pytest.fixture
def home(tmp_path_factory):
    return tmp_path_factory.mktemp("home")


def test_serve_task_with_approval_and_editor_context(project, home):
    server = FakeModelServer([
        {"tool_calls": [{"index": 0, "id": "p", "function": {"name": "update_plan", "arguments": json.dumps({"steps": ["Create hello.py"]})}}]},
        {"tool_calls": [{"index": 0, "id": "w", "function": {"name": "write_file", "arguments": json.dumps({"path": "hello.py", "content": "print('hi')\n"})}}]},
        {"content": "Created hello.py."},
    ])
    sp = ServeProcess(project, server.url, home)
    try:
        ready, _ = sp.wait_for("ready")
        assert ready["model"] == "fake-model" and ready["mode"] == "ask" and ready["local"] is True
        sp.send({"type": "task", "text": "create hello.py", "context": {"file": "src/app.py", "selection": {"start": 4, "end": 5, "text": "def add(a, b):"}}})
        req, _ = sp.wait_for("approval_requested")
        assert req["description"] == "Create hello.py (+1 -0)"
        assert req["change"]["after"] == "print('hi')\n" and req["change"]["action"] == "create"
        assert not (project / "hello.py").exists()
        sp.send({"type": "approval", "request_id": req["request_id"], "approved": True})
        done, seen = sp.wait_for("run_finished")
        assert done["result"]["status"] == "completed" and done["result"]["files_changed"] == ["hello.py"]
        assert (project / "hello.py").read_text() == "print('hi')\n"
        sp.wait_for("idle")
        first_prompt = server.requests[0]["messages"][-1]["content"]
        assert "selected lines 4-5 of src/app.py" in first_prompt
    finally:
        sp.close()
        server.close()


def test_serve_denied_approval_and_cancel(project, home):
    server = FakeModelServer([
        {"tool_calls": [{"index": 0, "id": "p", "function": {"name": "update_plan", "arguments": json.dumps({"steps": ["Delete README.md"]})}}]},
        {"tool_calls": [{"index": 0, "id": "d", "function": {"name": "delete_file", "arguments": json.dumps({"path": "README.md"})}}]},
        {"content": "OK, not deleting."},
    ])
    sp = ServeProcess(project, server.url, home)
    try:
        sp.wait_for("ready")
        sp.send({"type": "task", "text": "delete the readme"})
        req, _ = sp.wait_for("approval_requested")
        sp.send({"type": "approval", "request_id": req["request_id"], "approved": False, "feedback": "keep it"})
        finished, seen = sp.wait_for("tool_call_finished")
        assert not finished["ok"] and "keep it" in finished["output"]
        sp.wait_for("run_finished")
        assert (project / "README.md").exists()
    finally:
        sp.close()
        server.close()


def test_serve_reports_fatal_setup_error(project, home):
    sp = ServeProcess(project, "http://127.0.0.1:9/v1", home)
    try:
        ev, _ = sp.wait_for("fatal")
        assert "Cannot reach the model server" in ev["message"]
    finally:
        sp.proc.kill()


def test_serve_rejects_malformed_and_unknown_requests(project, home):
    server = FakeModelServer([])
    sp = ServeProcess(project, server.url, home)
    try:
        sp.wait_for("ready")
        sp.proc.stdin.write("not json\n")
        sp.proc.stdin.flush()
        ev, _ = sp.wait_for("notice")
        assert "malformed" in ev["message"]
        sp.send({"type": "set_mode", "mode": "auto-edit"})
        ev, _ = sp.wait_for("mode")
        assert ev["mode"] == "auto-edit"
    finally:
        sp.close()
        server.close()


def test_serve_sessions_list_resume_and_new(project, home):
    server = FakeModelServer([{"content": "First answer."}, {"content": "Second answer."}, {"content": "Resumed answer."}])
    sp = ServeProcess(project, server.url, home)
    try:
        ready, _ = sp.wait_for("ready")
        first_id = ready["session_id"]
        sp.send({"type": "task", "text": "first question"})
        sp.wait_for("idle")
        sp.send({"type": "clear"})
        cleared, _ = sp.wait_for("cleared")
        assert cleared["session_id"] != first_id
        sp.send({"type": "task", "text": "second question"})
        sp.wait_for("idle")
        sp.send({"type": "list_sessions"})
        listed, _ = sp.wait_for("sessions")
        assert [i["title"] for i in listed["items"]] == ["second question", "first question"]
        sp.send({"type": "resume", "session_id": first_id})
        history, _ = sp.wait_for("history")
        assert [(i["role"], i["text"]) for i in history["items"]] == [("user", "first question"), ("assistant", "First answer.")]
        sp.send({"type": "task", "text": "follow-up"})
        sp.wait_for("idle")
        sent = [m["content"] for m in server.requests[-1]["messages"]]
        assert "first question" in sent and "second question" not in sent  # the resumed conversation, not the other one
    finally:
        sp.close()


def test_serve_questions_and_editor_diagnostics(project, home):
    server = FakeModelServer([
        {"tool_calls": [{"index": 0, "id": "q", "function": {"name": "ask_user", "arguments": json.dumps({"question": "Greeting?", "options": ["hi", "hello"]})}}]},
        {"tool_calls": [{"index": 0, "id": "p", "function": {"name": "update_plan", "arguments": json.dumps({"steps": ["Create hello.py"]})}}]},
        {"tool_calls": [{"index": 0, "id": "w", "function": {"name": "write_file", "arguments": json.dumps({"path": "hello.py", "content": "print(hello)\n"})}}]},
        {"content": "Created hello.py."},
    ])
    sp = ServeProcess(project, server.url, home, "--diagnostics", "--mode", "auto-edit")
    problem = {"path": "hello.py", "line": 1, "column": 7, "severity": "error", "message": "'hello' is not defined", "source": "pyflakes"}
    try:
        sp.wait_for("ready")
        sp.send({"type": "task", "text": "create hello.py"})
        q, _ = sp.wait_for("question")
        assert q["question"] == "Greeting?" and q["options"] == ["hi", "hello"]
        sp.send({"type": "answer", "request_id": q["request_id"], "text": "hello"})
        plan, _ = sp.wait_for("plan_updated")
        assert plan["steps"] == [{"step": "Create hello.py", "status": "pending"}]
        requests = 0
        while True:
            ev, _ = sp.wait_for_any({"diagnostics_request", "run_finished"})
            if ev["type"] == "run_finished":
                break
            requests += 1
            assert ev["paths"] == ["hello.py"]
            sp.send({"type": "diagnostics", "request_id": ev["request_id"], "items": [problem]})
        assert requests == 2  # after the edit, and for the final report
        rep = ev["result"]["report"]
        assert rep["problems"][0]["message"] == "'hello' is not defined"
        assert any("1 error(s)" in r for r in rep["remaining"])
        tool_results = [m["content"] for m in server.requests[3]["messages"] if m["role"] == "tool"]
        assert "The user answered: hello" in tool_results[0]
        assert "hello.py:1:7 error: 'hello' is not defined [pyflakes]" in tool_results[2]
    finally:
        sp.close()
        server.close()


def test_serve_rewind_and_permission_rules(project, home):
    server = FakeModelServer([
        {"tool_calls": [{"index": 0, "id": "p", "function": {"name": "update_plan", "arguments": json.dumps({"steps": ["Create hello.py"]})}}]},
        {"tool_calls": [{"index": 0, "id": "w", "function": {"name": "write_file", "arguments": json.dumps({"path": "hello.py", "content": "print('hi')\n"})}}]},
        {"content": "Created hello.py."},
    ])
    sp = ServeProcess(project, server.url, home, "--mode", "auto-edit")
    try:
        sp.wait_for("ready")
        sp.send({"type": "task", "text": "create hello.py"})
        cp, _ = sp.wait_for("checkpoint")
        assert cp["prompt"] == "create hello.py"
        sp.wait_for("idle")
        assert (project / "hello.py").exists()
        sp.send({"type": "list_checkpoints"})
        listing, _ = sp.wait_for("checkpoints")
        assert listing["items"][0]["id"] == cp["checkpoint_id"] and listing["items"][0]["files"] == ["hello.py"]

        sp.send({"type": "rewind", "checkpoint_id": cp["checkpoint_id"], "mode": "both"})
        rewound, _ = sp.wait_for("rewound")
        assert rewound["removed"] == ["hello.py"] and rewound["prompt"] == "create hello.py"
        history, _ = sp.wait_for("history")
        assert history["rewound"] is True and not [i for i in history["items"] if i["role"] == "user"]
        assert not (project / "hello.py").exists()

        sp.send({"type": "add_permission", "kind": "allow", "rule": "Bash(npm test:*)"})
        perms, _ = sp.wait_for("permissions")
        assert {"kind": "allow", "rule": "Bash(npm test:*)", "source": "saved"} in perms["rules"]
        sp.send({"type": "remove_permission", "kind": "allow", "rule": "Bash(npm test:*)"})
        perms, _ = sp.wait_for("permissions")
        assert perms["rules"] == []
    finally:
        sp.close()
        server.close()


def test_serve_custom_commands(project, home):
    (project / ".lca" / "commands").mkdir(parents=True, exist_ok=True)
    (project / ".lca" / "commands" / "explain.md").write_text("---\ndescription: Explain a file\n---\nExplain $ARGUMENTS in two sentences.")
    server = FakeModelServer([{"content": "It adds numbers."}])
    sp = ServeProcess(project, server.url, home)
    try:
        sp.wait_for("ready")
        sp.send({"type": "list_commands"})
        cmds, _ = sp.wait_for("commands")
        assert [i for i in cmds["items"] if i["source"] == "project"] == [{"name": "explain", "description": "Explain a file", "argument_hint": "", "source": "project", "kind": "command"}]
        assert {"init", "commit", "review", "pr"} <= {i["name"] for i in cmds["items"]}
        sp.send({"type": "task", "text": "/nope"})
        err, _ = sp.wait_for("notice")
        assert "Unknown command /nope" in err["message"]
        sp.send({"type": "task", "text": "/explain src/app.py"})
        sp.wait_for("run_finished")
        assert server.requests[0]["messages"][-1]["content"] == "Explain src/app.py in two sentences."
    finally:
        sp.close()
        server.close()


def test_serve_images_and_status_line(project, home):
    py = sys.executable.replace("\\", "/")
    (home / "sl.py").write_text("import json, sys\nd = json.load(sys.stdin)\nprint(d['mode'] + ' | ' + d['model'] + ' | runs ' + str(d['last_run'].get('status')))\n")
    (home / "config.toml").write_text(f"[status_line]\ncommand = '\"{py}\" \"{(home / 'sl.py').as_posix()}\"'\n")
    png = "data:image/png;base64,iVBORw0KGgo="
    server = FakeModelServer([{"content": "I see a screenshot."}])
    sp = ServeProcess(project, server.url, home)
    try:
        ready, _ = sp.wait_for("ready")
        assert ready["vision"] is True
        first, _ = sp.wait_for("status_line")
        assert first["text"] == "ask | fake-model | runs None"
        sp.send({"type": "task", "text": "what is in this image?", "images": [png, "javascript:alert(1)"]})
        sp.wait_for("run_finished")
        content = server.requests[0]["messages"][-1]["content"]
        assert content == [{"type": "text", "text": "what is in this image?"}, {"type": "image_url", "image_url": {"url": png}}]
        after, _ = sp.wait_for("status_line")
        assert after["text"] == "ask | fake-model | runs completed"
    finally:
        sp.close()
        server.close()
