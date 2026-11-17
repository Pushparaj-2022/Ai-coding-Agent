"""A custom status line: your command prints one or two lines shown under the chat input (VS Code) or above the prompt
(lca chat), like Claude Code's statusLine.

    [status_line]                      # ~/.lca/config.toml only (it runs a command)
    command = "python ~/.lca/statusline.py"

The command gets a JSON object on stdin: model, workspace, session_id, mode, git_branch, context {used, window,
percent}, tokens {prompt, completion} (this session), last_run {status, duration_s, files_changed}, version. It runs
after every task and when the mode or session changes. ANSI colour codes are removed for VS Code.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any

log = logging.getLogger("lca.statusline")

ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
MAX_CHARS = 300


class StatusLine:
    def __init__(self, command: str, workspace: Path, shell: Any = None, timeout_s: float = 5) -> None:
        self.command = command
        self.workspace = workspace
        self.shell = shell  # lca.tools.terminal.Shell, as for the agent's commands
        self.timeout_s = timeout_s

    def render(self, data: dict[str, Any]) -> str:
        """The command's output (at most two lines), or "" if it fails."""
        argv: Any = self.shell.argv(self.command) if self.shell is not None else self.command
        try:
            proc = subprocess.run(
                argv, shell=self.shell is None, cwd=self.workspace, input=json.dumps(data, default=str), capture_output=True,
                text=True, encoding="utf-8", errors="replace", timeout=self.timeout_s,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"}, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.TimeoutExpired) as e:
            log.warning("status line command failed: %s", e)
            return ""
        if proc.returncode != 0:
            log.warning("status line command exited %s: %s", proc.returncode, proc.stderr.strip()[:200])
            return ""
        lines = [ANSI.sub("", ln).rstrip() for ln in proc.stdout.splitlines() if ln.strip()]
        return "\n".join(lines[:2])[:MAX_CHARS]
