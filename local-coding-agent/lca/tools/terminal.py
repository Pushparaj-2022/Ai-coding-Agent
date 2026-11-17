"""Shell tools: run_command (foreground or background), command_output, stop_command.

Safety layers, all enforced in code:
- classify_command() sorts every command into safe / normal / sensitive / blocked; the permission policy uses it.
- Commands run in the workspace with stdin closed (no interactive hangs), a timeout and bounded output capture.
- Environment variables that look like credentials are removed from the child's environment, and their values
  are scrubbed from output; output also passes through generic secret redaction.
- Cancellation and timeouts kill the whole process tree.
"""

from __future__ import annotations

import atexit
import itertools
import os
import re
import shlex
import shutil
import signal
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import BaseModel, Field

from lca.security import is_secret_path
from lca.tools.base import Tool, ToolContext, ToolError, ToolOutput

HEAD_BYTES = 16_000
TAIL_BYTES = 48_000
MAX_TIMEOUT_S = 1800
BACKGROUND_STARTUP_S = 3.0

Level = Literal["safe", "normal", "sensitive", "blocked"]

# ---------------------------------------------------------------- shell selection


@dataclass(frozen=True)
class Shell:
    kind: Literal["bash", "powershell", "cmd", "sh"]
    path: str

    def argv(self, command: str) -> list[str] | str:
        if self.kind == "powershell":
            # -Command reports only success/failure (0/1); propagate the real exit code of native programs.
            script = (
                "[Console]::OutputEncoding=[System.Text.Encoding]::UTF8; $ProgressPreference='SilentlyContinue'; "
                "$global:LASTEXITCODE=$null\n"
                f"{command}\n"
                "$lcaOk=$?\n"
                "if ($null -ne $LASTEXITCODE -and $LASTEXITCODE -ne 0) { exit $LASTEXITCODE }\n"
                "if (-not $lcaOk) { exit 1 }\n"
                "exit 0"
            )
            return [self.path, "-NoProfile", "-NonInteractive", "-Command", script]
        if self.kind == "cmd":
            # cmd.exe does not understand the \" escaping Python applies to list arguments; with /s it strips
            # exactly one pair of outer quotes and runs the rest verbatim.
            return f'"{self.path}" /d /s /c "{command}"'
        return [self.path, "-c", command]

    @property
    def label(self) -> str:
        return {"bash": "bash (Git Bash)" if os.name == "nt" else "bash", "powershell": "PowerShell", "cmd": "cmd.exe", "sh": "sh"}[self.kind]


def _git_bash() -> str | None:
    candidates = []
    git = shutil.which("git")
    if git:
        root = Path(git).resolve().parent.parent  # ...\Git\cmd\git.exe -> ...\Git
        candidates += [root / "bin" / "bash.exe", root.parent / "bin" / "bash.exe"]
    for base in (os.environ.get("ProgramFiles"), os.environ.get("ProgramW6432"), os.environ.get("LOCALAPPDATA")):
        if base:
            candidates += [Path(base) / "Git" / "bin" / "bash.exe", Path(base) / "Programs" / "Git" / "bin" / "bash.exe"]
    for c in candidates:
        if c.is_file():
            return str(c)
    return None


def detect_shell(preference: str = "auto") -> Shell:
    """auto: bash on POSIX; on Windows Git Bash if installed, else PowerShell. (System32\\bash.exe is WSL, not used.)"""
    if os.name != "nt":
        bash = shutil.which("bash")
        if preference in ("auto", "bash") and bash:
            return Shell("bash", bash)
        return Shell("sh", shutil.which("sh") or "/bin/sh")
    if preference in ("auto", "bash"):
        bash = _git_bash()
        if bash:
            return Shell("bash", bash)
        if preference == "bash":
            raise ToolError("Git Bash was requested but not found.")
    if preference in ("auto", "powershell"):
        ps = shutil.which("pwsh") or shutil.which("powershell")
        if ps:
            return Shell("powershell", ps)
    return Shell("cmd", os.environ.get("COMSPEC", "cmd.exe"))


# ---------------------------------------------------------------- command classification

_BLOCKED = [
    (r"\brm\s+(-[a-zA-Z]*\s+)*-[a-zA-Z]*[rR][a-zA-Z]*\s+(-[a-zA-Z]+\s+)*(/|~|/\*|\*|[a-zA-Z]:[\\/]?\*?|\$HOME)(\s|$)", "recursive delete of a root or home directory"),
    (r"\b(mkfs(\.\w+)?|diskpart|fdisk|format\s+[a-zA-Z]:)", "formats or partitions a disk"),
    (r"\bdd\s+.*\bof=/dev/", "writes raw data to a device"),
    (r">\s*/dev/sd[a-z]", "writes raw data to a disk"),
    (r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:", "fork bomb"),
    (r"\b(shutdown|reboot|halt|poweroff)\b|\bStop-Computer\b|\bRestart-Computer\b", "shuts down or restarts the machine"),
    (r"\b(curl|wget|iwr|irm|Invoke-WebRequest|Invoke-RestMethod)\b[^|]*\|\s*(sudo\s+)?(sh|bash|zsh|python\d?|iex|Invoke-Expression)\b", "runs a script downloaded from the internet"),
    (r"\b(del|erase|rd|rmdir)\b.*\s/[sS]\b.*\s[a-zA-Z]:[\\/]?\s*$", "recursive delete of a drive"),
    (r"\bRemove-Item\b.*-Recurse.*\s([a-zA-Z]:[\\/]?\*?|~|\$HOME|\$env:USERPROFILE)\s*(-|$)", "recursive delete of a drive or home directory"),
    (r"\breg(\.exe)?\s+delete\s+HK(LM|EY_LOCAL_MACHINE)", "deletes machine-wide registry keys"),
    (r"\b(vssadmin\s+delete|wbadmin\s+delete|bcdedit|cipher\s+/w)", "destroys backups or boot configuration"),
    (r"\bchmod\s+(-R\s+)?[0-7]*777\s+/(\s|$)", "makes the whole filesystem writable"),
]

_SENSITIVE = [
    (r"\bgit\s+push\b", "pushes to a remote repository"),
    (r"\bgit\s+(reset\s+--hard|clean\s+-[a-zA-Z]*f|checkout\s+--\s|restore\s+--source|branch\s+-D|stash\s+(drop|clear)|rebase|filter-branch)\b", "can discard work in git"),
    (r"\brm\s+(-[a-zA-Z]*\s+)*-[a-zA-Z]*[rRf]", "deletes files recursively or forcibly"),
    (r"\b(Remove-Item\b.*-Recurse|rd\s+/s|rmdir\s+/s|del\s+/[sqSQ])", "deletes files recursively"),
    (r"\b(npm|pnpm|yarn)\s+publish\b|\btwine\s+upload\b|\bcargo\s+publish\b|\bgem\s+push\b", "publishes a package"),
    (r"(?:^|[\s;&|(])(curl|wget|iwr|irm|Invoke-WebRequest|Invoke-RestMethod|scp|sftp|ssh|rsync|ftp|nc|ncat|telnet)(?:\.exe)?(?=\s|$)", "uses the network"),
    (r"\b(sudo|runas|su)\b|Start-Process\b.*-Verb\s+RunAs", "runs with elevated privileges"),
    (r"\b(chmod|chown|icacls|takeown|attrib)\b", "changes file permissions"),
    (r"\b(setx|reg(\.exe)?\s+(add|delete)|Set-ItemProperty\s+.*HK)", "changes system or user settings"),
    (r"\b(kill|pkill|killall|taskkill|Stop-Process)\b", "stops processes"),
    (r"\bdocker\s+(system\s+prune|volume\s+rm|rm\s+-f|rmi)\b", "deletes Docker data"),
    (r"\b(crontab|schtasks|systemctl|service|launchctl)\b", "changes scheduled tasks or services"),
    (r"(?:^|[;&|]\s*)(printenv|env|set)\s*$|\bGet-ChildItem\s+env:|\bdir\s+env:|\$env:\w+", "reads environment variables"),
    (r"(^|\s)(~|/etc/|/root/|\$HOME|\$env:USERPROFILE|%USERPROFILE%|C:\\Users\\)", "touches files outside the project"),
]

_SAFE_PROGRAMS = {
    "ls", "dir", "pwd", "cat", "type", "head", "tail", "wc", "grep", "rg", "findstr", "find", "tree", "file", "stat",
    "which", "where", "echo", "date", "whoami", "uname", "du", "df", "sort", "uniq", "diff", "cmp", "less", "more",
    "Get-ChildItem", "Get-Content", "Get-Location", "Select-String", "Get-Item", "Test-Path", "Measure-Object",
}
_SAFE_GIT = {"status", "log", "diff", "show", "branch", "remote", "rev-parse", "ls-files", "blame", "describe", "tag", "shortlog", "grep"}
_VERSION_FLAGS = {"--version", "-v", "-V", "version", "--help", "-h", "help"}
_SAFE_SUBCOMMANDS = {
    ("npm", "ls"), ("npm", "list"), ("npm", "outdated"), ("npm", "view"), ("pip", "list"), ("pip", "show"), ("pip", "freeze"),
    ("pip3", "list"), ("pip3", "show"), ("cargo", "tree"), ("go", "env"), ("go", "list"), ("dotnet", "--info"),
}
_METACHARS = re.compile(r"[;&|<>`]|\$\(|\n")


@dataclass(frozen=True)
class Assessment:
    level: Level
    reason: str = ""


_TOKEN_SPLIT = re.compile(r"[\s;&|<>()'\"`,=]+")


def touches_secret_file(command: str) -> bool:
    """True if any word of the command (including inside pipelines) names a secret file or directory."""
    for token in _TOKEN_SPLIT.split(command):
        parts = [p for p in re.split(r"[\\/]+", token) if p and p not in (".", "..")]
        if parts and is_secret_path(parts):
            return True
    return False


# Programs that only look at files; the file tools do this better (gitignore-aware, secret-safe, line numbers).
_VIEWERS = {
    "cat": "read_file", "head": "read_file (start_line/end_line)", "tail": "read_file (start_line/end_line)",
    "less": "read_file", "more": "read_file", "type": "read_file", "get-content": "read_file", "gc": "read_file",
    "ls": "list_files or repo_overview", "dir": "list_files or repo_overview", "tree": "repo_overview",
    "find": "list_files (with glob)", "get-childitem": "list_files", "gci": "list_files", "stat": "list_files",
    "wc": "read_file (it reports the line count)", "grep": "search_code", "rg": "search_code", "findstr": "search_code",
    "select-string": "search_code",
}
_NEUTRAL = {"echo", "printf", "true", ":", "pwd", "cd"}
_DEP_DIRS = ("node_modules", ".venv", "venv", "site-packages", "dist", "build", "target", "__pycache__")


def file_viewer_hint(command: str) -> str | None:
    """If the command only views files/directories, return guidance to use the file tools instead."""
    segments = [s.strip() for s in re.split(r"&&|\|\||;|\||\n", command) if s.strip()]
    if not segments or any(d in command for d in _DEP_DIRS) or ">" in re.sub(r"\d?>\s*/dev/null|2>&1", "", command):
        return None
    tools: list[str] = []
    for seg in segments:
        prog = Path(seg.split()[0].strip("\"'")).name.lower().removesuffix(".exe")
        if prog in _VIEWERS:
            tools.append(_VIEWERS[prog])
        elif prog not in _NEUTRAL:
            return None  # does real work (e.g. `npm test | tail -20`)
    if not tools:
        return None
    suggestions = ", ".join(dict.fromkeys(tools))
    return (
        f"Next time prefer {suggestions} for looking at files: they respect .gitignore and suggest the right path "
        "when a file does not exist."
    )


def classify_command(command: str, workspace_root: Path | None = None) -> Assessment:
    cmd = command.strip()
    if not cmd:
        return Assessment("blocked", "empty command")
    for pattern, reason in _BLOCKED:
        if re.search(pattern, cmd, re.I):
            return Assessment("blocked", reason)
    if touches_secret_file(cmd):
        return Assessment("sensitive", "touches a protected file (secrets, keys or .git internals)")
    for pattern, reason in _SENSITIVE:
        if re.search(pattern, cmd, re.I):
            return Assessment("sensitive", reason)
    if _METACHARS.search(cmd):
        return Assessment("normal", "compound command")
    try:
        tokens = shlex.split(cmd, posix=os.name != "nt")
    except ValueError:
        return Assessment("normal", "")
    if not tokens:
        return Assessment("normal", "")
    prog = Path(tokens[0].strip("\"'")).name
    prog_base = prog[:-4] if prog.lower().endswith((".exe", ".cmd", ".bat")) else prog
    args = tokens[1:]

    def touches_protected_path() -> bool:
        for a in args:
            a = a.strip("\"'")
            if a.startswith("-") and "=" not in a:
                continue
            value = a.split("=", 1)[-1]
            parts = re.split(r"[\\/]+", value)
            if ".." in parts or is_secret_path([p for p in parts if p]):
                return True
            if (value.startswith("/") or re.match(r"^[a-zA-Z]:", value)) and workspace_root is not None:
                try:
                    if not Path(value).resolve().is_relative_to(workspace_root):
                        return True
                except (OSError, ValueError):
                    return True
        return False

    safe = (
        prog_base in _SAFE_PROGRAMS
        or (prog_base == "git" and args and args[0] in _SAFE_GIT)
        or (len(args) == 1 and args[0] in _VERSION_FLAGS)
        or (bool(args) and (prog_base, args[0]) in _SAFE_SUBCOMMANDS)
    )
    if safe and prog_base == "find" and any(a in ("-delete", "-exec", "-execdir", "-ok") for a in args):
        safe = False
    if safe and prog_base == "git" and args[:1] == ["branch"] and any(a.startswith("-") and a not in ("-a", "-r", "-v", "-vv", "--list") for a in args[1:]):
        safe = False
    if safe and touches_protected_path():
        return Assessment("normal", "reads a file outside the project or a protected file")
    return Assessment("safe") if safe else Assessment("normal")


# ---------------------------------------------------------------- environment and output hygiene

_SECRET_ENV = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|PRIVATE|_AUTH|AUTH_|SESSION|COOKIE|CONNECTION_STRING|DSN)", re.I)
_ENV_KEEP = {"SSH_AUTH_SOCK", "GPG_AGENT_INFO", "XDG_SESSION_TYPE", "SESSIONNAME"}
_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07]*\x07|\x1b[@-Z\\-_]")


def child_environment(passthrough: set[str] | None = None) -> tuple[dict[str, str], list[str]]:
    """Returns (environment for commands, removed secret values to scrub from output)."""
    keep = (passthrough or set()) | _ENV_KEEP
    env: dict[str, str] = {}
    removed: list[str] = []
    for k, v in os.environ.items():
        if k not in keep and _SECRET_ENV.search(k):
            if len(v) >= 6:
                removed.append(v)
            continue
        env[k] = v
    env.update({
        "CI": "1", "NO_COLOR": "1", "FORCE_COLOR": "0", "TERM": "dumb", "PAGER": "cat", "GIT_PAGER": "cat",
        "GIT_TERMINAL_PROMPT": "0", "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", "npm_config_yes": "true",
        # agents edit and re-run within the same second; stale .pyc files (same mtime and size) would hide the fix
        "PYTHONDONTWRITEBYTECODE": "1",
        "DEBIAN_FRONTEND": "noninteractive",
    })
    return env, sorted(removed, key=len, reverse=True)


def clean_output(text: str, secrets: list[str]) -> str:
    text = _ANSI.sub("", text).replace("\r\n", "\n")
    # progress bars rewrite the line with \r: keep only the final state of each line
    text = "\n".join(line.rsplit("\r", 1)[-1] for line in text.split("\n"))
    for s in secrets:
        text = text.replace(s, "[REDACTED ENV VALUE]")
    return text


class OutputBuffer:
    """Keeps the first HEAD_BYTES and the last TAIL_BYTES of a stream, plus unread data for polling."""

    def __init__(self) -> None:
        self.head = bytearray()
        self.tail = bytearray()
        self.total = 0
        self.unread = bytearray()
        self.lock = threading.Lock()

    def add(self, chunk: bytes) -> None:
        with self.lock:
            self.total += len(chunk)
            self.unread += chunk
            if len(self.unread) > TAIL_BYTES:
                del self.unread[: len(self.unread) - TAIL_BYTES]
            room = HEAD_BYTES - len(self.head)
            if room > 0:
                self.head += chunk[:room]
                chunk = chunk[room:]
            if chunk:
                self.tail += chunk
                if len(self.tail) > TAIL_BYTES:
                    del self.tail[: len(self.tail) - TAIL_BYTES]

    def text(self) -> str:
        with self.lock:
            dropped = self.total - len(self.head) - len(self.tail)
            body = self.head.decode("utf-8", "replace")
            if dropped > 0:
                body += f"\n\n[... {dropped} bytes of output omitted ...]\n\n"
            return body + self.tail.decode("utf-8", "replace")

    def take_unread(self) -> str:
        with self.lock:
            data = bytes(self.unread)
            self.unread.clear()
        return data.decode("utf-8", "replace")


def _kill_tree(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], stdin=subprocess.DEVNULL, capture_output=True, timeout=15, check=False)
        else:
            os.killpg(proc.pid, signal.SIGKILL)
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        proc.kill()
    except OSError:
        pass


def _spawn(shell: Shell, command: str, cwd: Path, env: dict[str, str]) -> subprocess.Popen[bytes]:
    kwargs: dict = {}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | getattr(subprocess, "CREATE_NO_WINDOW", 0)
    else:
        kwargs["start_new_session"] = True
    try:
        return subprocess.Popen(
            shell.argv(command), cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **kwargs
        )
    except OSError as e:
        raise ToolError(f"Could not start {shell.label}: {e}") from e


# ---------------------------------------------------------------- background processes


@dataclass
class BackgroundProcess:
    id: str
    command: str
    proc: subprocess.Popen[bytes]
    buffer: OutputBuffer
    secrets: list[str]
    started: float = field(default_factory=time.monotonic)

    @property
    def status(self) -> str:
        code = self.proc.poll()
        return "running" if code is None else f"exited with code {code}"


class ProcessManager:
    """Tracks background commands for the lifetime of the session; all are killed on exit."""

    _ids = itertools.count(1)

    def __init__(self) -> None:
        self.processes: dict[str, BackgroundProcess] = {}
        atexit.register(self.stop_all)

    def start(self, shell: Shell, command: str, cwd: Path, env: dict[str, str], secrets: list[str]) -> BackgroundProcess:
        running = [p for p in self.processes.values() if p.proc.poll() is None]
        if len(running) >= 5:
            raise ToolError("Too many background commands are running (5). Stop one with stop_command first.")
        proc = _spawn(shell, command, cwd, env)
        bp = BackgroundProcess(id=f"bg{next(self._ids)}", command=command, proc=proc, buffer=OutputBuffer(), secrets=secrets)

        def pump() -> None:
            assert proc.stdout is not None
            read = proc.stdout.read1 if hasattr(proc.stdout, "read1") else proc.stdout.read
            for chunk in iter(lambda: read(4096), b""):
                bp.buffer.add(chunk)

        threading.Thread(target=pump, daemon=True).start()
        self.processes[bp.id] = bp
        return bp

    def adopt(self, command: str, proc: subprocess.Popen[bytes], buffer: OutputBuffer, secrets: list[str]) -> BackgroundProcess:
        """Take over a command that is already running in the foreground (the user moved it to the background)."""
        bp = BackgroundProcess(id=f"bg{next(self._ids)}", command=command, proc=proc, buffer=buffer, secrets=secrets)
        self.processes[bp.id] = bp
        return bp

    def get(self, pid: str) -> BackgroundProcess:
        if pid not in self.processes:
            known = ", ".join(self.processes) or "none"
            raise ToolError(f"No background command with id '{pid}'. Known ids: {known}.")
        return self.processes[pid]

    def stop_all(self) -> None:
        for bp in self.processes.values():
            _kill_tree(bp.proc)


# ---------------------------------------------------------------- tools


class RunCommandArgs(BaseModel):
    command: str = Field(min_length=1, description="The shell command to run")
    description: str = Field(default="", description="Short explanation of what the command does, shown to the user")
    cwd: str = Field(default=".", description="Working directory relative to the workspace root")
    timeout_s: int = Field(default=120, ge=1, le=MAX_TIMEOUT_S, description="Kill the command after this many seconds")
    background: bool = Field(
        default=False,
        description="Start a long-running process (dev server, watcher) and return immediately; read it with command_output",
    )


class RunCommand(Tool):
    name = "run_command"
    risk: ClassVar = "execute"
    Args = RunCommandArgs
    description = ""  # set per instance (depends on the shell)

    def __init__(self, shell: Shell | None = None, manager: ProcessManager | None = None, passthrough_env: set[str] | None = None) -> None:
        self.shell = shell or detect_shell()
        self.manager = manager or ProcessManager()
        self.passthrough_env = passthrough_env or set()
        self.description = (
            f"Run a shell command in the workspace using {self.shell.label}. Use it to install dependencies, build, run tests, "
            "linters and git. Returns the exit code and output. Commands cannot read keyboard input. For servers or watchers set "
            "background=true and check them with command_output. Prefer read_file/search_code over cat/grep."
        )


    def assess(self, args: RunCommandArgs, ctx: ToolContext) -> Assessment:
        return classify_command(args.command, ctx.workspace.root)

    def describe_call(self, args: RunCommandArgs) -> str:
        return f"Run: {args.command}"

    def run(self, args: RunCommandArgs, ctx: ToolContext) -> ToolOutput:
        cwd = ctx.workspace.resolve(args.cwd)
        if not cwd.is_dir():
            raise ToolError(f"cwd is not a directory: {args.cwd}")
        env, secrets = child_environment(self.passthrough_env)
        if args.background:
            bp = self.manager.start(self.shell, args.command, cwd, env, secrets)
            if ctx.cancel.wait(BACKGROUND_STARTUP_S):
                _kill_tree(bp.proc)
                ctx.cancel.raise_if_cancelled()
            early = clean_output(bp.buffer.text(), secrets)
            bp.buffer.take_unread()
            return ToolOutput(
                content=f"Started background command {bp.id} ({bp.status}).\nOutput so far:\n{early.strip() or '(none yet)'}\n"
                f"Use command_output with id '{bp.id}' to check it and stop_command to stop it.",
                is_error=bp.proc.poll() not in (None, 0),
            )

        started = time.monotonic()
        background = getattr(ctx, "background", None)  # threading.Event: the user asked to move this command to the background
        if background is not None:
            background.clear()  # a request made while nothing was running does not apply to this command
        proc = _spawn(self.shell, args.command, cwd, env)
        buffer = OutputBuffer()
        progress: Callable[[str], None] | None = getattr(ctx, "progress", None)
        live = [True]  # stop streaming to the tool row once the command is in the background

        def pump() -> None:
            assert proc.stdout is not None
            read = proc.stdout.read1 if hasattr(proc.stdout, "read1") else proc.stdout.read
            for chunk in iter(lambda: read(4096), b""):
                buffer.add(chunk)
                if progress and live[0]:
                    try:
                        progress(clean_output(chunk.decode("utf-8", "replace"), secrets))
                    except Exception:
                        pass

        reader = threading.Thread(target=pump, daemon=True)
        reader.start()
        unregister = ctx.cancel.on_cancel(lambda: _kill_tree(proc))
        timed_out = detached = False
        deadline = started + args.timeout_s
        try:
            while True:
                try:
                    proc.wait(timeout=0.25)
                    break
                except subprocess.TimeoutExpired:
                    if background is not None and background.is_set():
                        background.clear()
                        detached = True
                        break
                    if time.monotonic() > deadline:
                        timed_out = True
                        _kill_tree(proc)
                        proc.wait(timeout=10)
                        break
        finally:
            unregister()
        if detached:
            live[0] = False
            bp = self.manager.adopt(args.command, proc, buffer, secrets)
            so_far = clean_output(buffer.text(), secrets).strip()
            buffer.take_unread()
            return ToolOutput(content=(
                f"The user moved this command to the background as {bp.id}; it is still running.\n"
                f"Output so far:\n{so_far or '(none yet)'}\n"
                f"Use command_output with id '{bp.id}' to check on it and stop_command to stop it. Carry on with the task meanwhile."
            ))
        reader.join(timeout=5)
        ctx.cancel.raise_if_cancelled()
        seconds = round(time.monotonic() - started, 1)
        output = clean_output(buffer.text(), secrets).strip() or "(no output)"
        if timed_out:
            return ToolOutput(
                content=f"Command timed out after {args.timeout_s}s and was stopped.\n{output}\n"
                "If this is a server or watcher, run it with background=true. Otherwise raise timeout_s.",
                is_error=True,
            )
        code = proc.returncode
        content = f"Exit code {code} ({seconds}s)\n{output}"
        if code != 0:
            from lca.agent.environment import output_hints

            for hint in output_hints(args.command, code, output):
                content += f"\n[hint] {hint}"
        if viewer := file_viewer_hint(args.command):
            content += f"\n[hint] {viewer}"  # ran anyway; the file tools are just more reliable
        return ToolOutput(content=content, is_error=code != 0, exit_code=code)


class CommandOutputArgs(BaseModel):
    id: str = Field(description="Background command id, e.g. 'bg1'")
    wait_s: int = Field(default=0, ge=0, le=60, description="Wait this long for new output before returning")


class CommandOutput(Tool):
    name = "command_output"
    description = "Read new output from a background command started with run_command(background=true), and its status."
    Args = CommandOutputArgs

    def __init__(self, manager: ProcessManager) -> None:
        self.manager = manager

    def run(self, args: CommandOutputArgs, ctx: ToolContext) -> str:
        bp = self.manager.get(args.id)
        if args.wait_s:
            deadline = time.monotonic() + args.wait_s
            while time.monotonic() < deadline and bp.proc.poll() is None and not bp.buffer.unread:
                if ctx.cancel.wait(0.25):
                    break
        new = clean_output(bp.buffer.take_unread(), bp.secrets).strip()
        return f"{bp.id} ({bp.command}): {bp.status}\n{new or '(no new output)'}"


class StopCommandArgs(BaseModel):
    id: str = Field(description="Background command id, e.g. 'bg1'")


class StopCommand(Tool):
    name = "stop_command"
    description = "Stop a background command and everything it started."
    Args = StopCommandArgs

    def __init__(self, manager: ProcessManager) -> None:
        self.manager = manager

    def run(self, args: StopCommandArgs, ctx: ToolContext) -> str:
        bp = self.manager.get(args.id)
        if bp.proc.poll() is not None:
            return f"{bp.id} had already {bp.status}."
        _kill_tree(bp.proc)
        try:
            bp.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        tail = clean_output(bp.buffer.take_unread(), bp.secrets).strip()
        return f"Stopped {bp.id} ({bp.command}).{chr(10) + tail if tail else ''}"


def shell_tools(shell_preference: str = "auto", passthrough_env: set[str] | None = None) -> list[Tool]:
    manager = ProcessManager()
    return [RunCommand(detect_shell(shell_preference), manager, passthrough_env), CommandOutput(manager), StopCommand(manager)]


__all__ = [
    "Assessment", "Shell", "detect_shell", "classify_command", "child_environment", "clean_output", "OutputBuffer",
    "ProcessManager", "RunCommand", "CommandOutput", "StopCommand", "shell_tools",
]
