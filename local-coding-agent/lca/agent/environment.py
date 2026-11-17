"""Detect which developer commands actually work on this machine, so the model does not guess.

On Windows, `python`/`python3` are often Microsoft Store placeholders (they print nothing and exit 49);
the real interpreter is then the `py` launcher.
"""

from __future__ import annotations

import os
import re
import shutil
from functools import lru_cache

COMMANDS = ["node", "npm", "pnpm", "yarn", "bun", "deno", "git", "cargo", "go", "java", "mvn", "gradle", "dotnet", "make", "docker", "rg"]


def is_store_stub(path: str | None) -> bool:
    return bool(path) and os.name == "nt" and "\\windowsapps\\" in path.lower()


@lru_cache(maxsize=1)
def python_command() -> tuple[str | None, str]:
    """Returns (command to use, note for the model)."""
    for name in ("python", "python3"):
        p = shutil.which(name)
        if p and not is_store_stub(p):
            return name, ""
    if shutil.which("py"):
        return "py", "the `python` command is not installed (only the Microsoft Store placeholder); use `py` instead, e.g. `py -m pytest`"
    return None, "Python is not installed"


@lru_cache(maxsize=1)
def toolchain_summary() -> str:
    parts = []
    py, note = python_command()
    if py:
        parts.append(f"python → `{py}`" + (f" ({note})" if note else ""))
    else:
        parts.append("python: not installed")
    parts += [name for name in COMMANDS if shutil.which(name)]
    return ", ".join(parts)


_PY_MISSING = re.compile(r"ModuleNotFoundError: No module named '([\w.]+)'")
_NODE_MISSING = re.compile(r"Cannot find (?:module|package) '([^']+)'")


def output_hints(command: str, exit_code: int | None, output: str) -> list[str]:
    """Explain common failures (missing commands, missing dependencies, test runners finding no tests)."""
    hints = []
    if h := command_not_found_hint(command, exit_code, output):
        hints.append(h)
    py, _ = python_command()
    py = py or "python"
    if "NO TESTS RAN" in output or (exit_code == 5 and "unittest" in command):
        hints.append(
            "unittest found no tests. unittest only runs methods named test_* inside classes that subclass "
            "unittest.TestCase (plain test_* functions are pytest style). Fix the test file, then run it again."
        )
    if exit_code == 5 and "pytest" in command:
        hints.append("pytest collected no tests: test files must be named test_*.py and contain test_* functions.")
    if "No module named pytest" in output:
        hints.append(f"pytest is not installed. Install it with `{py} -m pip install pytest`, or use unittest.")
    elif m := _PY_MISSING.search(output):
        mod = m.group(1).split(".")[0]
        hints.append(
            f"The Python package for '{mod}' is missing. If it is a third-party package, install it "
            f"(`{py} -m pip install <package>`) and add it to requirements.txt; if it is a project module, check the "
            "file name and import path."
        )
    if (m := _NODE_MISSING.search(output)) and not m.group(1).startswith((".", "/")):
        hints.append(f"The npm package '{m.group(1)}' is missing. Install it with `npm install {m.group(1)}` (it is added to package.json).")
    if exit_code not in (0, None) and re.search(r"(^|[\s'\"=])(/(?:Users|home|mnt|[a-zA-Z])/|[A-Za-z]:[\\/])", command):
        hints.append("The command uses an absolute path. Commands already run in the workspace root: use relative paths "
                     "(e.g. `ls`, `ls src`), and list the workspace with list_files if you are unsure where things are.")
    if "Missing script:" in output:
        hints.append("That npm script does not exist. Read package.json to see the available scripts.")
    if "npm ERR! enoent" in output.lower() and "package.json" in output:
        hints.append("There is no package.json here. Create one (e.g. `npm init -y`) or run the command in the right folder.")
    return hints


def command_not_found_hint(command: str, exit_code: int | None, output: str) -> str:
    """Explain common 'command not found' failures in a way that tells the model what to do next."""
    low = output.lower()
    first = command.strip().split()[0].strip("\"'") if command.strip() else ""
    prog = os.path.basename(first).lower().removesuffix(".exe")
    py, _ = python_command()
    if prog in ("python", "python3") and (exit_code == 49 or is_store_stub(shutil.which(prog)) or "was not found" in low):
        if py and py != prog:
            return f"`{prog}` is the Microsoft Store placeholder here, not a real Python. Use `{py}` instead (e.g. `{py} -m unittest`)."
    not_found = (
        exit_code in (127, 9009)
        or "command not found" in low
        or "is not recognized as" in low
        or "commandnotfoundexception" in low
    )
    if not_found:
        return f"`{first}` was not found on this machine. Available commands: {toolchain_summary()}."
    return ""
