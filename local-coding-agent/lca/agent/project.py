"""Work out how to install, build and test a project from its files."""

from __future__ import annotations

import json
from pathlib import Path

from lca.agent.environment import python_command


def _node_pm(root: Path) -> str:
    if (root / "pnpm-lock.yaml").exists():
        return "pnpm"
    if (root / "yarn.lock").exists():
        return "yarn"
    if (root / "bun.lockb").exists() or (root / "bun.lock").exists():
        return "bun"
    return "npm"


def _scripts(root: Path) -> dict[str, str]:
    try:
        data = json.loads((root / "package.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    scripts = data.get("scripts") if isinstance(data, dict) else None
    return {k: v for k, v in (scripts or {}).items() if isinstance(v, str)}


def install_commands(root: Path) -> list[str]:
    """Dependency installation that is clearly missing (e.g. package.json without node_modules)."""
    cmds: list[str] = []
    if (root / "package.json").is_file() and not (root / "node_modules").is_dir():
        cmds.append(f"{_node_pm(root)} install")
    return cmds


def _uses_unittest(root: Path) -> bool:
    """Whether the test files are unittest classes. Plain `def test_*` functions only run under pytest."""
    files = [*root.glob("test_*.py"), *root.glob("*_test.py"), *(root / "tests").glob("**/test_*.py")][:50]
    for f in files:
        try:
            if "unittest" in f.read_text(encoding="utf-8", errors="replace"):
                return True
        except OSError:
            pass
    return False


def verify_commands(root: Path) -> list[str]:
    """Commands that prove the project works: type check, build and tests, in that order."""
    if (root / "package.json").is_file():
        pm = _node_pm(root)
        scripts = _scripts(root)
        cmds = []
        for name in ("typecheck", "build", "test"):
            body = scripts.get(name, "")
            if body and "no test specified" not in body:
                cmds.append(f"{pm} test" if name == "test" else f"{pm} run {name}")
        return cmds
    py, _ = python_command()
    has_python_tests = (root / "tests").is_dir() or any(root.glob("test_*.py")) or any(root.glob("*_test.py"))
    if py and ((root / "pyproject.toml").exists() or (root / "setup.py").exists() or (root / "requirements.txt").exists() or has_python_tests):
        text = ""
        for name in ("pyproject.toml", "requirements.txt", "requirements-dev.txt", "setup.cfg"):
            try:
                text += (root / name).read_text(encoding="utf-8", errors="replace")
            except OSError:
                pass
        if "pytest" in text or (root / "conftest.py").exists() or (root / "pytest.ini").exists():
            return [f"{py} -m pytest -q"]
        if has_python_tests:
            return [f"{py} -m unittest discover -v" if _uses_unittest(root) else f"{py} -m pytest -q"]
        return []
    if (root / "Cargo.toml").exists():
        return ["cargo build", "cargo test"]
    if (root / "go.mod").exists():
        return ["go build ./...", "go test ./..."]
    if (root / "pom.xml").exists():
        return ["mvn -q -B test"]
    if (root / "build.gradle").exists() or (root / "build.gradle.kts").exists():
        return ["gradle test" if not (root / "gradlew").exists() else "./gradlew test"]
    if any(root.glob("*.sln")) or any(root.glob("*.csproj")):
        return ["dotnet build", "dotnet test"]
    return []
