from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from lca.messages import ToolCall
from lca.tools import ListFiles, ReadFile, RepoOverview, SearchCode, ToolRegistry, Workspace, WorkspaceError, default_registry
from lca.tools.base import Tool, ToolContext
from lca.tools.filesystem import glob_to_regex
from lca.tools.search import SearchCodeArgs


def run(registry: ToolRegistry, ctx: ToolContext, name: str, **args):
    return registry.execute(ToolCall(id="t", name=name, arguments=args), ctx)


@pytest.fixture
def reg() -> ToolRegistry:
    return default_registry()


# ------------------------------------------------------------------ workspace safety


@pytest.mark.parametrize("bad", ["../outside.txt", "..\\outside.txt", "src/../../x", "/etc/passwd", "C:\\Windows\\win.ini"])
def test_paths_outside_workspace_rejected(project: Path, bad: str):
    with pytest.raises(WorkspaceError, match="outside the workspace"):
        Workspace(project).resolve(bad)


@pytest.mark.parametrize("secret", [".env", "config/.env.production", "deploy/id_rsa", "certs/server.pem", ".git/config", ".ssh/known_hosts"])
def test_secret_paths_rejected(project: Path, secret: str):
    with pytest.raises(WorkspaceError, match="secrets"):
        Workspace(project).resolve(secret)


def test_secret_templates_allowed(project: Path):
    assert Workspace(project).resolve(".env.example").name == ".env.example"


def test_symlink_escape_rejected(project: Path, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside")
    (outside / "secret.txt").write_text("nope")
    link = project / "link"
    try:
        os.symlink(outside, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("creating symlinks needs extra privileges on this system")
    with pytest.raises(WorkspaceError, match="outside"):
        Workspace(project).resolve("link/secret.txt")


# ------------------------------------------------------------------ read_file


def test_read_file_with_line_numbers(reg, ctx):
    out = run(reg, ctx, "read_file", path="src/app.py")
    assert not out.is_error
    assert out.content.startswith("src/app.py (lines 1-9 of 9)")
    assert "     4\tdef add(a, b):" in out.content
    assert "src/app.py" in ctx.files_read


def test_read_file_range_and_continuation_hint(reg, ctx, project):
    (project / "big.txt").write_text("\n".join(f"line {i}" for i in range(1, 1501)), encoding="utf-8")
    out = run(reg, ctx, "read_file", path="big.txt", start_line=100, end_line=102)
    assert "(lines 100-102 of 1500)" in out.content and "line 101" in out.content
    assert "start_line=103" in out.content
    capped = run(reg, ctx, "read_file", path="big.txt", end_line=1500)
    assert "(lines 1-1000 of 1500)" in capped.content


def test_read_file_errors(reg, ctx, project):
    (project / "img.bin").write_bytes(b"\x89PNG\0\0data")
    assert "binary" in run(reg, ctx, "read_file", path="img.bin").content
    assert "directory" in run(reg, ctx, "read_file", path="src").content
    assert "not found" in run(reg, ctx, "read_file", path="nope.py").content
    assert "past the end" in run(reg, ctx, "read_file", path="src/util.py", start_line=50).content
    assert run(reg, ctx, "read_file", path=".env").is_error


def test_read_file_validation_errors(reg, ctx):
    out = run(reg, ctx, "read_file")
    assert out.is_error and "path" in out.content
    # out-of-range numbers are clamped instead of rejected (small models often overshoot limits)
    out = run(reg, ctx, "read_file", path="src/app.py", start_line=0)
    assert not out.is_error and "(lines 1-9 of 9)" in out.content
    out = run(reg, ctx, "list_files", max_results=99999)
    assert not out.is_error
    out = run(reg, ctx, "read_file", path="src/app.py", start_line="abc")
    assert out.is_error and "start_line" in out.content


def test_read_file_coerces_string_numbers(reg, ctx):
    out = run(reg, ctx, "read_file", path="src/app.py", start_line="4", end_line="5")
    assert "(lines 4-5 of 9)" in out.content


# ------------------------------------------------------------------ list / overview


def test_list_files_ignores_deps_and_secrets(reg, ctx):
    out = run(reg, ctx, "list_files").content.splitlines()
    assert "src/app.py" in out and "README.md" in out and ".env.example" in out
    assert ".env" not in out and not any("node_modules" in line for line in out)


def test_list_files_glob_and_limit(reg, ctx):
    out = run(reg, ctx, "list_files", glob="*.py").content
    assert out.splitlines() == ["src/app.py", "src/util.py"]
    limited = run(reg, ctx, "list_files", max_results=1).content
    assert "more files" in limited


@pytest.mark.skipif(not shutil.which("git"), reason="git not installed")
def test_list_files_respects_gitignore(reg, project):
    subprocess.run(["git", "init", "-q"], cwd=project, check=True)
    (project / ".gitignore").write_text("generated/\n")
    (project / "generated").mkdir()
    (project / "generated" / "out.py").write_text("x = 1\n")
    ctx = ToolContext(workspace=Workspace(project), cancel=__import__("lca.cancel", fromlist=["CancelToken"]).CancelToken(), policy=None)
    out = run(reg, ctx, "list_files").content
    assert "src/app.py" in out and "generated/out.py" not in out
    assert ".env\n" not in out and not out.endswith(".env")
    overview = run(reg, ctx, "repo_overview").content
    assert "Git: branch" in overview


def test_repo_overview(reg, ctx):
    out = run(reg, ctx, "repo_overview").content
    assert "Languages: Python 2" in out
    assert "README.md" in out and "src/  (2 files)" in out


def test_glob_to_regex():
    assert glob_to_regex("*.py").match("a/b/c.py")
    assert glob_to_regex("src/**/*.ts").match("src/x.ts")
    assert glob_to_regex("src/**/*.ts").match("src/a/b/x.ts")
    assert not glob_to_regex("src/*.ts").match("src/a/x.ts")
    assert glob_to_regex("*.{js,json}").match("pkg/package.json")


# ------------------------------------------------------------------ search


@pytest.fixture(params=["python", "ripgrep"])
def search_backend(request, monkeypatch):
    if request.param == "python":
        monkeypatch.setattr("lca.tools.search.shutil.which", lambda name: None)
    elif not shutil.which("rg"):
        pytest.skip("ripgrep not installed")
    return request.param


def test_search_finds_matches_with_context(reg, ctx, search_backend):
    out = run(reg, ctx, "search_code", pattern=r"def \w+", context_lines=1).content
    assert "src/app.py" in out and "src/util.py" in out
    assert "    4: def add(a, b):" in out
    assert "    5-     return a + b" in out
    assert "matching line(s)" in out


def test_search_never_reads_secrets(reg, ctx, search_backend):
    out = run(reg, ctx, "search_code", pattern="supersecretvalue").content
    assert out.startswith("No matches")


def test_search_options(reg, ctx, search_backend):
    assert "No matches" in run(reg, ctx, "search_code", pattern="DEF ADD", case_sensitive=True).content
    assert "app.py" in run(reg, ctx, "search_code", pattern="DEF ADD").content
    assert "app.py" in run(reg, ctx, "search_code", pattern="add(2", fixed_string=True).content
    assert "util.py" not in run(reg, ctx, "search_code", pattern="def", glob="app.py").content


def test_search_invalid_regex(reg, ctx, search_backend):
    out = run(reg, ctx, "search_code", pattern="(unclosed")
    assert out.is_error


def test_search_max_results(ctx, search_backend):
    out = SearchCode().run(SearchCodeArgs(pattern="def", max_results=1), ctx)
    assert "showing 1" in out


# ------------------------------------------------------------------ registry behaviour


def test_unknown_tool(reg, ctx):
    out = run(reg, ctx, "delete_everything")
    assert out.is_error and "Available tools" in out.content


def test_crashing_tool_is_contained(ctx):
    from pydantic import BaseModel

    class Boom(Tool):
        name = "boom"
        description = "explodes"

        class Args(BaseModel):
            pass

        def run(self, args, ctx):
            raise RuntimeError("kaboom")

    out = ToolRegistry([Boom()]).execute(ToolCall(id="1", name="boom"), ctx)
    assert out.is_error and "RuntimeError: kaboom" in out.content


def test_output_is_redacted_and_truncated(ctx, project):
    (project / "settings.py").write_text("API_KEY = 'sk-abcdefghijklmnopqrstuvwxyz123456'\n" + "x = 1\n" * 5000, encoding="utf-8")
    out = ToolRegistry([ReadFile()], output_limit=2000).execute(ToolCall(id="1", name="read_file", arguments={"path": "settings.py"}), ctx)
    assert "sk-abcdefghijklmnop" not in out.content and "[REDACTED" in out.content
    assert len(out.content) < 2200 and "characters omitted" in out.content


def test_schemas_are_compact():
    for spec in default_registry().specs():
        text = str(spec.parameters)
        assert "title" not in text and "anyOf" not in text
    assert {"repo_overview", "list_files", "read_file", "search_code", "write_file", "edit_file", "multi_edit", "delete_file", "run_command", "command_output", "stop_command"} == set(default_registry().names)


@pytest.mark.skipif(os.name != "nt", reason="Git Bash paths only exist on Windows")
def test_git_bash_style_paths_inside_workspace_are_accepted(project):
    ws = Workspace(project)
    drive = project.drive[0].lower()
    bash_path = f"/{drive}{project.as_posix()[2:]}/src/app.py"
    assert ws.resolve(bash_path) == (project / "src" / "app.py").resolve()
    assert ws.resolve(f"/mnt{bash_path}") == (project / "src" / "app.py").resolve()
    with pytest.raises(WorkspaceError, match="outside"):
        ws.resolve(f"/{drive}/Windows/win.ini")
