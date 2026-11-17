from typing import Any

from lca.tools.base import FileChange, FileWriteTool, Tool, ToolContext, ToolError, ToolOutput, ToolRegistry
from lca.tools.edit import DeleteFile, EditFile, MultiEdit, WriteFile
from lca.tools.filesystem import ListFiles, ReadFile, RepoOverview
from lca.tools.interaction import AskUser, ExitPlanMode, GetDiagnostics, LoadSkill, Task, UpdatePlan, ViewImage
from lca.tools.search import SearchCode
from lca.tools.web import WebFetch, WebSearch
from lca.tools.terminal import CommandOutput, RunCommand, StopCommand, classify_command, detect_shell, shell_tools
from lca.tools.workspace import Workspace, WorkspaceError

READ_TOOLS = ["repo_overview", "list_files", "read_file", "search_code"]
WRITE_TOOLS = ["write_file", "edit_file", "multi_edit", "delete_file"]
SHELL_TOOLS = ["run_command", "command_output", "stop_command"]
MEMORY_TOOLS = ["remember", "forget"]
WEB_TOOLS = ["web_fetch", "web_search"]
INTERACTION_TOOLS = ["update_plan", "ask_user", "get_diagnostics", "task", "exit_plan_mode", "load_skill", "view_image"]


def default_registry(
    output_limit: int = 12000,
    shell: str = "auto",
    passthrough_env: list[str] | None = None,
    memory: bool = False,
    interaction: bool = False,
    web: Any = None,  # lca.config.WebSettings; None = no web tools
) -> ToolRegistry:
    from lca.tools.memory_tools import Forget, Remember

    return ToolRegistry(
        [
            RepoOverview(), ListFiles(), ReadFile(), SearchCode(), WriteFile(), EditFile(), MultiEdit(), DeleteFile(),
            *shell_tools(shell, set(passthrough_env or [])),
            *([Remember(), Forget()] if memory else []),
            *([UpdatePlan(), AskUser(), GetDiagnostics(), Task(), ExitPlanMode(), LoadSkill(), ViewImage()] if interaction else []),
            *([WebFetch(web.allow_private), WebSearch(web.search, web.searxng_url)] if web is not None and web.enabled else []),
        ],
        output_limit=output_limit,
    )


__all__ = [
    "Tool", "FileWriteTool", "FileChange", "ToolContext", "ToolError", "ToolOutput", "ToolRegistry", "Workspace",
    "WorkspaceError", "RepoOverview", "ListFiles", "ReadFile", "SearchCode", "WriteFile", "EditFile", "MultiEdit", "DeleteFile",
    "READ_TOOLS", "WRITE_TOOLS", "WEB_TOOLS", "WebFetch", "WebSearch", "SHELL_TOOLS", "MEMORY_TOOLS", "INTERACTION_TOOLS", "default_registry", "RunCommand",
    "CommandOutput", "StopCommand", "AskUser", "GetDiagnostics", "UpdatePlan", "Task", "ExitPlanMode", "classify_command", "detect_shell", "shell_tools",
]
