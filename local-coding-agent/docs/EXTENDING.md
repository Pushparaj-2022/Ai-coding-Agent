# Extending the agent: hooks, custom commands, skills

## Hooks

Hooks are your own shell commands that run at fixed points in the agent's work. Use them to block actions,
format or lint files after edits, add context to every message, or refuse to stop until something is done.

```toml
# ~/.lca/config.toml
[[hooks.PreToolUse]]
matcher = "Bash"                       # regex on the tool name; aliases: Bash, Edit, Write, Read, Grep
command = "python ~/.lca/hooks/no_npm.py"

[[hooks.PostToolUse]]
matcher = "Edit"
command = "npx prettier --write \"$(python -c 'import json,sys; print(json.load(sys.stdin)[\"arguments\"][\"path\"])')\""
timeout = 30

[[hooks.Stop]]
command = "python .lca/hooks/require_changelog.py"
```

| Event | When | Exit code 2 (or `{"decision": "block", "reason": ...}`) | stdout on success |
|---|---|---|---|
| `PreToolUse` | before a tool call | the call is refused; the reason goes to the agent | ignored |
| `PostToolUse` | after a tool call | the reason is added to the tool result as a problem | added to the tool result |
| `UserPromptSubmit` | before a message is sent to the agent | the message is refused | added to the message as context |
| `Stop` | when the agent is about to finish | the agent continues with the reason as feedback (up to 3 times) | ignored |
| `SessionStart` | once per conversation | ignored | added to the system prompt |

Each hook gets a JSON object on stdin: `event`, `workspace`, `session_id`, `agent` (null for the main agent, or the
sub-agent's role), plus `tool` and `arguments` (tool events), `ok` and `output` (PostToolUse), `prompt`
(UserPromptSubmit), `final_message`, `files_changed` and `verification` (Stop). The environment has
`LCA_HOOK_EVENT` and `LCA_WORKSPACE`. Hooks run in the agent's shell (Git Bash or PowerShell on Windows), in the
workspace folder. A hook that exits with another non-zero code, crashes or times out is reported as a warning; the
agent goes on.

Example `no_npm.py`:

```python
import json, sys
event = json.load(sys.stdin)
if event["arguments"].get("command", "").startswith("npm "):
    print("This project uses pnpm.", file=sys.stderr)
    sys.exit(2)
```

**Project hooks.** A project can define hooks in its `.lca/config.toml`, but they run only after you review and
trust them: `lca hooks --trust`. Trust covers the exact configuration; when the project's hooks change they stop
running until you trust them again. `lca hooks` lists the hooks that run.

## MCP servers

Tools from [MCP](https://modelcontextprotocol.io) servers become agent tools named `mcp__<server>__<tool>`.

```toml
# ~/.lca/config.toml
[mcp.servers.github]
command = "npx"
args = ["-y", "@modelcontextprotocol/server-github"]
env = { GITHUB_PERSONAL_ACCESS_TOKEN = "${GITHUB_TOKEN}" }   # ${VAR} comes from your environment
tools = ["get_issue", "list_issues", "create_pull_request"]  # optional: only these (fewer tools = better small-model accuracy)

[mcp.servers.docs]
url = "https://example.com/mcp"                              # streamable HTTP
headers = { Authorization = "Bearer ${DOCS_TOKEN}" }
timeout = 60
```

Servers start in the background when the agent starts. `lca mcp` connects and shows each server's status and
tools; `/mcp` shows the same in chat. MCP tools need approval in ask and auto-edit modes (they can do anything the
server can); allow the ones you trust with rules such as `allow = ["mcp__github__get_issue"]`. A project's own
`[mcp.servers]` start only after `lca mcp --trust`.

## Web

`web_fetch` reads a page as text (long pages are condensed by the model to what the agent asked for) and
`web_search` searches with DuckDuckGo. Both need approval except in full mode. web_fetch never reaches localhost
or the local network unless you set `web.allow_private = true` in your user config.

```toml
[web]
enabled = true
search = "duckduckgo"        # or "searxng" (with searxng_url) or "brave" (LCA_BRAVE_API_KEY)
searxng_url = ""
```

## Custom slash commands

A Markdown file is a command: `.lca/commands/review.md` (shared with the project) or `~/.lca/commands/review.md`
(yours, in every project) becomes `/review`. A project command overrides a personal one with the same name; files
in sub-folders are named `folder:name`.

```markdown
---
description: Review a part of the code for bugs
argument-hint: [file or folder]
---
Review $ARGUMENTS for bugs, missing error handling and security problems. Report each with path:line.
```

Built-in commands (`/init`, `/commit`, `/review`, `/pr`) are files of the same kind; a personal or project
command with the same name replaces them.

`$ARGUMENTS` is everything typed after the command; `$1`, `$2`, ... are single words. Without placeholders, the
arguments are appended. Commands appear in the `/` menu in VS Code; `lca commands` lists them, and they work in
`lca chat` and `lca run "/review src"` too.

## Skills

A skill is a folder with instructions for one kind of task, loaded only when needed:

```
.lca/skills/release/SKILL.md        (or ~/.lca/skills/release/SKILL.md)
.lca/skills/release/checklist.md    any other files the skill refers to
```

```markdown
---
name: release
description: How to cut a release of this project (version bump, changelog, tag)
---
# Release
1. ...
```

The agent sees only each skill's name and description. When a task matches, it loads the skill with `load_skill`
(and its other files with `load_skill(name, file=...)`), so skills cost no context until they are used. `/release
2.0` runs a skill directly.
