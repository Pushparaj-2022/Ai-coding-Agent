# Local Coding Agent (`lca`)

A local-first coding agent for the terminal (and VS Code, from Milestone 5). It explores a codebase through
controlled tools, runs on local models served by llama.cpp or Ollama, and never sends code off your
machine unless you explicitly allow a remote provider.

**Status:** the agent explores a codebase and creates, edits and deletes files. Each change is reviewed as a
diff first. It works from the terminal and from a VS Code extension. Running commands and tests (M3) comes
next. See [Roadmap](#roadmap).

## VS Code extension

```powershell
cd extension
npm install
npx tsc -p ./
.\node_modules\.bin\vsce package --allow-missing-repository --skip-license -o local-coding-agent.vsix
code --install-extension local-coding-agent.vsix
```

Open the **Local Coding Agent** icon in the activity bar, or press `Ctrl+Alt+L`. The extension runs
`lca serve` in the background. It finds `lca.exe` through the `lca.executable` setting, `PATH`, the
workspace `.venv`, or `~/.lca/bin`. See `extension/README.md` for details.

## Requirements

| Tool | Needed for | Check |
|---|---|---|
| Python 3.11+ | the agent | `py -3.12 --version` |
| A model server | inference | llama.cpp `llama-server` or Ollama |
| Git | `.gitignore`-aware file listing, repo status | `git --version` |
| ripgrep (optional) | fast search in large repos; a Python fallback is used without it | `winget install BurntSushi.ripgrep.MSVC` |

## Setup (Windows, PowerShell)

```powershell
cd D:\Projects\ai\local-coding-agent
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Start a model server. For example, llama.cpp with your GGUF on the GPU (Vulkan build):

```powershell
llama-server -m "<path to model>.gguf" --port 8080 --jinja -ngl 99 -c 32768 -np 2
```

- `--jinja` enables native tool calling. Without it, `lca` switches to a text tool-call format by itself.
- `-ngl 99` puts every layer on the GPU. On a 6 GB card, lower it if the model doesn't fit; models also run fully on CPU (`-ngl 0`).
- `-c 32768 -np 2` gives two parallel slots of 16k context each.

Or use Ollama: `ollama serve`, then pass `--provider ollama --model qwen2.5-coder:7b`.

Check everything:

```powershell
.\.venv\Scripts\lca.exe doctor --root D:\path\to\project
```

## Usage

```powershell
# one task, then exit (exit code: 0 completed, 3 blocked, 4 iteration limit, 130 cancelled, 1 error)
lca run "How is authentication implemented? Cite files." --root D:\path\to\project

# interactive session (history carries over between messages; /clear, /exit)
lca chat --root D:\path\to\project

# machine-readable event stream (used by the VS Code extension)
lca run "..." -o json

lca tools        # list tools and their JSON schemas
lca doctor       # environment and model checks
```

Press Ctrl+C once to cancel a run cleanly, or twice to quit immediately.

## Configuration

Settings are merged in this order, with later sources winning: built-in defaults, `~/.lca/config.toml`,
`<project>/.lca/config.toml`, environment variables, then command-line flags.

```toml
# ~/.lca/config.toml
allow_remote = false          # must be true (or --allow-remote) to use a non-localhost server

[provider]
kind = "openai"               # "openai" (llama.cpp, LM Studio, vLLM, hosted APIs) or "ollama"
base_url = "http://127.0.0.1:8080/v1"
model = ""                    # empty = first model the server reports
temperature = 0.2
max_tokens = 4096
tool_mode = "auto"            # auto | native | text
context_window = 16384        # optional; detected from llama.cpp /props or Ollama /api/show
timeout_s = 300

[agent]
max_iterations = 25
max_tool_calls_per_turn = 8
max_consecutive_failures = 3
max_identical_calls = 2
tool_output_chars = 12000
require_plan = "remind"       # update_plan: "remind" (default), "enforce" (no edits without a plan) or "off"
auto_verify = true            # after changing files, run the project's build/tests and send failures back
verify_rounds = 2             # how often a failed verification goes back to the agent

[permissions]                 # deny > ask > allow; see "Permission rules" below
allow = ["run_command(npm test:*)", "run_command(npm run lint:*)"]
ask   = ["run_command(npm publish:*)"]
deny  = ["Bash(git push:*)", "Edit(.github/**)"]
```

### Permission rules

| Rule | Matches |
|---|---|
| `run_command` / `Bash` | any command |
| `run_command(npm test)` | exactly `npm test` |
| `run_command(npm test:*)` | `npm test` and anything after it |
| `run_command(git log *)` | a glob on the command |
| `edit_file(src/**)`, `Edit(src/**)` | file changes under `src/` (`Edit` = write, edit and delete) |
| `read_file(secrets/**)`, `Read(...)` | reading files |

A command with several parts (`a && b`, `a | b`) is allowed only if every part is, and denied if any part is.
Commands with `>`, `<`, `$(...)` or backticks are never allowed by a rule. Risky commands (git push, `rm -rf`, secret
files) always ask, and blocked ones never run, whatever the rules say. A project's `.lca/config.toml` may add `ask`
and `deny` rules but not `allow`. "Always allow" on an approval, and `lca permissions --allow/--ask/--deny/--remove`,
save rules for one project in `~/.lca/permissions/`, outside the repository.

### Images, background commands, status line

- **Images**: paste or drop screenshots into the chat box, mention `@design.png`, or `lca run --image shot.png "..."`.
  The agent can also look at image files itself (`view_image`). Needs a vision model: llama-server with the model's
  `--mmproj` file, or an Ollama vision model. Set `provider.vision = false` for text-only models.
- **Background**: while a command runs, click *Background* on it (or press Ctrl+B) and the agent continues while the
  command keeps running; it checks on it with `command_output`.
- **Parallel sub-agents**: several `task` calls in one reply run at the same time (`agent.max_parallel_agents`, 3).
- **Status line**: `[status_line] command = "..."` in `~/.lca/config.toml`; your command gets the session as JSON on
  stdin (model, mode, context use, tokens, git branch, last run) and its output is shown under the chat box.
- **MCP sign-in**: HTTP MCP servers that use OAuth show *Sign in* under `/mcp` (or `lca mcp --login <server>`).

### Commands and @-mentions

Built in: `/init` writes AGENTS.md (build/test commands, architecture, conventions) for future sessions,
`/commit` commits in the repository's style, `/review` reviews uncommitted changes or a branch, `/pr` opens a pull
request with `gh`. Type `@` to attach a file (`@src/app.py`) or a folder listing to your message. The agent knows
the current branch, uncommitted changes and recent commits, and edits several places of a file at once with
`multi_edit`.

### Hooks, custom commands, skills

See [docs/EXTENDING.md](docs/EXTENDING.md): hooks run your own commands before/after tool calls and at
other points (a project's hooks run only after `lca hooks --trust`), `.lca/commands/*.md` become slash commands,
and `.lca/skills/*/SKILL.md` are instructions the agent loads when a task needs them.

### Plan mode and sub-agents

Pick **Plan** in the mode menu (or `--mode plan`). The agent only researches, then shows its plan. Choose "Yes,
auto-accept edits", "Yes, approve each edit", or tell it what to change. After approval it implements the plan in
the same task.

For broad research the agent can start sub-agents (`task` tool): an `explore` agent reads as many files as it needs
in its own context and returns only its findings, which keeps the main conversation small. A `general` sub-agent
can make changes, under your permission mode, and its edits can be rewound like any other.

### Undo: checkpoints and rewind

Before the agent writes, edits or deletes a file, the file is saved under the message that caused the change
(`.lca/checkpoints/`). To go back, hover a message in VS Code and click ↶ (or type `/rewind`), or use `/rewind` in
`lca chat`. Choose: code and conversation, conversation only, or code only. The rewound message returns to the input
box. Changes made by shell commands (installs, generators) are not tracked.

A project's own `.lca/config.toml` comes from the repository, so it is **not trusted**. It cannot set
`allow_remote`, API keys, or a non-local `base_url`; those settings are ignored with a warning.

### Environment variables

| Variable | Meaning |
|---|---|
| `LCA_PROVIDER` | `openai` or `ollama` |
| `LCA_BASE_URL` | Model server URL |
| `LCA_MODEL` | Model name |
| `LCA_API_KEY` | Bearer token for servers that need one. Read only from the environment, never written to logs |
| `LCA_TOOL_MODE` | `auto`, `native` or `text` |
| `LCA_ALLOW_REMOTE` | `1` to allow non-localhost model servers |
| `LCA_LOG_LEVEL` | `DEBUG`, `INFO`, `WARNING` or `ERROR` |
| `LCA_HOME` | Where config and logs live (default `~/.lca`) |

### Project instructions

Instructions are loaded from `~/.lca/AGENTS.md`, `<project>/AGENTS.md`, `<project>/LCA.md` and
`<project>/.lca/instructions.md`. Use them for build commands, conventions and architecture notes. They
guide the agent's work, but they cannot override its safety rules or permissions.

## Architecture

```
lca/
  cli.py                 Typer CLI, text/JSON renderers, Ctrl+C handling
  config.py              settings and trust boundaries
  messages.py            provider-neutral Message / ToolCall / ModelResponse
  cancel.py              cancellation token shared across threads
  security.py            secret file patterns and output redaction
  logging_setup.py       JSON-lines logs in ~/.lca/logs (no prompt or file contents)
  providers/
    base.py              Provider interface, retries, tool-mode fallback, error types
    openai_compatible.py SSE streaming /v1/chat/completions
    ollama.py            NDJSON streaming /api/chat
    toolcall_parser.py   <tool_call> JSON and LFM/pythonic call parsing, stream filter
    text_tools.py        text tool-call format for models without native tool calling
  tools/
    workspace.py         path validation, symlink/secret protection, gitignore-aware listing
    base.py              Tool interface, registry (validation → permission → run → redact → truncate)
    filesystem.py        repo_overview, list_files, read_file
    search.py            search_code (ripgrep, or Python fallback)
  agent/
    loop.py              the agent loop
    context.py           context-window budgeting
    permissions.py       read-only / ask / auto-edit / full policy
    prompts.py           system prompt
    events.py            event types (the JSON protocol)
  memory/instructions.py project instruction files
```

One change from the suggested layout: everything lives under one `lca` package. Top-level packages
named `agent` or `tools` would collide with other installed packages.

**How it scales to large repositories.** The agent never loads the whole repository. It works the way
an engineer would: `repo_overview` for the map, `search_code` to find what matters, then `read_file` on
specific line ranges. File listing goes through `git ls-files`, so it respects `.gitignore` and stays fast
on big repos. Tool output is capped. Before each model call, the context budget trims old tool output,
using the server's real token counts to keep its estimate accurate.

**Small-model robustness.**
- Tool calls are accepted in native, `<tool_call>` JSON, and pythonic (LFM2) formats.
- Every call is validated against a Pydantic schema before it runs.
- Malformed or truncated calls, unknown tools and invalid arguments go back to the model as errors it can correct.
- Identical calls repeated too often are blocked.
- The run stops as `blocked` after repeated all-failing turns, and as `max_iterations` at the safety limit.
- The final summary lists the tool calls that actually ran and their real results. It never treats the model's own claim of success as proof.

## Security model

- File tools can only reach the workspace. `..`, absolute paths and symlinks that resolve outside it are rejected.
- `.env*` files (except templates such as `.env.example`), private keys, certificates, credential files and `.git/`, `.ssh/`, `.aws/` and similar directories are never read or searched.
- Secret-looking values (API keys, tokens, private keys, JWTs, `password = ...`) are redacted from all tool output before it reaches the model, and from logs.
- File contents and tool output are labelled as untrusted data in the system prompt. Permissions are enforced in code, never left to the model.
- File changes need approval in `ask` mode, the default. The agent must read a file before changing it. Changes are refused if the file changed on disk since then, for example because you edited it. Declining a change stops the task.
- `--mode read-only` disables all file changes. `--yes` (`auto-edit`) applies them without asking.
- Remote inference must be explicitly enabled, and a warning is shown whenever it is.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```

121 tests cover:
- tool-call parsing, streaming, retries and fallbacks
- malformed and truncated model output
- path traversal and secret protection
- the permission matrix and config trust boundaries
- the agent loop's limits, cancellation and context trimming
- end-to-end CLI runs against a fake model server

Tests that need ripgrep or symlink privileges are skipped when those aren't available.

## Roadmap

1. ✅ **CLI, providers, read-only tools, agent loop**
2. ✅ **File creation, editing and deletion**: diffs reviewed before applying, read-before-write, conflict detection with your own edits, trash instead of delete
5. ✅ **VS Code extension** (done ahead of M3/M4, at your request): sidebar chat, streaming, approvals with the diff editor, cancel, editor-selection context
3. Terminal execution with timeouts, output limits, process tracking and cancellation; test feedback and debugging loop
4. Summarizing context compaction, Git awareness, SQLite sessions
6. LSP diagnostics, skills, hooks, MCP
7. Sub-agents with concurrency limits and write-conflict prevention

## Known limitations

- Answer quality depends on the model. A 2.6B model can find the right code and cite correct lines, but it can misdescribe what the code does. Check its claims.
- Without ripgrep, search in very large repositories is slower and stops after 30 seconds.
- Context compaction currently trims old tool output. Summarizing older history comes in Milestone 4.
