# Local Coding Agent for VS Code

A Claude Code-style chat panel that works on your project with a **local** model (llama.cpp or Ollama).
The agent explores your code, then creates, edits and deletes files. Each change is shown as a diff for
you to approve first.

## Requirements

1. The agent itself (the Python `lca` package). See the main README for setup.
2. A running model server, for example:
   `llama-server -m model.gguf --port 8080 --jinja -ngl 99 -c 32768 -np 2`

## Using it

- Open the **Local Coding Agent** icon in the activity bar, or press `Ctrl+Alt+L`.
- Type a task. The active file, or your selection, is sent along. Click the chip to turn this off.
- When the agent wants to change a file, a diff opens. Choose **Allow**, **Allow all edits this session**,
  or **Deny**, or type what it should do instead.
- **Esc** or **■** stops the current task.
- Select code and press `Ctrl+Alt+K` (or right-click → *Ask About Selection*) to ask about it.
- Deleted files are moved to `.lca/trash/` in your project, so you can restore them.

## Settings

| Setting | Meaning |
|---|---|
| `lca.executable` | Path to `lca.exe`. When empty, the extension looks on `PATH`, in the workspace `.venv`, and in `~/.lca/bin` |
| `lca.mode` | `ask` (approve each change), `auto-edit` or `read-only` |
| `lca.baseURL`, `lca.provider`, `lca.model` | Model server settings. When empty, the agent's own config is used |
| `lca.allowRemote` | Allow a non-local model server. Off by default: your code stays on this machine |
| `lca.openDiffOnApproval` | Open the diff editor automatically when a change needs approval |

Logs: run *Local Coding Agent: Show Agent Logs* from the Command Palette.
