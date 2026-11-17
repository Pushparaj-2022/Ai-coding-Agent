# Architecture

```
VS Code extension (TypeScript)          terminal (lca chat / run / build)
        │  JSON lines over stdin/stdout (docs/PROTOCOL.md)
        ▼
lca serve ──► AgentLoop  (LangGraph StateGraph)     BuildPipeline (LangGraph StateGraph)
                 compact → agent → tools → …           plan → implement → install → verify ⇄ fix
                 ├ verify (build/tests after changes)   → review → revise → verify again
                 └ wrapup (answer when stopping)
                         │                                   (each step is a fresh AgentLoop)
                         ▼
               LocalChatModel (LangChain BaseChatModel)
                         │  native tool calls, or text / LFM tool-call parsing
                         ▼
               Provider: llama.cpp / LM Studio / vLLM (OpenAI API) or Ollama
```

## Agent graph (`lca/graph/agent_graph.py`)

State is the LangChain message list with the `add_messages` reducer. Nodes:

- **compact**: repairs history after an interruption (every tool call gets a result), then keeps the
  conversation inside the context window (`lca/graph/compaction.py`). In order: shorten old tool output,
  summarize earlier turns with the model, summarize finished steps of the current task, trim. If the
  model can't summarize, a mechanical digest is used.
- **agent**: one model call with the tools bound. Streams text, handles a context overflow by
  compacting aggressively and retrying, and nudges after an empty reply.
- **tools**: runs the calls through the tool registry (validation, permissions, approvals, diffs,
  secret redaction). Tracks declines, consecutive failures and turns with no progress.
- **verify**: when the agent gives its final answer after changing files, runs the project's build and tests
  (`lca/agent/project.py`; a command the agent already ran successfully after its last change is not repeated).
  A failure goes back to the agent with the output, up to `agent.verify_rounds` times (default 2).
- **wrapup**: when a run has to stop (limit reached, or no progress), makes one last tool-free call,
  so the user still gets an answer.

## Working loop (chat agent)

| Step | How it is enforced |
|---|---|
| Understand | Questions are answered directly; a request that changes files must go through a plan |
| Inspect | Read tools (`repo_overview`, `search_code`, `read_file`); edits to a file are refused until it was read |
| Plan mode | Mode `plan`: read-only research, then `exit_plan_mode` shows the plan for approval; on approval the mode switches (ask or auto-edit), the plan's numbered steps become the run's plan, and the same task continues. The tool list and system prompt follow the mode at every run |
| Plan | `update_plan`, shown as a checklist. `agent.require_plan`: `remind` (default; the first change without a plan gets a reminder), `enforce` (changes refused until there is a plan) or `off`. Small models often never plan, and a refusal stalls them |
| Honest finish | If the agent stops while its last test/build run failed and nothing changed since, the verify step sends it back and the report says verification failed |
| Clarify | `ask_user` asks the user and waits (VS Code card, or a terminal prompt). Without a user, the agent is told to assume and say so |
| Execute | Tools with permissions and approvals. `task` hands a self-contained job to a sub-agent (`explore` and `plan` read only; `general` may edit under the user's permissions) with its own context; only its report comes back. Sub-agent edits join the message's checkpoint and count for verification |
| Observe | Tool output and exit codes, hints for common failures, and editor diagnostics (VS Code problems) appended to every file edit's result; `get_diagnostics` on demand |
| Repair and verify | The **verify** node (above) |
| Report | `AgentResult.report`: plan, changed files, verification, editor errors left, remaining issues, declined actions |

`/build` specialists get their plan from the planner and are verified by the pipeline, so they have no `update_plan` or
`ask_user` and do not auto-verify.

The system prompt, including long-term memory, is rebuilt for every call and is not stored in the
state. Updated instructions and memory therefore apply to resumed sessions too.

## Memory

| Kind | Where | What |
|---|---|---|
| Conversation history | `<project>/.lca/sessions.db` (LangGraph `SqliteSaver` plus a session index) | Every chat is saved after each step. Resume with History in VS Code, `lca chat --resume last`, or `/resume <id>`. Only the newest checkpoint per session is kept. |
| Project memory | `<project>/.lca/memory.md` | Facts saved with the `remember` tool: commands that work, conventions, key files |
| User memory | `~/.lca/memory.md` | Your preferences, applied in every project |

Both memory files are plain Markdown that you can edit. `lca memory` shows them. `.lca/` has its own
`.gitignore`, so it never ends up in commits.

## Limits and safety

Permission rules (`lca/agent/rules.py`) are checked in `PermissionPolicy.check` before the mode: deny rules block,
ask rules force a prompt even in auto modes, allow rules skip the prompt. Blocked and risky commands keep their
fixed behaviour. Checkpoints (`lca/memory/checkpoints.py`): the tool registry calls `ToolContext.on_change` right
before applying a file change; each user message has its own checkpoint, tied to the message by its id, so a rewind
can restore files and cut the conversation at the same point.

| Rule | Default | Why |
|---|---|---|
| Turns per task | 60 | Hard stop; the run ends with a wrap-up answer |
| Every call failed, turns in a row | 6 | Stops loops on wrong paths and unknown tools |
| Nothing succeeded, turns in a row | 10 | Stops command loops while still allowing debugging |
| The same call failed this often | 3 | Refused with the earlier error. Successful repeats are always allowed |
| Missing path | n/a | The error lists the closest existing paths |
| Shell file viewers (`cat`, `ls`…) | allowed | They run, with a hint to prefer the file tools |
| Risky commands (`git push`, secret files, `rm -rf`…) | always ask | Even in "Allow everything" mode |
| Destructive commands (`rm -rf /`, `format`, `curl \| sh`) | never run | |
