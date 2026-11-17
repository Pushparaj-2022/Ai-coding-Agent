# Agent event protocol (JSON lines)

`lca run "<task>" -o json` writes one JSON object per line to stdout. Diagnostics go to stderr. The VS Code
extension (Milestone 5) starts `lca` as a local process and reads this stream, so the extension and the CLI
always run the same agent code.

Every event has a `type` field. Event types are defined in `lca/agent/events.py`, which is the source of truth.

| `type` | Fields | Meaning |
|---|---|---|
| `turn_started` | `iteration` | The agent is about to call the model |
| `text_delta` | `text` | Streamed answer text, with tool-call markup already removed |
| `reasoning_delta` | `text` | Streamed model reasoning, if the model produces any |
| `tool_call_started` | `call_id`, `name`, `arguments` | A validated tool call is about to run |
| `tool_call_finished` | `call_id`, `name`, `ok`, `summary`, `output`, `duration_ms` | The real result, as sent to the model (redacted and truncated) |
| `notice` | `level` (`info`/`warning`/`error`), `message` | Context trimming, tool-format fallback, and similar |
| `checkpoint` | `checkpoint_id`, `message_id`, `prompt` | A user message started. Files the agent changes from here on can be rewound to their state before it |
| `plan_updated` | `steps` [{`step`, `status`: pending/in_progress/done/skipped}] | The agent recorded or updated its plan (`update_plan`) |
| `run_finished` | `result` | Final `AgentResult`: `status`, `final_message`, `iterations`, `tool_calls[]`, `files_read[]`, `files_changed[]`, `usage`, `duration_s`, `error`, `report` |

`result.report` is built by code from what actually happened, not from the model's answer:

| Field | Meaning |
|---|---|
| `plan` | The final plan steps with their status |
| `files_changed` | Files the agent created, edited or deleted |
| `verification` | `{status, commands: [{command, exit_code}], note}`; `status` is `not_needed` (no changes), `passed`, `failed` or `unverified` (no command found, declined, turned off, or the run stopped first) |
| `problems` | Editor errors left in changed files: [{`path`, `line`, `column`, `severity`, `message`, `source`, `code`}] |
| `remaining` | Open issues: failing commands, editor errors, plan steps not marked done, why a run stopped |
| `needs_approval` | Actions the user declined, as `tool: target` |

`result.status` is one of `completed`, `blocked`, `max_iterations`, `cancelled` or `error`. The process exit
code follows it: 0, 3, 4, 130 and 1 respectively.

Example:

```json
{"type":"turn_started","iteration":1}
{"type":"tool_call_started","call_id":"call_1","name":"search_code","arguments":{"pattern":"isBlockedCommand"}}
{"type":"tool_call_finished","call_id":"call_1","name":"search_code","ok":true,"summary":"src/permissions.js","output":"src/permissions.js\n     18: export function isBlockedCommand(cmd) {\n...","duration_ms":16}
{"type":"text_delta","text":"The check happens in "}
{"type":"run_finished","result":{"status":"completed","iterations":3,"...":"..."}}
```

## `lca serve` (used by the VS Code extension)

`lca serve --root <folder> [--mode ask|auto-edit|full|plan|read-only] [--base-url ...] [--diagnostics]` reads requests from stdin and writes
the events above, plus these, to stdout, one JSON object per line.

Requests (stdin):

| `type` | Fields | Effect |
|---|---|---|
| `task` | `text`, optional `context` `{file, selection: {start, end, text}}`, optional `images` [data URLs] | Start a task. Only one task runs at a time. `/name args` runs a custom command or skill (docs/EXTENDING.md); an unknown `/name` is refused with a notice |
| `build` | same as `task` | Run the multi-agent build pipeline (plan → implement → install → verify ⇄ fix → review) |
| `cancel` | | Cancel the running task |
| `background` | | Move the running foreground command to the background; the agent gets its id and continues |
| `mcp_login` | `server` | OAuth sign-in for an HTTP MCP server. The agent sends `open_url` for the sign-in page, then a notice and `mcp` |
| `approval` | `request_id`, `approved`, `remember`, `feedback`, `always`, `mode` | Answer an `approval_requested` event. `always` saves its `suggested_rule` as an allow rule for the project. For a plan (`plan` set), `mode` is the mode to implement it in (`ask` or `auto-edit`); declining with `feedback` keeps the agent planning |
| `answer` | `request_id`, `text` | Answer a `question` event |
| `diagnostics` | `request_id`, `items` [{`path`, `line`, `column`, `severity`, `message`, `source`, `code`}] | Answer a `diagnostics_request` (paths relative to the workspace, errors and warnings only) |
| `set_mode` | `mode` | Change the permission mode |
| `clear` | | Start a new conversation |
| `list_sessions` | | Reply with a `sessions` event |
| `resume` | `session_id` | Switch to a saved session; reply with a `history` event |
| `delete_session` | `session_id` | Delete a saved session (not the current one); reply with `sessions` |
| `list_checkpoints` | | Reply with `checkpoints`: `items` [{id, seq, prompt, created, message_id, files[]}], oldest first |
| `rewind` | `checkpoint_id`, `mode` (`both`, `code`, `conversation`) | Go back to before that message. Replies `rewound` and, unless `mode` is `code`, a `history` event with `rewound: true` |
| `list_mcp` | | Reply with `mcp`: `servers` [{name, status: starting/ready/failed, error, tools[], transport, source}] |
| `list_commands` | | Reply with `commands`: `items` [{name, description, argument_hint, source, kind: command/skill}] |
| `list_permissions` | | Reply with `permissions`: `rules` [{kind, rule, source}], `file` (where saved rules are kept) |
| `add_permission` / `remove_permission` | `kind` (`allow`, `ask`, `deny`), `rule` | Edit the saved rules for this project; replies `permissions` |
| `shutdown` | | Exit |

Extra events (stdout):

| `type` | Meaning |
|---|---|
| `ready` | Setup succeeded: `model`, `base_url`, `local`, `context_window`, `workspace`, `mode`, `session_id`, `vision` |
| `fatal` | Setup failed (`message`). The process then exits |
| `busy` / `idle` | A task started / finished |
| `status_line` | `text`: the output of the user's status line command (sent after start, each task, mode and session changes) |
| `open_url` | `url`: a page the user must open (OAuth sign-in); the client opens it in the browser |
| `approval_requested` | `request_id`, `tool`, `risk`, `description`, `command`, `level`, `reason`, `suggested_rule` (what "always allow" would save, or null), `plan` (Markdown, when the agent asks to leave plan mode; `tool` is then `exit_plan_mode`), `change` `{path, action, is_dir, before, after, diff}` |
| `approval_cancelled` | The task was cancelled while an approval was pending |
| `question` | `request_id`, `question`, `options[]`: the agent asks the user (`ask_user`) and waits for an `answer` |
| `question_cancelled` | The task was cancelled while a question was pending |
| `diagnostics_request` | `request_id`, `paths[]` (empty = whole workspace). Sent only with `--diagnostics`: after file edits, for the final report, and for `get_diagnostics`. The client replies with `diagnostics` within 15 s, or the agent goes on without them |
| `mode`, `cleared` | Responses to `set_mode` and `clear` (`cleared` carries the new `session_id`). `mode` is also sent when an approved plan switches the mode |
| `sessions` | `current`, `items` [{id, title, created, updated, turns}], newest first |
| `rewound` | `checkpoint_id`, `mode`, `restored[]`, `removed[]` (files the agent had created), `skipped[]`, `prompt` (the message rewound to) |
| `history` | `session_id`, `title`, `rewound` (true after a rewind), `items`: [{role: user, text, checkpoint_id}, {role: assistant, text}, {role: tool, name, arguments, ok, summary}, {role: notice, text}] |
| `stage_started` | `/build` step: `stage` (plan, implement, install, tests, verify, fix, review, revise), `title` |
| `build_finished` | `status` (passed, failed, unverified, cancelled, error), `summary`, `files_changed`, `commands` [{command, exit_code}], `review` |

Every agent event also has an `agent` field. It is `null` for the chat agent. For `/build` it names the specialist that produced the event (`planner`, `coder`, `tester`, `debugger` or `reviewer`), or `verify` for the pipeline's own verification commands. `tool_output_delta` (`call_id`, `text`) streams live command output.
