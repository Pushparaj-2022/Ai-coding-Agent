// The chat sidebar: owns the agent process, relays protocol events to the webview, and handles approvals.
import * as crypto from 'crypto';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import * as vscode from 'vscode';
import { AgentMessage, AgentProcess } from './agentProcess';
import { collectDiagnostics } from './diagnostics';
import { DiffView, ProposedChange } from './diff_view';
import { DEFAULT_BASE_URL, ModelServer, isLocalUrl, isReachable } from './modelServer';

type Status = 'starting' | 'ready' | 'busy' | 'offline' | 'error';

export class ChatViewProvider implements vscode.WebviewViewProvider {
  private view: vscode.WebviewView | undefined;
  private agent: AgentProcess | undefined;
  private status: Status = 'offline';
  private statusDetail = '';
  private model = '';
  private includeEditorContext = true;
  private restartPending = false;
  private readonly backlog: unknown[] = []; // messages produced before the webview was visible
  private launching: Promise<void> | undefined;
  private readonly readyWaiters: ((ok: boolean) => void)[] = [];
  private readonly modelServer: ModelServer;

  constructor(
    private readonly context: vscode.ExtensionContext,
    private readonly diffs: DiffView,
    private readonly output: vscode.OutputChannel,
    private readonly statusItem: vscode.StatusBarItem,
  ) {
    this.modelServer = new ModelServer((line) => this.output.appendLine(line));
  }

  // ------------------------------------------------------------ webview

  resolveWebviewView(view: vscode.WebviewView): void {
    this.view = view;
    const media = vscode.Uri.joinPath(this.context.extensionUri, 'media');
    view.webview.options = { enableScripts: true, localResourceRoots: [media] };
    view.webview.html = this.html(view.webview, media);
    view.webview.onDidReceiveMessage((m) => this.onWebviewMessage(m));
    view.onDidDispose(() => (this.view = undefined));
  }

  private post(msg: unknown): void {
    if (this.view) {
      void this.view.webview.postMessage(msg);
    } else {
      this.backlog.push(msg);
    }
  }

  private async onWebviewMessage(m: any): Promise<void> {
    switch (m?.type) {
      case 'ready':
        for (const msg of this.backlog.splice(0)) {
          void this.view?.webview.postMessage(msg);
        }
        this.pushStatus();
        this.pushEditorContext();
        this.ensureAgent();
        break;
      case 'send':
        this.sendTask(String(m.text ?? ''), false, Array.isArray(m.images) ? m.images.map(String) : []);
        break;
      case 'build':
        this.sendTask(String(m.text ?? ''), true);
        break;
      case 'stop':
        this.stop();
        break;
      case 'newChat':
        this.newChat();
        break;
      case 'listSessions':
        this.ensureAgent();
        if (await this.whenReady()) {
          this.agent?.send({ type: 'list_sessions' });
        }
        break;
      case 'resume':
        this.ensureAgent();
        if (await this.whenReady()) {
          this.agent?.send({ type: 'resume', session_id: String(m.session_id ?? '') });
        }
        break;
      case 'deleteSession':
        this.agent?.send({ type: 'delete_session', session_id: String(m.session_id ?? '') });
        break;
      case 'setMode':
        await vscode.workspace.getConfiguration('lca').update('mode', m.mode, vscode.ConfigurationTarget.Global);
        this.agent?.send({ type: 'set_mode', mode: m.mode });
        break;
      case 'approval':
        this.agent?.send({
          type: 'approval',
          request_id: m.request_id,
          approved: !!m.approved,
          remember: !!m.remember,
          always: !!m.always,
          mode: m.mode ? String(m.mode) : undefined,
          feedback: m.feedback ?? '',
        });
        break;
      case 'rewind':
        this.agent?.send({ type: 'rewind', checkpoint_id: String(m.checkpoint_id ?? ''), mode: String(m.mode ?? 'both') });
        break;
      case 'listCheckpoints':
      case 'listPermissions':
        this.ensureAgent();
        if (await this.whenReady()) {
          this.agent?.send({ type: m.type === 'listCheckpoints' ? 'list_checkpoints' : 'list_permissions' });
        }
        break;
      case 'findFiles':
        void this.findFiles(String(m.query ?? ''));
        break;
      case 'background':
        this.agent?.send({ type: 'background' });
        break;
      case 'mcpLogin':
        this.agent?.send({ type: 'mcp_login', server: String(m.server ?? '') });
        break;
      case 'listMcp':
        this.ensureAgent();
        if (await this.whenReady()) {
          this.agent?.send({ type: 'list_mcp' });
        }
        break;
      case 'listCommands':
        if (this.status === 'ready' || this.status === 'busy') {
          this.agent?.send({ type: 'list_commands' });
        }
        break;
      case 'addPermission':
      case 'removePermission':
        this.agent?.send({
          type: m.type === 'addPermission' ? 'add_permission' : 'remove_permission',
          kind: String(m.kind ?? 'allow'),
          rule: String(m.rule ?? ''),
        });
        break;
      case 'answer':
        this.agent?.send({ type: 'answer', request_id: m.request_id, text: String(m.text ?? '') });
        break;
      case 'openDiff':
        await this.diffs.open(String(m.request_id));
        break;
      case 'openFile':
        await this.openFile(String(m.path ?? ''), Number(m.line) || undefined);
        break;
      case 'toggleContext':
        this.includeEditorContext = !this.includeEditorContext;
        this.pushEditorContext();
        break;
      case 'retry':
        this.restart();
        break;
      case 'locateExecutable':
        await this.locateExecutable();
        break;
      case 'openSettings':
        void vscode.commands.executeCommand('workbench.action.openSettings', '@ext:redsand.local-coding-agent');
        break;
      case 'showLogs':
        this.output.show(true);
        break;
    }
  }

  // ------------------------------------------------------------ agent process

  private get root(): string | undefined {
    return vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
  }

  private resolveExecutable(): string | undefined {
    const configured = vscode.workspace.getConfiguration('lca').get<string>('executable', '').trim();
    if (configured) {
      return configured.replace(/^~(?=$|[\\/])/, os.homedir());
    }
    const exe = process.platform === 'win32' ? 'lca.exe' : 'lca';
    const candidates = [
      ...(process.env.PATH ?? '').split(path.delimiter).filter(Boolean).map((d) => path.join(d, exe)),
      ...(vscode.workspace.workspaceFolders ?? []).map((f) =>
        path.join(f.uri.fsPath, '.venv', process.platform === 'win32' ? 'Scripts' : 'bin', exe),
      ),
      path.join(os.homedir(), '.lca', 'bin', exe),
    ];
    return candidates.find((c) => {
      try {
        return fs.statSync(c).isFile();
      } catch {
        return false;
      }
    });
  }

  private agentArgs(): string[] {
    const cfg = vscode.workspace.getConfiguration('lca');
    const args = ['--mode', cfg.get<string>('mode', 'ask')];
    const baseURL = cfg.get<string>('baseURL', '').trim();
    const provider = cfg.get<string>('provider', '').trim();
    const model = cfg.get<string>('model', '').trim();
    if (provider) {
      args.push('--provider', provider);
    }
    if (baseURL) {
      args.push('--base-url', baseURL);
    }
    if (model) {
      args.push('--model', model);
    }
    if (cfg.get<boolean>('allowRemote', false)) {
      args.push('--allow-remote');
    }
    if (cfg.get<boolean>('editorDiagnostics', true)) {
      args.push('--diagnostics');
    }
    return args;
  }

  /** Start the model server (if needed) and the agent. Safe to call repeatedly. */
  private ensureAgent(): void {
    if (this.agent?.running || this.launching) {
      return;
    }
    this.launching = this.launch().finally(() => (this.launching = undefined));
  }

  private async launch(): Promise<void> {
    const root = this.root;
    if (!root) {
      this.setStatus('error', 'Open a folder first (File → Open Folder). The agent works inside the open workspace.');
      return;
    }
    const executable = this.resolveExecutable();
    if (!executable) {
      this.setStatus('error', 'Could not find the lca executable. Install the agent, or locate lca.exe.');
      this.post({ type: 'needExecutable' });
      return;
    }
    const cfg = vscode.workspace.getConfiguration('lca');
    const baseURL = cfg.get<string>('baseURL', '').trim() || DEFAULT_BASE_URL;
    if (cfg.get<boolean>('server.autoStart', true) && isLocalUrl(baseURL) && !(await isReachable(baseURL))) {
      this.setStatus('starting', 'Starting the model server…');
      const result = await this.modelServer.start(
        baseURL,
        {
          executable: cfg.get<string>('server.executable', '').trim(),
          model: cfg.get<string>('server.model', '').trim(),
          gpuLayers: cfg.get<number>('server.gpuLayers', 99),
          contextSize: cfg.get<number>('server.contextSize', 32768),
          parallel: cfg.get<number>('server.parallel', 1),
          extraArgs: cfg.get<string[]>('server.extraArgs', []),
        },
        (msg) => this.setStatus('starting', msg, false),
      );
      if (!result.ok) {
        this.setStatus('error', `The model server could not be started: ${result.message}`);
        return;
      }
    }
    const agent = new AgentProcess({ executable, root, args: this.agentArgs() }, (line) => this.output.appendLine(line));
    this.agent = agent;
    agent.on('message', (msg: AgentMessage) => this.onAgentMessage(msg));
    agent.on('spawnError', (err: Error) => {
      this.setStatus('error', `Could not start ${executable}: ${err.message}`);
      this.post({ type: 'needExecutable' });
    });
    agent.on('exit', (code: number | null, _signal: unknown, stopping: boolean) => {
      void vscode.commands.executeCommand('setContext', 'lca.busy', false);
      if (this.agent === agent) {
        this.agent = undefined;
      }
      if (this.restartPending) {
        this.restartPending = false;
        this.ensureAgent();
        return;
      }
      if (!stopping && this.status !== 'error') {
        this.setStatus('offline', `Agent stopped (exit code ${code}). It restarts when you send a message.`);
      }
    });
    this.setStatus('starting', 'Starting agent…');
    agent.start();
  }

  /** Resolves true once the agent reports ready, false if starting fails. */
  private whenReady(): Promise<boolean> {
    if (this.status === 'ready' || this.status === 'busy') {
      return Promise.resolve(true);
    }
    return new Promise((resolve) => this.readyWaiters.push(resolve));
  }

  private onAgentMessage(msg: AgentMessage): void {
    switch (msg.type) {
      case 'ready':
        this.model = String(msg.model ?? '');
        this.setStatus('ready', `${msg.model} · ctx ${msg.context_window}${msg.local ? '' : ' · REMOTE'}`);
        break;
      case 'fatal':
        // the chat renders the fatal event itself; avoid a second copy of the message
        this.setStatus('error', String(msg.message ?? 'The agent failed to start.'), false);
        break;
      case 'busy':
        this.setStatus('busy', this.statusDetail);
        void vscode.commands.executeCommand('setContext', 'lca.busy', true);
        break;
      case 'idle':
        this.setStatus('ready', this.statusDetail);
        void vscode.commands.executeCommand('setContext', 'lca.busy', false);
        break;
      case 'approval_requested': {
        const change = msg.change as ProposedChange | null;
        if (change) {
          this.diffs.remember(String(msg.request_id), change);
          if (vscode.workspace.getConfiguration('lca').get<boolean>('openDiffOnApproval', true)) {
            void this.diffs.open(String(msg.request_id));
          }
        }
        break;
      }
      case 'mode':
        // the agent changed mode itself (a plan was approved): keep the setting in step for the next start
        if (msg.mode && msg.mode !== vscode.workspace.getConfiguration('lca').get<string>('mode', 'ask')) {
          void vscode.workspace.getConfiguration('lca').update('mode', String(msg.mode), vscode.ConfigurationTarget.Global);
        }
        break;
      case 'open_url': {
        const url = String(msg.url ?? '');
        if (/^https?:\/\//i.test(url)) {
          void vscode.env.openExternal(vscode.Uri.parse(url));
        }
        return;
      }
      case 'diagnostics_request':
        this.answerDiagnostics(msg); // handled here; the chat does not show it
        return;
    }
    this.post({ type: 'agentEvent', event: msg });
  }

  /** Workspace files for @-mention completion. */
  private async findFiles(query: string): Promise<void> {
    const root = this.root;
    if (!root) {
      return;
    }
    const q = query.replace(/\\/g, '/').replace(/[*?[\]{}!]/g, '');
    const pattern = !q ? '**/*' : q.includes('/') ? `**/${q}*` : `**/*${q}*`;
    const excluded = '**/{node_modules,.git,.lca,dist,build,out,.venv,venv,__pycache__,.next,target}/**';
    const uris = await vscode.workspace.findFiles(new vscode.RelativePattern(root, pattern), excluded, 200);
    const items = uris
      .map((u) => path.relative(root, u.fsPath).split(path.sep).join('/'))
      .sort((a, b) => a.length - b.length || a.localeCompare(b))
      .slice(0, 12);
    this.post({ type: 'fileMatches', query, items });
  }

  private answerDiagnostics(msg: AgentMessage): void {
    const agent = this.agent;
    const root = this.root;
    const paths = Array.isArray(msg.paths) ? msg.paths.map(String) : [];
    const reply = (items: unknown[]) => agent?.send({ type: 'diagnostics', request_id: msg.request_id as string, items });
    if (!root) {
      reply([]);
      return;
    }
    collectDiagnostics(root, paths).then(reply, (err) => {
      this.output.appendLine(`[diagnostics] ${err}`);
      reply([]);
    });
  }

  private setStatus(status: Status, detail: string, notify = true): void {
    this.status = status;
    this.statusDetail = detail;
    const icon = { starting: '$(sync~spin)', ready: '$(hubot)', busy: '$(loading~spin)', offline: '$(circle-slash)', error: '$(error)' }[status];
    this.statusItem.text = `${icon} Agent${status === 'busy' ? ': working' : status === 'error' ? ': error' : ''}`;
    this.statusItem.tooltip = detail || status;
    if (status === 'ready' || status === 'error' || status === 'offline') {
      const ok = status === 'ready';
      for (const resolve of this.readyWaiters.splice(0)) {
        resolve(ok);
      }
    }
    this.pushStatus(notify);
  }

  private pushStatus(notify = false): void {
    this.post({
      type: 'status',
      status: this.status,
      detail: this.statusDetail,
      model: this.model,
      notify,
      mode: vscode.workspace.getConfiguration('lca').get<string>('mode', 'ask'),
    });
  }

  // ------------------------------------------------------------ actions

  sendTask(text: string, build = false, images: string[] = []): void {
    text = text.trim();
    if (!text) {
      return;
    }
    this.post({ type: 'userMessage', text: build ? `/build ${text}` : text, images });
    const context = this.includeEditorContext ? this.editorContext() : undefined;
    this.ensureAgent();
    // if the model server or agent is still starting, the task is sent as soon as it is ready
    void this.whenReady().then((ok) => {
      if (ok) {
        this.agent?.send({ type: build ? 'build' : 'task', text, context: context?.payload, images });
      }
    });
  }

  stop(): void {
    this.agent?.send({ type: 'cancel' });
  }

  newChat(): void {
    this.stop();
    this.agent?.send({ type: 'clear' });
    this.post({ type: 'clearLog' });
  }

  restart(): void {
    if (this.agent?.running) {
      this.restartPending = true;
      this.agent.stop();
    } else {
      this.setStatus('offline', '');
      this.ensureAgent();
    }
  }

  dispose(): void {
    this.agent?.stop();
    this.modelServer.stop(); // only stops a server this extension started
  }

  focusInput(): void {
    this.post({ type: 'focusInput' });
  }

  async locateExecutable(): Promise<void> {
    const picked = await vscode.window.showOpenDialog({
      canSelectMany: false,
      openLabel: 'Use this lca executable',
      filters: process.platform === 'win32' ? { Executable: ['exe', 'cmd', 'bat'] } : undefined,
    });
    if (!picked?.[0]) {
      return;
    }
    await vscode.workspace.getConfiguration('lca').update('executable', picked[0].fsPath, vscode.ConfigurationTarget.Global);
    this.restart();
  }

  onConfigChanged(e: vscode.ConfigurationChangeEvent): void {
    if (['executable', 'baseURL', 'provider', 'model', 'allowRemote', 'editorDiagnostics'].some((k) => e.affectsConfiguration(`lca.${k}`))) {
      this.restart();
    } else if (e.affectsConfiguration('lca.mode')) {
      this.agent?.send({ type: 'set_mode', mode: vscode.workspace.getConfiguration('lca').get<string>('mode', 'ask') });
      this.pushStatus();
    }
  }

  // ------------------------------------------------------------ editor context

  private editorContext(): { label: string; payload: object } | undefined {
    const ed = vscode.window.activeTextEditor;
    const root = this.root;
    if (!ed || !root || ed.document.uri.scheme !== 'file') {
      return undefined;
    }
    const rel = path.relative(root, ed.document.uri.fsPath).split(path.sep).join('/');
    if (rel.startsWith('..') || path.isAbsolute(rel)) {
      return undefined;
    }
    const sel = ed.selection;
    if (!sel.isEmpty) {
      const start = sel.start.line + 1;
      const end = sel.end.line + 1;
      return {
        label: `${rel}:${start}${end !== start ? '-' + end : ''}`,
        payload: { file: rel, selection: { start, end, text: ed.document.getText(sel) } },
      };
    }
    return { label: rel, payload: { file: rel } };
  }

  pushEditorContext(): void {
    this.post({ type: 'editorContext', label: this.editorContext()?.label ?? '', enabled: this.includeEditorContext });
  }

  enableEditorContext(): void {
    this.includeEditorContext = true;
    this.pushEditorContext();
  }

  private async openFile(rel: string, line?: number): Promise<void> {
    const root = this.root;
    if (!root || !rel) {
      return;
    }
    const abs = path.resolve(root, rel);
    if (path.relative(root, abs).startsWith('..')) {
      return;
    }
    try {
      const doc = await vscode.workspace.openTextDocument(abs);
      const options: vscode.TextDocumentShowOptions = { preview: true };
      if (line) {
        options.selection = new vscode.Range(line - 1, 0, line - 1, 0);
      }
      await vscode.window.showTextDocument(doc, options);
    } catch {
      void vscode.window.showWarningMessage(`Cannot open ${rel}`);
    }
  }

  // ------------------------------------------------------------ html

  private html(webview: vscode.Webview, media: vscode.Uri): string {
    const nonce = crypto.randomBytes(16).toString('base64');
    const css = webview.asWebviewUri(vscode.Uri.joinPath(media, 'chat.css'));
    const js = webview.asWebviewUri(vscode.Uri.joinPath(media, 'chat.js'));
    return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data:; style-src ${webview.cspSource}; script-src 'nonce-${nonce}';">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link rel="stylesheet" href="${css}">
<title>Local Coding Agent</title>
</head>
<body>
<div id="progress"><span></span></div>
<header id="top">
  <div class="brand" id="titleWrap">
    <span class="sessionTitle" id="sessionTitle">New chat</span>
  </div>
  <div class="actions">
    <button class="icon" id="logsBtn" title="Agent logs" aria-label="Agent logs">
      <svg viewBox="0 0 16 16"><path d="M4.5 2h7a1 1 0 0 1 1 1v11l-4.5-2.8L3.5 14V3a1 1 0 0 1 1-1z"/></svg>
    </button>
    <button class="icon" id="historyBtn" title="History" aria-label="History">
      <svg viewBox="0 0 16 16"><circle cx="8" cy="8" r="6"/><path d="M8 4.8V8l2.2 1.4"/></svg>
    </button>
    <button class="icon" id="newBtn" title="New chat" aria-label="New chat">
      <svg viewBox="0 0 16 16"><circle cx="8" cy="8" r="6"/><path d="M8 5.3v5.4M5.3 8h5.4"/></svg>
    </button>
  </div>
</header>
<div id="historyPanel" class="hidden" role="dialog" aria-label="Past sessions">
  <div class="hpHead"><span>Past sessions</span><button class="icon" id="historyClose" aria-label="Close"><svg viewBox="0 0 16 16"><path d="M4 4l8 8M12 4l-8 8"/></svg></button></div>
  <div id="historyList"><div class="hpEmpty">Loading…</div></div>
</div>
<main id="log" aria-live="polite">
  <section id="welcome">
    <div class="welcomeBox">
      <h2><span class="logo">✻</span> Welcome to <b>Local Coding Agent</b></h2>
      <p class="lead">Explores your project, edits files and runs commands on your local model. You approve every change.</p>
    </div>
    <div id="boot" class="hidden"></div>
    <div class="suggestions">
      <button class="suggestion" data-text="Explain how this project is structured">Explain how this project is structured</button>
      <button class="suggestion" data-text="Run the tests and fix whatever fails">Run the tests and fix whatever fails</button>
      <button class="suggestion" data-text="Find and fix bugs in the open file">Find and fix bugs in the open file</button>
      <button class="suggestion build" data-text="/build ">
        <b>/build</b>&nbsp;a feature with a team of agents
      </button>
    </div>
    <p class="hint"><kbd>Enter</kbd> send · <kbd>Shift</kbd>+<kbd>Enter</kbd> new line · <kbd>↑</kbd> last message · <kbd>Esc</kbd> stop</p>
  </section>
</main>
<button id="scrollBtn" class="hidden" title="Jump to latest" aria-label="Jump to latest">
  <svg viewBox="0 0 16 16"><path d="M8 3v10M3.5 8.5 8 13l4.5-4.5"/></svg>
</button>
<footer>
  <div id="working" class="hidden">
    <span class="star"></span>
    <span id="workText">Working…</span>
    <span id="workTime"></span>
    <button id="stopLink" class="linkish" title="Stop the agent">esc to interrupt</button>
  </div>
  <div id="slashMenu" class="menu hidden" role="menu"></div>
  <div id="modeMenu" class="menu right hidden" role="menu"></div>
  <div class="composer">
    <div class="attachments hidden"><button id="contextChip" class="chip" title="Send the active file or selection as context"></button></div>
    <div id="imageTray" class="imageTray hidden" aria-label="Attached images"></div>
    <textarea id="input" rows="1" placeholder="Ask anything · / for commands · @ for files · paste images"></textarea>
    <div class="composerBar">
      <button class="tbtn" id="attachBtn" title="Attach the open file / selection as context" aria-label="Attach context">
        <svg viewBox="0 0 16 16"><path d="M8 3v10M3 8h10"/></svg>
      </button>
      <button class="tbtn" id="slashBtn" title="Commands" aria-label="Commands">
        <svg viewBox="0 0 16 16"><rect x="2.5" y="2.5" width="11" height="11" rx="2"/><path d="M9.6 5 6.4 11"/></svg>
      </button>
      <span class="ring" id="ctxRing" title="Context used">
        <svg viewBox="0 0 20 20"><circle class="track" cx="10" cy="10" r="7"/><circle class="val" id="ctxVal" cx="10" cy="10" r="7"/></svg>
      </span>
      <span class="meta" id="sessionClock" title="Session time">
        <svg viewBox="0 0 16 16"><circle cx="8" cy="8" r="6"/><path d="M8 4.8V8l2.2 1.4"/></svg><span id="clockText">0m</span>
      </span>
      <button class="modelPill" id="brand" title="Model">
        <span class="dot" id="statusDot"></span><span id="statusText">Starting…</span><span class="sub" id="modelSub"></span>
      </button>
      <span class="spacer"></span>
      <button class="modeBtn" id="modeBtn" title="Permission mode" aria-haspopup="menu">
        <svg viewBox="0 0 16 16"><path d="M9 1.5 3.5 9H8l-1 5.5L12.5 7H8z"/></svg><span id="modeLabel">Ask</span>
      </button>
      <select id="mode" class="hidden" aria-hidden="true" tabindex="-1">
        <option value="ask">Ask</option>
        <option value="auto-edit">Auto-edit</option>
        <option value="full">Auto</option>
        <option value="plan">Plan</option>
        <option value="read-only">Read only</option>
      </select>
      <button id="sendBtn" class="send" title="Send (Enter)" aria-label="Send">
        <svg class="sendIcon" viewBox="0 0 16 16"><path d="M8 13V3M3.5 7.5 8 3l4.5 4.5"/></svg>
        <svg class="stopIcon" viewBox="0 0 16 16"><rect x="4" y="4" width="8" height="8" rx="1.5"/></svg>
      </button>
    </div>
  </div>
  <div id="statusLine" class="hidden"></div>
</footer>
<script nonce="${nonce}" src="${js}"></script>
</body>
</html>`;
  }
}
