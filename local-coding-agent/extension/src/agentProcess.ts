// Runs `lca serve` as a child process and exchanges JSON-lines messages with it (see docs/PROTOCOL.md).
import { ChildProcess, spawn } from 'child_process';
import { EventEmitter } from 'events';
import * as readline from 'readline';

export interface AgentMessage {
  type: string;
  [key: string]: unknown;
}

export interface AgentProcessOptions {
  executable: string;
  root: string;
  args: string[];
}

export class AgentProcess extends EventEmitter {
  private proc: ChildProcess | undefined;
  private stopping = false;

  constructor(
    private readonly opts: AgentProcessOptions,
    private readonly log: (line: string) => void,
  ) {
    super();
  }

  get running(): boolean {
    return !!this.proc && this.proc.exitCode === null && !this.proc.killed;
  }

  start(): void {
    if (this.running) {
      return;
    }
    this.stopping = false;
    const args = ['serve', '--root', this.opts.root, ...this.opts.args];
    this.log(`$ ${this.opts.executable} ${args.join(' ')}`);
    const proc = spawn(this.opts.executable, args, {
      cwd: this.opts.root,
      env: { ...process.env, PYTHONUTF8: '1', PYTHONIOENCODING: 'utf-8' },
      windowsHide: true,
    });
    this.proc = proc;

    readline.createInterface({ input: proc.stdout! }).on('line', (line) => {
      let msg: AgentMessage;
      try {
        msg = JSON.parse(line);
      } catch {
        this.log(`[agent stdout] ${line}`);
        return;
      }
      if (msg && typeof msg.type === 'string') {
        this.emit('message', msg);
      }
    });
    proc.stderr!.setEncoding('utf8');
    proc.stderr!.on('data', (chunk: string) => this.log(chunk.trimEnd()));
    proc.on('error', (err) => this.emit('spawnError', err));
    proc.on('exit', (code, signal) => {
      if (this.proc === proc) {
        this.proc = undefined;
      }
      this.emit('exit', code, signal, this.stopping);
    });
  }

  send(msg: AgentMessage): boolean {
    const stdin = this.proc?.stdin;
    if (!stdin || !stdin.writable) {
      return false;
    }
    stdin.write(JSON.stringify(msg) + '\n');
    return true;
  }

  stop(): void {
    const proc = this.proc;
    if (!proc) {
      return;
    }
    this.stopping = true;
    this.send({ type: 'shutdown' });
    setTimeout(() => {
      if (proc.exitCode === null) {
        proc.kill();
      }
    }, 3000);
  }
}
