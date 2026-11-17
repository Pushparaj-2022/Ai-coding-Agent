// Starts a local llama.cpp server when none is running, so the user does not have to manage it by hand.
import { ChildProcess, spawn, spawnSync } from 'child_process';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';

export interface ServerSettings {
  executable: string;
  model: string;
  gpuLayers: number;
  contextSize: number;
  parallel: number;
  extraArgs: string[];
}

export const DEFAULT_BASE_URL = 'http://127.0.0.1:8080/v1';
const READY_TIMEOUT_MS = 180_000;

const isWindows = process.platform === 'win32';

function isFile(p: string): boolean {
  try {
    return fs.statSync(p).isFile();
  } catch {
    return false;
  }
}

function listDir(p: string): string[] {
  try {
    return fs.readdirSync(p);
  } catch {
    return [];
  }
}

export function isLocalUrl(url: string): boolean {
  try {
    const host = new URL(url).hostname.replace(/^\[|\]$/g, '');
    return host === 'localhost' || host === '127.0.0.1' || host === '::1';
  } catch {
    return false;
  }
}

export async function isReachable(baseURL: string, timeoutMs = 2000): Promise<boolean> {
  try {
    const res = await fetch(`${baseURL.replace(/\/+$/, '')}/models`, { signal: AbortSignal.timeout(timeoutMs) });
    return res.ok; // llama-server answers 503 while the model is still loading
  } catch {
    return false;
  }
}

export function findLlamaServer(configured: string): string | undefined {
  if (configured) {
    return configured.replace(/^~(?=$|[\\/])/, os.homedir());
  }
  const exe = isWindows ? 'llama-server.exe' : 'llama-server';
  const candidates = (process.env.PATH ?? '').split(path.delimiter).filter(Boolean).map((d) => path.join(d, exe));
  if (isWindows && process.env.LOCALAPPDATA) {
    const winget = path.join(process.env.LOCALAPPDATA, 'Microsoft', 'WinGet');
    candidates.push(path.join(winget, 'Links', exe));
    for (const pkg of listDir(path.join(winget, 'Packages'))) {
      if (pkg.toLowerCase().startsWith('ggml.llamacpp')) {
        candidates.push(path.join(winget, 'Packages', pkg, exe));
      }
    }
  } else {
    candidates.push('/opt/homebrew/bin/llama-server', '/usr/local/bin/llama-server');
  }
  return candidates.find(isFile);
}

/** GGUF models already downloaded on this machine (Hugging Face and llama.cpp caches), newest first. */
export function findModels(): string[] {
  const found: string[] = [];
  const addGguf = (dir: string, depth: number) => {
    for (const name of listDir(dir)) {
      const full = path.join(dir, name);
      if (name.toLowerCase().endsWith('.gguf') && !name.toLowerCase().includes('mmproj') && isFile(full)) {
        found.push(full);
      } else if (depth > 0 && !name.includes('.')) {
        addGguf(full, depth - 1);
      }
    }
  };
  const hfHome = process.env.HF_HOME ?? path.join(os.homedir(), '.cache', 'huggingface');
  const hub = process.env.HF_HUB_CACHE ?? path.join(hfHome, 'hub');
  for (const repo of listDir(hub).filter((n) => n.startsWith('models--'))) {
    for (const snapshot of listDir(path.join(hub, repo, 'snapshots'))) {
      addGguf(path.join(hub, repo, 'snapshots', snapshot), 1);
    }
  }
  const llamaCache = process.env.LLAMA_CACHE ?? (isWindows && process.env.LOCALAPPDATA
    ? path.join(process.env.LOCALAPPDATA, 'llama.cpp')
    : path.join(os.homedir(), '.cache', 'llama.cpp'));
  addGguf(llamaCache, 0);
  const mtime = (p: string) => {
    try {
      return fs.statSync(p).mtimeMs;
    } catch {
      return 0;
    }
  };
  return [...new Set(found)].sort((a, b) => mtime(b) - mtime(a));
}

export class ModelServer {
  private proc: ChildProcess | undefined;
  private tail: string[] = [];

  constructor(private readonly log: (line: string) => void) {}

  get startedByUs(): boolean {
    return !!this.proc && this.proc.exitCode === null;
  }

  async start(baseURL: string, s: ServerSettings, progress: (msg: string) => void): Promise<{ ok: boolean; message: string }> {
    const exe = findLlamaServer(s.executable);
    if (!exe) {
      return {
        ok: false,
        message: 'Could not find llama-server. Install llama.cpp (winget install ggml.llamacpp), or set "lca.server.executable".',
      };
    }
    const model = s.model || findModels()[0];
    if (!model) {
      return {
        ok: false,
        message: 'No downloaded .gguf model was found. Download one (e.g. llama-server -hf <user>/<repo>:Q8_0) or set "lca.server.model".',
      };
    }
    if (!isFile(model)) {
      return { ok: false, message: `Model file not found: ${model}` };
    }
    const port = new URL(baseURL).port || '8080';
    const args = [
      '-m', model, '--host', '127.0.0.1', '--port', port, '--jinja',
      '-ngl', String(s.gpuLayers), '-c', String(s.contextSize), '-np', String(s.parallel), ...s.extraArgs,
    ];
    this.log(`$ ${exe} ${args.join(' ')}`);
    this.tail = [];
    let exitReason: string | undefined;
    const proc = spawn(exe, args, { windowsHide: true });
    this.proc = proc;
    const onData = (chunk: Buffer) => {
      for (const line of chunk.toString('utf8').split(/\r?\n/)) {
        if (line.trim()) {
          this.log(`[llama-server] ${line}`);
          this.tail.push(line);
          if (this.tail.length > 20) {
            this.tail.shift();
          }
        }
      }
    };
    proc.stdout?.on('data', onData);
    proc.stderr?.on('data', onData);
    proc.on('error', (err) => (exitReason = `Could not start llama-server: ${err.message}`));
    proc.on('exit', (code) => {
      exitReason ??= `llama-server exited with code ${code}`;
      if (this.proc === proc) {
        this.proc = undefined;
      }
    });

    progress(`Loading ${path.basename(model)}… (first start can take a minute)`);
    const deadline = Date.now() + READY_TIMEOUT_MS;
    while (Date.now() < deadline) {
      if (exitReason) {
        const hint = this.tail.some((l) => /bind|address already in use|port/i.test(l))
          ? `\nPort ${port} may be used by another program.`
          : this.tail.some((l) => /out of (device )?memory|OutOfDeviceMemory|failed to allocate/i.test(l))
            ? '\nThe model does not fit in GPU memory. Lower "lca.server.gpuLayers" or "lca.server.contextSize".'
            : '';
        return { ok: false, message: `${exitReason}.${hint}\n${this.tail.slice(-5).join('\n')}` };
      }
      if (await isReachable(baseURL, 1500)) {
        return { ok: true, message: '' };
      }
      await new Promise((r) => setTimeout(r, 700));
    }
    this.stop();
    return { ok: false, message: 'llama-server did not become ready within 3 minutes. See "Show Agent Logs".' };
  }

  stop(): void {
    const proc = this.proc;
    this.proc = undefined;
    if (!proc || proc.exitCode !== null || proc.pid === undefined) {
      return;
    }
    if (isWindows) {
      spawnSync('taskkill', ['/T', '/F', '/PID', String(proc.pid)], { windowsHide: true });
    } else {
      proc.kill();
    }
  }
}
