// Answers the agent's diagnostics_request with VS Code's problems (type checker, linter, compiler) for the given files.
import * as path from 'path';
import * as vscode from 'vscode';

export interface AgentDiagnostic {
  path: string;
  line: number;
  column: number;
  severity: 'error' | 'warning';
  message: string;
  source: string;
  code: string;
}

const FIRST_CHANGE_MS = 2500; // no diagnostics change at all within this time: nothing new is coming
const SETTLE_MS = 600; // quiet this long after the last change: the language services are done
const MAX_WAIT_MS = 8000;
const MAX_ITEMS = 100;

/** Resolves once diagnostics for `wanted` stop changing (or nothing changes, or the wait limit is reached). */
function waitForSettle(wanted: Set<string>): Promise<void> {
  return new Promise((resolve) => {
    let quiet: NodeJS.Timeout = setTimeout(done, FIRST_CHANGE_MS);
    const cap = setTimeout(done, MAX_WAIT_MS);
    const sub = vscode.languages.onDidChangeDiagnostics((e) => {
      if (e.uris.some((u) => wanted.has(u.toString()))) {
        clearTimeout(quiet);
        quiet = setTimeout(done, SETTLE_MS);
      }
    });
    function done(): void {
      clearTimeout(quiet);
      clearTimeout(cap);
      sub.dispose();
      resolve();
    }
  });
}

function codeOf(d: vscode.Diagnostic): string {
  const c = d.code;
  if (c === undefined) {
    return '';
  }
  return typeof c === 'object' ? String(c.value) : String(c);
}

/** Problems for `paths` (relative to `root`), or for the whole workspace when `paths` is empty. */
export async function collectDiagnostics(root: string, paths: string[]): Promise<AgentDiagnostic[]> {
  const inside = (fsPath: string) => {
    const rel = path.relative(root, fsPath);
    return !!rel && !rel.startsWith('..') && !path.isAbsolute(rel);
  };
  const uris = paths.map((p) => vscode.Uri.file(path.resolve(root, p))).filter((u) => inside(u.fsPath));
  let entries: [vscode.Uri, readonly vscode.Diagnostic[]][];
  if (uris.length) {
    const settled = waitForSettle(new Set(uris.map((u) => u.toString())));
    // most language servers only check open documents: opening one (without showing it) gets it checked
    await Promise.all(uris.map((u) => vscode.workspace.openTextDocument(u).then(() => undefined, () => undefined)));
    await settled;
    entries = uris.map((u) => [u, vscode.languages.getDiagnostics(u)]);
  } else {
    entries = vscode.languages.getDiagnostics().filter(([u]) => u.scheme === 'file' && inside(u.fsPath));
  }
  const items: AgentDiagnostic[] = [];
  for (const [uri, diags] of entries) {
    const rel = path.relative(root, uri.fsPath).split(path.sep).join('/');
    for (const d of diags) {
      if (d.severity !== vscode.DiagnosticSeverity.Error && d.severity !== vscode.DiagnosticSeverity.Warning) {
        continue;
      }
      items.push({
        path: rel,
        line: d.range.start.line + 1,
        column: d.range.start.character + 1,
        severity: d.severity === vscode.DiagnosticSeverity.Error ? 'error' : 'warning',
        message: d.message,
        source: d.source ?? '',
        code: codeOf(d),
      });
    }
  }
  // errors first, so a long list never hides them
  items.sort((a, b) => (a.severity === b.severity ? 0 : a.severity === 'error' ? -1 : 1));
  return items.slice(0, MAX_ITEMS);
}
