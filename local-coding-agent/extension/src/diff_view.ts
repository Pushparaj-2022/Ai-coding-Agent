// Shows a proposed file change in VS Code's diff editor before the user approves it.
import * as path from 'path';
import * as vscode from 'vscode';

export const DIFF_SCHEME = 'lca-diff';

export interface ProposedChange {
  path: string;
  action: 'create' | 'modify' | 'delete';
  is_dir: boolean;
  before: string;
  after: string;
  diff: string;
}

export class DiffView implements vscode.TextDocumentContentProvider {
  private readonly contents = new Map<string, string>();
  private readonly changes = new Map<string, ProposedChange>();

  register(): vscode.Disposable {
    return vscode.workspace.registerTextDocumentContentProvider(DIFF_SCHEME, this);
  }

  provideTextDocumentContent(uri: vscode.Uri): string {
    return this.contents.get(uri.path) ?? '';
  }

  remember(requestId: string, change: ProposedChange): void {
    this.changes.set(requestId, change);
    if (this.changes.size > 200) {
      const oldest = this.changes.keys().next().value;
      if (oldest !== undefined) {
        this.changes.delete(oldest);
      }
    }
  }

  async open(requestId: string): Promise<void> {
    const change = this.changes.get(requestId);
    if (!change || change.is_dir) {
      return;
    }
    const name = path.posix.basename(change.path);
    const left = `/${requestId}/before/${name}`;
    const right = `/${requestId}/after/${name}`;
    this.contents.set(left, change.before);
    this.contents.set(right, change.after);
    const verb = change.action === 'create' ? 'new file' : change.action === 'delete' ? 'delete' : 'proposed edit';
    await vscode.commands.executeCommand(
      'vscode.diff',
      vscode.Uri.from({ scheme: DIFF_SCHEME, path: left }),
      vscode.Uri.from({ scheme: DIFF_SCHEME, path: right }),
      `${change.path} (${verb})`,
      { preview: true, preserveFocus: true },
    );
  }
}
