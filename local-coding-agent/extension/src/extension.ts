import * as vscode from 'vscode';
import { DiffView } from './diff_view';
import { ChatViewProvider } from './sidebar';

let provider: ChatViewProvider | undefined;

export function activate(context: vscode.ExtensionContext): void {
  const output = vscode.window.createOutputChannel('Local Coding Agent');
  const statusItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
  statusItem.text = '$(hubot) Agent';
  statusItem.command = 'lca.focus';
  statusItem.show();

  const diffs = new DiffView();
  provider = new ChatViewProvider(context, diffs, output, statusItem);
  const chat = provider;

  context.subscriptions.push(
    output,
    statusItem,
    diffs.register(),
    vscode.window.registerWebviewViewProvider('lca.chat', chat, { webviewOptions: { retainContextWhenHidden: true } }),
    vscode.commands.registerCommand('lca.focus', () => vscode.commands.executeCommand('lca.chat.focus')),
    vscode.commands.registerCommand('lca.newTask', async () => {
      await vscode.commands.executeCommand('lca.chat.focus');
      chat.newChat();
      chat.focusInput();
    }),
    vscode.commands.registerCommand('lca.stop', () => chat.stop()),
    vscode.commands.registerCommand('lca.restart', () => chat.restart()),
    vscode.commands.registerCommand('lca.showLogs', () => output.show(true)),
    vscode.commands.registerCommand('lca.askAboutSelection', async () => {
      await vscode.commands.executeCommand('lca.chat.focus');
      chat.enableEditorContext();
      chat.focusInput();
    }),
    vscode.window.onDidChangeActiveTextEditor(() => chat.pushEditorContext()),
    vscode.window.onDidChangeTextEditorSelection(() => chat.pushEditorContext()),
    vscode.workspace.onDidChangeConfiguration((e) => chat.onConfigChanged(e)),
  );
}

export function deactivate(): void {
  provider?.dispose();
}
