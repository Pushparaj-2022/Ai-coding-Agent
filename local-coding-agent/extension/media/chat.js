// Chat panel UI: renders agent protocol events (docs/PROTOCOL.md) relayed by the extension.
(() => {
  const vscode = acquireVsCodeApi();
  const $ = (id) => document.getElementById(id);
  const log = $('log');
  const input = $('input');
  const sendBtn = $('sendBtn');
  const modeSel = $('mode');
  const chip = $('contextChip');
  const working = $('working');
  const scrollBtn = $('scrollBtn');
  const welcomeHtml = $('welcome').outerHTML;

  // ------------------------------------------------------------ icons and tool metadata
  const ICONS = {
    file: '<path d="M4 1.5h5l3 3v10H4z"/><path d="M9 1.5v3h3"/>',
    folder: '<path d="M1.5 4h4l1.5 1.5h7.5v8h-13z"/>',
    search: '<circle cx="7" cy="7" r="4.5"/><path d="m10.5 10.5 3.5 3.5"/>',
    map: '<path d="M1.5 3.5 5.5 2l5 1.5 4-1.5v10.5l-4 1.5-5-1.5-4 1.5z"/><path d="M5.5 2v10.5M10.5 3.5V14"/>',
    pencil: '<path d="M11 2.5 13.5 5 6 12.5H3.5V10z"/>',
    plus: '<path d="M8 3v10M3 8h10"/>',
    trash: '<path d="M2.5 4h11M6 4V2.5h4V4M4 4l.7 9.5h6.6L12 4"/>',
    terminal: '<rect x="1.5" y="2.5" width="13" height="11" rx="1.5"/><path d="m4 6 2.5 2L4 10M8 10.5h4"/>',
    stop: '<rect x="4" y="4" width="8" height="8" rx="1"/>',
    chev: '<path d="m6 4 4 4-4 4"/>',
    explore: '<circle cx="6.5" cy="6.5" r="4"/><path d="m9.5 9.5 4 4"/><path d="M4.5 6.5h4"/>',
    shield: '<path d="M8 1.5 13 3.5v4c0 3-2.2 5.6-5 7-2.8-1.4-5-4-5-7v-4z"/>',
    list: '<path d="M6 4h7.5M6 8h7.5M6 12h7.5M2.5 4h.5M2.5 8h.5M2.5 12h.5"/>',
    help: '<circle cx="8" cy="8" r="6"/><path d="M6.3 6.2a1.8 1.8 0 1 1 2.4 1.7c-.5.2-.7.6-.7 1.1v.4M8 11.6h.01"/>',
    alert: '<path d="M8 2 14.5 13.5h-13z"/><path d="M8 6.5v3.2M8 11.7h.01"/>',
    globe: '<circle cx="8" cy="8" r="6"/><path d="M2 8h12M8 2c1.8 1.7 2.7 3.7 2.7 6S9.8 12.3 8 14c-1.8-1.7-2.7-3.7-2.7-6S6.2 3.7 8 2z"/>',
    plug: '<path d="M6 2v3M10 2v3M4.5 5h7v2.5a3.5 3.5 0 0 1-7 0zM8 11v3"/>',
    undo: '<path d="M4 6.5h6a3.5 3.5 0 0 1 0 7H7"/><path d="M6.5 4 4 6.5 6.5 9"/>',
  };
  const svg = (name, cls = '') => `<svg class="${cls}" viewBox="0 0 16 16">${ICONS[name] || ''}</svg>`;
  const TOOL = {
    read_file: ['Read', 'file'], list_files: ['List', 'folder'], search_code: ['Search', 'search'],
    repo_overview: ['Overview', 'map'], write_file: ['Write', 'plus'], edit_file: ['Update', 'pencil'], multi_edit: ['Update', 'pencil'],
    delete_file: ['Delete', 'trash'], run_command: ['Bash', 'terminal'], command_output: ['BashOutput', 'terminal'],
    stop_command: ['KillShell', 'stop'], update_plan: ['Update Todos', 'list'], ask_user: ['AskUser', 'help'],
    get_diagnostics: ['Diagnostics', 'alert'], task: ['Task', 'explore'], exit_plan_mode: ['ExitPlanMode', 'list'], load_skill: ['Skill', 'map'], web_fetch: ['Fetch', 'globe'], web_search: ['WebSearch', 'search'],
  };
  const EXPLORE = new Set(['read_file', 'list_files', 'search_code', 'repo_overview']);
  const AGENT_COLORS = { planner: 'var(--planner)', explore: 'var(--planner)', general: 'var(--coder)', coder: 'var(--coder)', tester: 'var(--tester)', verify: 'var(--verify)', debugger: 'var(--debugger)', reviewer: 'var(--reviewer)' };

  // ------------------------------------------------------------ state
  let busy = false;
  let stick = true;
  let turn = null; // container of the current exchange
  let stageBody = null; // container of the current /build stage
  let textBlock = null;
  let planCard = null; // the current task's plan checklist, updated in place
  let group = null; // current "Exploring" group
  const tools = new Map(); // call_id -> row element
  let history = [];
  let historyIndex = -1;

  // ------------------------------------------------------------ helpers
  const el = (tag, cls, text) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  };
  const esc = (s) => String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  const badge = (agent) => (agent ? `<span class="agentBadge ${esc(agent)}">${esc(agent)}</span>` : '');
  // the first n non-empty lines, with "… +k lines" for the rest (Claude Code style result preview)
  const firstLines = (text, n) => {
    const all = String(text || '').split('\n').filter((l) => l.trim());
    return all.slice(0, n).join('\n') + (all.length > n ? `\n… +${all.length - n} lines (click to expand)` : '');
  };
  const fmtSecs = (ms) => (ms < 1000 ? `${ms}ms` : `${(ms / 1000).toFixed(ms < 10000 ? 1 : 0)}s`);

  const atBottom = () => log.scrollHeight - log.scrollTop - log.clientHeight < 60;
  log.addEventListener('scroll', () => {
    stick = atBottom();
    scrollBtn.classList.toggle('hidden', stick);
  });
  const scroll = () => {
    if (stick) log.scrollTop = log.scrollHeight;
  };
  scrollBtn.addEventListener('click', () => {
    stick = true;
    log.scrollTop = log.scrollHeight;
    scrollBtn.classList.add('hidden');
  });

  function inline(s) {
    return s
      .replace(/`([^`]+)`/g, '<code>$1</code>')
      .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
      .replace(/(^|[^*\w])\*([^*\n]+)\*/g, '$1<em>$2</em>');
  }

  function blocks(t) {
    let html = '';
    let list = null;
    let para = [];
    let table = [];
    const flushP = () => {
      if (para.length) html += `<p>${inline(para.join('<br>'))}</p>`;
      para = [];
    };
    const flushL = () => {
      if (list) html += `</${list}>`;
      list = null;
    };
    const flushT = () => {
      if (!table.length) return;
      const rows = table.filter((r) => !/^\|?\s*:?-{2,}/.test(r));
      html += '<table>' + rows.map((r, i) => {
        const cells = r.replace(/^\||\|$/g, '').split('|').map((c) => inline(c.trim()));
        const tag = i === 0 ? 'th' : 'td';
        return '<tr>' + cells.map((c) => `<${tag}>${c}</${tag}>`).join('') + '</tr>';
      }).join('') + '</table>';
      table = [];
    };
    for (const line of t.split('\n')) {
      let m;
      if (/^\s*\|.*\|\s*$/.test(line)) {
        flushP(); flushL(); table.push(line.trim());
        continue;
      }
      flushT();
      if ((m = /^(#{1,4})\s+(.*)/.exec(line))) {
        flushP(); flushL();
        const n = Math.min(m[1].length + 2, 6);
        html += `<h${n}>${inline(m[2])}</h${n}>`;
      } else if ((m = /^\s*[-*]\s+(.*)/.exec(line))) {
        flushP();
        if (list !== 'ul') { flushL(); html += '<ul>'; list = 'ul'; }
        html += `<li>${inline(m[1])}</li>`;
      } else if ((m = /^\s*\d+[.)]\s+(.*)/.exec(line))) {
        flushP();
        if (list !== 'ol') { flushL(); html += '<ol>'; list = 'ol'; }
        html += `<li>${inline(m[1])}</li>`;
      } else if (!line.trim()) {
        flushP(); flushL();
      } else {
        flushL();
        para.push(line);
      }
    }
    flushT(); flushP(); flushL();
    return html;
  }

  function markdown(src) {
    return src
      .split(/(```[\s\S]*?(?:```|$))/g)
      .map((part) => {
        if (!part.startsWith('```')) return blocks(esc(part));
        const m = /^```([\w+-]*)\n?([\s\S]*?)(?:```)?$/.exec(part);
        return `<div class="code"><div class="codeHead"><span>${esc(m[1] || 'code')}</span><button class="copy">Copy</button></div><pre><code>${esc(m[2])}</code></pre></div>`;
      })
      .join('');
  }

  function linkFiles(root) {
    for (const code of root.querySelectorAll('p code, li code, td code')) {
      const m = /^([\w@.\/\\-]+\.[\w]+)(?::(\d+))?/.exec(code.textContent);
      if (m && !/^https?:/.test(m[1])) {
        code.classList.add('filelink');
        code.dataset.path = m[1];
        if (m[2]) code.dataset.line = m[2];
      }
    }
  }

  function diffHtml(diff, maxLines) {
    const lines = diff.replace(/\n$/, '').split('\n');
    const shown = lines.slice(0, maxLines).map((l) => {
      const cls = l.startsWith('+++') || l.startsWith('---') ? 'dh' : l.startsWith('+') ? 'add' : l.startsWith('-') ? 'del' : l.startsWith('@@') ? 'hunk' : '';
      return `<span class="${cls}">${esc(l) || ' '}</span>`;
    });
    if (lines.length > maxLines) shown.push(`<span class="dh">… ${lines.length - maxLines} more lines</span>`);
    return `<pre class="diff">${shown.join('')}</pre>`;
  }

  // ------------------------------------------------------------ containers
  function ensureTurn() {
    if (!turn) {
      $('welcome')?.remove();
      turn = el('div', 'turn');
      log.appendChild(turn);
    }
    return turn;
  }
  // every agent reply in a turn sits under one "Agent" header
  function replyArea() {
    const t = ensureTurn();
    let r = t.querySelector(':scope > .reply');
    if (!r) {
      r = el('div', 'reply');
      r.innerHTML = '<div class="replyHead"><span class="avatar"><svg viewBox="0 0 16 16"><path d="M8 1.8 9.6 6.4 14.2 8 9.6 9.6 8 14.2 6.4 9.6 1.8 8 6.4 6.4z"/></svg></span><span class="who">Agent</span></div><div class="replyBody"></div>';
      t.appendChild(r);
    }
    return r.querySelector('.replyBody');
  }
  const target = () => stageBody || replyArea();

  // typing indicator shown until the first text / tool of a turn arrives
  function removeThinking() {
    for (const n of document.querySelectorAll('.thinking')) n.remove();
  }
  function showThinking(label) {
    removeThinking();
    const d = el('div', 'thinking');
    d.innerHTML = `<span class="star"></span><span class="shimmer">${esc(label || 'Thinking')}…</span>`;
    target().appendChild(d);
    scroll();
  }
  const endGroup = () => {
    group = null;
  };

  function appendText(text, agent) {
    endGroup();
    if (!textBlock || textBlock.agent !== (agent || null)) {
      textBlock = el('div', 'md' + (agent ? ' sub' : ''));
      textBlock.raw = '';
      textBlock.agent = agent || null;
      target().appendChild(textBlock);
    }
    const block = textBlock;
    block.raw += text;
    if (!block.pending) {
      block.pending = true;
      requestAnimationFrame(() => {
        block.pending = false;
        block.innerHTML = markdown(block.raw.replace(/^\s+/, ''));
        linkFiles(block);
        scroll();
      });
    }
  }

  // ------------------------------------------------------------ tool rows
  function argSummary(name, a) {
    if (!a) return '';
    if (a.command) return a.command;
    if (name === 'read_file' && a.path && (a.start_line > 1 || a.end_line)) return `${a.path}:${a.start_line || 1}${a.end_line ? '-' + a.end_line : ''}`;
    if (a.pattern) return `"${a.pattern}"${a.path && a.path !== '.' ? ' in ' + a.path : ''}${a.glob ? ' · ' + a.glob : ''}`;
    if (a.path) return a.path + (a.glob ? ' · ' + a.glob : '');
    if (a.glob) return a.glob;
    if (a.id) return a.id;
    return '';
  }

  function makeRow(ev) {
    // MCP tools are named mcp__<server>__<tool>
    const mcp = /^mcp__(.+?)__(.+)$/.exec(ev.name);
    const [verb, icon] = TOOL[ev.name] || (mcp ? [`${mcp[1]} · ${mcp[2]}`, 'plug'] : [ev.name, 'terminal']);
    const row = el('div', 'tool running');
    row.dataset.name = ev.name;
    const head = el('div', 'toolHead');
    const arg = esc(argSummary(ev.name, ev.arguments));
    const fileArg = ['read_file', 'write_file', 'edit_file'].includes(ev.name) && ev.arguments?.path;
    // Claude Code style: ● Verb(argument), with the result on a ⎿ line below
    const desc = ev.name === 'run_command' && ev.arguments?.description ? esc(ev.arguments.description) : '';
    head.innerHTML =
      '<span class="bullet"></span>' +
      (group || stageBody ? '' : badge(ev.agent)) + // inside a group/stage the agent is already shown
      svg(icon, 'tIcon') +
      `<span class="tname">${verb}</span>` +
      (arg ? `<span class="paren">(</span><span class="targ${fileArg ? ' filelink' : ''}"${fileArg ? ` data-path="${esc(ev.arguments.path)}"` : ''} title="${desc || arg}">${arg}</span><span class="paren">)</span>` : '') +
      `<span class="tmeta"><span class="spinner"></span></span>` +
      svg('chev', 'chev');
    head.addEventListener('click', (e) => {
      if (!e.target.closest('.filelink')) row.classList.toggle('open');
    });
    const note = el('div', 'tnote');
    if (desc) note.textContent = ev.arguments.description;
    else note.classList.add('hidden');
    const body = el('div', 'tbody');
    row.append(head, note, body);
    return row;
  }

  function exploreGroup(ev) {
    if (group && group.agent === (ev.agent || null) && group.el.isConnected && group.el === target().lastElementChild) return group;
    const g = el('div', 'tool group running open');
    g.innerHTML =
      `<div class="toolHead"><span class="bullet"></span>${stageBody ? '' : badge(ev.agent)}${svg('explore', 'tIcon')}<span class="tname">Exploring</span>` +
      `<span class="targ"></span><span class="tmeta"><span class="spinner"></span></span>${svg('chev', 'chev')}</div><div class="tbody"></div>`;
    g.querySelector('.toolHead').addEventListener('click', () => g.classList.toggle('open'));
    target().appendChild(g);
    group = { el: g, agent: ev.agent || null, body: g.querySelector('.tbody'), counts: {} };
    return group;
  }

  function updateGroup(gr) {
    const c = gr.counts;
    const parts = [];
    if (c.read_file) parts.push(`${c.read_file} file${c.read_file > 1 ? 's' : ''}`);
    if (c.search_code) parts.push(`${c.search_code} search${c.search_code > 1 ? 'es' : ''}`);
    if (c.list_files) parts.push(`${c.list_files} listing${c.list_files > 1 ? 's' : ''}`);
    if (c.repo_overview) parts.push('overview');
    gr.el.querySelector(':scope > .toolHead .targ').textContent = parts.join(', ');
    const running = gr.body.querySelector('.tool.running');
    const failed = gr.body.querySelectorAll('.tool.failed').length;
    gr.el.classList.toggle('running', !!running);
    const meta = gr.el.querySelector(':scope > .toolHead .tmeta');
    if (!running) {
      gr.el.querySelector(':scope > .toolHead .tname').textContent = 'Explored';
      meta.innerHTML = failed ? `<span class="pill warn">${failed} failed</span>` : '';
      gr.el.classList.remove('open'); // collapse once done: details stay one click away
    } else {
      meta.innerHTML = '<span class="spinner"></span>';
    }
  }

  function toolStarted(ev) {
    textBlock = null;
    let parent = target();
    let gr = null;
    if (EXPLORE.has(ev.name)) {
      gr = exploreGroup(ev);
      parent = gr.body;
    } else {
      endGroup();
    }
    const row = makeRow(ev);
    row.group = gr;
    row.args = ev.arguments || {}; // tool_call_finished does not repeat the arguments
    parent.appendChild(row);
    if (gr) {
      gr.counts[ev.name] = (gr.counts[ev.name] || 0) + 1;
      updateGroup(gr);
    }
    if (ev.name === 'run_command') {
      const pre = el('pre', 'out term');
      pre.innerHTML = `<span class="prompt">$ </span>${esc(ev.arguments?.command || '')}\n`;
      row.querySelector('.tbody').appendChild(pre);
    }
    tools.set(ev.call_id, row);
    setWorking(`${ev.agent ? ev.agent + ' · ' : ''}${(TOOL[ev.name] || [ev.name])[0]} ${argSummary(ev.name, ev.arguments)}`);
    scroll();
  }

  function toolOutput(ev) {
    const row = tools.get(ev.call_id);
    const pre = row?.querySelector('.out.term');
    if (!pre) return;
    row.classList.add('open');
    pre.append(document.createTextNode(ev.text));
    if (pre.textContent.length > 40000) pre.textContent = pre.textContent.slice(-30000);
    pre.scrollTop = pre.scrollHeight;
    scroll();
  }

  function addBackgroundButton(callId) {
    const head = tools.get(callId)?.querySelector(':scope > .toolHead');
    if (!head) return;
    const b = el('button', 'bgBtn', 'Background');
    b.title = 'Keep it running in the background and let the agent continue (Ctrl+B)';
    b.addEventListener('click', (e) => {
      e.stopPropagation();
      vscode.postMessage({ type: 'background' });
      b.disabled = true;
      b.textContent = 'Moving…';
    });
    head.appendChild(b);
  }

  function toolFinished(ev) {
    const row = tools.get(ev.call_id);
    if (!row) return;
    row.querySelector(':scope > .toolHead .bgBtn')?.remove();
    row.classList.remove('running');
    row.classList.add(ev.ok ? 'ok' : 'failed');
    row.querySelector(':scope > .perm')?.remove();
    const meta = row.querySelector(':scope > .toolHead .tmeta');
    const note = row.querySelector(':scope > .tnote');
    const body = row.querySelector(':scope > .tbody');
    const output = ev.output || '';
    const lines = output.split('\n');
    const dur = ev.duration_ms == null ? "" : `<span class="dur">${fmtSecs(ev.duration_ms)}</span>`;
    const showNote = (text) => {
      note.textContent = text;
      note.classList.remove('hidden');
    };

    if (/^Permission denied:/.test(output)) {
      const said = /said: (.*?)\. Follow/.exec(output);
      meta.innerHTML = '<span class="pill bad">denied</span>';
      showNote(said ? `You said: ${said[1]}` : 'You declined this action.');
      body.innerHTML = '';
    } else if (/^Not run: /.test(output)) {
      meta.innerHTML = '<span class="pill warn">redirected</span>';
      showNote(output.replace(/^Not run: /, '').split('.')[0] + '.');
      body.innerHTML = '';
      row.classList.replace('failed', 'ok');
    } else if (ev.name === 'run_command' && ev.exit_code !== null && ev.exit_code !== undefined) {
      meta.innerHTML = `<span class="pill ${ev.exit_code === 0 ? 'ok' : 'bad'}">${ev.exit_code === 0 ? 'exit 0' : 'exit ' + ev.exit_code}</span>${dur}`;
      const pre = body.querySelector('.out.term') || body.appendChild(el('pre', 'out term'));
      const rest = lines.slice(1).join('\n').replace(/\n\[hint\][^\n]*/g, '').trim();
      pre.innerHTML = `<span class="prompt">$ </span>${esc(row.args.command || '')}\n${esc(rest || '(no output)')}`;
      const hints = lines.filter((l) => l.startsWith('[hint]')).map((l) => l.slice(7));
      if (hints.length) showNote('Hint: ' + hints.join(' '));
      else showNote(firstLines(rest, 3) || '(no output)');
      row.classList.toggle('open', !ev.ok);
    } else if (/^(Created|Rewrote|Edited) /.test(output) && output.includes('\n@@')) {
      const stats = /\(\+(\d+) -(\d+) lines\)/.exec(lines[0]);
      meta.innerHTML = '';
      if (stats) {
        const n = (k, w) => `${k} ${w}${k === '1' ? '' : 's'}`;
        showNote(`${lines[0].split(' ')[0]} ${row.args.path || ''} with ${n(stats[1], 'addition')} and ${n(stats[2], 'removal')}`);
      }
      body.innerHTML = diffHtml(output.slice(output.indexOf('\n') + 1), 300);
      row.classList.add('open');
    } else if (!ev.ok) {
      meta.innerHTML = '<span class="pill bad">error</span>';
      showNote(lines[0].replace(/^Error: /, ''));
      if (lines.length > 1) body.innerHTML = `<pre class="out">${esc(output)}</pre>`;
    } else {
      meta.innerHTML = ev.name === 'read_file' ? '' : dur;
      const rest = lines[0].trim() === (ev.summary || '').trim() ? lines.slice(1).join('\n').trim() : output;
      body.innerHTML = rest ? `<pre class="out">${esc(rest)}</pre>` : '';
      const range = /\(lines (\d+)-(\d+)/.exec(lines[0]);
      if (ev.name === 'read_file' && range) showNote(`Read ${+range[2] - +range[1] + 1} lines`);
      else if (ev.name === 'update_plan' || ev.name === 'ask_user') note.classList.add('hidden');
      else showNote(ev.summary || firstLines(rest, 1) || 'Done');
    }
    if (!body.innerHTML) row.querySelector(':scope > .toolHead .chev')?.classList.add('hidden');
    if (row.group) updateGroup(row.group);
    scroll();
  }

  // ------------------------------------------------------------ approvals
  function planApproval(ev) {
    endGroup();
    textBlock = null;
    const card = el('div', 'perm planApproval');
    card.dataset.id = ev.request_id;
    card.innerHTML = `
      <div class="permTitle">${svg('list')}Ready to implement this plan?</div>
      <div class="md planBody">${markdown(ev.plan || '')}</div>
      <div class="permBtns">
        <button class="btn primary" data-m="auto-edit">Yes, auto-accept edits</button>
        <button class="btn" data-m="ask">Yes, approve each edit</button>
        <button class="btn danger" data-c="no">No</button>
      </div>
      <div class="permFeedback"><input placeholder="Or tell the agent what to change in the plan…"><button class="btn" data-c="feedback">Keep planning</button></div>`;
    linkFiles(card.querySelector('.planBody'));
    const fb = card.querySelector('input');
    const reply = (approved, mode, feedback) => {
      vscode.postMessage({ type: 'approval', request_id: ev.request_id, approved, mode, feedback: feedback || '' });
      resolveCard(card, approved ? `Plan approved (${mode === 'auto-edit' ? 'auto-accept edits' : 'approve each edit'})` : feedback ? `Keep planning: ${feedback}` : 'Plan rejected', approved);
      setWorking(approved ? 'Implementing the plan…' : feedback ? 'Revising the plan…' : 'Stopping…');
    };
    card.addEventListener('click', (e) => {
      const b = e.target.closest('button');
      if (!b) return;
      if (b.dataset.m) return reply(true, b.dataset.m);
      if (b.dataset.c === 'no') return reply(false);
      if (b.dataset.c === 'feedback') return fb.value.trim() ? reply(false, undefined, fb.value.trim()) : fb.focus();
    });
    fb.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && fb.value.trim()) reply(false, undefined, fb.value.trim());
    });
    target().appendChild(card);
    setWorking('Waiting for your approval of the plan');
    card.querySelector('.btn.primary').focus({ preventScroll: true });
    stick = true;
    scroll();
  }

  function approvalRequested(ev) {
    if (ev.plan) return planApproval(ev);
    textBlock = null;
    const change = ev.change;
    const isCmd = !!ev.command;
    const sensitive = ev.level === 'sensitive';
    const card = el('div', 'perm' + (sensitive ? ' sensitive' : ''));
    card.dataset.id = ev.request_id;
    const title = isCmd ? (sensitive ? 'Allow this risky command?' : 'Allow this command?') : (change?.action === 'delete' ? 'Delete this?' : 'Apply this change?');
    card.innerHTML = `
      <div class="permTitle">${svg('shield')}${title}</div>
      ${ev.reason ? `<div class="permReason">This command ${esc(ev.reason)}.</div>` : ''}
      ${isCmd ? `<pre class="out term permCmd"><span class="prompt">$ </span>${esc(ev.command)}</pre>` : `<div class="permDetail">${esc(ev.description)}</div>`}
      ${change && change.diff ? `<div class="permDiff">${diffHtml(change.diff, 60)}</div>` : ''}
      <div class="permBtns">
        <button class="btn primary" data-c="allow">Allow</button>
        ${ev.suggested_rule ? `<button class="btn" data-c="forever" title="Saves the rule ${esc(ev.suggested_rule)} for this project">Always allow <code>${esc(ev.suggested_rule.replace(/^run_command\((.*):\*\)$/, '$1'))}</code></button>` : ''}
        ${sensitive ? '' : `<button class="btn" data-c="always">Allow all ${isCmd ? 'commands' : 'edits'} this session</button>`}
        <button class="btn danger" data-c="deny">Deny</button>
        ${change && !change.is_dir ? '<button class="btn link" data-c="diff">Open diff</button>' : ''}
      </div>
      <div class="permFeedback"><input placeholder="Or tell the agent what to do instead…"><button class="btn" data-c="feedback">Send</button></div>`;
    const fb = card.querySelector('input');
    const host = [...tools.values()].reverse().find((r) => r.classList.contains('running') && r.isConnected);
    const reply = (approved, remember, feedback, always) => {
      vscode.postMessage({ type: 'approval', request_id: ev.request_id, approved, remember, feedback: feedback || '', always: !!always });
      if (host) {
        card.remove(); // the tool row itself shows what happened next
      } else {
        resolveCard(card, approved ? 'Allowed' : feedback ? `Denied: ${feedback}` : 'Denied', approved);
      }
      setWorking(approved ? 'Continuing…' : 'Stopping…');
    };
    card.addEventListener('click', (e) => {
      const b = e.target.closest('button');
      if (!b) return;
      const c = b.dataset.c;
      if (c === 'diff') return vscode.postMessage({ type: 'openDiff', request_id: ev.request_id });
      if (c === 'allow') return reply(true, false);
      if (c === 'always') return reply(true, true);
      if (c === 'forever') return reply(true, false, '', true);
      if (c === 'deny') return reply(false, false);
      if (c === 'feedback') return fb.value.trim() ? reply(false, false, fb.value.trim()) : fb.focus();
    });
    fb.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && fb.value.trim()) reply(false, false, fb.value.trim());
    });
    if (host) {
      host.classList.remove('open'); // the card already shows the command / diff
      host.insertBefore(card, host.querySelector(':scope > .tbody'));
      if (host.group) host.group.el.classList.add('open');
    } else {
      target().appendChild(card);
    }
    setWorking('Waiting for your approval');
    card.querySelector('.btn.primary').focus({ preventScroll: true });
    stick = true;
    scroll();
  }

  // ------------------------------------------------------------ questions and the plan
  function questionAsked(ev) {
    endGroup();
    textBlock = null;
    const options = ev.options || [];
    const card = el('div', 'perm question');
    card.dataset.id = ev.request_id;
    const btns = options.map((o, i) => `<button class="btn${i === 0 ? ' primary' : ''}" data-i="${i}">${esc(o)}</button>`).join('');
    card.innerHTML = `
      <div class="permTitle">${svg('help')}<span>${esc(ev.question)}</span></div>
      ${btns ? `<div class="permBtns">${btns}</div>` : ''}
      <div class="permFeedback"><input placeholder="${btns ? 'Or type another answer…' : 'Type your answer…'}"><button class="btn${btns ? '' : ' primary'}" data-c="answer">Send</button></div>`;
    const fb = card.querySelector('input');
    const reply = (text) => {
      vscode.postMessage({ type: 'answer', request_id: ev.request_id, text });
      resolveCard(card, `${ev.question} → ${text}`, true);
      setWorking('Continuing…');
    };
    card.addEventListener('click', (e) => {
      const b = e.target.closest('button');
      if (!b) return;
      if (b.dataset.i !== undefined) return reply(options[Number(b.dataset.i)]);
      if (b.dataset.c === 'answer') return fb.value.trim() ? reply(fb.value.trim()) : fb.focus();
    });
    fb.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && fb.value.trim()) reply(fb.value.trim());
    });
    target().appendChild(card);
    setWorking('Waiting for your answer');
    (card.querySelector('.btn.primary') || fb).focus({ preventScroll: true });
    stick = true;
    scroll();
  }

  const PLAN_MARK = { pending: '☐', in_progress: '☐', done: '☒', skipped: '☒' };
  function planUpdated(ev) {
    endGroup();
    textBlock = null;
    if (!planCard || !planCard.isConnected) {
      planCard = el('div', 'plan');
      target().appendChild(planCard);
    }
    const done = ev.steps.filter((s) => s.status === 'done').length;
    planCard.innerHTML =
      `<div class="planHead"><span class="bullet"></span><span>Update Todos</span><span class="meta">${done}/${ev.steps.length}</span></div>` +
      `<ol class="planList">${ev.steps.map((s) => `<li class="${esc(s.status)}"><span class="mark">${PLAN_MARK[s.status] || '○'}</span><span>${esc(s.step)}</span></li>`).join('')}</ol>`;
    scroll();
  }

  // ------------------------------------------------------------ rewind
  let pendingRewind = null; // shown once the re-rendered history arrives
  function attachRewind(userEl, checkpointId) {
    if (!userEl || !checkpointId || userEl.querySelector('.rewindBtn')) return;
    userEl.dataset.checkpoint = checkpointId;
    const b = el('button', 'rewindBtn');
    b.title = 'Rewind to before this message';
    b.setAttribute('aria-label', b.title);
    b.innerHTML = svg('undo');
    b.addEventListener('click', (e) => {
      e.stopPropagation();
      rewindMenu(checkpointId);
    });
    userEl.appendChild(b);
  }
  function rewindMenu(id) {
    if (busy) {
      ensureTurn();
      notice('Stop the running task before rewinding.', 'warning');
      return;
    }
    const go = (mode) => vscode.postMessage({ type: 'rewind', checkpoint_id: id, mode });
    openMenu(modeMenu, [
      { name: 'Restore code and conversation', desc: 'Undo the agent\'s file changes from this message on, and remove these messages', run: () => go('both') },
      { name: 'Restore conversation only', desc: 'Remove these messages, keep the files as they are now', run: () => go('conversation') },
      { name: 'Restore code only', desc: 'Undo the file changes, keep the conversation', run: () => go('code') },
    ]);
  }
  function rewound(ev) {
    const parts = [];
    if (ev.mode !== 'conversation') {
      parts.push(`Restored ${ev.restored.length} file(s)${ev.removed.length ? `, removed ${ev.removed.length} the agent had created` : ''}.`);
    }
    if (ev.skipped.length) parts.push(`Not restored: ${ev.skipped.join(', ')}.`);
    const text = `${ev.mode === 'code' ? 'Code rewound' : 'Rewound'} to before “${ev.prompt.slice(0, 80)}”. ${parts.join(' ')}`.trim();
    if (ev.mode === 'code') {
      ensureTurn();
      notice(text, 'info');
    } else {
      pendingRewind = { text, prompt: ev.prompt };
    }
  }
  function checkpointsMenu(ev) {
    if (!ev.items.length) {
      ensureTurn();
      notice('Nothing to rewind yet.', 'info');
      return;
    }
    openMenu(modeMenu, ev.items.slice().reverse().slice(0, 15).map((cp) => ({
      name: cp.prompt.length > 70 ? cp.prompt.slice(0, 69) + '…' : cp.prompt,
      desc: `${ago(cp.created)}${cp.files.length ? ` · ${cp.files.length} file(s) changed` : ''}`,
      run: () => rewindMenu(cp.id),
    })));
  }

  // ------------------------------------------------------------ permission rules
  function renderPermissions(ev) {
    let card = document.querySelector('.permsCard');
    if (!card) {
      card = el('div', 'summary permsCard');
      card.addEventListener('click', (e) => {
        const b = e.target.closest('button');
        if (!b) return;
        if (b.dataset.r) vscode.postMessage({ type: 'removePermission', kind: b.dataset.k, rule: b.dataset.r });
        if (b.dataset.c === 'add') addRule(card);
      });
      card.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && e.target.matches('input')) addRule(card);
      });
      ensureTurn().appendChild(card);
    }
    const PILL = { allow: 'ok', ask: 'warn', deny: 'bad' };
    const rows = ev.rules.map((r) =>
      `<div class="ruleRow"><span class="pill ${PILL[r.kind]}">${r.kind}</span><code>${esc(r.rule)}</code><span class="meta">${esc(r.source)}</span>` +
      (r.source === 'saved' ? `<button class="btn link" data-k="${r.kind}" data-r="${esc(r.rule)}">Remove</button>` : '') + '</div>').join('');
    card.innerHTML =
      '<div class="sumHead"><span class="label">Permission rules</span><span class="meta">deny › ask › allow</span></div>' +
      (rows || '<div class="sumMsg">No rules yet. Choose “Always allow” on a command, or add one below.</div>') +
      `<div class="permFeedback"><select aria-label="Rule kind"><option>allow</option><option>ask</option><option>deny</option></select>` +
      `<input placeholder="run_command(npm test:*)  ·  Edit(src/**)  ·  Bash(git push:*)"><button class="btn" data-c="add">Add</button></div>` +
      `<div class="ruleHint">Saved rules: ${esc(ev.file)}. Rules from config files are edited in those files.</div>`;
    scroll();
  }
  function renderMcp(ev) {
    const PILL = { ready: 'ok', starting: 'warn', needs_login: 'warn' };
    const card = el('div', 'summary');
    const rows = ev.servers.map((srv) =>
      `<div class="ruleRow"><span class="pill ${PILL[srv.status] || 'bad'}">${esc(srv.status)}</span><code>${esc(srv.name)}</code>` +
      `<span class="meta">${esc(srv.transport)}, ${esc(srv.source)}</span></div>` +
      (srv.status === 'needs_login' ? `<div class="btnRow"><button class="btn primary" data-login="${esc(srv.name)}">Sign in to ${esc(srv.name)}</button></div>` : '') +
      (srv.error ? `<div class="ruleHint">${esc(srv.error)}</div>` : '') +
      (srv.tools.length ? `<div class="ruleHint">${srv.tools.map(esc).join(' · ')}</div>` : '')).join('');
    card.innerHTML = '<div class="sumHead"><span class="label">MCP servers</span></div>' +
      (rows || '<div class="sumMsg">No MCP servers. Add [mcp.servers.&lt;name&gt;] to ~/.lca/config.toml — see docs/EXTENDING.md.</div>');
    card.addEventListener('click', (e) => {
      const b = e.target.closest('button[data-login]');
      if (!b) return;
      vscode.postMessage({ type: 'mcpLogin', server: b.dataset.login });
      b.disabled = true;
      b.textContent = 'Waiting for the browser…';
    });
    ensureTurn().appendChild(card);
    scroll();
  }

  function addRule(card) {
    const rule = card.querySelector('input').value.trim();
    if (rule) vscode.postMessage({ type: 'addPermission', kind: card.querySelector('select').value, rule });
  }

  function resolveCard(card, label, yes) {
    if (!card || card.classList.contains('done')) return;
    card.classList.add('done', yes ? 'yes' : 'no');
    for (const sel of ['.permBtns', '.permFeedback', '.permDiff', '.permCmd', '.permReason']) card.querySelector(sel)?.remove();
    card.querySelector('.permTitle').textContent = label;
  }

  // ------------------------------------------------------------ notices, stages, summaries
  let lastQuiet = '';
  function notice(text, level, quiet) {
    if (quiet && text === lastQuiet) return null;
    if (quiet) lastQuiet = text;
    endGroup();
    const n = el('div', `notice ${level || 'info'}${quiet ? ' quiet' : ''}`, text);
    target().appendChild(n);
    textBlock = null;
    scroll();
    return n;
  }

  function errorBox(message, buttons) {
    const n = notice(message, 'error');
    const row = el('div', 'btnRow');
    for (const [label, type, primary] of buttons) {
      const b = el('button', 'btn' + (primary ? ' primary' : ''), label);
      b.onclick = () => vscode.postMessage({ type });
      row.appendChild(b);
    }
    n.appendChild(row);
    return n;
  }

  const cmdSection = (cmds) =>
    !cmds.length ? '' :
      '<div class="section"><h5>Commands</h5>' + cmds.slice(-8).map((c) =>
        `<div class="cmdLine"><span class="pill ${c.exit_code === 0 ? 'ok' : 'bad'}">${c.exit_code === 0 ? '✓' : '✗ ' + c.exit_code}</span><span>${esc(c.command)}</span></div>`).join('') + '</div>';
  const fileSection = (files) =>
    !files || !files.length ? '' :
      `<div class="section"><h5>Changed files</h5><div class="files">${files.map((f) => `<code class="filelink" data-path="${esc(f)}">${esc(f)}</code>`).join('')}</div></div>`;

  const VERIFY_LABEL = { passed: ['ok', '✓ passed'], failed: ['bad', '✗ failed'], unverified: ['warn', 'not verified'] };
  // the run report: built by the agent from real results (exit codes, editor problems), not from the model's claims
  function reportSections(rep) {
    if (!rep) return '';
    let html = '';
    const v = rep.verification;
    if (v && v.status !== 'not_needed') {
      const [cls, text] = VERIFY_LABEL[v.status] || ['warn', v.status];
      const detail = v.note || (v.commands || []).map((c) => c.command).join(' → ');
      html += `<div class="section"><h5>Verification</h5><div class="cmdLine"><span class="pill ${cls}">${text}</span><span>${esc(detail)}</span></div></div>`;
    }
    if (rep.problems && rep.problems.length) {
      html += '<div class="section"><h5>Problems in changed files</h5>' + rep.problems.slice(0, 10).map((d) =>
        `<div class="cmdLine"><code class="filelink" data-path="${esc(d.path)}" data-line="${d.line}">${esc(d.path)}:${d.line}</code><span>${esc(d.message)}</span></div>`).join('') + '</div>';
    }
    const list = (title, items) => (items && items.length
      ? `<div class="section"><h5>${title}</h5><ul class="sumList">${items.map((t) => `<li>${esc(t)}</li>`).join('')}</ul></div>` : '');
    return html + list('Remaining', rep.remaining) + list('Needs your approval', rep.needs_approval);
  }

  function runFinished(r) {
    endGroup();
    textBlock = null;
    stageBody = null;
    const rep = r.report;
    const verifyFailed = rep && rep.verification && rep.verification.status === 'failed';
    const ok = r.status === 'completed' && !verifyFailed;
    const cmds = r.tool_calls.filter((t) => t.name === 'run_command' && t.exit_code !== null && t.exit_code !== undefined).map((t) => ({ command: t.arguments.command, exit_code: t.exit_code }));
    // a plain answer without tools needs no summary card
    if (ok && !r.tool_calls.length) return;
    const unsure = r.status === 'cancelled' || (ok && rep && (rep.remaining.length || rep.needs_approval.length || rep.verification.status === 'unverified'));
    const s = el('div', 'summary ' + (unsure ? 'warn' : ok ? 'good' : 'bad'));
    const label = verifyFailed && r.status === 'completed' ? 'Done, but verification fails'
      : { completed: 'Done', blocked: 'Stopped', max_iterations: 'Stopped at the turn limit', cancelled: 'Cancelled', error: 'Error' }[r.status] || r.status;
    s.innerHTML =
      `<div class="sumHead"><span class="label">${esc(label)}</span><span class="meta">${r.duration_s}s · ${r.tool_calls.length} tool call${r.tool_calls.length === 1 ? '' : 's'}</span></div>` +
      (r.status !== 'completed' && r.final_message ? `<div class="sumMsg">${esc(r.final_message)}</div>` : '') +
      fileSection(r.files_changed) + reportSections(rep) + cmdSection(cmds);
    ensureTurn().appendChild(s);
    scroll();
  }

  function stageStarted(ev) {
    endGroup();
    textBlock = null;
    const s = el('div', 'stage');
    s.style.setProperty('--stage-color', AGENT_COLORS[ev.agent] || 'var(--link)');
    s.innerHTML = `<span class="stageTitle">${esc(ev.title)}</span>${badge(ev.agent)}`;
    stageBody = null;
    replyArea().appendChild(s);
    stageBody = el('div', 'stageBody');
    replyArea().appendChild(stageBody);
    setWorking(`${ev.title}${ev.agent ? ' · ' + ev.agent : ''}`);
    scroll();
  }

  function buildFinished(ev) {
    endGroup();
    textBlock = null;
    stageBody = null;
    const cls = ev.status === 'passed' ? 'good' : ev.status === 'unverified' || ev.status === 'cancelled' ? 'warn' : 'bad';
    const label = { passed: '✓ Build verified', failed: '✗ Build still failing', unverified: 'Build not verified', cancelled: 'Build stopped', error: 'Build error' }[ev.status] || ev.status;
    const s = el('div', 'summary ' + cls);
    s.innerHTML =
      `<div class="sumHead"><span class="label">${label}</span><span class="meta">${ev.duration_s}s</span></div>` +
      `<div class="sumMsg">${esc(ev.summary)}</div>` + cmdSection(ev.commands || []) + fileSection(ev.files_changed);
    replyArea().appendChild(s);
    scroll();
  }

  function handleAgentEvent(ev) {
    if (!['turn_started', 'reasoning_delta', 'notice', 'busy', 'mode', 'sessions'].includes(ev.type)) removeThinking();
    switch (ev.type) {
      case 'turn_started':
        setWorking(`${ev.agent ? ev.agent + ' · ' : ''}Thinking${ev.iteration > 1 ? ` (turn ${ev.iteration})` : ''}…`);
        if (!textBlock || !textBlock.isConnected || textBlock !== target().lastElementChild) showThinking(ev.agent ? `${ev.agent} is thinking` : 'Thinking');
        break;
      case 'text_delta':
        appendText(ev.text, ev.agent);
        break;
      case 'tool_call_started':
        toolStarted(ev);
        if (ev.name === 'run_command' && !(ev.arguments && ev.arguments.background)) addBackgroundButton(ev.call_id);
        break;
      case 'tool_output_delta':
        toolOutput(ev);
        break;
      case 'tool_call_finished':
        toolFinished(ev);
        break;
      case 'approval_requested':
        approvalRequested(ev);
        break;
      case 'approval_cancelled':
      case 'question_cancelled':
        document.querySelector(`.perm[data-id="${ev.request_id}"]`)?.remove();
        break;
      case 'question':
        questionAsked(ev);
        break;
      case 'checkpoint':
        attachRewind(turn && [...turn.querySelectorAll(':scope > .user')].pop(), ev.checkpoint_id);
        break;
      case 'checkpoints':
        checkpointsMenu(ev);
        break;
      case 'rewound':
        rewound(ev);
        break;
      case 'permissions':
        renderPermissions(ev);
        break;
      case 'commands':
        customSlash = ev.items || [];
        break;
      case 'mcp':
        renderMcp(ev);
        break;
      case 'status_line': {
        const sl = $('statusLine');
        sl.textContent = ev.text || '';
        sl.classList.toggle('hidden', !ev.text);
        break;
      }
      case 'plan_updated':
        if (!ev.agent) planUpdated(ev);
        break;
      case 'notice':
        notice(ev.message, ev.level, ev.quiet);
        break;
      case 'run_finished':
        if (!ev.agent) runFinished(ev.result);
        break;
      case 'stage_started':
        stageStarted(ev);
        break;
      case 'build_finished':
        buildFinished(ev);
        break;
      case 'ready':
        visionModel = ev.vision !== false;
        break;
      case 'fatal':
        errorBox(ev.message, [['Retry', 'retry', true], ['Show logs', 'showLogs'], ['Settings', 'openSettings']]);
        break;
      case 'busy':
        setBusy(true);
        break;
      case 'idle':
        setBusy(false);
        for (const row of document.querySelectorAll('.tool.running')) {
          row.classList.replace('running', 'failed');
          row.querySelector(':scope > .perm')?.remove();
          const meta = row.querySelector(':scope > .toolHead .tmeta');
          if (meta) meta.innerHTML = '<span class="pill warn">stopped</span>';
        }
        break;
      case 'mode':
        setModeUi(ev.mode);
        break;
      case 'context':
        if (!ev.agent) setContext(ev.used, ev.window);
        break;
      case 'cleared':
        resetSession('');
        break;
      case 'sessions':
        renderSessions(ev);
        break;
      case 'history':
        renderHistory(ev);
        break;
    }
  }

  // ------------------------------------------------------------ working indicator
  let startedAt = 0;
  let timer = null;
  function setWorking(text) {
    $('workText').textContent = text;
  }
  function setBusy(b) {
    busy = b;
    sendBtn.classList.toggle('stop', b);
    sendBtn.title = b ? 'Stop (Esc)' : 'Send (Enter)';
    working.classList.toggle('hidden', !b);
    document.body.classList.toggle('busy', b);
    if (!b) removeThinking();
    clearInterval(timer);
    if (b) {
      startedAt = Date.now();
      setWorking('Thinking…');
      $('workTime').textContent = '0s';
      timer = setInterval(() => {
        const s = Math.floor((Date.now() - startedAt) / 1000);
        $('workTime').textContent = s < 60 ? `${s}s` : `${Math.floor(s / 60)}m ${s % 60}s`;
      }, 1000);
    }
  }
  $('stopLink').addEventListener('click', () => vscode.postMessage({ type: 'stop' }));

  function shortModel(name) {
    let n = String(name || 'Ready').split(/[\\/]/).pop().replace(/\.gguf$/i, '');
    const quant = (/[-_.](q\d[\w]*|f16|bf16)$/i.exec(n) || [])[1];
    if (quant) n = n.slice(0, -(quant.length + 1));
    const parts = n.split(/[-_]/);
    let short = parts.slice(0, 2).join('-');
    if (short.length < 8 && parts[2]) short += '-' + parts[2];
    return short + (quant ? ' ' + quant.toUpperCase() : '');
  }

  // ------------------------------------------------------------ start-up screen
  const BOOT_STEPS = [
    ['server', 'Starting model server'],
    ['load', 'Loading the model'],
    ['agent', 'Starting the agent'],
  ];
  function boot(status, detail) {
    const b = $('boot');
    if (!b) return;
    document.body.classList.toggle('booting', status === 'starting');
    if (status !== 'starting') {
      b.classList.add('hidden');
      return;
    }
    const d = detail.toLowerCase();
    const current = d.includes('loading') ? 1 : d.includes('agent') ? 2 : 0;
    b.classList.remove('hidden');
    b.innerHTML = `<div class="bootTitle">Getting ready</div><div class="bootSub">${esc(detail || 'Starting…')}</div>
      <ol class="steps">${BOOT_STEPS.map(([, label], i) =>
        `<li class="${i < current ? 'done' : i === current ? 'active' : ''}"><span class="mark">${i < current ? '✓' : ''}</span>${label}</li>`).join('')}</ol>
      <div class="bar"><span style="width:${[18, 55, 88][current]}%"></span></div>`;
  }

  // ------------------------------------------------------------ messages from the extension
  window.addEventListener('message', ({ data: m }) => {
    switch (m.type) {
      case 'agentEvent':
        handleAgentEvent(m.event);
        break;
      case 'userMessage': {
        turn = null;
        stageBody = null;
        textBlock = null;
        planCard = null;
        endGroup();
        const u = el('div', 'user');
        const build = /^\/build\s/.exec(m.text);
        u.innerHTML = build ? `<span class="cmdWord">/build</span> ${esc(m.text.slice(build[0].length))}` : esc(m.text);
        if (m.images && m.images.length) {
          u.insertAdjacentHTML('beforeend', `<div class="userImages">${m.images.map((url) => `<img src="${esc(url)}" alt="">`).join('')}</div>`);
        }
        ensureTurn().appendChild(u);
        if ($('sessionTitle').textContent === 'New chat') setTitle(m.text.replace(/^\/build\s+/, '').slice(0, 80));
        showThinking('Thinking');
        stick = true;
        scroll();
        break;
      }
      case 'status': {
        boot(m.status, m.detail || '');
        $('statusDot').className = 'dot ' + m.status;
        const text = m.status === 'ready' || m.status === 'busy' ? m.model || 'Ready'
          : m.status === 'starting' ? m.detail || 'Starting…'
          : m.status === 'offline' ? 'Agent offline' : 'Agent error';
        $('statusText').textContent = m.status === 'ready' || m.status === 'busy' ? shortModel(m.model) : text;
        $('modelSub').textContent = m.status === 'ready' || m.status === 'busy' ? 'Local' : '';
        $('brand').title = m.detail || '';
        if (m.mode) setModeUi(m.mode);
        if (m.status === 'error' && m.detail && m.notify) {
          errorBox(m.detail, [['Retry', 'retry', true], ['Show logs', 'showLogs'], ['Settings', 'openSettings']]);
        }
        break;
      }
      case 'needExecutable':
        errorBox('Point the extension at the agent: choose lca.exe (in the agent\'s .venv\\Scripts folder), or set "lca.executable".',
          [['Locate lca.exe…', 'locateExecutable', true], ['Settings', 'openSettings']]);
        break;
      case 'editorContext':
        chip.innerHTML = svg('file') + `<span>${esc(m.label || 'No file')}</span>`;
        chip.classList.toggle('off', !m.enabled || !m.label);
        document.querySelector('.attachments').classList.toggle('hidden', !m.enabled || !m.label);
        $('attachBtn').classList.toggle('on', !!(m.enabled && m.label));
        chip.title = m.label ? (m.enabled ? 'Sent as context. Click to stop sending it.' : 'Not sent. Click to send it as context.') : 'Open a file to give the agent context';
        break;
      case 'clearLog':
        resetSession('');
        log.innerHTML = welcomeHtml;
        bindSuggestions();
        turn = null;
        stageBody = null;
        textBlock = null;
        planCard = null;
        endGroup();
        tools.clear();
        setBusy(false);
        break;
      case 'focusInput':
        input.focus();
        break;
      case 'fileMatches':
        fileMatches(m);
        break;
    }
  });

  // ------------------------------------------------------------ input
  const HELP = [
    'Commands',
    '  /build <task>   a team of agents: plan → code → install → test ⇄ fix → review',
    '  /clear          start a new conversation (the old one stays in History)',
    '  /history        browse and resume past sessions',
    '  /rewind         undo to before a message (or use ↶ on any of your messages)',
    '  /permissions    allow / ask / deny rules, e.g. run_command(npm test:*)',
    '  /mcp            MCP servers and their tools',
    '  /init /commit /review /pr   built-in commands; add your own in .lca/commands/*.md',
    '  @path           attach a file (type @ to pick one)',
    '  /help           show this help',
  ].join('\n');

  // ------------------------------------------------------------ pasted / dropped images
  let pendingImages = [];
  const MAX_IMAGE = 5_000_000;
  function renderImageTray() {
    const tray = $('imageTray');
    tray.innerHTML = pendingImages.map((url, i) =>
      `<span class="thumb"><img src="${url}" alt="attached image ${i + 1}"><button data-i="${i}" title="Remove" aria-label="Remove image">×</button></span>`).join('');
    tray.classList.toggle('hidden', !pendingImages.length);
  }
  let visionModel = true; // from the agent's ready event
  function addImageFile(file) {
    if (!file || !file.type.startsWith('image/')) return false;
    if (!visionModel) {
      ensureTurn();
      notice('This model cannot read images (the model server reports no vision support). Use a vision model, e.g. llama-server with the model’s --mmproj file.', 'warning');
      return true;
    }
    const reader = new FileReader();
    reader.onload = () => {
      shrinkImage(String(reader.result)).then((url) => {
        if (url.length > MAX_IMAGE * 1.37) {
          ensureTurn();
          notice(`${file.name || 'The image'} is still larger than 5 MB after shrinking.`, 'warning');
        } else if (pendingImages.length < 6) {
          pendingImages.push(url);
        }
        renderImageTray();
      });
    };
    reader.readAsDataURL(file);
    return true;
  }
  // Vision models work at about this size; larger images only cost context and upload time.
  const MAX_SIDE = 1568;
  function shrinkImage(url) {
    return new Promise((resolve) => {
      const img = new Image();
      img.onload = () => {
        const scale = Math.min(1, MAX_SIDE / Math.max(img.naturalWidth, img.naturalHeight));
        if (scale === 1 && url.length < 1_500_000) return resolve(url);
        const canvas = document.createElement('canvas');
        canvas.width = Math.max(1, Math.round(img.naturalWidth * scale));
        canvas.height = Math.max(1, Math.round(img.naturalHeight * scale));
        canvas.getContext('2d').drawImage(img, 0, 0, canvas.width, canvas.height);
        // keep PNG for small screenshots (sharp text); JPEG for photos and anything still large
        const png = canvas.toDataURL('image/png');
        resolve(png.length < 1_500_000 ? png : canvas.toDataURL('image/jpeg', 0.85));
      };
      img.onerror = () => resolve(url);
      img.src = url;
    });
  }
  $('imageTray').addEventListener('click', (e) => {
    const b = e.target.closest('button');
    if (!b) return;
    pendingImages.splice(Number(b.dataset.i), 1);
    renderImageTray();
  });
  input.addEventListener('paste', (e) => {
    let took = false;
    for (const item of e.clipboardData?.items || []) {
      if (item.kind === 'file') took = addImageFile(item.getAsFile()) || took;
    }
    if (took) e.preventDefault();
  });
  const composer = document.querySelector('.composer');
  composer.addEventListener('dragover', (e) => {
    if ([...(e.dataTransfer?.items || [])].some((i) => i.kind === 'file')) {
      e.preventDefault();
      composer.classList.add('dropping');
    }
  });
  composer.addEventListener('dragleave', () => composer.classList.remove('dropping'));
  composer.addEventListener('drop', (e) => {
    composer.classList.remove('dropping');
    const files = [...(e.dataTransfer?.files || [])].filter((f) => f.type.startsWith('image/'));
    if (!files.length) return;
    e.preventDefault();
    files.forEach(addImageFile);
  });

  function submit(text) {
    text = text.trim();
    if (!text || busy) return false;
    history = [text, ...history.filter((h) => h !== text)].slice(0, 50);
    historyIndex = -1;
    if (text === '/history' || text === '/sessions') {
      input.value = '';
      openHistory();
      return true;
    }
    if (text === '/help') {
      ensureTurn();
      notice(HELP, 'info');
      return true;
    }
    if (text === '/clear' || text === '/new') {
      vscode.postMessage({ type: 'newChat' });
      return true;
    }
    if (text === '/rewind' || text === '/undo') {
      vscode.postMessage({ type: 'listCheckpoints' });
      return true;
    }
    if (text === '/mcp') {
      vscode.postMessage({ type: 'listMcp' });
      return true;
    }
    if (text === '/permissions') {
      vscode.postMessage({ type: 'listPermissions' });
      return true;
    }
    if (/^\/build\b/.test(text)) {
      const task = text.replace(/^\/build\b/, '').trim();
      if (!task) {
        ensureTurn();
        notice('Usage: /build <what to build>, e.g. /build a CLI todo app in Python with tests', 'warning');
        return false;
      }
      vscode.postMessage({ type: 'build', text: task });
      return true;
    }
    vscode.postMessage({ type: 'send', text, images: pendingImages });
    pendingImages = [];
    renderImageTray();
    return true;
  }

  function send() {
    if (submit(input.value)) {
      input.value = '';
      autosize();
    }
  }

  function autosize() {
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 240) + 'px';
  }

  input.addEventListener('input', autosize);
  input.addEventListener('keydown', (e) => {
    // Tab (or Enter while picking a file) takes the first entry of an open menu
    const first = !slashMenu.classList.contains('hidden') && slashMenu.querySelector('.mItem');
    if (first && (e.key === 'Tab' || (e.key === 'Enter' && mention))) {
      e.preventDefault();
      first.click();
      return;
    }
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      send();
    } else if (e.key === 'ArrowUp' && !input.value.includes('\n') && input.selectionStart === 0 && history.length) {
      e.preventDefault();
      historyIndex = Math.min(historyIndex + 1, history.length - 1);
      input.value = history[historyIndex];
      autosize();
    } else if (e.key === 'ArrowDown' && historyIndex >= 0 && !input.value.includes('\n')) {
      e.preventDefault();
      historyIndex -= 1;
      input.value = historyIndex >= 0 ? history[historyIndex] : '';
      autosize();
    }
  });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && busy) vscode.postMessage({ type: 'stop' });
  });

  sendBtn.addEventListener('click', () => (busy ? vscode.postMessage({ type: 'stop' }) : send()));
  // ------------------------------------------------------------ composer toolbar
  const MODES = [
    ['ask', 'Ask', 'Approve every file change and command'],
    ['auto-edit', 'Auto-edit', 'Apply file edits automatically, ask before commands'],
    ['full', 'Auto', 'Allow everything except risky commands'],
    ['plan', 'Plan', 'Research first, then approve a plan before any change'],
    ['read-only', 'Read only', 'Never change files'],
  ];
  function setModeUi(value) {
    modeSel.value = value;
    const m = MODES.find((x) => x[0] === value) || MODES[0];
    $('modeLabel').textContent = m[1];
    $('modeBtn').dataset.mode = m[0];
    $('modeBtn').title = `Permission mode: ${m[1]}. ${m[2]}.`;
  }
  const slashMenu = $('slashMenu');
  const modeMenu = $('modeMenu');
  const closeMenus = () => {
    slashMenu.classList.add('hidden');
    modeMenu.classList.add('hidden');
  };
  function openMenu(menu, items) {
    closeMenus();
    menu.innerHTML = items.map((it, i) =>
      `<button class="mItem${it.active ? ' active' : ''}" data-i="${i}" role="menuitem"><span class="mName">${esc(it.name)}</span><span class="mDesc">${esc(it.desc)}</span>${it.active ? '<span class="mCheck">✓</span>' : ''}</button>`).join('');
    menu.querySelectorAll('.mItem').forEach((b) => b.addEventListener('click', () => {
      closeMenus();
      items[Number(b.dataset.i)].run();
    }));
    menu.classList.remove('hidden');
  }
  const SLASH = [
    { name: '/build', desc: 'Plan, code, test and review with a team of agents', run: () => { input.value = '/build '; autosize(); input.focus(); } },
    { name: '/history', desc: 'Browse and resume past sessions', run: () => openHistory() },
    { name: '/clear', desc: 'Start a new chat (the old one stays in History)', run: () => vscode.postMessage({ type: 'newChat' }) },
    { name: '/rewind', desc: 'Undo to before a message: code, conversation or both', run: () => submit('/rewind') },
    { name: '/mcp', desc: 'MCP servers: status and tools', run: () => submit('/mcp') },
    { name: '/permissions', desc: 'Allow, ask and deny rules for tools and commands', run: () => submit('/permissions') },
    { name: '/settings', desc: 'Model server, model and permission settings', run: () => vscode.postMessage({ type: 'openSettings' }) },
    { name: '/logs', desc: 'Show the agent log', run: () => vscode.postMessage({ type: 'showLogs' }) },
    { name: '/help', desc: 'Keyboard shortcuts and commands', run: () => submit('/help') },
  ];
  // the project's own commands (.lca/commands) and skills (.lca/skills), refreshed whenever the menu opens
  let customSlash = [];
  const allSlash = () => [...SLASH, ...customSlash.filter((c) => !SLASH.some((b) => b.name === '/' + c.name)).map((c) => ({
    name: '/' + c.name + (c.argument_hint ? ' ' + c.argument_hint : ''),
    desc: `${c.description || ''}${c.description ? ' · ' : ''}${c.kind === 'skill' ? 'skill' : 'command'}, ${c.source}`,
    run: () => { input.value = '/' + c.name + ' '; autosize(); input.focus(); },
  }))];
  $('slashBtn').addEventListener('click', (e) => {
    e.stopPropagation();
    vscode.postMessage({ type: 'listCommands' });
    slashMenu.classList.contains('hidden') ? openMenu(slashMenu, allSlash()) : closeMenus();
  });
  $('modeBtn').addEventListener('click', (e) => {
    e.stopPropagation();
    if (!modeMenu.classList.contains('hidden')) return closeMenus();
    openMenu(modeMenu, MODES.map(([value, name, desc]) => ({
      name, desc, active: modeSel.value === value,
      run: () => { setModeUi(value); vscode.postMessage({ type: 'setMode', mode: value }); },
    })));
  });
  document.addEventListener('click', (e) => { if (!e.target.closest('.menu')) closeMenus(); });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeMenus(); });
  document.addEventListener('keydown', (e) => {
    // Ctrl+B while a command runs: move it to the background (like Claude Code)
    if (e.ctrlKey && !e.shiftKey && !e.altKey && e.key.toLowerCase() === 'b' && busy && document.querySelector('.bgBtn:not([disabled])')) {
      e.preventDefault();
      document.querySelector('.bgBtn:not([disabled])').click();
    }
  });
  // "@" mentions: pick a workspace file; its contents are attached to the message
  let mention = null; // {start, query} of the @token being typed
  function insertMention(path) {
    if (!mention) return;
    const end = input.selectionStart;
    input.value = input.value.slice(0, mention.start) + '@' + path + ' ' + input.value.slice(end);
    const pos = mention.start + path.length + 2;
    input.setSelectionRange(pos, pos);
    mention = null;
    closeMenus();
    autosize();
    input.focus();
  }
  function fileMatches(m) {
    if (!mention || m.query !== mention.query) return; // the user typed on
    if (!m.items.length) return closeMenus();
    openMenu(slashMenu, m.items.map((p) => ({ name: p, desc: '', run: () => insertMention(p) })));
  }
  input.addEventListener('input', () => {
    // typing "/" in an empty box opens the command menu, filtered as you type
    const v = input.value;
    if (v === '/') vscode.postMessage({ type: 'listCommands' });
    if (/^\/[\w:.-]*$/.test(v)) {
      mention = null;
      const items = allSlash().filter((c) => c.name.startsWith(v));
      if (items.length) return openMenu(slashMenu, items);
    }
    const at = /(?:^|\s)@([\w./\\-]*)$/.exec(v.slice(0, input.selectionStart));
    if (at) {
      mention = { start: input.selectionStart - at[1].length - 1, query: at[1] };
      vscode.postMessage({ type: 'findFiles', query: at[1] });
      return;
    }
    mention = null;
    if (!slashMenu.classList.contains('hidden')) slashMenu.classList.add('hidden');
  });
  $('attachBtn').addEventListener('click', () => vscode.postMessage({ type: 'toggleContext' }));
  $('brand').addEventListener('click', () => vscode.postMessage({ type: 'openSettings' }));

  // context ring and session clock
  function setContext(used, window) {
    const pct = window ? Math.min(1, used / window) : 0;
    const c = 2 * Math.PI * 7;
    const val = $('ctxVal');
    val.style.strokeDasharray = `${c}`;
    val.style.strokeDashoffset = `${c * (1 - pct)}`;
    const ring = $('ctxRing');
    ring.classList.toggle('warn', pct >= 0.7 && pct < 0.9);
    ring.classList.toggle('full', pct >= 0.9);
    const k = (n) => (n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(n));
    ring.title = window ? `Context used: ${Math.round(pct * 100)}% (${k(used)} of ${k(window)} tokens). Older messages are summarized automatically.` : 'Context used';
  }
  let sessionStart = Date.now();
  function tickClock() {
    const m = Math.floor((Date.now() - sessionStart) / 60000);
    $('clockText').textContent = m < 60 ? `${m}m` : `${Math.floor(m / 60)}h ${m % 60}m`;
  }
  setInterval(tickClock, 15000);
  setContext(0, 0);
  setModeUi(modeSel.value);
  function setTitle(text) {
    $('sessionTitle').textContent = text || 'New chat';
    $('sessionTitle').title = text || '';
  }
  function resetSession(title) {
    sessionStart = Date.now();
    tickClock();
    setContext(0, 0);
    setTitle(title);
  }
  chip.addEventListener('click', () => vscode.postMessage({ type: 'toggleContext' }));
  $('newBtn').addEventListener('click', () => vscode.postMessage({ type: 'newChat' }));
  $('logsBtn').addEventListener('click', () => vscode.postMessage({ type: 'showLogs' }));
  const historyPanel = $('historyPanel');
  const openHistory = () => {
    historyPanel.classList.remove('hidden');
    $('historyList').innerHTML = '<div class="hpEmpty">Loading…</div>';
    vscode.postMessage({ type: 'listSessions' });
  };
  $('historyBtn').addEventListener('click', () => (historyPanel.classList.contains('hidden') ? openHistory() : historyPanel.classList.add('hidden')));
  $('historyClose').addEventListener('click', () => historyPanel.classList.add('hidden'));
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && !historyPanel.classList.contains('hidden')) historyPanel.classList.add('hidden');
  });

  function ago(ts) {
    const s = Math.max(0, Date.now() / 1000 - ts);
    if (s < 60) return 'just now';
    if (s < 3600) return `${Math.floor(s / 60)} min ago`;
    if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
    return new Date(ts * 1000).toLocaleDateString();
  }

  function renderSessions(ev) {
    const list = $('historyList');
    if (!ev.items.length) {
      list.innerHTML = '<div class="hpEmpty">No saved sessions yet. Conversations are saved automatically.</div>';
      return;
    }
    list.innerHTML = '';
    for (const it of ev.items) {
      const row = el('div', 'hpItem' + (it.id === ev.current ? ' current' : ''));
      row.innerHTML = `<div class="hpTitle">${esc(it.title || '(untitled)')}</div>
        <div class="hpMeta">${ago(it.updated)} · ${it.turns} message${it.turns === 1 ? '' : 's'}${it.id === ev.current ? ' · current' : ''}</div>`;
      if (it.id !== ev.current) {
        const del = el('button', 'icon hpDel');
        del.title = 'Delete session';
        del.innerHTML = '<svg viewBox="0 0 16 16"><path d="M3 4.5h10M6.5 4.5V3h3v1.5M5 4.5l.6 8.5h4.8l.6-8.5"/></svg>';
        del.addEventListener('click', (e) => {
          e.stopPropagation();
          vscode.postMessage({ type: 'deleteSession', session_id: it.id });
        });
        row.appendChild(del);
        row.addEventListener('click', () => {
          historyPanel.classList.add('hidden');
          vscode.postMessage({ type: 'resume', session_id: it.id });
        });
      }
      list.appendChild(row);
    }
  }

  function renderHistory(ev) {
    resetSession(ev.title);
    log.innerHTML = '';
    turn = null;
    stageBody = null;
    textBlock = null;
    endGroup();
    tools.clear();
    let n = 0;
    for (const it of ev.items) {
      if (it.role === 'user') {
        turn = null;
        textBlock = null;
        endGroup();
        const u = el('div', 'user');
        u.textContent = it.text;
        if (it.images) u.appendChild(el('div', 'userImageNote', `${it.images} image${it.images === 1 ? '' : 's'} attached`));
        ensureTurn().appendChild(u);
        attachRewind(u, it.checkpoint_id);
      } else if (it.role === 'assistant') {
        appendText(it.text);
        textBlock = null;
      } else if (it.role === 'tool') {
        const id = `h${n++}`;
        toolStarted({ call_id: id, name: it.name, arguments: it.arguments || {} });
        toolFinished({ call_id: id, name: it.name, ok: it.ok, summary: it.summary, output: it.summary, duration_ms: null });
      } else if (it.role === 'notice') {
        notice(it.text, 'info', true);
      }
    }
    if (ev.rewound) {
      if (pendingRewind) {
        ensureTurn().appendChild(el('div', 'notice info', pendingRewind.text));
        // like Claude Code: the rewound message comes back into the input box, ready to edit and resend
        input.value = pendingRewind.prompt;
        autosize();
        input.focus();
        pendingRewind = null;
      }
    } else {
      const banner = el('div', 'notice quiet', `Resumed “${ev.title || 'session'}”. Continue the conversation below.`);
      ensureTurn().appendChild(banner);
    }
    turn = null;
    textBlock = null;
    stick = true;
    scroll();
  }

  function bindSuggestions() {
    for (const b of document.querySelectorAll('.suggestion')) {
      b.addEventListener('click', () => {
        const text = b.dataset.text || '';
        if (text.endsWith(' ')) {
          input.value = text; // e.g. "/build ": let the user finish the sentence
          input.focus();
          input.setSelectionRange(text.length, text.length);
          autosize();
        } else {
          submit(text);
        }
      });
    }
  }
  bindSuggestions();

  log.addEventListener('click', (e) => {
    const f = e.target.closest('.filelink');
    if (f) {
      e.stopPropagation();
      return vscode.postMessage({ type: 'openFile', path: f.dataset.path, line: Number(f.dataset.line) || undefined });
    }
    const copy = e.target.closest('.copy');
    if (copy) {
      navigator.clipboard.writeText(copy.closest('.code').querySelector('code').textContent);
      copy.textContent = 'Copied';
      setTimeout(() => (copy.textContent = 'Copy'), 1200);
    }
  });

  vscode.postMessage({ type: 'ready' });
  input.focus();
})();
