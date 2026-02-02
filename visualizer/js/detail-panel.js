// detail-panel.js — Side panel for node details

export class DetailPanel {
  constructor(container) {
    this.container = container;
    this.container.innerHTML = '<p class="placeholder">Click a node to see details</p>';
  }

  show(node) {
    if (node.isGroupSummary) {
      this.showGroup(node);
      return;
    }

    const lines = [];
    lines.push(`<h3>${escHtml(node.fullOp || node.op)}</h3>`);
    lines.push(`<div class="detail-row"><span class="detail-label">SSA Name</span><span>${escHtml(node.ssaName)}</span></div>`);
    if (node.dialect) {
      lines.push(`<div class="detail-row"><span class="detail-label">Dialect</span><span>${escHtml(node.dialect)}</span></div>`);
    }
    if (node.type) {
      lines.push(`<div class="detail-row"><span class="detail-label">Type</span><span>${escHtml(node.type)}</span></div>`);
    }
    if (node.group) {
      lines.push(`<div class="detail-row"><span class="detail-label">Group</span><span>${escHtml(node.group)}</span></div>`);
    }

    if (Object.keys(node.attrs).length > 0) {
      lines.push('<h4>Attributes</h4>');
      for (const [k, v] of Object.entries(node.attrs)) {
        lines.push(`<div class="detail-row"><span class="detail-label">${escHtml(k)}</span><span>${escHtml(v)}</span></div>`);
      }
    }

    if (node.simValue != null) {
      lines.push('<h4>Simulation Value</h4>');
      const val =
        typeof node.simValue === 'object'
          ? JSON.stringify(node.simValue, null, 2)
          : String(node.simValue);
      lines.push(`<pre class="sim-value">${escHtml(val)}</pre>`);
    }

    this.container.innerHTML = lines.join('\n');
  }

  showGroup(node, childNodes) {
    const lines = [];
    lines.push(`<h3>@${escHtml(node.groupName)}</h3>`);
    lines.push(`<div class="detail-row"><span class="detail-label">Type</span><span>Function group</span></div>`);
    lines.push(`<div class="detail-row"><span class="detail-label">Ops</span><span>${node.childCount}</span></div>`);
    if (node.type) {
      lines.push(`<div class="detail-row"><span class="detail-label">Return</span><span>${escHtml(node.type)}</span></div>`);
    }

    lines.push('<h4>Op Breakdown</h4>');
    const sorted = Object.entries(node.opCounts).sort((a, b) => b[1] - a[1]);
    for (const [op, count] of sorted) {
      lines.push(`<div class="detail-row"><span class="detail-label">${escHtml(op)}</span><span>${count}</span></div>`);
    }

    // Show simulation values for child nodes if available
    if (childNodes) {
      const withValues = childNodes.filter(n => n.simValue != null);
      if (withValues.length > 0) {
        lines.push(`<h4>Simulation Values (${withValues.length}/${childNodes.length})</h4>`);
        for (const n of withValues) {
          const val = typeof n.simValue === 'object'
            ? JSON.stringify(n.simValue) : String(n.simValue);
          const short = val.length > 40 ? val.slice(0, 38) + '..' : val;
          lines.push(`<div class="detail-row"><span class="detail-label">${escHtml(n.ssaName || n.op)}</span><span style="color:#4fc3f7">${escHtml(short)}</span></div>`);
        }
      }
    }

    lines.push('<p class="placeholder" style="padding-top:12px">Double-click to expand</p>');

    this.container.innerHTML = lines.join('\n');
  }

  clear() {
    this.container.innerHTML = '<p class="placeholder">Click a node to see details</p>';
  }
}

function escHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
