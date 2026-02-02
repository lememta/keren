// graph-renderer.js — Dagre layout + D3 SVG rendering with collapsible groups

export class GraphRenderer {
  constructor(container, onNodeClick) {
    this.container = container;
    this.onNodeClick = onNodeClick;
    this.selectedNodeId = null;
    this.svg = null;
    this.g = null;
    this.funcs = null;
    // Collapse state per group id: true = collapsed
    this.collapsed = {};
  }

  render(funcs) {
    this.funcs = funcs;
    // Start all groups collapsed
    for (const f of funcs) {
      if (!(f.id in this.collapsed)) {
        this.collapsed[f.id] = true;
      }
    }
    this._draw();
  }

  toggleGroup(groupId) {
    this.collapsed[groupId] = !this.collapsed[groupId];
    this._draw();
  }

  _draw() {
    const funcs = this.funcs;
    this.container.innerHTML = '';
    if (!funcs || funcs.length === 0) return;

    // Build visible nodes and edges based on collapse state
    const visibleNodes = [];
    const visibleEdges = [];
    const collapsedGroupIds = new Set();
    const nodeToGroup = {}; // nodeId -> groupId
    const groupSummaryId = {}; // groupId -> summary node id

    for (const f of funcs) {
      const gid = f.id;
      for (const n of f.nodes) {
        nodeToGroup[n.id] = gid;
      }

      if (this.collapsed[gid]) {
        collapsedGroupIds.add(gid);
        // Create a summary node for the collapsed group
        const summaryId = `__group__${gid}`;
        groupSummaryId[gid] = summaryId;
        const opCounts = {};
        let simCount = 0;
        for (const n of f.nodes) {
          opCounts[n.op] = (opCounts[n.op] || 0) + 1;
          if (n.simValue != null) simCount++;
        }
        visibleNodes.push({
          id: summaryId,
          ssaName: '',
          op: '__group__',
          dialect: '',
          type: f.returnType || '',
          attrs: {},
          simValue: simCount > 0 ? `${simCount}/${f.nodes.length} ops have values` : null,
          label: `@${f.name}`,
          isGroupSummary: true,
          groupId: gid,
          groupName: f.name,
          childCount: f.nodes.length,
          opCounts,
          simCount,
        });
      } else {
        for (const n of f.nodes) {
          visibleNodes.push(n);
        }
      }
    }

    // Resolve edges: remap edges where source/target is inside a collapsed group
    for (const f of funcs) {
      for (const e of f.edges) {
        const fromGroup = nodeToGroup[e.from];
        const toGroup = nodeToGroup[e.to];
        const fromCollapsed = collapsedGroupIds.has(fromGroup);
        const toCollapsed = collapsedGroupIds.has(toGroup);
        const from = fromCollapsed ? groupSummaryId[fromGroup] : e.from;
        const to = toCollapsed ? groupSummaryId[toGroup] : e.to;
        // Skip self-loops on summary nodes
        if (from === to) continue;
        // Deduplicate
        const key = `${from}->${to}`;
        if (!visibleEdges.find(x => `${x.from}->${x.to}` === key)) {
          visibleEdges.push({ from, to });
        }
      }
    }

    // Determine which groups are expanded (for drawing boundaries)
    const expandedGroups = funcs.filter(f => !this.collapsed[f.id]);

    // Build dagre graph
    const dg = new dagre.graphlib.Graph({ compound: true });
    dg.setGraph({ rankdir: 'TB', nodesep: 30, ranksep: 50, marginx: 30, marginy: 30 });
    dg.setDefaultEdgeLabel(() => ({}));

    // Add cluster parent nodes for expanded groups
    for (const f of expandedGroups) {
      dg.setNode(f.id, { label: f.name, clusterLabelPos: 'top' });
    }

    for (const node of visibleNodes) {
      const hasSim = node.simValue != null;
      const w = Math.max(node.label.length * 9 + 30, hasSim ? 140 : 80);
      const h = node.isGroupSummary ? (hasSim ? 64 : 50) : (hasSim ? 56 : 40);
      dg.setNode(node.id, { label: node.label, width: w, height: h, node });
      // Assign to parent cluster if expanded
      if (!node.isGroupSummary && node.group && !this.collapsed[node.group]) {
        dg.setParent(node.id, node.group);
      }
    }
    for (const edge of visibleEdges) {
      dg.setEdge(edge.from, edge.to);
    }

    // Add invisible ordering edges between consecutive groups so they stack
    // vertically instead of spreading horizontally
    const groupNodeIds = funcs.map(f => {
      if (this.collapsed[f.id]) return groupSummaryId[f.id];
      // For expanded groups, pick the first node as anchor
      return f.nodes.length > 0 ? f.nodes[0].id : null;
    }).filter(Boolean);
    for (let gi = 0; gi < groupNodeIds.length - 1; gi++) {
      const a = groupNodeIds[gi];
      const b = groupNodeIds[gi + 1];
      if (!dg.hasEdge(a, b)) {
        dg.setEdge(a, b, { style: 'invis', weight: 0, minlen: 1 });
      }
    }

    dagre.layout(dg);

    const graph = dg.graph();
    const totalW = (graph.width || 400) + 60;
    const totalH = (graph.height || 300) + 60;

    // Create SVG
    const svg = d3
      .select(this.container)
      .append('svg')
      .attr('width', '100%')
      .attr('height', '100%')
      .attr('viewBox', `0 0 ${totalW} ${totalH}`);

    this.svg = svg;

    // Arrow marker
    svg
      .append('defs')
      .append('marker')
      .attr('id', 'arrowhead')
      .attr('viewBox', '0 0 10 10')
      .attr('refX', 10)
      .attr('refY', 5)
      .attr('markerWidth', 8)
      .attr('markerHeight', 8)
      .attr('orient', 'auto')
      .append('path')
      .attr('d', 'M 0 0 L 10 5 L 0 10 z')
      .attr('fill', '#888');

    const inner = svg.append('g');
    this.g = inner;

    // Zoom/pan
    const zoom = d3.zoom().scaleExtent([0.1, 5]).on('zoom', (e) => {
      inner.attr('transform', e.transform);
    });
    svg.call(zoom);

    // Draw expanded group backgrounds
    const self = this;
    for (const f of expandedGroups) {
      const clusterNode = dg.node(f.id);
      if (!clusterNode) continue;
      const cx = clusterNode.x;
      const cy = clusterNode.y;
      const cw = clusterNode.width + 30;
      const ch = clusterNode.height + 40;

      const gg = inner.append('g')
        .attr('class', 'group-boundary');

      // Background rect
      gg.append('rect')
        .attr('x', cx - cw / 2)
        .attr('y', cy - ch / 2)
        .attr('width', cw)
        .attr('height', ch)
        .attr('rx', 8)
        .attr('ry', 8)
        .attr('fill', 'rgba(255,255,255,0.03)')
        .attr('stroke', '#444')
        .attr('stroke-width', 1)
        .attr('stroke-dasharray', '6,3');

      // Group label + collapse toggle
      const labelG = gg.append('g')
        .attr('transform', `translate(${cx - cw / 2 + 10}, ${cy - ch / 2 + 6})`)
        .style('cursor', 'pointer')
        .on('click', function (event) {
          event.stopPropagation();
          self.toggleGroup(f.id);
        });

      // Triangle (expanded = down)
      labelG.append('text')
        .attr('x', 0)
        .attr('y', 10)
        .attr('fill', '#4fc3f7')
        .attr('font-size', '10px')
        .attr('font-family', 'monospace')
        .text('\u25BC'); // down triangle

      labelG.append('text')
        .attr('x', 14)
        .attr('y', 10)
        .attr('fill', '#888')
        .attr('font-size', '11px')
        .attr('font-family', 'monospace')
        .text(`@${f.name}`);
    }

    // Draw edges
    for (const edgeKey of dg.edges()) {
      const edgeData = dg.edge(edgeKey);
      if (!edgeData) continue;
      // Skip invisible ordering edges
      if (edgeData.style === 'invis') continue;
      const points = edgeData.points;
      const line = d3.line().x(d => d.x).y(d => d.y).curve(d3.curveBasis);
      inner
        .append('path')
        .attr('d', line(points))
        .attr('fill', 'none')
        .attr('stroke', '#666')
        .attr('stroke-width', 1.5)
        .attr('marker-end', 'url(#arrowhead)');
    }

    // Draw nodes
    for (const nodeId of dg.nodes()) {
      const n = dg.node(nodeId);
      if (!n || !n.node) continue;
      const nodeData = n.node;

      const group = inner
        .append('g')
        .attr('transform', `translate(${n.x - n.width / 2}, ${n.y - n.height / 2})`)
        .attr('class', nodeData.isGroupSummary ? 'graph-node group-summary' : 'graph-node')
        .attr('data-id', nodeData.id)
        .style('cursor', 'pointer');

      if (nodeData.isGroupSummary) {
        // Collapsed group summary node
        group.on('click', function (event) {
          self.selectNode(nodeData.id);
          if (self.onNodeClick) {
            // Find child nodes for this group
            const groupFunc = funcs.find(f => f.id === nodeData.groupId);
            self.onNodeClick(nodeData, groupFunc ? groupFunc.nodes : null);
          }
        });
        group.on('dblclick', function (event) {
          event.stopPropagation();
          self.toggleGroup(nodeData.groupId);
        });

        // Summary rect — distinct style
        group
          .append('rect')
          .attr('width', n.width)
          .attr('height', n.height)
          .attr('rx', 8)
          .attr('ry', 8)
          .attr('fill', '#2c3e50')
          .attr('stroke', '#4fc3f7')
          .attr('stroke-width', 2)
          .attr('stroke-dasharray', '4,2');

        // Expand triangle (right = collapsed)
        group
          .append('text')
          .attr('x', 10)
          .attr('y', n.height / 2 + 4)
          .attr('fill', '#4fc3f7')
          .attr('font-size', '11px')
          .attr('font-family', 'monospace')
          .text('\u25B6'); // right triangle

        // Label
        group
          .append('text')
          .attr('x', 24)
          .attr('y', n.height / 2 + 4)
          .attr('fill', '#eee')
          .attr('font-size', '12px')
          .attr('font-family', 'monospace')
          .text(`${nodeData.label} (${nodeData.childCount} ops)`);
      } else {
        // Normal node
        group.on('click', function () {
          self.selectNode(nodeData.id);
          if (self.onNodeClick) self.onNodeClick(nodeData);
        });

        const color = getNodeColor(nodeData);
        group
          .append('rect')
          .attr('width', n.width)
          .attr('height', n.height)
          .attr('rx', 6)
          .attr('ry', 6)
          .attr('fill', color)
          .attr('stroke', '#555')
          .attr('stroke-width', 1.5);

        const hasSimVal = nodeData.simValue != null;

        group
          .append('text')
          .attr('x', n.width / 2)
          .attr('y', hasSimVal ? n.height / 2 - 2 : n.height / 2 + 4)
          .attr('text-anchor', 'middle')
          .attr('fill', '#eee')
          .attr('font-size', '12px')
          .attr('font-family', 'monospace')
          .text(nodeData.label);

        // Simulation value annotation on the node
        if (hasSimVal) {
          const valStr =
            typeof nodeData.simValue === 'object'
              ? JSON.stringify(nodeData.simValue)
              : String(nodeData.simValue);
          const short = valStr.length > 24 ? valStr.slice(0, 22) + '..' : valStr;
          group
            .append('text')
            .attr('class', 'sim-badge')
            .attr('x', n.width / 2)
            .attr('y', n.height / 2 + 14)
            .attr('text-anchor', 'middle')
            .attr('fill', '#4fc3f7')
            .attr('font-size', '10px')
            .attr('font-family', 'monospace')
            .text(short);
        }
      }
    }
  }

  selectNode(nodeId) {
    this.selectedNodeId = nodeId;
    if (!this.g) return;
    this.g.selectAll('.graph-node rect')
      .attr('stroke', function () {
        const el = d3.select(this.parentNode);
        return el.classed('group-summary') ? '#4fc3f7' : '#555';
      })
      .attr('stroke-width', function () {
        const el = d3.select(this.parentNode);
        return el.classed('group-summary') ? 2 : 1.5;
      });
    this.g
      .selectAll(`.graph-node[data-id="${CSS.escape(nodeId)}"] rect`)
      .attr('stroke', '#ffab40')
      .attr('stroke-width', 3);
  }

  applySimValues(valueMap) {
    if (!this.funcs) return;
    // Store values on node data objects so detail panel can show them
    for (const f of this.funcs) {
      for (const n of f.nodes) {
        const ssaName = n.ssaName || n.id.split('/').pop();
        // Try exact match first, then with/without % prefix
        if (valueMap[ssaName] != null) {
          n.simValue = valueMap[ssaName];
        } else {
          const bare = ssaName.startsWith('%') ? ssaName.slice(1) : ssaName;
          const prefixed = ssaName.startsWith('%') ? ssaName : `%${ssaName}`;
          if (valueMap[bare] != null) {
            n.simValue = valueMap[bare];
          } else if (valueMap[prefixed] != null) {
            n.simValue = valueMap[prefixed];
          }
        }
      }
    }
    // Re-draw to show badges on nodes
    this._draw();
  }
}

function getNodeColor(node) {
  switch (node.op) {
    case 'arg':
      return '#37474f';
    case 'return':
      return '#4e342e';
    case 'constant':
      return '#1b5e20';
    default:
      return '#1a237e';
  }
}
