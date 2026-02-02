// mlir-parser.js — Parse StableHLO MLIR text into a graph structure

export function parseMLIR(text) {
  const functions = [];
  const lines = text.split('\n');
  let i = 0;

  while (i < lines.length) {
    const funcMatch = lines[i].match(
      /func\.func\s+@(\w+)\s*\(([^)]*)\)\s*(?:->\s*(.+?))?\s*\{/
    );
    if (funcMatch) {
      const func = {
        id: `func/${funcMatch[1]}`,
        name: funcMatch[1],
        args: parseArgs(funcMatch[2]),
        returnType: (funcMatch[3] || '').trim(),
        nodes: [],
        edges: [],
      };
      i++;
      i = parseBody(lines, i, func);
      functions.push(func);
    } else {
      i++;
    }
  }

  if (functions.length === 0) {
    const func = { id: 'func/main', name: 'main', args: [], returnType: '', nodes: [], edges: [] };
    parseBody(lines, 0, func);
    if (func.nodes.length > 0) functions.push(func);
  }

  for (const f of functions) {
    for (const node of f.nodes) {
      node.group = f.id;
    }
  }

  return functions;
}

function parseArgs(argStr) {
  if (!argStr.trim()) return [];
  return argStr.split(',').map((a, idx) => {
    const m = a.trim().match(/(%\w+)\s*:\s*(.+)/);
    return m
      ? { id: m[1], type: m[2].trim() }
      : { id: `%arg${idx}`, type: a.trim() };
  });
}

// Extract all %name references from a string, stopping at `:` that
// begins a type annotation (but not inside `<...>` or `(...)`)
function extractOperands(str) {
  const operands = [];
  // Find all %identifier tokens that appear before the type annotation
  // The type annotation starts with ` : ` followed by a type like `tensor<...>`
  // We split on ` : ` but need to be careful about nested colons in types.
  //
  // Strategy: scan for all %word tokens, but skip those inside type positions.
  // A simple heuristic: everything before the last top-level ` : ` is operands.
  const colonIdx = findTypeColon(str);
  const operandPart = colonIdx >= 0 ? str.slice(0, colonIdx) : str;

  const re = /%[\w#.]+(?::\d+)?/g;
  let m;
  while ((m = re.exec(operandPart)) !== null) {
    operands.push(m[0]);
  }
  return operands;
}

// Find the index of the ` : ` that starts the type annotation.
// We look for `: ` preceded by space (or start) that's not inside parens/angles/braces.
function findTypeColon(str) {
  let depth = 0;
  for (let i = 0; i < str.length; i++) {
    const ch = str[i];
    if (ch === '(' || ch === '<' || ch === '{' || ch === '[') depth++;
    else if (ch === ')' || ch === '>' || ch === '}' || ch === ']') depth--;
    else if (ch === ':' && depth === 0) {
      // Check it's followed by a space and looks like a type (tensor, i32, f32, etc.)
      const after = str.slice(i + 1).trimStart();
      if (after.match(/^(?:tensor|memref|[ifu]\d|index|none|\(|!)/)) {
        return i;
      }
    }
  }
  return -1;
}

function parseBody(lines, start, func) {
  let i = start;
  const valueMap = {}; // %name -> node id

  // Add function arguments as nodes
  for (const arg of func.args) {
    const nodeId = `${func.name}/${arg.id}`;
    func.nodes.push({
      id: nodeId,
      ssaName: arg.id,
      op: 'arg',
      dialect: 'func',
      type: arg.type,
      attrs: {},
      simValue: null,
      label: `${arg.id}: ${shortType(arg.type)}`,
    });
    valueMap[arg.id] = nodeId;
  }

  while (i < lines.length) {
    const line = lines[i].trim();
    if (line === '}') return i + 1;
    if (!line || line.startsWith('//')) { i++; continue; }

    // Check for keren-value annotation
    let simValue = null;
    const simMatch = line.match(/\/\/\s*keren-value:\s*(.+)$/);
    if (simMatch) {
      try { simValue = JSON.parse(simMatch[1]); }
      catch { simValue = simMatch[1].trim(); }
    }

    // Match: return %name, %name : type
    const retMatch = line.match(/^(?:stablehlo\.return|return|func\.return)\s+(.+)/);
    if (retMatch) {
      const retPart = retMatch[1];
      const operands = extractOperands(retPart);
      const nodeId = `${func.name}/__return`;
      func.nodes.push({
        id: nodeId, ssaName: '__return', op: 'return', dialect: 'func',
        type: '', attrs: {}, simValue, label: 'return',
      });
      for (const operand of operands) {
        const fromId = valueMap[operand];
        if (fromId) func.edges.push({ from: fromId, to: nodeId });
      }
      i++; continue;
    }

    // Match: %result = op_name ...rest...
    // Handles both:
    //   %c = stablehlo.add %a, %b : tensor<...>
    //   %0 = stablehlo.dot_general(%a, %b) ... : type
    //   %0 = "stablehlo.constant"() {value = ...} : type
    const assignMatch = line.match(
      /^(%[\w#.]+(?::\d+)?(?:\s*,\s*%[\w#.]+(?::\d+)?)*)\s*=\s*(?:"([^"]+)"|(\S+?))\s*(.*)/
    );

    if (assignMatch) {
      const results = assignMatch[1].split(',').map(r => r.trim());
      const opName = assignMatch[2] || assignMatch[3];
      const rest = assignMatch[4] || '';

      const dotIdx = opName.indexOf('.');
      const dialect = dotIdx >= 0 ? opName.slice(0, dotIdx) : '';
      const shortOp = dotIdx >= 0 ? opName.slice(dotIdx + 1) : opName;

      // Extract operands from the rest (handles both bare and parenthesized)
      const operands = extractOperands(rest);
      const { attrs, type } = parseAttrsAndType(rest);

      for (const result of results) {
        const nodeId = `${func.name}/${result}`;
        func.nodes.push({
          id: nodeId, ssaName: result, op: shortOp, fullOp: opName,
          dialect, type, attrs, simValue, label: `${result} = ${shortOp}`,
        });
        valueMap[result] = nodeId;

        for (const operand of operands) {
          const fromId = valueMap[operand];
          if (fromId) func.edges.push({ from: fromId, to: nodeId });
        }
      }
    }

    i++;
  }
  return i;
}

function parseAttrsAndType(rest) {
  let attrs = {};
  let type = '';

  // Extract attributes in { ... } (top-level only)
  const attrStart = rest.indexOf('{');
  const attrEnd = rest.indexOf('}', attrStart);
  if (attrStart >= 0 && attrEnd > attrStart) {
    const attrStr = rest.slice(attrStart + 1, attrEnd);
    for (const pair of attrStr.split(',')) {
      const eq = pair.indexOf('=');
      if (eq > 0) {
        attrs[pair.slice(0, eq).trim()] = pair.slice(eq + 1).trim();
      }
    }
  }

  // Extract type from the type annotation position
  const colonIdx = findTypeColon(rest);
  if (colonIdx >= 0) {
    let t = rest.slice(colonIdx + 1).trim();
    // Remove trailing comments
    const commentIdx = t.indexOf('//');
    if (commentIdx >= 0) t = t.slice(0, commentIdx).trim();
    type = t;
  }

  return { attrs, type };
}

function shortType(t) {
  // Abbreviate long tensor types for labels
  if (t.length > 20) {
    const m = t.match(/tensor<([^>]+)>/);
    if (m) return `tensor<${m[1]}>`;
  }
  return t;
}
