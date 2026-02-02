// app.js — Main controller

import { parseMLIR } from './mlir-parser.js';
import { GraphRenderer } from './graph-renderer.js';
import { DetailPanel } from './detail-panel.js';

const SAMPLE_MLIR = `func.func @main() -> tensor<2x2xi32> {
  %a = stablehlo.constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>
  %b = stablehlo.constant dense<[[10, 20], [30, 40]]> : tensor<2x2xi32>
  %c = stablehlo.add %a, %b : tensor<2x2xi32>
  return %c : tensor<2x2xi32>
}`;

const SAMPLE_MLIR_2 = `func.func @dot_relu(%arg0: tensor<4x3xf32>, %arg1: tensor<3x2xf32>) -> tensor<4x2xf32> {
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<4x3xf32>, tensor<3x2xf32>) -> tensor<4x2xf32>
  %1 = stablehlo.constant dense<0.0> : tensor<4x2xf32>
  %2 = stablehlo.maximum %0, %1 : tensor<4x2xf32>
  return %2 : tensor<4x2xf32>
}`;

const SAMPLE_MLIR_3 = `func.func @multi_op() -> tensor<2x2xf32> {
  %a = stablehlo.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
  %b = stablehlo.constant dense<[[5.0, 6.0], [7.0, 8.0]]> : tensor<2x2xf32>
  %sum = stablehlo.add %a, %b : tensor<2x2xf32>
  %prod = stablehlo.multiply %a, %b : tensor<2x2xf32>
  %result = stablehlo.add %sum, %prod : tensor<2x2xf32>
  return %result : tensor<2x2xf32>
}`;

const SAMPLE_MLIR_MULTI = `func.func @encoder(%input: tensor<4x8xf32>, %weights: tensor<8x8xf32>) -> tensor<4x8xf32> {
  %0 = stablehlo.dot_general %input, %weights, contracting_dims = [1] x [0] : (tensor<4x8xf32>, tensor<8x8xf32>) -> tensor<4x8xf32>
  %1 = stablehlo.constant dense<0.0> : tensor<4x8xf32>
  %2 = stablehlo.maximum %0, %1 : tensor<4x8xf32>
  return %2 : tensor<4x8xf32>
}

func.func @decoder(%input: tensor<4x8xf32>, %weights: tensor<8x4xf32>) -> tensor<4x4xf32> {
  %0 = stablehlo.dot_general %input, %weights, contracting_dims = [1] x [0] : (tensor<4x8xf32>, tensor<8x4xf32>) -> tensor<4x4xf32>
  %1 = stablehlo.logistic %0 : tensor<4x4xf32>
  return %1 : tensor<4x4xf32>
}

func.func @loss(%pred: tensor<4x4xf32>, %target: tensor<4x4xf32>) -> tensor<f32> {
  %diff = stablehlo.subtract %pred, %target : tensor<4x4xf32>
  %sq = stablehlo.multiply %diff, %diff : tensor<4x4xf32>
  %0 = stablehlo.constant dense<0.0> : tensor<f32>
  %sum = stablehlo.reduce(%sq init: %0) applies stablehlo.add across dimensions = [0, 1] : (tensor<4x4xf32>, tensor<f32>) -> tensor<f32>
  return %sum : tensor<f32>
}`;

const SAMPLE_SIM_JSON = `{
  "%c": [[11, 22], [33, 44]],
  "%a": [[1, 2], [3, 4]],
  "%b": [[10, 20], [30, 40]]
}`;

const SAMPLE_INPUT_JSON = `[
  [[1, 2], [3, 4]],
  [[10, 20], [30, 40]]
]`;

document.addEventListener('DOMContentLoaded', () => {
  const mlirInput = document.getElementById('mlir-input');
  const simInput = document.getElementById('sim-input');
  const parseBtn = document.getElementById('parse-btn');
  const simBtn = document.getElementById('load-sim-btn');
  const collapseBtn = document.getElementById('collapse-btn');
  const graphContainer = document.getElementById('graph-container');
  const detailContainer = document.getElementById('detail-panel');
  const presetSelect = document.getElementById('preset-select');
  const fileInput = document.getElementById('file-input');
  const valuesFileInput = document.getElementById('values-file-input');

  // Transport controls
  const simRunBtn = document.getElementById('sim-run-btn');
  const simStopBtn = document.getElementById('sim-stop-btn');
  const simStepBtn = document.getElementById('sim-step-btn');
  const simProgressBar = document.getElementById('sim-progress-bar');
  const simStatus = document.getElementById('sim-status');

  const toggleLeftBtn = document.getElementById('toggle-left');
  const toggleRightBtn = document.getElementById('toggle-right');
  const leftPanel = document.getElementById('left-panel');

  toggleLeftBtn.addEventListener('click', () => {
    leftPanel.classList.toggle('collapsed');
    toggleLeftBtn.textContent = leftPanel.classList.contains('collapsed') ? '\u203A' : '\u2039';
  });

  toggleRightBtn.addEventListener('click', () => {
    detailContainer.classList.toggle('collapsed');
    toggleRightBtn.textContent = detailContainer.classList.contains('collapsed') ? '\u2039' : '\u203A';
  });

  const detailPanel = new DetailPanel(detailContainer);
  let renderer = new GraphRenderer(graphContainer, (node) => detailPanel.show(node));
  let currentFuncs = null;
  let allCollapsed = true;
  let simAbort = null; // AbortController for cancelling requests
  let stepIndex = 0;   // for step-through mode

  const presets = {
    '': '',
    'add': SAMPLE_MLIR,
    'dot_relu': SAMPLE_MLIR_2,
    'multi_op': SAMPLE_MLIR_3,
    'multi_func': SAMPLE_MLIR_MULTI,
  };

  presetSelect.addEventListener('change', async () => {
    const val = presetSelect.value;
    if (val === 'gpt2') {
      // Fetch both MLIR and input values from server
      try {
        const [mlirResp, inputsResp] = await Promise.all([
          fetch('examples/gpt2.mlir'),
          fetch('examples/gpt2_inputs.json'),
        ]);
        mlirInput.value = await mlirResp.text();
        simInput.value = await inputsResp.text();
      } catch (err) {
        alert('Failed to load GPT-2 files: ' + err.message);
      }
    } else if (presets[val]) {
      mlirInput.value = presets[val];
    }
  });

  fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => { mlirInput.value = reader.result; };
    reader.readAsText(file);
  });

  valuesFileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => { simInput.value = reader.result; };
    reader.readAsText(file);
  });

  // Parse & Render
  parseBtn.addEventListener('click', () => {
    const text = mlirInput.value;
    if (!text.trim()) return;
    try {
      currentFuncs = parseMLIR(text);
      renderer = new GraphRenderer(graphContainer, (node) => detailPanel.show(node));
      renderer.render(currentFuncs);
      detailPanel.clear();
      allCollapsed = true;
      collapseBtn.textContent = 'Expand All';
      stepIndex = 0;
      setSimStatus('Ready', '');
      simProgressBar.style.width = '0%';
    } catch (err) {
      graphContainer.innerHTML = `<p class="error">Parse error: ${err.message}</p>`;
    }
  });

  // Collapse All / Expand All
  collapseBtn.addEventListener('click', () => {
    if (!currentFuncs || !renderer.funcs) return;
    allCollapsed = !allCollapsed;
    for (const f of currentFuncs) {
      renderer.collapsed[f.id] = allCollapsed;
    }
    renderer._draw();
    collapseBtn.textContent = allCollapsed ? 'Expand All' : 'Collapse All';
    detailPanel.clear();
  });

  // Load manual simulation values
  simBtn.addEventListener('click', () => {
    const text = simInput.value;
    if (!text.trim() || !renderer) return;
    try {
      const raw = JSON.parse(text);
      const valueMap = normalizeSimValues(raw);
      renderer.applySimValues(valueMap);
    } catch (err) {
      alert('Invalid JSON: ' + err.message);
    }
  });

  // --- Simulation transport controls ---

  // Run: call /api/simulate, show progress, apply results
  simRunBtn.addEventListener('click', () => {
    const mlir = mlirInput.value.trim();
    if (!mlir) return;
    // Parse input values from the sim input textarea
    let inputs = [];
    const simText = simInput.value.trim();
    if (simText) {
      try {
        const parsed = JSON.parse(simText);
        if (Array.isArray(parsed)) {
          inputs = parsed;
        }
      } catch (e) {
        // If not valid JSON array, ignore — will run without inputs
      }
    }
    runSimulation(mlir, inputs);
  });

  // Stop: abort in-flight request
  simStopBtn.addEventListener('click', () => {
    if (simAbort) {
      simAbort.abort();
      simAbort = null;
    }
    setSimIdle();
    setSimStatus('Stopped', '');
  });

  // Step: highlight nodes one at a time with sim values
  simStepBtn.addEventListener('click', () => {
    if (!currentFuncs) return;

    // Collect all non-arg nodes in order
    const allNodes = [];
    for (const f of currentFuncs) {
      for (const n of f.nodes) {
        if (n.op !== 'arg') allNodes.push(n);
      }
    }

    if (allNodes.length === 0) return;

    if (stepIndex >= allNodes.length) stepIndex = 0;

    const node = allNodes[stepIndex];
    renderer.selectNode(node.id);
    detailPanel.show(node);

    const pct = Math.round(((stepIndex + 1) / allNodes.length) * 100);
    simProgressBar.style.width = `${pct}%`;
    simProgressBar.classList.remove('indeterminate');
    setSimStatus(`Step ${stepIndex + 1}/${allNodes.length}: ${node.op}`, '');

    stepIndex++;
  });

  async function runSimulation(mlir, inputs = []) {
    setSimRunning();

    simAbort = new AbortController();
    try {
      const resp = await fetch('/api/simulate-trace', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mlir, inputs }),
        signal: simAbort.signal,
      });

      const data = await resp.json();

      if (data.success) {
        // Build a value map from trace data (per-op intermediate values)
        // and final results, keyed by SSA name
        const valueMap = {};

        // Map input values to arg nodes
        if (inputs.length > 0 && currentFuncs) {
          for (const f of currentFuncs) {
            let argIdx = 0;
            for (const n of f.nodes) {
              if (n.op === 'arg' && argIdx < inputs.length) {
                const ssaName = n.ssaName || n.id.split('/').pop();
                valueMap[ssaName] = inputs[argIdx];
                argIdx++;
              }
            }
          }
        }

        // Trace values: keyed by probe ID (e.g. "probe1")
        // We need to map them to SSA names.
        // The probes are inserted in program order, so we map them
        // to non-arg nodes in order.
        if (data.trace && currentFuncs) {
          const allOps = [];
          for (const f of currentFuncs) {
            for (const n of f.nodes) {
              if (n.op !== 'arg' && n.op !== 'return') allOps.push(n);
            }
          }
          // Probes are numbered sequentially: probe1, probe2, ...
          const probeKeys = Object.keys(data.trace)
            .filter(k => k.startsWith('probe'))
            .sort((a, b) => {
              const na = parseInt(a.replace('probe', ''));
              const nb = parseInt(b.replace('probe', ''));
              return na - nb;
            });
          for (let i = 0; i < probeKeys.length && i < allOps.length; i++) {
            const traceEntry = data.trace[probeKeys[i]];
            const node = allOps[i];
            const ssaName = node.ssaName || node.id.split('/').pop();
            if (traceEntry.value != null) {
              valueMap[ssaName] = traceEntry.value;
            }
          }
        }

        // Also include final results (mapped to return values)
        if (data.results) {
          const normalized = normalizeSimValues(data.results);
          Object.assign(valueMap, normalized);
        }

        if (renderer) {
          renderer.applySimValues(valueMap);
        }

        simProgressBar.style.width = '100%';
        simProgressBar.classList.remove('indeterminate');
        setSimStatus('Simulation complete', 'success');
      } else {
        setSimStatus(data.error || 'Simulation failed', 'error');
        simProgressBar.style.width = '0%';
        simProgressBar.classList.remove('indeterminate');
      }
    } catch (err) {
      if (err.name === 'AbortError') {
        // user cancelled
      } else {
        setSimStatus(`Server error: ${err.message}`, 'error');
        simProgressBar.style.width = '0%';
        simProgressBar.classList.remove('indeterminate');
      }
    } finally {
      simAbort = null;
      setSimIdle();
    }
  }

  function setSimRunning() {
    simRunBtn.disabled = true;
    simRunBtn.classList.add('running');
    simStopBtn.disabled = false;
    simStepBtn.disabled = true;
    simProgressBar.style.width = '30%';
    simProgressBar.classList.add('indeterminate');
    setSimStatus('Running keren-sim...', '');
  }

  function setSimIdle() {
    simRunBtn.disabled = false;
    simRunBtn.classList.remove('running');
    simStopBtn.disabled = true;
    simStepBtn.disabled = false;
  }

  function setSimStatus(text, cls) {
    simStatus.textContent = text;
    simStatus.className = cls || '';
  }

  // Load defaults
  mlirInput.value = SAMPLE_MLIR;
  simInput.value = SAMPLE_INPUT_JSON;
});

function normalizeSimValues(raw) {
  const result = {};
  for (const [key, val] of Object.entries(raw)) {
    if (val && typeof val === 'object' && !Array.isArray(val) && 'value' in val) {
      result[key] = val.value;
    } else {
      result[key] = val;
    }
  }
  return result;
}
