/**
 * Sensor Transformer Web UI — Frontend Application
 *
 * Vanilla JS single-page app with tabbed interface.
 * Communicates with Express backend via REST + SSE.
 */

// ═══════════════════════════════════════════
// Global State
// ═══════════════════════════════════════════
const STATE = {
  currentCSV: null,       // { name, path, headers, totalRows }
  boundary: [],           // selected boundary signal names
  target: [],             // selected target signal names
  configPath: null,       // saved config JSON path
  stage1Result: null,     // { model_path, scaler_path, config_path }
  stage2Result: null,
  residualsPath: null,
  predictionsCSV: null,
  resultsJSON: null,
};

// ═══════════════════════════════════════════
// Tab Navigation
// ═══════════════════════════════════════════
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('tab-' + btn.dataset.tab).classList.add('active');

    // Refresh selectors when switching to certain tabs
    const tab = btn.dataset.tab;
    if (['residuals', 'stage2', 'inference', 'results'].includes(tab)) {
      refreshSelectors();
    }
  });
});

// ═══════════════════════════════════════════
// Utility Functions
// ═══════════════════════════════════════════
function $(id) { return document.getElementById(id); }

function setStatus(text) { $('globalStatus').textContent = text; }

function appendLog(panelId, text, cls = '') {
  const panel = $(panelId);
  const line = document.createElement('div');
  line.className = 'log-line ' + cls;
  line.textContent = text;
  panel.appendChild(line);
  panel.scrollTop = panel.scrollHeight;
}

function clearLog(panelId) { $(panelId).innerHTML = ''; }

function show(el) {
  if (typeof el === 'string') el = $(el);
  el.classList.remove('hidden');
}

function hide(el) {
  if (typeof el === 'string') el = $(el);
  el.classList.add('hidden');
}

function formatNum(n, digits = 4) {
  return typeof n === 'number' ? n.toFixed(digits) : n;
}

// ═══════════════════════════════════════════
// SSE Helper — connect to streaming endpoint
// ═══════════════════════════════════════════
function connectSSE(url, { onMessage, onDone, onError, logPanel }) {
  const es = new EventSource(url);
  es.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.type === 'done') {
        es.close();
        if (onDone) onDone(data);
        return;
      }
      if (onMessage) onMessage(data);
      // Auto-log
      if (logPanel) {
        if (data.type === 'error') {
          appendLog(logPanel, data.message || data.text || JSON.stringify(data), 'error');
        } else if (data.type === 'info' || data.type === 'log') {
          appendLog(logPanel, data.message || data.text, 'info');
        } else if (data.type === 'stderr') {
          appendLog(logPanel, data.text, 'error');
        } else if (data.type === 'epoch') {
          appendLog(logPanel,
            `Epoch ${data.epoch}/${data.total} | loss: ${formatNum(data.train_loss,6)}/${formatNum(data.val_loss,6)} | R²: ${formatNum(data.train_r2)}/${formatNum(data.val_r2)} | ${data.elapsed}s`,
            'epoch');
        }
      }
    } catch {
      if (logPanel) appendLog(logPanel, event.data, '');
    }
  };
  es.onerror = () => {
    es.close();
    if (onError) onError();
  };
  return es;
}

// ═══════════════════════════════════════════
// TAB 1: Data & Config
// ═══════════════════════════════════════════

// CSV Upload
const csvUploadZone = $('csvUploadZone');
const csvFileInput = $('csvFileInput');

csvUploadZone.addEventListener('click', () => csvFileInput.click());
csvUploadZone.addEventListener('dragover', (e) => { e.preventDefault(); csvUploadZone.classList.add('dragover'); });
csvUploadZone.addEventListener('dragleave', () => csvUploadZone.classList.remove('dragover'));
csvUploadZone.addEventListener('drop', (e) => {
  e.preventDefault();
  csvUploadZone.classList.remove('dragover');
  if (e.dataTransfer.files.length) uploadCSV(e.dataTransfer.files[0]);
});
csvFileInput.addEventListener('change', () => {
  if (csvFileInput.files.length) uploadCSV(csvFileInput.files[0]);
});

async function uploadCSV(file) {
  setStatus('Uploading...');
  const fd = new FormData();
  fd.append('file', file);
  const res = await fetch('/api/data/upload-csv', { method: 'POST', body: fd });
  const data = await res.json();
  if (data.error) { alert(data.error); setStatus('Error'); return; }

  // Preview
  const preview = await (await fetch(`/api/data/preview-csv?path=${encodeURIComponent(data.path)}`)).json();
  STATE.currentCSV = { name: data.name, path: data.path, headers: preview.headers, totalRows: preview.totalRows };

  renderCSVPreview(preview);
  renderSignalSelector(preview.headers);
  refreshCSVList();
  setStatus('Ready');
}

function renderCSVPreview(preview) {
  show('dataPreviewCard');
  $('dataInfo').textContent = `${preview.totalRows} rows x ${preview.headers.length} columns`;

  let html = '<table><thead><tr>';
  preview.headers.forEach(h => { html += `<th>${h}</th>`; });
  html += '</tr></thead><tbody>';
  preview.rows.forEach(row => {
    html += '<tr>';
    preview.headers.forEach(h => { html += `<td>${row[h] ?? ''}</td>`; });
    html += '</tr>';
  });
  html += '</tbody></table>';
  $('dataTable').innerHTML = html;
}

// Signal Selector
let selectedAvailable = new Set();
let selectedBoundary = new Set();
let selectedTarget = new Set();

function renderSignalSelector(headers) {
  show('signalConfigCard');
  const avail = $('availableSignals');

  // Filter out timestamp-like columns
  const signals = headers.filter(h =>
    !h.toLowerCase().startsWith('time') &&
    !h.toLowerCase().startsWith('date') &&
    !h.match(/^20\d{2}/)
  );

  // Remove already assigned signals
  const assigned = new Set([...STATE.boundary, ...STATE.target]);
  const available = signals.filter(s => !assigned.has(s));

  avail.innerHTML = '';
  available.forEach(sig => {
    const div = document.createElement('div');
    div.className = 'signal-item' + (selectedAvailable.has(sig) ? ' selected' : '');
    div.textContent = sig;
    div.addEventListener('click', () => {
      if (selectedAvailable.has(sig)) selectedAvailable.delete(sig);
      else selectedAvailable.add(sig);
      div.classList.toggle('selected');
    });
    avail.appendChild(div);
  });
  $('availableCount').textContent = available.length;
  renderAssignedSignals();
}

function renderAssignedSignals() {
  // Boundary
  const bDiv = $('boundarySignals');
  bDiv.innerHTML = '';
  STATE.boundary.forEach(sig => {
    const div = document.createElement('div');
    div.className = 'signal-item' + (selectedBoundary.has(sig) ? ' selected' : '');
    div.textContent = sig;
    div.addEventListener('click', () => {
      if (selectedBoundary.has(sig)) selectedBoundary.delete(sig);
      else selectedBoundary.add(sig);
      div.classList.toggle('selected');
    });
    bDiv.appendChild(div);
  });
  $('boundaryCount').textContent = STATE.boundary.length;

  // Target
  const tDiv = $('targetSignals');
  tDiv.innerHTML = '';
  STATE.target.forEach(sig => {
    const div = document.createElement('div');
    div.className = 'signal-item' + (selectedTarget.has(sig) ? ' selected' : '');
    div.textContent = sig;
    div.addEventListener('click', () => {
      if (selectedTarget.has(sig)) selectedTarget.delete(sig);
      else selectedTarget.add(sig);
      div.classList.toggle('selected');
    });
    tDiv.appendChild(div);
  });
  $('targetCount').textContent = STATE.target.length;
}

// Arrow buttons
$('btnAddBoundary').addEventListener('click', () => {
  selectedAvailable.forEach(s => STATE.boundary.push(s));
  selectedAvailable.clear();
  if (STATE.currentCSV) renderSignalSelector(STATE.currentCSV.headers);
});

$('btnAddTarget').addEventListener('click', () => {
  selectedAvailable.forEach(s => STATE.target.push(s));
  selectedAvailable.clear();
  if (STATE.currentCSV) renderSignalSelector(STATE.currentCSV.headers);
});

$('btnRemoveSignal').addEventListener('click', () => {
  selectedBoundary.forEach(s => { STATE.boundary = STATE.boundary.filter(x => x !== s); });
  selectedTarget.forEach(s => { STATE.target = STATE.target.filter(x => x !== s); });
  selectedBoundary.clear();
  selectedTarget.clear();
  if (STATE.currentCSV) renderSignalSelector(STATE.currentCSV.headers);
});

// Save config
$('btnSaveConfig').addEventListener('click', async () => {
  if (!STATE.boundary.length || !STATE.target.length) {
    alert('Please select boundary and target signals first.');
    return;
  }
  const name = prompt('Config name:', 'signals_config');
  if (!name) return;
  const res = await fetch('/api/data/save-config', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, boundary: STATE.boundary, target: STATE.target }),
  });
  const data = await res.json();
  STATE.configPath = data.path;
  alert('Config saved: ' + data.path);
});

// Load config
$('btnLoadConfig').addEventListener('click', () => $('configFileInput').click());
$('configFileInput').addEventListener('change', async () => {
  const file = $('configFileInput').files[0];
  if (!file) return;
  const fd = new FormData();
  fd.append('file', file);
  const res = await fetch('/api/data/upload-config', { method: 'POST', body: fd });
  const data = await res.json();
  if (data.error) { alert(data.error); return; }
  STATE.boundary = data.content.boundary || [];
  STATE.target = data.content.target || [];
  STATE.configPath = data.path;
  if (STATE.currentCSV) renderSignalSelector(STATE.currentCSV.headers);
});

async function refreshCSVList() {
  const res = await fetch('/api/data/csv-files');
  const files = await res.json();
  $('csvFileList').innerHTML = files.map(f =>
    `<span class="file-chip ${f.path === STATE.currentCSV?.path ? 'selected' : ''}"
      onclick="selectCSV('${f.path}', '${f.name}')">${f.name}</span>`
  ).join(' ');
}

window.selectCSV = async (path, name) => {
  const preview = await (await fetch(`/api/data/preview-csv?path=${encodeURIComponent(path)}`)).json();
  STATE.currentCSV = { name, path, headers: preview.headers, totalRows: preview.totalRows };
  renderCSVPreview(preview);
  renderSignalSelector(preview.headers);
};

// ═══════════════════════════════════════════
// Refresh Selectors (for later tabs)
// ═══════════════════════════════════════════
async function refreshSelectors() {
  // Models
  const modelsRes = await fetch('/api/pipeline/models');
  const modelsData = await modelsRes.json();

  const pthFiles = modelsData.models || [];
  const configFiles = modelsData.configs || [];
  const scalerFiles = pthFiles.map(f => ({
    ...f,
    name: f.name.replace('.pth', '_scalers.pkl'),
    path: f.path.replace('.pth', '_scalers.pkl'),
  }));

  fillSelect('resModelSelect', pthFiles);
  fillSelect('resScalerSelect', scalerFiles);
  fillSelect('infS1Model', pthFiles.filter(f => f.name.includes('stage1') || !f.name.includes('stage2')));
  fillSelect('infS1Scalers', scalerFiles.filter(f => f.name.includes('stage1') || !f.name.includes('stage2')));
  fillSelect('infS2Model', pthFiles.filter(f => f.name.includes('stage2')));
  fillSelect('infS2Scalers', scalerFiles.filter(f => f.name.includes('stage2')));
  fillSelect('s2Stage1ConfigSelect', configFiles);

  // CSV files
  const csvRes = await fetch('/api/data/csv-files');
  const csvFiles = await csvRes.json();
  fillSelect('resDataSelect', csvFiles);
  fillSelect('infDataSelect', csvFiles);

  // Config files
  const configRes = await fetch('/api/data/config-files');
  const cfgFiles = await configRes.json();
  fillSelect('resConfigSelect', cfgFiles);
  fillSelect('s2ConfigSelect', cfgFiles);
  fillSelect('infConfigSelect', cfgFiles);

  // Residual files
  const resFilesRes = await fetch('/api/pipeline/residual-files');
  const resFiles = await resFilesRes.json();
  fillSelect('s2ResidualsSelect', resFiles.filter(f => f.name.endsWith('.csv')));
  fillSelect('infResMetrics', resFiles.filter(f => f.name.endsWith('.json')), true);
}

function fillSelect(id, files, keepFirst = false) {
  const sel = $(id);
  const first = keepFirst ? sel.options[0] : null;
  sel.innerHTML = '';
  if (first) sel.appendChild(first);
  else sel.innerHTML = '<option value="">-- select --</option>';
  files.forEach(f => {
    const opt = document.createElement('option');
    opt.value = f.path;
    opt.textContent = f.name;
    sel.appendChild(opt);
  });
}

// ═══════════════════════════════════════════
// TAB 2: Stage1 Training
// ═══════════════════════════════════════════
let s1Chart = null;
const s1History = { train_loss: [], val_loss: [], train_r2: [], val_r2: [] };

$('btnTrainS1').addEventListener('click', () => {
  if (!STATE.currentCSV) { alert('Please upload data first (Tab 1).'); return; }
  if (!STATE.boundary.length || !STATE.target.length) { alert('Please configure signals first (Tab 1).'); return; }

  // Save config first if not already saved
  if (!STATE.configPath) {
    const tmpName = '_tmp_signals';
    fetch('/api/data/save-config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: tmpName, boundary: STATE.boundary, target: STATE.target }),
    }).then(r => r.json()).then(data => {
      STATE.configPath = data.path;
      startStage1Training();
    });
  } else {
    startStage1Training();
  }
});

function startStage1Training() {
  clearLog('s1Log');
  show('s1ProgressCard');
  hide('s1ResultCard');
  show('btnAbortS1');
  $('btnTrainS1').disabled = true;
  setStatus('Stage1 Training...');

  // Reset
  s1History.train_loss = []; s1History.val_loss = [];
  s1History.train_r2 = []; s1History.val_r2 = [];
  if (s1Chart) { s1Chart.destroy(); s1Chart = null; }

  const params = new URLSearchParams({
    data: STATE.currentCSV.path,
    config: STATE.configPath,
    name: $('s1Name').value || '',
    d_model: $('s1Dmodel').value,
    nhead: $('s1Nhead').value,
    num_layers: $('s1Layers').value,
    dropout: $('s1Dropout').value,
    epochs: $('s1Epochs').value,
    batch_size: $('s1Batch').value,
    lr: $('s1Lr').value,
    weight_decay: $('s1Wd').value,
    grad_clip: $('s1GradClip').value,
    patience: $('s1Patience').value,
    test_size: $('s1TestSize').value,
    val_size: $('s1ValSize').value,
  });

  connectSSE(`/api/pipeline/train-stage1?${params}`, {
    logPanel: 's1Log',
    onMessage: (msg) => {
      if (msg.type === 'epoch') {
        const pct = (msg.epoch / msg.total * 100).toFixed(0);
        $('s1ProgressBar').style.width = pct + '%';
        $('s1EpochInfo').textContent = `Epoch ${msg.epoch} / ${msg.total}`;

        s1History.train_loss.push(msg.train_loss);
        s1History.val_loss.push(msg.val_loss);
        s1History.train_r2.push(msg.train_r2);
        s1History.val_r2.push(msg.val_r2);

        // Live metrics
        $('s1MetricsLive').innerHTML = `
          <div class="metric-card"><div class="value">${formatNum(msg.val_loss,6)}</div><div class="label">Val Loss</div></div>
          <div class="metric-card"><div class="value">${formatNum(msg.val_r2)}</div><div class="label">Val R²</div></div>
          <div class="metric-card"><div class="value">${formatNum(msg.train_loss,6)}</div><div class="label">Train Loss</div></div>
          <div class="metric-card"><div class="value">${formatNum(msg.train_r2)}</div><div class="label">Train R²</div></div>
        `;

        updateTrainingChart('s1LossChart', s1History, (c) => { s1Chart = c; });
      }
      if (msg.type === 'test_metrics') {
        show('s1ResultCard');
        $('s1TestMetrics').innerHTML = `
          <div class="metric-card"><div class="value">${formatNum(msg.overall_r2)}</div><div class="label">Overall R²</div></div>
          <div class="metric-card"><div class="value">${formatNum(msg.overall_mae, 6)}</div><div class="label">Overall MAE</div></div>
          <div class="metric-card"><div class="value">${formatNum(msg.overall_rmse, 6)}</div><div class="label">Overall RMSE</div></div>
        `;
        if (msg.per_signal) {
          renderPerSignalTable('s1PerSignalMetrics', msg.per_signal);
        }
      }
      if (msg.type === 'complete') {
        STATE.stage1Result = msg;
        appendLog('s1Log', `Model saved: ${msg.model_path}`, 'success');
      }
    },
    onDone: () => {
      hide('btnAbortS1');
      $('btnTrainS1').disabled = false;
      setStatus('Stage1 Complete');
    },
    onError: () => {
      hide('btnAbortS1');
      $('btnTrainS1').disabled = false;
      setStatus('Stage1 Error');
    },
  });
}

$('btnAbortS1').addEventListener('click', () => {
  fetch('/api/pipeline/abort', { method: 'POST' });
  appendLog('s1Log', 'Aborting...', 'error');
});

// ═══════════════════════════════════════════
// TAB 3: Residual Extraction
// ═══════════════════════════════════════════
$('btnExtractRes').addEventListener('click', () => {
  const model = $('resModelSelect').value;
  const scalers = $('resScalerSelect').value;
  const data = $('resDataSelect').value;
  const config = $('resConfigSelect').value;

  if (!model || !scalers || !data || !config) {
    alert('Please select all required fields.');
    return;
  }

  clearLog('resLog');
  hide('resResultCard');
  show('btnAbortRes');
  setStatus('Extracting Residuals...');

  const params = new URLSearchParams({ model, scalers, data, config });

  connectSSE(`/api/pipeline/extract-residuals?${params}`, {
    logPanel: 'resLog',
    onMessage: (msg) => {
      if (msg.type === 'metrics') {
        show('resResultCard');
        $('resMetrics').innerHTML = `
          <div class="metric-card"><div class="value">${formatNum(msg.overall_r2)}</div><div class="label">Stage1 Overall R²</div></div>
        `;
        if (msg.per_signal) renderPerSignalTable('resPerSignal', msg.per_signal);
      }
      if (msg.type === 'complete') {
        STATE.residualsPath = msg.residuals_csv;
        appendLog('resLog', `Residuals saved: ${msg.residuals_csv}`, 'success');
      }
    },
    onDone: () => { hide('btnAbortRes'); setStatus('Residuals Extracted'); },
    onError: () => { hide('btnAbortRes'); setStatus('Error'); },
  });
});

$('btnAbortRes').addEventListener('click', () => {
  fetch('/api/pipeline/abort', { method: 'POST' });
});

// ═══════════════════════════════════════════
// TAB 4: Stage2 Training
// ═══════════════════════════════════════════
let s2Chart = null;
const s2History = { train_loss: [], val_loss: [], train_r2: [], val_r2: [] };

$('btnTrainS2').addEventListener('click', () => {
  const residuals = $('s2ResidualsSelect').value;
  const config = $('s2ConfigSelect').value;
  const stage1Config = $('s2Stage1ConfigSelect').value;

  if (!residuals || !config || !stage1Config) {
    alert('Please select residuals CSV, signal config, and Stage1 config.');
    return;
  }

  clearLog('s2Log');
  show('s2ProgressCard');
  hide('s2ResultCard');
  show('btnAbortS2');
  $('btnTrainS2').disabled = true;
  setStatus('Stage2 Training...');

  s2History.train_loss = []; s2History.val_loss = [];
  s2History.train_r2 = []; s2History.val_r2 = [];
  if (s2Chart) { s2Chart.destroy(); s2Chart = null; }

  const params = new URLSearchParams({
    residuals,
    config,
    stage1_config: stage1Config,
    name: $('s2Name').value || '',
    d_model: $('s2Dmodel').value,
    nhead: $('s2Nhead').value,
    num_layers: $('s2Layers').value,
    dropout: $('s2Dropout').value,
    epochs: $('s2Epochs').value,
    batch_size: $('s2Batch').value,
    lr: $('s2Lr').value,
    patience: $('s2Patience').value,
  });

  connectSSE(`/api/pipeline/train-stage2?${params}`, {
    logPanel: 's2Log',
    onMessage: (msg) => {
      if (msg.type === 'epoch') {
        const pct = (msg.epoch / msg.total * 100).toFixed(0);
        $('s2ProgressBar').style.width = pct + '%';
        $('s2EpochInfo').textContent = `Epoch ${msg.epoch} / ${msg.total}`;

        s2History.train_loss.push(msg.train_loss);
        s2History.val_loss.push(msg.val_loss);
        s2History.train_r2.push(msg.train_r2);
        s2History.val_r2.push(msg.val_r2);

        $('s2MetricsLive').innerHTML = `
          <div class="metric-card"><div class="value">${formatNum(msg.val_loss,6)}</div><div class="label">Val Loss</div></div>
          <div class="metric-card"><div class="value">${formatNum(msg.val_r2)}</div><div class="label">Val R²</div></div>
          <div class="metric-card"><div class="value">${formatNum(msg.train_loss,6)}</div><div class="label">Train Loss</div></div>
          <div class="metric-card"><div class="value">${formatNum(msg.train_r2)}</div><div class="label">Train R²</div></div>
        `;

        updateTrainingChart('s2LossChart', s2History, (c) => { s2Chart = c; });
      }
      if (msg.type === 'test_metrics') {
        show('s2ResultCard');
        $('s2TestMetrics').innerHTML = `
          <div class="metric-card"><div class="value">${formatNum(msg.overall_r2)}</div><div class="label">Overall R² (Residual)</div></div>
        `;
      }
      if (msg.type === 'complete') {
        STATE.stage2Result = msg;
        appendLog('s2Log', `Stage2 model saved: ${msg.model_path}`, 'success');
      }
    },
    onDone: () => { hide('btnAbortS2'); $('btnTrainS2').disabled = false; setStatus('Stage2 Complete'); },
    onError: () => { hide('btnAbortS2'); $('btnTrainS2').disabled = false; setStatus('Error'); },
  });
});

$('btnAbortS2').addEventListener('click', () => {
  fetch('/api/pipeline/abort', { method: 'POST' });
});

// ═══════════════════════════════════════════
// TAB 5: Ensemble Inference
// ═══════════════════════════════════════════
$('btnRunInference').addEventListener('click', () => {
  const s1m = $('infS1Model').value;
  const s1s = $('infS1Scalers').value;
  const s2m = $('infS2Model').value;
  const s2s = $('infS2Scalers').value;
  const data = $('infDataSelect').value;
  const config = $('infConfigSelect').value;

  if (!s1m || !s1s || !s2m || !s2s || !data || !config) {
    alert('Please select all required fields.');
    return;
  }

  clearLog('infLog');
  hide('infResultCard');
  show('btnAbortInf');
  setStatus('Running Inference...');

  const params = new URLSearchParams({
    stage1_model: s1m,
    stage1_scalers: s1s,
    stage2_model: s2m,
    stage2_scalers: s2s,
    data,
    config,
    r2_threshold: $('infR2Threshold').value,
  });
  const resMetrics = $('infResMetrics').value;
  if (resMetrics) params.set('residual_metrics', resMetrics);

  connectSSE(`/api/pipeline/inference-ensemble?${params}`, {
    logPanel: 'infLog',
    onMessage: (msg) => {
      if (msg.type === 'comparison') {
        show('infResultCard');
        $('infOverallMetrics').innerHTML = `
          <div class="metric-card"><div class="value">${formatNum(msg.stage1_r2)}</div><div class="label">Stage1 R²</div></div>
          <div class="metric-card"><div class="value">${formatNum(msg.ensemble_r2)}</div><div class="label">Ensemble R²</div></div>
          <div class="metric-card"><div class="value ${msg.delta_r2 >= 0 ? 'text-success' : 'text-danger'}">${msg.delta_r2 >= 0 ? '+' : ''}${formatNum(msg.delta_r2)}</div><div class="label">Delta R²</div></div>
        `;
        if (msg.per_signal) renderEnsembleTable('infPerSignal', msg.per_signal);
      }
      if (msg.type === 'complete') {
        STATE.predictionsCSV = msg.predictions_csv;
        STATE.resultsJSON = msg.results_json;
        show('btnGenCharts');
        appendLog('infLog', `Predictions saved: ${msg.predictions_csv}`, 'success');
      }
    },
    onDone: () => { hide('btnAbortInf'); setStatus('Inference Complete'); },
    onError: () => { hide('btnAbortInf'); setStatus('Error'); },
  });
});

$('btnAbortInf').addEventListener('click', () => {
  fetch('/api/pipeline/abort', { method: 'POST' });
});

// Generate Charts
$('btnGenCharts').addEventListener('click', () => {
  if (!STATE.predictionsCSV || !STATE.resultsJSON) {
    alert('Run inference first.'); return;
  }
  setStatus('Generating Charts...');
  appendLog('infLog', 'Generating visualization charts...', 'info');

  const params = new URLSearchParams({
    predictions: STATE.predictionsCSV,
    results: STATE.resultsJSON,
  });

  connectSSE(`/api/pipeline/generate-charts?${params}`, {
    logPanel: 'infLog',
    onMessage: (msg) => {
      if (msg.type === 'complete') {
        appendLog('infLog', `Generated ${msg.total} charts`, 'success');
      }
    },
    onDone: () => { setStatus('Charts Generated'); },
  });
});

// ═══════════════════════════════════════════
// TAB 6: Results
// ═══════════════════════════════════════════
$('btnRefreshCharts').addEventListener('click', loadCharts);
$('btnDownloadPredCSV').addEventListener('click', () => {
  if (STATE.predictionsCSV) {
    window.open(`/api/download?path=${encodeURIComponent(STATE.predictionsCSV)}`);
  }
});
$('btnDownloadResultJSON').addEventListener('click', () => {
  if (STATE.resultsJSON) {
    window.open(`/api/download?path=${encodeURIComponent(STATE.resultsJSON)}`);
  }
});

async function loadCharts() {
  const res = await fetch('/api/pipeline/result-files');
  const files = await res.json();
  const pngs = files.filter(f => f.name.endsWith('.png'));

  // Also check charts subdirectory
  const grid = $('chartGrid');
  const noMsg = $('noChartsMsg');

  // Try loading from workspace/results/charts/
  const chartsFiles = [];
  try {
    const res2 = await fetch('/api/pipeline/result-files');
    const all = await res2.json();
    all.forEach(f => {
      if (f.name.endsWith('.png')) chartsFiles.push(f);
    });
  } catch {}

  // Scan for chart images via workspace static serving
  if (pngs.length === 0 && chartsFiles.length === 0) {
    show(noMsg);
    grid.innerHTML = '';
    return;
  }

  hide(noMsg);
  const allPngs = [...pngs, ...chartsFiles];
  // Deduplicate by name
  const seen = new Set();
  const unique = allPngs.filter(f => {
    if (seen.has(f.name)) return false;
    seen.add(f.name);
    return true;
  });

  grid.innerHTML = unique.map(f => {
    // Build a workspace-relative URL
    const relPath = f.path.split('/workspace/')[1];
    return relPath
      ? `<div><img src="/workspace/${relPath}" alt="${f.name}" loading="lazy"><div class="text-muted" style="text-align:center;padding:4px">${f.name}</div></div>`
      : '';
  }).join('');
}

// ═══════════════════════════════════════════
// Shared: Training Chart Renderer
// ═══════════════════════════════════════════
function updateTrainingChart(canvasId, history, setRef) {
  const canvas = $(canvasId);
  const ctx = canvas.getContext('2d');

  // Destroy old chart from this canvas
  const existing = Chart.getChart(canvas);
  if (existing) existing.destroy();

  const labels = history.train_loss.map((_, i) => i + 1);

  const chart = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        { label: 'Train Loss', data: history.train_loss, borderColor: '#FF9800', borderWidth: 1.5, pointRadius: 0, tension: 0.3, yAxisID: 'y' },
        { label: 'Val Loss', data: history.val_loss, borderColor: '#f87171', borderWidth: 1.5, pointRadius: 0, tension: 0.3, yAxisID: 'y' },
        { label: 'Train R²', data: history.train_r2, borderColor: '#4f8cff', borderWidth: 1.5, pointRadius: 0, tension: 0.3, yAxisID: 'y1' },
        { label: 'Val R²', data: history.val_r2, borderColor: '#34d399', borderWidth: 1.5, pointRadius: 0, tension: 0.3, yAxisID: 'y1' },
      ],
    },
    options: {
      responsive: true,
      interaction: { mode: 'index', intersect: false },
      plugins: { legend: { labels: { color: '#8b90a5', font: { size: 11 } } } },
      scales: {
        x: { title: { display: true, text: 'Epoch', color: '#8b90a5' }, ticks: { color: '#8b90a5' }, grid: { color: 'rgba(45,50,72,0.5)' } },
        y: { type: 'linear', position: 'left', title: { display: true, text: 'Loss', color: '#8b90a5' }, ticks: { color: '#8b90a5' }, grid: { color: 'rgba(45,50,72,0.5)' } },
        y1: { type: 'linear', position: 'right', title: { display: true, text: 'R²', color: '#8b90a5' }, ticks: { color: '#8b90a5' }, grid: { drawOnChartArea: false } },
      },
    },
  });

  if (setRef) setRef(chart);
}

// ═══════════════════════════════════════════
// Shared: Per-Signal Metrics Table
// ═══════════════════════════════════════════
function renderPerSignalTable(containerId, perSignal) {
  const sigs = Object.keys(perSignal);
  let html = '<div class="table-wrap"><table><thead><tr><th>Signal</th><th>R²</th><th>MAE</th><th>RMSE</th></tr></thead><tbody>';
  sigs.forEach(sig => {
    const m = perSignal[sig];
    const r2Color = m.r2 >= 0.8 ? 'text-success' : m.r2 >= 0.4 ? 'text-primary' : 'text-danger';
    html += `<tr><td>${sig}</td><td class="${r2Color}">${formatNum(m.r2)}</td><td>${formatNum(m.mae, 6)}</td><td>${formatNum(m.rmse, 6)}</td></tr>`;
  });
  html += '</tbody></table></div>';
  $(containerId).innerHTML = html;
}

function renderEnsembleTable(containerId, perSignal) {
  const sigs = Object.keys(perSignal);
  let html = '<div class="table-wrap"><table><thead><tr><th>Signal</th><th>R²</th><th>MAE</th><th>RMSE</th><th>Delta R²</th></tr></thead><tbody>';
  sigs.forEach(sig => {
    const m = perSignal[sig];
    const r2Color = m.r2 >= 0.8 ? 'text-success' : m.r2 >= 0.4 ? 'text-primary' : 'text-danger';
    const deltaColor = (m.delta_r2 || 0) >= 0 ? 'text-success' : 'text-danger';
    const deltaStr = (m.delta_r2 || 0) >= 0 ? `+${formatNum(m.delta_r2)}` : formatNum(m.delta_r2);
    html += `<tr><td>${sig}</td><td class="${r2Color}">${formatNum(m.r2)}</td><td>${formatNum(m.mae,6)}</td><td>${formatNum(m.rmse,6)}</td><td class="${deltaColor}">${deltaStr}</td></tr>`;
  });
  html += '</tbody></table></div>';
  $(containerId).innerHTML = html;
}

// ═══════════════════════════════════════════
// Init
// ═══════════════════════════════════════════
refreshCSVList();
