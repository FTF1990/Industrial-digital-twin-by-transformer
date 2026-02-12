/**
 * Training routes — Stage1 & Stage2 training with SSE progress streaming.
 */

const express = require('express');
const path = require('path');
const { runPython, abortPython } = require('../utils/python-runner');
const { DIRS, listFiles } = require('../utils/file-manager');

const router = express.Router();

// ── Generic SSE training endpoint ──
function createTrainingEndpoint(scriptName, buildArgs) {
  return (req, res) => {
    // SSE headers
    res.writeHead(200, {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
    });

    const send = (data) => {
      res.write(`data: ${JSON.stringify(data)}\n\n`);
    };

    const args = buildArgs(req.query);
    send({ type: 'info', message: `Starting ${scriptName} ...`, args: args });

    runPython(
      scriptName,
      args,
      (msg) => send(msg),
      (stderr) => {
        // Only forward meaningful stderr (skip Python warnings)
        const text = stderr.trim();
        if (text && !text.includes('UserWarning') && !text.includes('FutureWarning')) {
          send({ type: 'stderr', text });
        }
      }
    ).then(({ code, stderr }) => {
      if (code !== 0) {
        send({ type: 'error', message: `Process exited with code ${code}`, stderr });
      }
      send({ type: 'done', code });
      res.end();
    }).catch((err) => {
      send({ type: 'error', message: err.message });
      res.end();
    });

    req.on('close', () => {
      abortPython();
    });
  };
}

// ── Stage1 Training (SSE) ──
router.get('/train-stage1', createTrainingEndpoint('train_stage1.py', (q) => {
  const args = [
    '--data', q.data,
    '--config', q.config,
    '--output', q.output || DIRS.models,
  ];
  if (q.name) args.push('--name', q.name);
  if (q.d_model) args.push('--d_model', q.d_model);
  if (q.nhead) args.push('--nhead', q.nhead);
  if (q.num_layers) args.push('--num_layers', q.num_layers);
  if (q.dropout) args.push('--dropout', q.dropout);
  if (q.epochs) args.push('--epochs', q.epochs);
  if (q.batch_size) args.push('--batch_size', q.batch_size);
  if (q.lr) args.push('--lr', q.lr);
  if (q.weight_decay) args.push('--weight_decay', q.weight_decay);
  if (q.grad_clip) args.push('--grad_clip', q.grad_clip);
  if (q.patience) args.push('--patience', q.patience);
  if (q.test_size) args.push('--test_size', q.test_size);
  if (q.val_size) args.push('--val_size', q.val_size);
  return args;
}));

// ── Residual Extraction (SSE) ──
router.get('/extract-residuals', createTrainingEndpoint('extract_residuals.py', (q) => {
  const args = [
    '--model', q.model,
    '--scalers', q.scalers,
    '--data', q.data,
    '--config', q.config,
    '--output', q.output || DIRS.residuals,
  ];
  if (q.batch_size) args.push('--batch_size', q.batch_size);
  return args;
}));

// ── Stage2 Training (SSE) ──
router.get('/train-stage2', createTrainingEndpoint('train_stage2.py', (q) => {
  const args = [
    '--residuals', q.residuals,
    '--config', q.config,
    '--stage1_config', q.stage1_config,
    '--output', q.output || DIRS.models,
  ];
  if (q.name) args.push('--name', q.name);
  if (q.d_model) args.push('--d_model', q.d_model);
  if (q.nhead) args.push('--nhead', q.nhead);
  if (q.num_layers) args.push('--num_layers', q.num_layers);
  if (q.dropout) args.push('--dropout', q.dropout);
  if (q.epochs) args.push('--epochs', q.epochs);
  if (q.batch_size) args.push('--batch_size', q.batch_size);
  if (q.lr) args.push('--lr', q.lr);
  if (q.weight_decay) args.push('--weight_decay', q.weight_decay);
  if (q.grad_clip) args.push('--grad_clip', q.grad_clip);
  if (q.patience) args.push('--patience', q.patience);
  if (q.stage2_mask_mode) args.push('--stage2_mask_mode', q.stage2_mask_mode);
  return args;
}));

// ── Ensemble Inference (SSE) ──
router.get('/inference-ensemble', createTrainingEndpoint('inference_ensemble.py', (q) => {
  const args = [
    '--stage1_model', q.stage1_model,
    '--stage1_scalers', q.stage1_scalers,
    '--stage2_model', q.stage2_model,
    '--stage2_scalers', q.stage2_scalers,
    '--data', q.data,
    '--config', q.config,
    '--output', q.output || DIRS.results,
  ];
  if (q.r2_threshold) args.push('--r2_threshold', q.r2_threshold);
  if (q.residual_metrics) args.push('--residual_metrics', q.residual_metrics);
  if (q.batch_size) args.push('--batch_size', q.batch_size);
  return args;
}));

// ── Generate Charts (SSE) ──
router.get('/generate-charts', createTrainingEndpoint('generate_charts.py', (q) => {
  const chartsDir = path.join(q.output || DIRS.results, 'charts');
  return [
    '--predictions', q.predictions,
    '--results', q.results,
    '--output', chartsDir,
  ];
}));

// ── List trained models ──
router.get('/models', (_req, res) => {
  const pthFiles = listFiles('models', '.pth');
  const configFiles = listFiles('models', '_config.json');
  res.json({ models: pthFiles, configs: configFiles });
});

// ── List residual files ──
router.get('/residual-files', (_req, res) => {
  res.json(listFiles('residuals'));
});

// ── List result files ──
router.get('/result-files', (_req, res) => {
  res.json(listFiles('results'));
});

// ── Abort running process ──
router.post('/abort', (_req, res) => {
  abortPython();
  res.json({ message: 'Abort signal sent' });
});

module.exports = router;
