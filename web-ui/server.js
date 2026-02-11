/**
 * Sensor Transformer Web UI — Express Server
 *
 * Lightweight local web interface for the Industrial Digital Twin
 * training & inference pipeline (Stage1 + Stage2 SST models).
 *
 * Usage:
 *   npm start            # starts on port 3000
 *   node server.js 8080  # custom port
 */

const express = require('express');
const path = require('path');
const fs = require('fs');
const { ensureWorkspace } = require('./src/utils/file-manager');

const app = express();
const PORT = process.argv[2] || process.env.PORT || 3000;

// ── Ensure workspace directories ──
ensureWorkspace();

// ── Middleware ──
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true }));

// ── Static files ──
app.use(express.static(path.join(__dirname, 'public')));

// ── Serve workspace files (for chart images, downloads, etc.) ──
app.use('/workspace', express.static(path.join(__dirname, 'workspace')));

// ── API Routes ──
app.use('/api/data', require('./src/routes/data'));
app.use('/api/pipeline', require('./src/routes/training'));

// ── File download helper ──
app.get('/api/download', (req, res) => {
  const filePath = req.query.path;
  if (!filePath || !fs.existsSync(filePath)) {
    return res.status(404).json({ error: 'File not found' });
  }
  res.download(filePath);
});

// ── Health check ──
app.get('/api/health', (_req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// ── Start ──
app.listen(PORT, () => {
  console.log(`\n  Sensor Transformer Web UI`);
  console.log(`  ========================`);
  console.log(`  Local:  http://localhost:${PORT}`);
  console.log(`  Press Ctrl+C to stop\n`);
});
