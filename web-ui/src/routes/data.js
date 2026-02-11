/**
 * Data routes — CSV upload, preview, signal config management.
 */

const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { DIRS, listFiles, previewCSV } = require('../utils/file-manager');

const router = express.Router();

// ── Multer for CSV upload ──
const csvStorage = multer.diskStorage({
  destination: (req, _file, cb) => cb(null, DIRS.uploads),
  filename: (req, file, cb) => cb(null, file.originalname),
});
const uploadCSV = multer({ storage: csvStorage, fileFilter: (_r, file, cb) => {
  cb(null, file.originalname.endsWith('.csv'));
}});

// ── Multer for config JSON upload ──
const configStorage = multer.diskStorage({
  destination: (req, _file, cb) => cb(null, DIRS.configs),
  filename: (req, file, cb) => cb(null, file.originalname),
});
const uploadConfig = multer({ storage: configStorage });

// ── Upload CSV ──
router.post('/upload-csv', uploadCSV.single('file'), (req, res) => {
  if (!req.file) return res.status(400).json({ error: 'No CSV file uploaded' });
  res.json({
    name: req.file.originalname,
    path: req.file.path,
    size: req.file.size,
  });
});

// ── List uploaded CSVs ──
router.get('/csv-files', (_req, res) => {
  res.json(listFiles('uploads', '.csv'));
});

// ── Preview CSV ──
router.get('/preview-csv', (req, res) => {
  const filePath = req.query.path;
  if (!filePath || !fs.existsSync(filePath)) {
    return res.status(400).json({ error: 'File not found' });
  }
  try {
    const preview = previewCSV(filePath, 30);
    res.json(preview);
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// ── Upload signal config JSON ──
router.post('/upload-config', uploadConfig.single('file'), (req, res) => {
  if (!req.file) return res.status(400).json({ error: 'No config file uploaded' });
  try {
    const content = JSON.parse(fs.readFileSync(req.file.path, 'utf-8'));
    res.json({ name: req.file.originalname, path: req.file.path, content });
  } catch (e) {
    res.status(400).json({ error: 'Invalid JSON: ' + e.message });
  }
});

// ── Save signal config (from UI) ──
router.post('/save-config', express.json(), (req, res) => {
  const { name, boundary, target } = req.body;
  if (!name || !boundary || !target) {
    return res.status(400).json({ error: 'name, boundary, and target are required' });
  }
  const filePath = path.join(DIRS.configs, name.endsWith('.json') ? name : `${name}.json`);
  const config = { boundary, target };
  fs.writeFileSync(filePath, JSON.stringify(config, null, 2));
  res.json({ path: filePath, config });
});

// ── List config files ──
router.get('/config-files', (_req, res) => {
  res.json(listFiles('configs', '.json'));
});

// ── Read a config file ──
router.get('/read-config', (req, res) => {
  const filePath = req.query.path;
  if (!filePath || !fs.existsSync(filePath)) {
    return res.status(400).json({ error: 'File not found' });
  }
  try {
    const content = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
    res.json(content);
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

module.exports = router;
