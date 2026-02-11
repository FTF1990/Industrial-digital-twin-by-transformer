/**
 * Local workspace file manager.
 * Provides helpers for the workspace/ directory structure.
 */

const fs = require('fs');
const path = require('path');

const WORKSPACE = path.join(__dirname, '..', '..', 'workspace');

const DIRS = {
  uploads:   path.join(WORKSPACE, 'uploads'),
  configs:   path.join(WORKSPACE, 'configs'),
  models:    path.join(WORKSPACE, 'models'),
  residuals: path.join(WORKSPACE, 'residuals'),
  results:   path.join(WORKSPACE, 'results'),
};

/** Ensure all workspace directories exist. */
function ensureWorkspace() {
  for (const dir of Object.values(DIRS)) {
    fs.mkdirSync(dir, { recursive: true });
  }
}

/** List files in a workspace sub-directory, with optional extension filter. */
function listFiles(dirKey, ext) {
  const dir = DIRS[dirKey];
  if (!dir || !fs.existsSync(dir)) return [];
  let files = fs.readdirSync(dir);
  if (ext) {
    files = files.filter(f => f.endsWith(ext));
  }
  return files.map(f => ({
    name: f,
    path: path.join(dir, f),
    size: fs.statSync(path.join(dir, f)).size,
    mtime: fs.statSync(path.join(dir, f)).mtime,
  }));
}

/** Read a JSON file from workspace. */
function readJSON(filePath) {
  return JSON.parse(fs.readFileSync(filePath, 'utf-8'));
}

/** Preview first N rows of a CSV file. */
function previewCSV(filePath, maxRows = 20) {
  const content = fs.readFileSync(filePath, 'utf-8');
  const lines = content.split('\n').filter(Boolean);
  const headers = lines[0].split(',').map(h => h.trim());
  const rows = [];
  for (let i = 1; i <= Math.min(maxRows, lines.length - 1); i++) {
    const vals = lines[i].split(',');
    const row = {};
    headers.forEach((h, idx) => { row[h] = vals[idx]?.trim(); });
    rows.push(row);
  }
  const totalRows = lines.length - 1;
  return { headers, rows, totalRows };
}

module.exports = { DIRS, ensureWorkspace, listFiles, readJSON, previewCSV };
