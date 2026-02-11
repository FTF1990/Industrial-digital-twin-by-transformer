/**
 * Python subprocess runner with JSON-line streaming.
 * Spawns Python scripts and parses stdout JSON messages for SSE forwarding.
 */

const { spawn } = require('child_process');
const path = require('path');

const PYTHON_DIR = path.join(__dirname, '..', '..', 'python');

/**
 * Run a Python script and stream JSON-line output.
 *
 * @param {string} script      - Script filename (e.g. 'train_stage1.py')
 * @param {string[]} args      - CLI arguments
 * @param {function} onMessage - Called with each parsed JSON object from stdout
 * @param {function} onError   - Called with stderr text
 * @returns {Promise<{code: number}>} resolves when process exits
 */
function runPython(script, args, onMessage, onError) {
  return new Promise((resolve, reject) => {
    const scriptPath = path.join(PYTHON_DIR, script);
    const proc = spawn('python3', ['-u', scriptPath, ...args], {
      cwd: PYTHON_DIR,
      env: { ...process.env, PYTHONUNBUFFERED: '1' },
    });

    let stderrBuf = '';

    proc.stdout.on('data', (chunk) => {
      const lines = chunk.toString().split('\n').filter(Boolean);
      for (const line of lines) {
        try {
          const msg = JSON.parse(line);
          if (onMessage) onMessage(msg);
        } catch {
          // Non-JSON output — treat as info
          if (onMessage) onMessage({ type: 'log', text: line });
        }
      }
    });

    proc.stderr.on('data', (chunk) => {
      stderrBuf += chunk.toString();
      if (onError) onError(chunk.toString());
    });

    proc.on('close', (code) => {
      if (code !== 0 && code !== null) {
        resolve({ code, stderr: stderrBuf });
      } else {
        resolve({ code: code || 0 });
      }
    });

    proc.on('error', (err) => {
      reject(err);
    });

    // Attach kill method so callers can abort
    runPython._currentProc = proc;
  });
}

/** Abort the currently running python process (if any). */
function abortPython() {
  if (runPython._currentProc) {
    runPython._currentProc.kill('SIGTERM');
    runPython._currentProc = null;
  }
}

module.exports = { runPython, abortPython, PYTHON_DIR };
