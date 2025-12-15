#!/bin/bash
# Quick start script for FastAPI inference service

echo "========================================"
echo "Starting FastAPI Inference Service"
echo "========================================"

# Change to repo root directory
cd "$(dirname "$0")/.." || exit 1

# Check if FastAPI is installed
if ! python -c "import fastapi" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install -r fastapi_inference/requirements.txt
fi

# Start server
echo "Starting server on http://0.0.0.0:8000"
echo "API Documentation: http://localhost:8000/docs"
echo "Press Ctrl+C to stop"
echo "========================================"

python -m fastapi_inference.main
