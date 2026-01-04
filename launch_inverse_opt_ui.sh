#!/bin/bash
# Launch Inverse Optimization Gradio Interface

echo "========================================="
echo "Inverse Optimization Web Interface"
echo "========================================="
echo ""
echo "Starting Gradio server..."
echo "Access the interface at: http://localhost:7861"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python gradio_apps/inverse_control_interface.py
