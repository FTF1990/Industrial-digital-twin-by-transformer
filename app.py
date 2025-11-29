#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hugging Face Spaces Deployment Version
Industrial Digital Twin with Residual Boost Training
"""
import os
import sys

# Set environment for Hugging Face Spaces
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Import the main application
from gradio_sensor_transformer_app import create_unified_interface

if __name__ == "__main__":
    print("🚀 Starting Industrial Digital Twin on Hugging Face Spaces...")
    print("="*80)

    # Create necessary directories
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("saved_models/stage2_boost", exist_ok=True)
    os.makedirs("saved_models/ensemble", exist_ok=True)
    os.makedirs("saved_models/reinference_results", exist_ok=True)
    os.makedirs("saved_models/residuals_data", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    print("✅ Created necessary directories")

    # Build the interface
    demo = create_unified_interface()
    print("✅ UI built successfully")
    print("="*80)

    # Launch for Hugging Face Spaces
    print("\n🌐 Launching on Hugging Face Spaces...")
    demo.launch(
        server_name="0.0.0.0",  # Important for HF Spaces
        server_port=7860,        # HF Spaces default port
        share=False,             # Not needed on HF Spaces
        debug=True,
        show_error=True
    )
