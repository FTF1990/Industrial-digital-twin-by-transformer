"""
FastAPI Configuration
"""
import os

# API Configuration
API_TITLE = "Industrial Digital Twin - Inference API"
API_DESCRIPTION = """
FastAPI service for Industrial Digital Twin inference using Stage1 SST and Residual Boost models.

## Features

* **Model Management**: Load and manage Stage1 and Residual Boost models
* **Ensemble Generation**: Create ensemble models using Delta R² strategy
* **Batch Inference**: Perform batch predictions on new data
* **Dynamic Threshold**: Update Delta R² threshold without re-inference

## Workflow

1. Load Stage1 model via `/api/v1/models/stage1/load`
2. Load Residual Boost model via `/api/v1/models/residual-boost/load`
3. Create ensemble via `/api/v1/ensemble/create` with evaluation data
4. Run inference via `/api/v1/inference/batch` with new data
5. Optionally update threshold via `/api/v1/ensemble/{name}/update-threshold`
"""
API_VERSION = "1.0.0"
API_CONTACT = {
    "name": "FTF1990",
    "url": "https://github.com/FTF1990/Industrial-digital-twin-by-transformer"
}

# Server Configuration
HOST = "0.0.0.0"
PORT = 8000
RELOAD = False  # Set to True for development

# CORS Configuration
ALLOW_ORIGINS = ["*"]  # Adjust for production
ALLOW_CREDENTIALS = True
ALLOW_METHODS = ["*"]
ALLOW_HEADERS = ["*"]

# Paths
ENSEMBLE_SAVE_DIR = "../saved_models/ensemble"
RESULTS_DIR = "results"

# Create directories
os.makedirs(ENSEMBLE_SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
