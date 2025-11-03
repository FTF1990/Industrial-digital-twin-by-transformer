# Saved Models Directory

This directory is used to store trained models, scalers, and inference configurations.

## Supported File Types

- **`*.pth`** - PyTorch model checkpoint files
- **`*_scalers.pkl`** - Scaler files (StandardScaler for input/output normalization)
- **`*_inference.json`** - Inference configuration files

## Usage

### 1. Place Your Files Here

Simply copy your trained model files into this directory:

```bash
saved_models/
├── my_sst_model.pth
├── my_sst_model_scalers.pkl
└── my_sst_model_inference.json
```

### 2. Load in Gradio Interface

#### Tab 2: SST Model Training
- Trained models are automatically saved here

#### Tab 3: Residual Extraction
Three loading options:
- **Load from Inference Config**: Select `*_inference.json` file
- **Load from Model File**: Select `*.pth` file
- **Load Scalers**: Select `*_scalers.pkl` file

Steps:
1. Open the Gradio interface
2. Navigate to Tab 3 (🔬 残差提取)
3. Click refresh buttons (🔄) to scan this folder
4. Select files from dropdown menus
5. Click load buttons (📥) to load

### 3. File Naming Convention

Recommended naming pattern:
```
<model_name>.pth                    # Model checkpoint
<model_name>_scalers.pkl            # Scalers
<model_name>_inference.json         # Inference config
```

Example:
```
SST_20250102_143025.pth
SST_20250102_143025_scalers.pkl
SST_20250102_143025_inference.json
```

## Notes

- Files in this directory are **not tracked by git** (except this README)
- You can organize files in subdirectories - the system will scan recursively
- Model files can be large - the `.gitignore` ensures they won't be committed
