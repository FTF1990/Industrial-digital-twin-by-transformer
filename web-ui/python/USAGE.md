# Python Scripts — Standalone Usage Guide

All scripts are self-contained CLI tools. They can be used independently without the Node.js web server.

Each script prints **JSON Lines** to stdout (one JSON object per line), making them easy to integrate with any automation tool.

---

## Prerequisites

```bash
cd web-ui
pip install -r requirements.txt
```

Ensure the project root is accessible (scripts auto-detect it via relative path).

---

## 1. train_stage1.py — Stage1 SST Model Training

Trains a StaticSensorTransformer (SST) model on boundary → target sensor mapping.

**Basic usage:**

```bash
python python/train_stage1.py \
    --data /path/to/sensor_data.csv \
    --config /path/to/signals.json \
    --output ./workspace/models/
```

**Full options:**

```bash
python python/train_stage1.py \
    --data ./data.csv \
    --config ./signals.json \
    --output ./workspace/models/ \
    --name my_stage1_model \
    --d_model 128 \
    --nhead 8 \
    --num_layers 3 \
    --dropout 0.1 \
    --epochs 100 \
    --batch_size 64 \
    --lr 0.001 \
    --weight_decay 1e-5 \
    --grad_clip 1.0 \
    --patience 25 \
    --test_size 0.2 \
    --val_size 0.2
```

**Signal config format** (`signals.json`):

```json
{
    "boundary": ["sensor_temp_1", "sensor_pressure_1", "sensor_flow_1"],
    "target": ["sensor_quality_1", "sensor_vibration_1"]
}
```

**Outputs:**
- `{name}.pth` — Model weights + config + training history
- `{name}_scalers.pkl` — StandardScaler for X and y (pickle)
- `{name}_config.json` — Full inference config (signals, architecture, metrics)

**Example stdout (JSON Lines):**

```json
{"type": "info", "message": "Using device: cuda"}
{"type": "status", "stage": "training", "total_epochs": 100}
{"type": "epoch", "epoch": 1, "total": 100, "train_loss": 0.98, "val_loss": 0.95, "train_r2": 0.02, "val_r2": 0.05, "lr": 0.001, "elapsed": 1.2}
{"type": "epoch", "epoch": 2, "total": 100, "train_loss": 0.85, "val_loss": 0.82, ...}
...
{"type": "test_metrics", "overall_r2": 0.85, "overall_mae": 0.12, "per_signal": {...}}
{"type": "complete", "model_name": "my_stage1_model", "model_path": "...", "overall_r2": 0.85}
```

---

## 2. extract_residuals.py — Residual Extraction

Computes residuals = y_true - y_pred (in **original scale**) from a trained Stage1 model.

```bash
python python/extract_residuals.py \
    --model ./workspace/models/stage1.pth \
    --scalers ./workspace/models/stage1_scalers.pkl \
    --data ./data.csv \
    --config ./signals.json \
    --output ./workspace/residuals/ \
    --batch_size 1024
```

**Outputs:**
- `residuals.csv` — Contains boundary signals + `{signal}_residual` + `{signal}_true` + `{signal}_pred` columns
- `residual_metrics.json` — Per-signal MAE, RMSE, R², residual mean/std

---

## 3. train_stage2.py — Stage2 Residual Boost Training

Trains a Stage2 SST model on the extracted residuals.

```bash
python python/train_stage2.py \
    --residuals ./workspace/residuals/residuals.csv \
    --config ./signals.json \
    --stage1_config ./workspace/models/stage1_config.json \
    --output ./workspace/models/ \
    --name my_stage2_model \
    --epochs 100 \
    --batch_size 64 \
    --lr 0.001 \
    --patience 25
```

**Key:** `--stage1_config` is required to ensure the same data split ratios as Stage1.

**Outputs:** Same as Stage1 (`.pth`, `_scalers.pkl`, `_config.json`)

---

## 4. inference_ensemble.py — Combined Stage1 + Stage2 Inference

Runs selective boosting: signals with Stage1 R² below the threshold get Stage2 correction.

```bash
python python/inference_ensemble.py \
    --stage1_model ./workspace/models/stage1.pth \
    --stage1_scalers ./workspace/models/stage1_scalers.pkl \
    --stage2_model ./workspace/models/stage2.pth \
    --stage2_scalers ./workspace/models/stage2_scalers.pkl \
    --data ./data.csv \
    --config ./signals.json \
    --output ./workspace/results/ \
    --r2_threshold 0.4 \
    --residual_metrics ./workspace/residuals/residual_metrics.json
```

**R² Threshold Logic:**
- If a signal's Stage1 R² < threshold → apply Stage2 boost: `y_final = y_stage1 + y_residual`
- If a signal's Stage1 R² >= threshold → use Stage1 only: `y_final = y_stage1`
- `--residual_metrics` provides pre-computed R² scores. If omitted, R² is computed from the data (requires target columns in CSV).

**Outputs:**
- `predictions.csv` — Columns: boundary signals, `{sig}_actual`, `{sig}_stage1`, `{sig}_ensemble`, `{sig}_residual_pred`
- `inference_results.json` — Per-signal metrics, boosting info, delta R²

---

## 5. generate_charts.py — Visualization

Generates per-signal comparison PNG charts + an overview R² bar chart.

```bash
python python/generate_charts.py \
    --predictions ./workspace/results/predictions.csv \
    --results ./workspace/results/inference_results.json \
    --output ./workspace/results/charts/ \
    --max_samples 2000
```

**Outputs:**
- `{signal_name}.png` — Per-signal: actual vs stage1 vs ensemble + residual plot
- `_overview_r2_comparison.png` — Bar chart comparing R² across all signals

---

## Full Pipeline Example (CLI only)

```bash
# 1. Train Stage1
python python/train_stage1.py \
    --data ./data/raw/sensors.csv \
    --config ./configs/signals.json \
    --output ./workspace/models/ \
    --name stage1_v1

# 2. Extract residuals
python python/extract_residuals.py \
    --model ./workspace/models/stage1_v1.pth \
    --scalers ./workspace/models/stage1_v1_scalers.pkl \
    --data ./data/raw/sensors.csv \
    --config ./configs/signals.json \
    --output ./workspace/residuals/

# 3. Train Stage2
python python/train_stage2.py \
    --residuals ./workspace/residuals/residuals.csv \
    --config ./configs/signals.json \
    --stage1_config ./workspace/models/stage1_v1_config.json \
    --output ./workspace/models/ \
    --name stage2_v1

# 4. Ensemble inference
python python/inference_ensemble.py \
    --stage1_model ./workspace/models/stage1_v1.pth \
    --stage1_scalers ./workspace/models/stage1_v1_scalers.pkl \
    --stage2_model ./workspace/models/stage2_v1.pth \
    --stage2_scalers ./workspace/models/stage2_v1_scalers.pkl \
    --data ./data/raw/sensors.csv \
    --config ./configs/signals.json \
    --output ./workspace/results/ \
    --r2_threshold 0.4 \
    --residual_metrics ./workspace/residuals/residual_metrics.json

# 5. Generate charts
python python/generate_charts.py \
    --predictions ./workspace/results/predictions.csv \
    --results ./workspace/results/inference_results.json \
    --output ./workspace/results/charts/
```

---

## JSON Lines Protocol

All scripts output structured JSON to stdout, one object per line:

| type | Description |
|------|-------------|
| `info` | Informational message (`message` field) |
| `status` | Pipeline stage change (`stage` field) |
| `epoch` | Training epoch metrics (loss, R², lr, elapsed) |
| `progress` | Batch progress for inference (`batch`, `total`, `pct`) |
| `test_metrics` | Final test evaluation (overall + per-signal) |
| `metrics` | Residual extraction metrics |
| `comparison` | Ensemble vs Stage1 comparison |
| `complete` | Task complete with output file paths |
| `error` | Error message |

This makes it trivial to parse output in any language:

```bash
# Filter only epoch lines
python python/train_stage1.py --data ... 2>/dev/null | grep '"type": "epoch"'

# Get final R² with jq
python python/train_stage1.py --data ... 2>/dev/null | grep '"type": "complete"' | jq '.overall_r2'
```
