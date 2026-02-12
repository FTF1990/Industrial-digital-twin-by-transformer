#!/usr/bin/env python3
"""
Residual Extraction Script

Extracts Stage1 residuals (y_true - y_pred) in original scale for Stage2 training.
Supports both SST and MaskedSST model types.

Usage:
    python extract_residuals.py --model stage1.pth --scalers stage1_scalers.pkl \
        --data data.csv --config signals.json --output ./residuals/
"""

import argparse
import json
import sys
import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import autocast
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

WEBUI_PYTHON = os.path.dirname(__file__)
sys.path.insert(0, WEBUI_PYTHON)


def emit(msg_type, **kwargs):
    msg = {"type": msg_type, **kwargs}
    print(json.dumps(msg, ensure_ascii=False), flush=True)


def load_model_auto(checkpoint, device):
    """Load model from checkpoint, auto-detecting SST vs MaskedSST."""
    model_type = checkpoint.get('model_type', 'SST')
    cfg = checkpoint['model_config']

    if model_type == 'MaskedSST':
        from models.masked_sst import MaskedSST
        signal_mapping_data = checkpoint.get('signal_mapping', {})
        mask = None
        if 'mask_matrix' in signal_mapping_data:
            mask = torch.tensor(signal_mapping_data['mask_matrix'], dtype=torch.float32)

        model = MaskedSST(
            num_input_signals=cfg['num_input_signals'],
            num_output_signals=cfg['num_output_signals'],
            d_model=cfg['d_model'],
            nhead=cfg['nhead'],
            num_layers=cfg['num_layers'],
            dropout=cfg.get('dropout', 0.1),
            mask_matrix=mask
        ).to(device)
        emit("info", message=f"Loaded MaskedSST model (Signal Mapping enabled)")
    else:
        from models.static_transformer import StaticSensorTransformer
        model = StaticSensorTransformer(
            num_boundary_sensors=cfg.get('num_boundary_sensors', cfg.get('num_input_signals')),
            num_target_sensors=cfg.get('num_target_sensors', cfg.get('num_output_signals')),
            d_model=cfg['d_model'],
            nhead=cfg['nhead'],
            num_layers=cfg['num_layers'],
            dropout=cfg.get('dropout', 0.1)
        ).to(device)
        emit("info", message="Loaded standard SST model")

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description='Extract Stage1 Residuals')
    parser.add_argument('--model', required=True, help='Path to Stage1 .pth model')
    parser.add_argument('--scalers', required=True, help='Path to Stage1 scalers .pkl')
    parser.add_argument('--data', required=True, help='Path to CSV data file')
    parser.add_argument('--config', required=True, help='Path to signal config JSON')
    parser.add_argument('--output', default='./residuals', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=1024)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    emit("info", message=f"Using device: {device}")

    # ── Load Config ──
    with open(args.config, 'r') as f:
        config = json.load(f)
    boundary_signals = config['boundary']
    target_signals = config['target']

    # ── Load Data ──
    emit("status", stage="loading_data")
    df = pd.read_csv(args.data)
    X = df[boundary_signals].values.astype(np.float32)
    y_true = df[target_signals].values.astype(np.float32)

    # Handle NaN
    nan_mask = np.isnan(X).any(axis=1) | np.isnan(y_true).any(axis=1)
    if nan_mask.any():
        emit("info", message=f"Dropping {nan_mask.sum()} rows with NaN")
        X = X[~nan_mask]
        y_true = y_true[~nan_mask]

    emit("info", message=f"Data: {X.shape[0]} samples, {X.shape[1]} boundary, {y_true.shape[1]} target")

    # ── Load Model ──
    emit("status", stage="loading_model")
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    model = load_model_auto(checkpoint, device)

    # ── Load Scalers ──
    with open(args.scalers, 'rb') as f:
        scalers = pickle.load(f)
    scaler_X = scalers['X']
    scaler_y = scalers['y']

    # ── Batch Inference ──
    emit("status", stage="extracting_residuals")
    X_scaled = scaler_X.transform(X)
    use_amp = device.type == 'cuda'

    y_pred_scaled_list = []
    total_batches = (len(X_scaled) + args.batch_size - 1) // args.batch_size

    with torch.no_grad():
        for i in range(0, len(X_scaled), args.batch_size):
            batch = torch.FloatTensor(X_scaled[i:i + args.batch_size]).to(device)
            if use_amp:
                with autocast():
                    pred = model(batch).cpu().numpy()
            else:
                pred = model(batch).cpu().numpy()
            y_pred_scaled_list.append(pred)

            batch_num = i // args.batch_size + 1
            if batch_num % 10 == 0 or batch_num == total_batches:
                emit("progress", batch=batch_num, total=total_batches,
                     pct=round(100 * batch_num / total_batches, 1))

    y_pred_scaled = np.vstack(y_pred_scaled_list)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    # ── Compute Residuals ──
    residuals = y_true - y_pred

    # ── Per-signal Metrics ──
    emit("status", stage="computing_metrics")
    per_signal_metrics = {}
    r2_scores = []

    for i, sig in enumerate(target_signals):
        mae = float(mean_absolute_error(y_true[:, i], y_pred[:, i]))
        rmse = float(np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i])))
        var = np.var(y_true[:, i])
        r2 = float(r2_score(y_true[:, i], y_pred[:, i])) if var > 1e-10 else 0.0
        r2_scores.append(r2)

        residual_mean = float(np.mean(residuals[:, i]))
        residual_std = float(np.std(residuals[:, i]))

        per_signal_metrics[sig] = {
            "mae": round(mae, 6),
            "rmse": round(rmse, 6),
            "r2": round(r2, 4),
            "residual_mean": round(residual_mean, 6),
            "residual_std": round(residual_std, 6),
        }

    overall_r2 = float(np.mean([m['r2'] for m in per_signal_metrics.values()]))

    emit("metrics", overall_r2=round(overall_r2, 4), per_signal=per_signal_metrics)

    # ── Save Outputs ──
    emit("status", stage="saving")
    os.makedirs(args.output, exist_ok=True)

    # Save residuals CSV (boundary signals + residual columns)
    residuals_df = pd.DataFrame()
    for sig in boundary_signals:
        residuals_df[sig] = df[sig].values[:len(X)] if not nan_mask.any() else df[sig].values[~nan_mask]

    for i, sig in enumerate(target_signals):
        residuals_df[f"{sig}_residual"] = residuals[:, i]
        residuals_df[f"{sig}_true"] = y_true[:, i]
        residuals_df[f"{sig}_pred"] = y_pred[:, i]

    residuals_csv_path = os.path.join(args.output, "residuals.csv")
    residuals_df.to_csv(residuals_csv_path, index=False)

    # Save metrics JSON
    metrics_path = os.path.join(args.output, "residual_metrics.json")
    metrics_data = {
        'overall_r2': round(overall_r2, 4),
        'per_signal': per_signal_metrics,
        'r2_scores': [round(r, 4) for r in r2_scores],
        'boundary_signals': boundary_signals,
        'target_signals': target_signals,
        'num_samples': len(X),
        'model_type': checkpoint.get('model_type', 'SST'),
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=2)

    emit("complete",
         residuals_csv=residuals_csv_path,
         metrics_json=metrics_path,
         overall_r2=round(overall_r2, 4),
         num_samples=len(X))


if __name__ == '__main__':
    main()
