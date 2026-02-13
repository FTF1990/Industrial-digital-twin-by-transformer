#!/usr/bin/env python3
"""
Ensemble Inference Script - Stage1 + Stage2 Combined

Performs inference with selective boosting:
  - Signals with high Stage1 R² -> use Stage1 only
  - Signals with low Stage1 R² -> apply Stage2 residual correction

Supports both SST and MaskedSST model types (auto-detected from checkpoint).

Usage:
    python inference_ensemble.py --stage1_model stage1.pth --stage1_scalers stage1_scalers.pkl \
        --stage2_model stage2.pth --stage2_scalers stage2_scalers.pkl \
        --data data.csv --config signals.json --output ./results/
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

WEBUI_PYTHON = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WEBUI_PYTHON)


def emit(msg_type, **kwargs):
    msg = {"type": msg_type, **kwargs}
    print(json.dumps(msg, ensure_ascii=False), flush=True)


def load_model_auto(model_path, device):
    """Load model from checkpoint, auto-detecting SST vs MaskedSST."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
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

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, model_type


def batch_predict(model, X, scaler_X, scaler_y, device, batch_size=512):
    """Batch inference: X (original scale) -> predictions (original scale)."""
    X_scaled = scaler_X.transform(X)
    use_amp = device.type == 'cuda'
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_scaled), batch_size):
            batch = torch.FloatTensor(X_scaled[i:i + batch_size]).to(device)
            if use_amp:
                with autocast():
                    p = model(batch).cpu().numpy()
            else:
                p = model(batch).cpu().numpy()
            preds.append(p)
    return scaler_y.inverse_transform(np.vstack(preds))


def main():
    parser = argparse.ArgumentParser(description='Ensemble Inference (Stage1 + Stage2)')
    parser.add_argument('--stage1_model', required=True, help='Stage1 .pth file')
    parser.add_argument('--stage1_scalers', required=True, help='Stage1 scalers .pkl')
    parser.add_argument('--stage2_model', required=True, help='Stage2 .pth file')
    parser.add_argument('--stage2_scalers', required=True, help='Stage2 scalers .pkl')
    parser.add_argument('--data', required=True, help='Input CSV data')
    parser.add_argument('--config', required=True, help='Signal config JSON')
    parser.add_argument('--output', default='./results', help='Output directory')
    parser.add_argument('--r2_threshold', type=float, default=0.4,
                        help='R² threshold for selective boosting (signals below this get boosted)')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--residual_metrics', default=None,
                        help='Path to residual_metrics.json (for R² scores). '
                             'If not provided, uses --r2_threshold as uniform cutoff.')
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

    has_targets = all(s in df.columns for s in target_signals)
    y_true = df[target_signals].values.astype(np.float32) if has_targets else None

    nan_mask = np.isnan(X).any(axis=1)
    if y_true is not None:
        nan_mask = nan_mask | np.isnan(y_true).any(axis=1)
    if nan_mask.any():
        emit("info", message=f"Dropping {nan_mask.sum()} rows with NaN")
        X = X[~nan_mask]
        if y_true is not None:
            y_true = y_true[~nan_mask]

    emit("info", message=f"Inference data: {X.shape[0]} samples")

    # ── Load Models ──
    emit("status", stage="loading_models")
    stage1_model, s1_type = load_model_auto(args.stage1_model, device)
    stage2_model, s2_type = load_model_auto(args.stage2_model, device)
    emit("info", message=f"Stage1: {s1_type}, Stage2: {s2_type}")

    with open(args.stage1_scalers, 'rb') as f:
        s1_scalers = pickle.load(f)
    with open(args.stage2_scalers, 'rb') as f:
        s2_scalers = pickle.load(f)

    # ── Stage1 Prediction ──
    emit("status", stage="stage1_inference")
    y_stage1 = batch_predict(stage1_model, X, s1_scalers['X'], s1_scalers['y'],
                             device, args.batch_size)
    emit("info", message="Stage1 inference complete")

    # ── Stage2 Residual Prediction ──
    emit("status", stage="stage2_inference")
    residual_pred = batch_predict(stage2_model, X, s2_scalers['X'], s2_scalers['residual'],
                                  device, args.batch_size)
    emit("info", message="Stage2 inference complete")

    # ── Determine R² scores for selective boosting ──
    signal_r2_scores = None
    if args.residual_metrics and os.path.exists(args.residual_metrics):
        with open(args.residual_metrics, 'r') as f:
            res_metrics = json.load(f)
        r2_list = []
        for sig in target_signals:
            r2_val = res_metrics.get('per_signal', {}).get(sig, {}).get('r2', 0.0)
            r2_list.append(r2_val)
        signal_r2_scores = np.array(r2_list)
        emit("info", message=f"Loaded per-signal R² from metrics file")
    elif y_true is not None:
        # Compute R² from actual data
        r2_list = []
        for i in range(len(target_signals)):
            var = np.var(y_true[:, i])
            r2 = float(r2_score(y_true[:, i], y_stage1[:, i])) if var > 1e-10 else 0.0
            r2_list.append(r2)
        signal_r2_scores = np.array(r2_list)
        emit("info", message="Computed per-signal R² from actual data")

    # ── Selective Boosting ──
    emit("status", stage="ensemble_boosting")
    y_ensemble = y_stage1.copy()
    boosting_info = {}

    if signal_r2_scores is not None:
        boosting_mask = signal_r2_scores < args.r2_threshold
        y_ensemble[:, boosting_mask] = y_stage1[:, boosting_mask] + residual_pred[:, boosting_mask]

        n_boosted = int(np.sum(boosting_mask))
        emit("info", message=f"Selective boosting: {n_boosted}/{len(target_signals)} signals boosted "
                             f"(R² < {args.r2_threshold})")

        for i, sig in enumerate(target_signals):
            boosting_info[sig] = {
                "stage1_r2": round(float(signal_r2_scores[i]), 4),
                "boosted": bool(boosting_mask[i]),
            }
    else:
        # Full boosting
        y_ensemble = y_stage1 + residual_pred
        emit("info", message="Full boosting applied (no R² scores available)")
        for sig in target_signals:
            boosting_info[sig] = {"boosted": True}

    # ── Compute Final Metrics ──
    results = {
        'stage1_metrics': {},
        'ensemble_metrics': {},
        'boosting_info': boosting_info,
        'r2_threshold': args.r2_threshold,
        'model_types': {'stage1': s1_type, 'stage2': s2_type},
    }

    if y_true is not None:
        emit("status", stage="computing_metrics")
        for i, sig in enumerate(target_signals):
            var = np.var(y_true[:, i])
            s1_r2 = float(r2_score(y_true[:, i], y_stage1[:, i])) if var > 1e-10 else 0.0
            s1_mae = float(mean_absolute_error(y_true[:, i], y_stage1[:, i]))
            s1_rmse = float(np.sqrt(mean_squared_error(y_true[:, i], y_stage1[:, i])))

            en_r2 = float(r2_score(y_true[:, i], y_ensemble[:, i])) if var > 1e-10 else 0.0
            en_mae = float(mean_absolute_error(y_true[:, i], y_ensemble[:, i]))
            en_rmse = float(np.sqrt(mean_squared_error(y_true[:, i], y_ensemble[:, i])))

            delta_r2 = en_r2 - s1_r2

            results['stage1_metrics'][sig] = {
                'r2': round(s1_r2, 4), 'mae': round(s1_mae, 6), 'rmse': round(s1_rmse, 6)
            }
            results['ensemble_metrics'][sig] = {
                'r2': round(en_r2, 4), 'mae': round(en_mae, 6), 'rmse': round(en_rmse, 6),
                'delta_r2': round(delta_r2, 4)
            }
            results['boosting_info'][sig]['delta_r2'] = round(delta_r2, 4)

        # Overall
        s1_r2_all = [v['r2'] for v in results['stage1_metrics'].values()]
        en_r2_all = [v['r2'] for v in results['ensemble_metrics'].values()]
        results['overall_stage1_r2'] = round(float(np.mean(s1_r2_all)), 4)
        results['overall_ensemble_r2'] = round(float(np.mean(en_r2_all)), 4)
        results['overall_delta_r2'] = round(results['overall_ensemble_r2'] - results['overall_stage1_r2'], 4)

        emit("comparison",
             stage1_r2=results['overall_stage1_r2'],
             ensemble_r2=results['overall_ensemble_r2'],
             delta_r2=results['overall_delta_r2'],
             per_signal=results['ensemble_metrics'])

    # ── Save Results ──
    emit("status", stage="saving")
    os.makedirs(args.output, exist_ok=True)

    # Save predictions CSV
    pred_df = pd.DataFrame()
    for sig in boundary_signals:
        vals = df[sig].values[:len(X)] if not nan_mask.any() else df[sig].values[~nan_mask]
        pred_df[sig] = vals

    for i, sig in enumerate(target_signals):
        if y_true is not None:
            pred_df[f"{sig}_actual"] = y_true[:, i]
        pred_df[f"{sig}_stage1"] = y_stage1[:, i]
        pred_df[f"{sig}_ensemble"] = y_ensemble[:, i]
        pred_df[f"{sig}_residual_pred"] = residual_pred[:, i]

    pred_csv = os.path.join(args.output, "predictions.csv")
    pred_df.to_csv(pred_csv, index=False)

    # Save results JSON
    results['num_samples'] = len(X)
    results['boundary_signals'] = boundary_signals
    results['target_signals'] = target_signals
    results_json = os.path.join(args.output, "inference_results.json")
    with open(results_json, 'w') as f:
        json.dump(results, f, indent=2)

    emit("complete",
         predictions_csv=pred_csv,
         results_json=results_json,
         num_samples=len(X))


if __name__ == '__main__':
    main()
