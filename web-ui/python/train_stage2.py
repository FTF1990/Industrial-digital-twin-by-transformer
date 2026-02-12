#!/usr/bin/env python3
"""
Stage2 Training Script - Residual Boost Model

Trains a Stage2 model on extracted residuals from Stage1.
Supports Signal Mapping Layer with configurable mask mode:
  - stage2_mask_mode='same': Use same mask as Stage1 (consistent physical semantics)
  - stage2_mask_mode='none': No masking (pure accuracy optimization, for comparison)

Usage:
    python train_stage2.py --residuals ./residuals/residuals.csv \
        --config signals.json --stage1_config ./models/stage1_config.json \
        --output ./models/ --stage2_mask_mode same
"""

import argparse
import json
import sys
import os
import time
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

WEBUI_PYTHON = os.path.dirname(__file__)
sys.path.insert(0, WEBUI_PYTHON)

from models.static_transformer import StaticSensorTransformer
from signal_mapping_utils import (
    is_mapping_enabled, generate_mask_matrix, generate_full_mask, describe_mask
)


def emit(msg_type, **kwargs):
    msg = {"type": msg_type, **kwargs}
    print(json.dumps(msg, ensure_ascii=False), flush=True)


def compute_r2_safe(y_true, y_pred):
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
    n_outputs = y_true.shape[1]
    per_r2 = []
    for i in range(n_outputs):
        var = np.var(y_true[:, i])
        if var < 1e-10:
            per_r2.append(0.0)
        else:
            try:
                r2 = r2_score(y_true[:, i], y_pred[:, i])
                if np.isfinite(r2) and r2 > -10:
                    per_r2.append(r2)
            except Exception:
                pass
    return float(np.mean(per_r2)) if per_r2 else -1.0


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    emit("info", message=f"Using device: {device}")

    # ── Load Signal Config ──
    with open(args.config, 'r') as f:
        config = json.load(f)
    boundary_signals = config['boundary']
    target_signals = config['target']
    residual_columns = [f"{sig}_residual" for sig in target_signals]

    # ── Check Signal Mapping ──
    use_mapping = is_mapping_enabled(config)
    mapping_config = config.get('signal_mapping', {})
    mask_mode = args.stage2_mask_mode

    if use_mapping:
        emit("info", message=f"Signal Mapping detected. Stage2 mask mode: '{mask_mode}'")
        if mask_mode == 'same':
            mask_matrix = generate_mask_matrix(boundary_signals, target_signals, mapping_config)
            mask_desc = describe_mask(mask_matrix, boundary_signals, target_signals)
            emit("info", message=f"Using SAME mask as Stage1: {mask_desc['blocked_connections']} connections blocked")
        else:
            mask_matrix = generate_full_mask(boundary_signals, target_signals)
            emit("info", message="Using NO mask (all connections allowed) for Stage2")
    else:
        mask_matrix = None
        emit("info", message="Standard mode (no signal mapping)")

    # ── Load Stage1 Config for data split consistency ──
    with open(args.stage1_config, 'r') as f:
        stage1_config = json.load(f)
    data_split = stage1_config.get('data_split', {})
    test_size = data_split.get('test_size', args.test_size)
    val_size = data_split.get('val_size', args.val_size)

    # ── Load Residuals ──
    emit("status", stage="loading_data")
    res_df = pd.read_csv(args.residuals)
    emit("info", message=f"Loaded residuals: {res_df.shape[0]} rows")

    X = res_df[boundary_signals].values.astype(np.float32)
    y = res_df[residual_columns].values.astype(np.float32)

    # Handle NaN
    nan_mask = np.isnan(X).any(axis=1) | np.isnan(y).any(axis=1)
    if nan_mask.any():
        emit("info", message=f"Dropping {nan_mask.sum()} rows with NaN")
        X = X[~nan_mask]
        y = y[~nan_mask]

    # ── Use same split ratios as Stage1 ──
    n = len(X)
    n_test = int(n * test_size)
    n_val = int(n * val_size)
    n_train = n - n_test - n_val

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]

    emit("info", message=f"Split: train={n_train}, val={n_val}, test={n_test}")

    # ── Standardize ──
    scaler_X = StandardScaler()
    scaler_residual = StandardScaler()

    X_train_s = scaler_X.fit_transform(X_train)
    y_train_s = scaler_residual.fit_transform(y_train)
    X_val_s = scaler_X.transform(X_val)
    y_val_s = scaler_residual.transform(y_val)
    X_test_s = scaler_X.transform(X_test)

    # ── DataLoaders ──
    bs = args.batch_size
    train_ds = TensorDataset(torch.FloatTensor(X_train_s), torch.FloatTensor(y_train_s))
    val_ds = TensorDataset(torch.FloatTensor(X_val_s), torch.FloatTensor(y_val_s))
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)

    # ── Build Model ──
    if use_mapping:
        from models.masked_sst import MaskedSST
        model = MaskedSST(
            num_input_signals=len(boundary_signals),
            num_output_signals=len(target_signals),
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dropout=args.dropout,
            mask_matrix=mask_matrix
        ).to(device)
        emit("info", message=f"Stage2 using MaskedSST (mask_mode='{mask_mode}')")
    else:
        model = StaticSensorTransformer(
            num_boundary_sensors=len(boundary_signals),
            num_target_sensors=len(target_signals),
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dropout=args.dropout
        ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    emit("info", message=f"Stage2 model parameters: {total_params:,}")

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    criterion = nn.MSELoss()
    scaler_amp = GradScaler() if device.type == 'cuda' else None

    # ── Training Loop ──
    emit("status", stage="training", total_epochs=args.epochs)
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    history = {
        'train_loss': [], 'val_loss': [],
        'train_r2': [], 'val_r2': []
    }

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train
        model.train()
        train_losses, train_preds, train_tgts = [], [], []
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            if scaler_amp:
                with autocast():
                    out = model(bx)
                    loss = criterion(out, by)
                scaler_amp.scale(loss).backward()
                scaler_amp.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                scaler_amp.step(optimizer)
                scaler_amp.update()
            else:
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                optimizer.step()

            train_losses.append(loss.item())
            train_preds.append(out.detach().cpu().numpy())
            train_tgts.append(by.detach().cpu().numpy())

        train_loss = float(np.mean(train_losses))
        train_r2 = compute_r2_safe(np.vstack(train_tgts), np.vstack(train_preds))

        # Validate
        model.eval()
        val_losses, val_preds, val_tgts = [], [], []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                if scaler_amp:
                    with autocast():
                        out = model(bx)
                        loss = criterion(out, by)
                else:
                    out = model(bx)
                    loss = criterion(out, by)
                val_losses.append(loss.item())
                val_preds.append(out.cpu().numpy())
                val_tgts.append(by.cpu().numpy())

        val_loss = float(np.mean(val_losses))
        val_r2 = compute_r2_safe(np.vstack(val_tgts), np.vstack(val_preds))

        scheduler.step(val_loss)
        elapsed = time.time() - t0

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_r2'].append(train_r2)
        history['val_r2'].append(val_r2)

        emit("epoch", epoch=epoch, total=args.epochs,
             train_loss=round(train_loss, 6), val_loss=round(val_loss, 6),
             train_r2=round(train_r2, 4), val_r2=round(val_r2, 4),
             lr=optimizer.param_groups[0]['lr'], elapsed=round(elapsed, 2))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= args.patience:
            emit("info", message=f"Early stopping at epoch {epoch}")
            break

    if best_model_state:
        model.load_state_dict(best_model_state)

    # ── Test Evaluation ──
    emit("status", stage="evaluating")
    model.eval()
    test_preds_list = []
    with torch.no_grad():
        for i in range(0, len(X_test_s), bs):
            batch = torch.FloatTensor(X_test_s[i:i+bs]).to(device)
            if scaler_amp:
                with autocast():
                    pred = model(batch).cpu().numpy()
            else:
                pred = model(batch).cpu().numpy()
            test_preds_list.append(pred)

    test_preds_s = np.vstack(test_preds_list)
    test_preds_orig = scaler_residual.inverse_transform(test_preds_s)

    per_signal_metrics = {}
    for i, sig in enumerate(target_signals):
        mae = float(mean_absolute_error(y_test[:, i], test_preds_orig[:, i]))
        rmse = float(np.sqrt(mean_squared_error(y_test[:, i], test_preds_orig[:, i])))
        var = np.var(y_test[:, i])
        r2 = float(r2_score(y_test[:, i], test_preds_orig[:, i])) if var > 1e-10 else 0.0
        per_signal_metrics[sig] = {"mae": round(mae, 6), "rmse": round(rmse, 6), "r2": round(r2, 4)}

    overall_r2 = compute_r2_safe(y_test, test_preds_orig)
    emit("test_metrics", overall_r2=round(overall_r2, 4), per_signal=per_signal_metrics)

    # ── Save ──
    emit("status", stage="saving")
    os.makedirs(args.output, exist_ok=True)
    model_name = args.name or f"stage2_{int(time.time())}"

    # Build checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_type': 'MaskedSST' if use_mapping else 'SST',
        'model_config': {
            'd_model': args.d_model,
            'nhead': args.nhead,
            'num_layers': args.num_layers,
            'dropout': args.dropout,
        },
        'training_config': {
            'epochs_trained': epoch,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'best_val_loss': best_val_loss,
        },
        'history': history
    }

    if use_mapping:
        checkpoint['model_config']['num_input_signals'] = len(boundary_signals)
        checkpoint['model_config']['num_output_signals'] = len(target_signals)
        checkpoint['signal_mapping'] = {
            'enabled': True,
            'mask_mode': mask_mode,
            'input_signals': boundary_signals,
            'output_signals': target_signals,
            'mask_matrix': mask_matrix.tolist(),
        }
    else:
        checkpoint['model_config']['num_boundary_sensors'] = len(boundary_signals)
        checkpoint['model_config']['num_target_sensors'] = len(target_signals)

    model_path = os.path.join(args.output, f"{model_name}.pth")
    torch.save(checkpoint, model_path)

    scaler_path = os.path.join(args.output, f"{model_name}_scalers.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump({'X': scaler_X, 'residual': scaler_residual}, f)

    config_path = os.path.join(args.output, f"{model_name}_config.json")
    stage2_conf = {
        'model_name': model_name,
        'model_path': model_path,
        'scaler_path': scaler_path,
        'stage': 2,
        'model_type': 'MaskedSST' if use_mapping else 'SST',
        'boundary_signals': boundary_signals,
        'target_signals': target_signals,
        'architecture': checkpoint['model_config'],
        'data_split': {
            'test_size': test_size,
            'val_size': val_size,
        },
        'test_metrics': {
            'overall_r2': round(overall_r2, 4),
            'per_signal': per_signal_metrics,
        }
    }
    if use_mapping:
        stage2_conf['signal_mapping'] = {
            'enabled': True,
            'mask_mode': mask_mode,
        }

    with open(config_path, 'w') as f:
        json.dump(stage2_conf, f, indent=2)

    emit("complete", model_name=model_name,
         model_path=model_path, scaler_path=scaler_path, config_path=config_path,
         overall_r2=round(overall_r2, 4), epochs_trained=epoch,
         stage2_mask_mode=mask_mode)


def main():
    parser = argparse.ArgumentParser(description='Stage2 Residual Boost Training')
    parser.add_argument('--residuals', required=True, help='Path to residuals CSV')
    parser.add_argument('--config', required=True, help='Path to signal config JSON')
    parser.add_argument('--stage1_config', required=True, help='Path to Stage1 config JSON (for split ratios)')
    parser.add_argument('--output', default='./models', help='Output directory')
    parser.add_argument('--name', default=None, help='Model name')

    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--patience', type=int, default=25)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--val_size', type=float, default=0.2)

    # Signal Mapping
    parser.add_argument('--stage2_mask_mode', default='same', choices=['same', 'none'],
                        help='Stage2 mask mode: "same" uses Stage1 mask, "none" disables masking')

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
