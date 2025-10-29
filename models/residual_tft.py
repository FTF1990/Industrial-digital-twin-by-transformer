"""
Residual TFT Module for Stage2 Boost Training

This module provides utilities for residual extraction and Stage2 model training
in the Industrial Digital Twin framework.

Key Components:
- ResidualExtractor: Extract residuals from trained SST models
- GroupedMultiTargetTFT: TFT-style model for residual prediction
- Utility functions for residual data preparation
- Mixed precision inference utilities
- Safe R² computation for multi-output scenarios
- Selective boosting based on R² thresholds
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import gc
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.cuda.amp import autocast


class GroupedMultiTargetTFT(nn.Module):
    """
    Grouped Multi-Target Temporal Fusion Transformer

    A TFT-style model for predicting multiple target sensors with optional grouping.
    This model is compatible with the StaticSensorTransformer architecture but adds
    support for signal grouping and temporal fusion capabilities.

    Args:
        num_targets (int): Number of target sensors to predict
        num_external_factors (int): Number of external/boundary condition factors
        d_model (int): Model dimension
        nhead (int): Number of attention heads
        num_layers (int): Number of transformer layers
        dropout (float): Dropout rate
        use_grouping (bool): Whether to use signal grouping
        signal_groups (List[List[int]], optional): Groups of signal indices
    """

    def __init__(
        self,
        num_targets: int,
        num_external_factors: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_grouping: bool = False,
        signal_groups: Optional[List[List[int]]] = None
    ):
        super(GroupedMultiTargetTFT, self).__init__()

        self.num_targets = num_targets
        self.num_external_factors = num_external_factors
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.use_grouping = use_grouping
        self.signal_groups = signal_groups

        # Input embedding
        self.input_embedding = nn.Linear(1, d_model)
        self.position_encoding = nn.Parameter(
            torch.randn(num_external_factors, d_model)
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output layers
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.output_projection = nn.Linear(d_model, num_targets)

        self._init_weights()

    def _init_weights(self):
        """Initialize model weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_external_factors)

        Returns:
            torch.Tensor: Predictions of shape (batch_size, num_targets)
        """
        batch_size = x.shape[0]

        # Embed inputs
        x = x.unsqueeze(-1)  # (batch, factors, 1)
        x = self.input_embedding(x) + self.position_encoding.unsqueeze(0)

        # Transform
        x = self.transformer(x)  # (batch, factors, d_model)

        # Pool and project
        x = x.permute(0, 2, 1)  # (batch, d_model, factors)
        x = self.global_pool(x).squeeze(-1)  # (batch, d_model)
        predictions = self.output_projection(x)  # (batch, num_targets)

        return predictions


class ResidualExtractor:
    """
    Utility class for extracting residuals from trained SST models

    This class provides methods to extract prediction residuals from trained
    StaticSensorTransformer models, which can then be used for Stage2 boost training.
    """

    @staticmethod
    def extract_residuals_from_trained_models(
        model_name: str,
        df: pd.DataFrame,
        global_state: Dict[str, Any],
        device: torch.device
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Extract residuals from a trained SST model

        Args:
            model_name (str): Name of the trained model in global_state
            df (pd.DataFrame): Input dataframe with sensor data
            global_state (Dict): Global state containing trained models
            device (torch.device): Device to run inference on

        Returns:
            Tuple[pd.DataFrame, Dict]:
                - DataFrame containing residuals and predictions
                - Dictionary with extraction info
        """
        try:
            # Get model from global state
            if model_name not in global_state.get('trained_models', {}):
                return pd.DataFrame(), {
                    'error': f"Model '{model_name}' not found in trained models"
                }

            model_info = global_state['trained_models'][model_name]
            model = model_info['model']
            boundary_signals = model_info['boundary_signals']
            target_signals = model_info['target_signals']

            # Get scalers
            scalers = global_state.get('scalers', {}).get(model_name, {})
            if not scalers:
                return pd.DataFrame(), {
                    'error': f"Scalers not found for model '{model_name}'"
                }

            scaler_X = scalers.get('X')
            scaler_y = scalers.get('y')

            # Prepare input data
            X = df[boundary_signals].values
            y_true = df[target_signals].values

            # Use improved residual computation with batch processing and mixed precision
            residuals = compute_residuals_correctly(
                X, y_true, model, scaler_X, scaler_y, device, batch_size=1024
            )

            # Also get predictions for visualization
            X_scaled = scaler_X.transform(X)
            model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_scaled).to(device)
                y_pred_list = []
                batch_size = 1024
                for i in range(0, len(X_tensor), batch_size):
                    batch = X_tensor[i:i+batch_size]
                    with autocast():
                        y_pred_batch = model(batch).cpu().numpy()
                    y_pred_list.append(y_pred_batch)
                y_pred_scaled = np.vstack(y_pred_list)
            y_pred = scaler_y.inverse_transform(y_pred_scaled)

            # Create output dataframe
            residuals_df = pd.DataFrame()

            # Add boundary signals
            for sig in boundary_signals:
                residuals_df[sig] = df[sig].values

            # Add residuals for each target signal
            residual_signals = []
            for i, sig in enumerate(target_signals):
                residual_col = f"{sig}_residual"
                residuals_df[residual_col] = residuals[:, i]
                residual_signals.append(residual_col)

                # Also add true and predicted values
                residuals_df[f"{sig}_true"] = y_true[:, i]
                residuals_df[f"{sig}_pred"] = y_pred[:, i]

            # Calculate per-signal metrics
            metrics = {}
            for i, sig in enumerate(target_signals):
                metrics[sig] = {
                    'mae': mean_absolute_error(y_true[:, i], y_pred[:, i]),
                    'rmse': np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i])),
                    'r2': r2_score(y_true[:, i], y_pred[:, i])
                }

            # Create info dictionary
            info = {
                'model_name': model_name,
                'boundary_signals': boundary_signals,
                'target_signals': target_signals,
                'residual_signals': residual_signals,
                'num_samples': len(residuals_df),
                'metrics': metrics
            }

            return residuals_df, info

        except Exception as e:
            import traceback
            return pd.DataFrame(), {
                'error': f"Failed to extract residuals: {str(e)}",
                'traceback': traceback.format_exc()
            }


def prepare_residual_sequence_data(
    residuals_df: pd.DataFrame,
    boundary_signals: List[str],
    residual_signals: List[str],
    sequence_length: int = 10,
    future_horizon: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare sequential residual data for TFT-style models

    Args:
        residuals_df (pd.DataFrame): DataFrame containing residuals
        boundary_signals (List[str]): List of boundary signal column names
        residual_signals (List[str]): List of residual signal column names
        sequence_length (int): Length of input sequences
        future_horizon (int): Number of steps to predict ahead

    Returns:
        Tuple[np.ndarray, np.ndarray]: X sequences and y targets
    """
    X_sequences = []
    y_sequences = []

    # Get data
    boundary_data = residuals_df[boundary_signals].values
    residual_data = residuals_df[residual_signals].values

    # Create sequences
    for i in range(len(residuals_df) - sequence_length - future_horizon + 1):
        # Input: boundary conditions + past residuals
        X_seq = np.concatenate([
            boundary_data[i:i + sequence_length],
            residual_data[i:i + sequence_length]
        ], axis=1)

        # Target: future residuals
        y_seq = residual_data[i + sequence_length:i + sequence_length + future_horizon]

        X_sequences.append(X_seq)
        y_sequences.append(y_seq)

    return np.array(X_sequences), np.array(y_sequences)


def train_residual_tft(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    config: Dict[str, Any],
    device: torch.device
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Train a Residual TFT model

    Args:
        model (nn.Module): The TFT model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        config (Dict): Training configuration
        device (torch.device): Device to train on

    Returns:
        Tuple[nn.Module, Dict]: Trained model and training history
    """
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get('lr', 0.001),
        weight_decay=config.get('weight_decay', 1e-5)
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config.get('scheduler_factor', 0.5),
        patience=config.get('scheduler_patience', 10)
    )

    criterion = nn.MSELoss()

    history = {
        'train_losses': [],
        'val_losses': [],
        'train_r2': [],
        'val_r2': []
    }

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    early_stop_patience = config.get('early_stop_patience', 25)

    for epoch in range(config.get('epochs', 100)):
        # Training
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()

            # Gradient clipping
            if config.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config['grad_clip']
                )

            optimizer.step()

            train_loss += loss.item()
            train_preds.append(outputs.detach().cpu().numpy())
            train_targets.append(batch_y.detach().cpu().numpy())

        train_loss /= len(train_loader)
        train_preds = np.vstack(train_preds)
        train_targets = np.vstack(train_targets)
        train_r2 = r2_score(train_targets, train_preds)

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)

                val_loss += loss.item()
                val_preds.append(outputs.cpu().numpy())
                val_targets.append(batch_y.cpu().numpy())

        val_loss /= len(val_loader)
        val_preds = np.vstack(val_preds)
        val_targets = np.vstack(val_targets)
        val_r2 = r2_score(val_targets, val_preds)

        # Record history
        history['train_losses'].append(train_loss)
        history['val_losses'].append(val_loss)
        history['train_r2'].append(train_r2)
        history['val_r2'].append(val_r2)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stop_patience:
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, history


def compute_r2_safe(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    method: str = 'per_output_mean'
) -> Tuple[float, np.ndarray]:
    """
    Safe R² calculation - avoid anomalies with multi-output

    This function computes R² scores per output signal and aggregates them
    to avoid anomalies that can occur with sklearn's default multi-output R² calculation.

    Args:
        y_true: Ground truth (n_samples, n_outputs) or (n_samples,)
        y_pred: Predictions (n_samples, n_outputs) or (n_samples,)
        method: Aggregation method
            - 'per_output_mean': Mean of per-output R² (filters out anomalies)
            - 'per_output_median': Median of per-output R²
            - 'sklearn_default': Use sklearn's default multioutput='uniform_average'
            - 'global': Treat all values as one global prediction

    Returns:
        r2: Overall R² score
        per_output_r2: R² for each output (for diagnostics)
    """
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)

    n_outputs = y_true.shape[1]
    per_output_r2 = np.zeros(n_outputs)

    # Compute R² for each output separately
    for i in range(n_outputs):
        y_t = y_true[:, i]
        y_p = y_pred[:, i]

        # Check variance
        var_true = np.var(y_t)
        if var_true < 1e-10:
            per_output_r2[i] = 0.0
        else:
            try:
                per_output_r2[i] = r2_score(y_t, y_p)
            except Exception:
                per_output_r2[i] = -1.0

    # Aggregate based on method
    if method == 'per_output_mean':
        # Filter out anomalies
        valid_r2 = per_output_r2[np.isfinite(per_output_r2) & (per_output_r2 > -10)]
        r2 = np.mean(valid_r2) if len(valid_r2) > 0 else -1.0
    elif method == 'per_output_median':
        valid_r2 = per_output_r2[np.isfinite(per_output_r2) & (per_output_r2 > -10)]
        r2 = np.median(valid_r2) if len(valid_r2) > 0 else -1.0
    elif method == 'sklearn_default':
        r2 = r2_score(y_true, y_pred, multioutput='uniform_average')
    elif method == 'global':
        # Flatten and treat as single output
        r2 = r2_score(y_true.flatten(), y_pred.flatten())
    else:
        r2 = r2_score(y_true, y_pred)

    return r2, per_output_r2


def compute_residuals_correctly(
    X_orig: np.ndarray,
    y_orig: np.ndarray,
    base_model: nn.Module,
    scaler_X: Any,
    scaler_y: Any,
    device: torch.device,
    batch_size: int = 1024
) -> np.ndarray:
    """
    Correctly compute residuals in original scale.

    CRITICAL: Residuals must be computed in the original (non-standardized) space.
    This ensures that Stage2 model learns meaningful residual patterns.

    Steps:
    1. Standardize input using scaler_X
    2. Predict in standardized space
    3. Inverse transform predictions to original space
    4. Compute residuals = y_true - y_pred (both in original space)

    Args:
        X_orig: Original input data (n_samples, n_features)
        y_orig: Original target data (n_samples, n_targets)
        base_model: Trained Stage1 model
        scaler_X: Input scaler (from Stage1 training)
        scaler_y: Output scaler (from Stage1 training)
        device: torch.device for inference
        batch_size: Batch size for memory-efficient inference

    Returns:
        residuals: Residuals in original space (n_samples, n_targets)
    """
    base_model.eval()

    # Step 1: Standardize input
    X_scaled = scaler_X.transform(X_orig)

    # Step 2: Predict in standardized space (with batching for memory efficiency)
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_scaled).to(device)
        y_pred_scaled_list = []

        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i+batch_size]
            with autocast():
                y_pred_batch = base_model(batch).cpu().numpy()
            y_pred_scaled_list.append(y_pred_batch)

        y_pred_scaled = np.vstack(y_pred_scaled_list)

    # Step 3: Inverse transform to original space
    y_pred_original = scaler_y.inverse_transform(y_pred_scaled)

    # Step 4: Compute residuals in original space
    residuals = y_orig - y_pred_original

    return residuals


def batch_inference(
    model: nn.Module,
    X_data: np.ndarray,
    scaler_X: Any,
    scaler_y: Any,
    device: torch.device,
    batch_size: int = 512,
    model_name: str = "Model"
) -> np.ndarray:
    """
    Batch processing inference to avoid GPU OOM

    This function performs inference in batches with automatic memory management
    to handle large datasets without running out of GPU memory.

    Args:
        model: Trained model
        X_data: Input data in original space (n_samples, n_features)
        scaler_X: Input scaler
        scaler_y: Output scaler
        device: torch.device for inference
        batch_size: Batch size for processing
        model_name: Name for logging

    Returns:
        y_pred: Predictions in original space (n_samples, n_targets)
    """
    model.eval()
    n_samples = X_data.shape[0]
    predictions_list = []

    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        batch_X = X_data[i:end_idx]

        # Standardize
        batch_X_scaled = scaler_X.transform(batch_X)
        batch_X_tensor = torch.FloatTensor(batch_X_scaled).to(device)

        # Inference with mixed precision
        with torch.no_grad():
            with autocast():
                batch_pred_scaled = model(batch_X_tensor).cpu().numpy()

        # Inverse transform
        batch_pred = scaler_y.inverse_transform(batch_pred_scaled)
        predictions_list.append(batch_pred)

        # Clean up GPU memory
        del batch_X_tensor, batch_pred_scaled
        if (i // batch_size + 1) % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    y_pred = np.vstack(predictions_list)
    return y_pred


def inference_with_boosting(
    X_test: np.ndarray,
    base_model: nn.Module,
    residual_model: nn.Module,
    scalers: Dict[str, Any],
    device: torch.device,
    signal_r2_scores: Optional[np.ndarray] = None,
    r2_threshold: float = 0.4,
    batch_size: int = 512,
    use_selective_boosting: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply boosting selectively based on R² scores

    This function implements intelligent model selection:
    - For signals with high R² (≥ threshold): Use Stage1 predictions only
    - For signals with low R² (< threshold): Apply Stage2 boost correction

    Args:
        X_test: Test data in original space
        base_model: Stage 1 model
        residual_model: Stage 2 residual model
        scalers: Dictionary containing:
            - 'stage1_scaler_X': Stage1 input scaler
            - 'stage1_scaler_y': Stage1 output scaler
            - 'stage2_scaler_X': Stage2 input scaler
            - 'stage2_scaler_residual': Stage2 residual scaler
        device: torch.device
        signal_r2_scores: R² score for each signal from validation (optional)
        r2_threshold: Threshold for determining weak signals
        batch_size: Batch size for inference
        use_selective_boosting: If True, only boost weak signals; if False, boost all

    Returns:
        y_pred_stage1: Stage 1 predictions only
        y_pred_boosted: Final predictions (with selective boosting)
        boosting_mask: Boolean mask indicating which signals were boosted
    """

    # Stage 1 prediction
    y_pred_stage1 = batch_inference(
        base_model, X_test,
        scalers['stage1_scaler_X'],
        scalers['stage1_scaler_y'],
        device,
        batch_size,
        "Stage 1"
    )

    # Stage 2 residual prediction
    residual_pred = batch_inference(
        residual_model, X_test,
        scalers['stage2_scaler_X'],
        scalers['stage2_scaler_residual'],
        device,
        batch_size,
        "Stage 2"
    )

    # Full boosting (all signals)
    y_pred_full_boosted = y_pred_stage1 + residual_pred

    # Selective boosting (only weak signals)
    if use_selective_boosting and signal_r2_scores is not None:
        # Identify weak signals (R² < threshold)
        boosting_mask = signal_r2_scores < r2_threshold
        num_boosted = np.sum(boosting_mask)

        print(f"Selective Boosting: {num_boosted}/{len(boosting_mask)} signals boosted (R² < {r2_threshold})")

        # Apply boosting only to weak signals
        y_pred_boosted = y_pred_stage1.copy()
        y_pred_boosted[:, boosting_mask] = y_pred_full_boosted[:, boosting_mask]
    else:
        # Use full boosting for all signals
        y_pred_boosted = y_pred_full_boosted
        boosting_mask = np.ones(y_pred_stage1.shape[1], dtype=bool)

    return y_pred_stage1, y_pred_boosted, boosting_mask


def compute_per_signal_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Compute detailed metrics for each signal

    Args:
        y_true: Ground truth (n_samples, n_signals)
        y_pred: Predictions (n_samples, n_signals)

    Returns:
        metrics: Dictionary with per-signal metrics
            - 'mae': Mean Absolute Error per signal
            - 'rmse': Root Mean Squared Error per signal
            - 'r2': R² score per signal
            - 'mape': Mean Absolute Percentage Error per signal
            - 'true_mean': Mean of true values per signal
            - 'true_std': Std of true values per signal
            - 'pred_mean': Mean of predictions per signal
            - 'pred_std': Std of predictions per signal
    """
    n_signals = y_true.shape[1]

    metrics = {
        'mae': np.zeros(n_signals),
        'rmse': np.zeros(n_signals),
        'r2': np.zeros(n_signals),
        'mape': np.zeros(n_signals),
        'true_mean': np.zeros(n_signals),
        'true_std': np.zeros(n_signals),
        'pred_mean': np.zeros(n_signals),
        'pred_std': np.zeros(n_signals)
    }

    for i in range(n_signals):
        y_t = y_true[:, i]
        y_p = y_pred[:, i]

        metrics['mae'][i] = mean_absolute_error(y_t, y_p)
        metrics['rmse'][i] = np.sqrt(mean_squared_error(y_t, y_p))

        var_true = np.var(y_t)
        if var_true < 1e-10:
            metrics['r2'][i] = 0.0
        else:
            try:
                metrics['r2'][i] = r2_score(y_t, y_p)
            except Exception:
                metrics['r2'][i] = -1.0

        # MAPE for non-zero values
        non_zero_mask = np.abs(y_t) > 1e-6
        if np.sum(non_zero_mask) > 0:
            mape = np.mean(np.abs((y_t[non_zero_mask] - y_p[non_zero_mask]) /
                                   y_t[non_zero_mask])) * 100
            metrics['mape'][i] = mape
        else:
            metrics['mape'][i] = np.nan

        metrics['true_mean'][i] = np.mean(y_t)
        metrics['true_std'][i] = np.std(y_t)
        metrics['pred_mean'][i] = np.mean(y_p)
        metrics['pred_std'][i] = np.std(y_p)

    return metrics


def clear_gpu_memory():
    """Clean up GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


def print_gpu_memory():
    """Print GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory: Allocated {allocated:.2f} GB | Reserved {reserved:.2f} GB")
