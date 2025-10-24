"""
Residual TFT Module for Stage2 Boost Training

This module provides utilities for residual extraction and Stage2 model training
in the Industrial Digital Twin framework.

Key Components:
- ResidualExtractor: Extract residuals from trained SST models
- GroupedMultiTargetTFT: TFT-style model for residual prediction
- Utility functions for residual data preparation
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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

            # Scale inputs
            X_scaled = scaler_X.transform(X)

            # Generate predictions
            model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_scaled).to(device)
                y_pred_scaled = model(X_tensor).cpu().numpy()

            # Inverse transform predictions
            y_pred = scaler_y.inverse_transform(y_pred_scaled)

            # Calculate residuals
            residuals = y_true - y_pred

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
