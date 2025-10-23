"""
HybridSensorTransformer (HST): Hybrid Temporal + Static Transformer

This module implements a hybrid Transformer model that combines temporal context
analysis with static sensor mapping for enhanced prediction accuracy in industrial
digital twin applications.

Formerly known as V4 or HybridTemporalTransformer in earlier versions.
"""

import torch
import torch.nn as nn


class HybridSensorTransformer(nn.Module):
    """
    HybridSensorTransformer (HST): 混合时序+静态Transformer

    An advanced hybrid architecture that combines:
    1. Temporal branch: Analyzes time-series patterns using context windows
    2. Static branch: Learns spatial sensor relationships
    3. Fusion mechanism: Intelligently combines both approaches

    This model is particularly effective for sensors with temporal dependencies
    while maintaining accuracy for static relationships.

    The model can operate in two modes:
    - **Hybrid mode**: When temporal_signals are specified, uses both branches
    - **Static-only mode**: Falls back to static mapping when no temporal context

    Args:
        num_boundary_sensors (int): Number of boundary condition sensors (input features)
        num_target_sensors (int): Number of target sensors to predict (output features)
        d_model (int): Dimension of the transformer model. Default: 64
        nhead (int): Number of attention heads. Default: 4
        num_layers (int): Number of transformer encoder layers. Default: 2
        dropout (float): Dropout rate. Default: 0.1
        use_temporal (bool): Whether to use temporal branch. Default: True
        context_window (int): Size of context window for temporal analysis. Default: 5

    Example:
        >>> model = HybridSensorTransformer(
        ...     num_boundary_sensors=10,
        ...     num_target_sensors=5,
        ...     use_temporal=True,
        ...     context_window=5
        ... )
        >>> # With temporal context
        >>> x_temporal = torch.randn(32, 11, 10)  # (batch, context=11, sensors=10)
        >>> predictions = model(x_temporal)       # output: (32, 5)
    """

    def __init__(self, num_boundary_sensors, num_target_sensors,
                 d_model=64, nhead=4, num_layers=2, dropout=0.1,
                 use_temporal=True, context_window=5):
        super(HybridSensorTransformer, self).__init__()

        self.num_boundary_sensors = num_boundary_sensors
        self.num_target_sensors = num_target_sensors
        self.d_model = d_model
        self.use_temporal = use_temporal
        self.context_window = context_window

        if self.use_temporal:
            context_size = 2 * context_window + 1

            # 时序分支
            self.temporal_embedding = nn.Linear(num_boundary_sensors, d_model)
            self.temporal_position_encoding = nn.Parameter(torch.randn(context_size, d_model))

            temporal_encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
                dropout=dropout, batch_first=True
            )
            self.temporal_transformer = nn.TransformerEncoder(temporal_encoder_layer, num_layers)
            self.temporal_head = nn.Linear(d_model, num_target_sensors)

        # 静态分支
        self.static_embedding = nn.Linear(1, d_model)
        self.static_position_encoding = nn.Parameter(torch.randn(num_boundary_sensors, d_model))

        static_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True
        )
        self.static_transformer = nn.TransformerEncoder(static_encoder_layer, num_layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.static_head = nn.Linear(d_model, num_target_sensors)

        if self.use_temporal:
            self.fusion = nn.Linear(num_target_sensors * 2, num_target_sensors)

        self._init_weights()

    def _init_weights(self):
        """Initialize model weights using Xavier uniform initialization"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.1)

    def forward_static(self, x):
        """
        Static branch forward pass

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_boundary_sensors)

        Returns:
            torch.Tensor: Predictions from static branch
        """
        x = x.unsqueeze(-1)
        x = self.static_embedding(x) + self.static_position_encoding.unsqueeze(0)
        x = self.static_transformer(x)
        x = x.permute(0, 2, 1)
        x = self.global_pool(x).squeeze(-1)
        return self.static_head(x)

    def forward_temporal(self, x):
        """
        Temporal branch forward pass

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, context_size, num_boundary_sensors)

        Returns:
            torch.Tensor: Predictions from temporal branch
        """
        x = self.temporal_embedding(x)
        x = x + self.temporal_position_encoding.unsqueeze(0)
        x = self.temporal_transformer(x)

        # Extract center timestep features
        center_idx = x.shape[1] // 2
        center_features = x[:, center_idx, :]
        return self.temporal_head(center_features)

    def forward(self, x):
        """
        Unified forward pass

        Args:
            x (torch.Tensor): Input tensor
                - If use_temporal=True: shape (batch_size, context_size, num_boundary_sensors)
                - If use_temporal=False: shape (batch_size, num_boundary_sensors)

        Returns:
            torch.Tensor: Predicted target sensor values of shape (batch_size, num_target_sensors)
        """
        if self.use_temporal and len(x.shape) == 3:
            # Hybrid mode: combine temporal and static predictions
            temporal_pred = self.forward_temporal(x)

            center_idx = x.shape[1] // 2
            center_data = x[:, center_idx, :]
            static_pred = self.forward_static(center_data)

            combined = torch.cat([temporal_pred, static_pred], dim=1)
            return self.fusion(combined)
        else:
            # Static-only mode
            return self.forward_static(x)


# Alias for backward compatibility and convenience
HST = HybridSensorTransformer

