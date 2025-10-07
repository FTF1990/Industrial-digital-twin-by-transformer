"""
StaticSensorTransformer (SST): Static Sensor Mapping Transformer

This module implements a static Transformer model for mapping boundary sensor
measurements to target sensor predictions in industrial digital twin applications.

Formerly known as V1 or CompactSensorTransformer in earlier versions.
"""

import torch
import torch.nn as nn


class StaticSensorTransformer(nn.Module):
    """
    StaticSensorTransformer (SST): 静态传感器映射Transformer

    A Transformer architecture designed for static sensor-to-sensor mapping
    in complex industrial systems. This model learns spatial relationships between
    boundary condition sensors and target sensors without temporal dependencies.

    This model is ideal for steady-state systems where sensor relationships
    are primarily spatial rather than temporal.

    Args:
        num_boundary_sensors (int): Number of boundary condition sensors (input features)
        num_target_sensors (int): Number of target sensors to predict (output features)
        d_model (int): Dimension of the transformer model. Default: 128
        nhead (int): Number of attention heads. Default: 8
        num_layers (int): Number of transformer encoder layers. Default: 3
        dropout (float): Dropout rate. Default: 0.1

    Example:
        >>> model = StaticSensorTransformer(
        ...     num_boundary_sensors=10,
        ...     num_target_sensors=5,
        ...     d_model=128,
        ...     nhead=8
        ... )
        >>> x = torch.randn(32, 10)  # batch_size=32, sensors=10
        >>> predictions = model(x)   # output: (32, 5)
    """

    def __init__(self, num_boundary_sensors, num_target_sensors,
                 d_model=128, nhead=8, num_layers=3, dropout=0.1):
        super(StaticSensorTransformer, self).__init__()

        self.num_boundary_sensors = num_boundary_sensors
        self.num_target_sensors = num_target_sensors
        self.d_model = d_model

        # 边界条件嵌入
        self.boundary_embedding = nn.Linear(1, d_model)
        self.boundary_position_encoding = nn.Parameter(torch.randn(num_boundary_sensors, d_model))

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出层
        self.output_projection = nn.Linear(d_model, num_target_sensors)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self._init_weights()

    def _init_weights(self):
        """Initialize model weights using Xavier uniform initialization"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, boundary_conditions):
        """
        Forward pass of the model

        Args:
            boundary_conditions (torch.Tensor): Input tensor of shape (batch_size, num_boundary_sensors)

        Returns:
            torch.Tensor: Predicted target sensor values of shape (batch_size, num_target_sensors)
        """
        batch_size = boundary_conditions.shape[0]

        # Embed boundary conditions
        x = boundary_conditions.unsqueeze(-1)  # (batch, sensors, 1)
        x = self.boundary_embedding(x) + self.boundary_position_encoding.unsqueeze(0)

        # Transform
        x = self.transformer(x)  # (batch, sensors, d_model)

        # Global pooling and projection
        x = x.permute(0, 2, 1)  # (batch, d_model, sensors)
        x = self.global_pool(x).squeeze(-1)  # (batch, d_model)
        predictions = self.output_projection(x)  # (batch, num_target_sensors)

        return predictions


# Alias for backward compatibility and convenience
SST = StaticSensorTransformer

