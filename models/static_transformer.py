"""
StaticSensorTransformer (SST): Sensor Sequence Transformer

This module implements a novel Transformer architecture that treats fixed sensor arrays
as sequences, replacing traditional token/word sequences in NLP. This innovation enables
spatial relationship learning between sensors in industrial digital twin applications.

Key Innovation:
- Sensors as Sequence Elements: Unlike NLP where tokens represent words, here each
  position represents a physical sensor with learned positional embeddings.
- Spatial Attention: Multi-head attention captures complex sensor inter-dependencies.
- Industrial-Specific Design: Optimized for boundary-to-target sensor mapping.
"""

import torch
import torch.nn as nn


class StaticSensorTransformer(nn.Module):
    """
    StaticSensorTransformer (SST): Sensor Sequence Transformer

    革新性架构 - 将固定传感器序列替代传统Transformer的语料序列

    Core Innovation:
    ---------------
    Traditional NLP Transformers:
        Input: [Token_1, Token_2, ..., Token_N] (words/subwords)
        Position: Learned position embeddings for word order
        Attention: Captures semantic relationships between words

    SST (This Model):
        Input: [Sensor_1, Sensor_2, ..., Sensor_N] (physical sensors)
        Position: Learned position embeddings for sensor locations
        Attention: Captures spatial relationships between sensors

    Key Differences from NLP:
    -------------------------
    1. Fixed Sequence Length: N sensors is predetermined by physical system
    2. Spatial Semantics: Position embeddings encode sensor locations, not temporal order
    3. Cross-Sensor Dependencies: Attention learns physical causality (e.g., temperature
       sensor affects pressure sensor in industrial processes)
    4. Domain-Specific: Designed for industrial sensor arrays, not language

    Architecture Details:
    ---------------------
    - Sensor Embedding: Projects each scalar sensor reading to d_model dimensions
    - Positional Encoding: Learnable parameters encoding sensor spatial positions
    - Multi-Head Attention: Captures complex inter-sensor relationships
    - Global Pooling: Aggregates sensor sequence information
    - Output Projection: Maps to target sensor predictions

    This design enables Transformers to excel at industrial digital twin tasks by
    treating sensor arrays as "sentences" where each sensor is a "word" with spatial
    rather than temporal semantics.

    Args:
        num_boundary_sensors (int): Number of boundary condition sensors (input sequence length)
        num_target_sensors (int): Number of target sensors to predict (output features)
        d_model (int): Transformer model dimension. Default: 128
        nhead (int): Number of attention heads. Default: 8
        num_layers (int): Number of transformer encoder layers. Default: 3
        dropout (float): Dropout rate for regularization. Default: 0.1

    Example:
        >>> model = StaticSensorTransformer(
        ...     num_boundary_sensors=10,  # 10 sensors in input sequence
        ...     num_target_sensors=5,     # Predict 5 target sensors
        ...     d_model=128,
        ...     nhead=8
        ... )
        >>> x = torch.randn(32, 10)      # Batch of 32 samples, 10 sensor readings
        >>> predictions = model(x)       # Output: (32, 5) target predictions
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

