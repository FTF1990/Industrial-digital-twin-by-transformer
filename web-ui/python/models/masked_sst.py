"""
MaskedSST: StaticSensorTransformer with Signal Mapping Layer

Extends the base SST architecture by replacing Global Average Pooling + Linear Output
with a learnable cross-attention based Signal Mapping Layer that supports configurable
input-to-output masking.

Key Innovation:
    - Each output signal has its own learned query vector
    - Cross-attention between output queries and encoder outputs
    - Mask matrix controls which input signals can influence which output signals
    - Enables Level 2/3 digital twin: signals can appear in both input and output
      while preventing self-prediction (and other configurable exclusions)

Architecture:
    Input (batch, N_in)
      -> Embedding (batch, N_in, d_model)
      -> Positional Encoding
      -> Transformer Encoder (batch, N_in, d_model)   [full self-attention]
      -> Signal Mapping Layer                           [masked cross-attention]
          - Output queries: (N_out, d_model)
          - Attention: query x encoder_output -> (batch, N_out, N_in)
          - Apply mask: blocked positions -> -inf before softmax
          - Weighted sum -> (batch, N_out, d_model)
          - Project -> (batch, N_out)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SignalMappingLayer(nn.Module):
    """
    Cross-attention based Signal Mapping Layer with configurable masking.

    For each output signal, learns which input signals to attend to,
    with hard constraints from the mask matrix.

    Args:
        d_model (int): Transformer model dimension
        num_output_signals (int): Number of output signals
        num_input_signals (int): Number of input signals
        mask_matrix (torch.Tensor, optional): Binary mask of shape (num_output, num_input).
            1 = allow attention, 0 = block attention. Default: all ones (no masking).
    """

    def __init__(self, d_model, num_output_signals, num_input_signals, mask_matrix=None):
        super().__init__()

        self.d_model = d_model
        self.num_output_signals = num_output_signals
        self.num_input_signals = num_input_signals

        # Learnable query vector for each output signal
        self.output_queries = nn.Parameter(torch.randn(num_output_signals, d_model))

        # Scaling factor for attention scores (standard scaled dot-product)
        self.scale = d_model ** 0.5

        # Output projection: d_model -> 1 per output signal
        self.output_projection = nn.Linear(d_model, 1)

        # Register mask as buffer (saved with model but not trained)
        if mask_matrix is not None:
            self.register_buffer('mask', mask_matrix.float())
        else:
            self.register_buffer('mask', torch.ones(num_output_signals, num_input_signals))

    def forward(self, encoder_output):
        """
        Args:
            encoder_output: (batch, num_input, d_model) from Transformer encoder

        Returns:
            (batch, num_output) predicted values for each output signal
        """
        # Compute attention scores
        # output_queries: (num_out, d_model) -> (1, num_out, d_model)
        # encoder_output: (batch, num_in, d_model) -> transpose -> (batch, d_model, num_in)
        # Result: (batch, num_out, num_in)
        attn_scores = torch.matmul(
            self.output_queries.unsqueeze(0),
            encoder_output.transpose(-1, -2)
        ) / self.scale

        # Apply mask: blocked positions get -inf so softmax produces 0
        attn_scores = attn_scores.masked_fill(self.mask.unsqueeze(0) == 0, float('-inf'))

        # Softmax over input dimension
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Handle edge case: if ALL inputs are masked for an output, softmax gives NaN
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        # Weighted sum of encoder outputs
        # attn_weights: (batch, num_out, num_in)
        # encoder_output: (batch, num_in, d_model)
        # Result: (batch, num_out, d_model)
        context = torch.matmul(attn_weights, encoder_output)

        # Project each output signal's context to a scalar
        # (batch, num_out, d_model) -> (batch, num_out, 1) -> (batch, num_out)
        output = self.output_projection(context).squeeze(-1)

        return output


class MaskedSST(nn.Module):
    """
    StaticSensorTransformer with Signal Mapping Layer.

    Supports configurable signal-to-signal masking for Level 2/3 digital twin.
    When mask is all ones, behaves similarly to the original SST but with
    learned per-output attention instead of global average pooling.

    Args:
        num_input_signals (int): Number of input (boundary) signals
        num_output_signals (int): Number of output (target) signals
        d_model (int): Transformer model dimension. Default: 128
        nhead (int): Number of attention heads. Default: 8
        num_layers (int): Number of transformer encoder layers. Default: 3
        dropout (float): Dropout rate. Default: 0.1
        mask_matrix (torch.Tensor, optional): Binary mask (num_output, num_input).
            1 = allow, 0 = block. Default: all ones.

    Example:
        >>> mask = torch.ones(5, 4)
        >>> mask[4, 1] = 0  # Block input[1] -> output[4]
        >>> model = MaskedSST(num_input_signals=4, num_output_signals=5, mask_matrix=mask)
        >>> x = torch.randn(32, 4)
        >>> y = model(x)  # (32, 5)
    """

    def __init__(self, num_input_signals, num_output_signals,
                 d_model=128, nhead=8, num_layers=3, dropout=0.1,
                 mask_matrix=None):
        super().__init__()

        self.num_input_signals = num_input_signals
        self.num_output_signals = num_output_signals
        self.d_model = d_model

        # Input signal embedding: scalar -> d_model
        self.input_embedding = nn.Linear(1, d_model)
        self.position_encoding = nn.Parameter(torch.randn(num_input_signals, d_model))

        # Transformer encoder (full self-attention among input signals)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Signal Mapping Layer (replaces global_pool + output_projection)
        self.signal_mapping = SignalMappingLayer(
            d_model=d_model,
            num_output_signals=num_output_signals,
            num_input_signals=num_input_signals,
            mask_matrix=mask_matrix
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input of shape (batch_size, num_input_signals)

        Returns:
            torch.Tensor: Predictions of shape (batch_size, num_output_signals)
        """
        # Embed each input signal
        x = x.unsqueeze(-1)  # (batch, num_input, 1)
        x = self.input_embedding(x) + self.position_encoding.unsqueeze(0)

        # Transformer encoder: signals interact via self-attention
        x = self.transformer(x)  # (batch, num_input, d_model)

        # Signal Mapping Layer: masked cross-attention to produce outputs
        output = self.signal_mapping(x)  # (batch, num_output)

        return output


def load_model_from_checkpoint(checkpoint, device, mask_override=None):
    """
    Load either MaskedSST or StaticSensorTransformer from a checkpoint.

    Automatically detects model type from checkpoint metadata.

    Args:
        checkpoint: Dict loaded from torch.load()
        device: torch device
        mask_override: Optional mask matrix to override the saved one

    Returns:
        nn.Module: Loaded model in eval mode
    """
    model_type = checkpoint.get('model_type', 'SST')
    cfg = checkpoint['model_config']

    if model_type == 'MaskedSST':
        # Reconstruct mask
        signal_mapping_data = checkpoint.get('signal_mapping', {})
        if mask_override is not None:
            mask = mask_override
        elif 'mask_matrix' in signal_mapping_data:
            mask = torch.tensor(signal_mapping_data['mask_matrix'], dtype=torch.float32)
        else:
            mask = None

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
        # Fallback to original SST
        import sys, os
        PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        sys.path.insert(0, PROJECT_ROOT)
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
    return model
