"""
Industrial Digital Twin Models by Transformer

This package contains Transformer-based models for industrial sensor prediction.

Enhanced version with Stage2 Residual Boost training system.
"""

from .static_transformer import StaticSensorTransformer, SST

__all__ = [
    'StaticSensorTransformer',
    'SST',
]

__version__ = '1.0.0'  # Enhanced with Stage2 Boost
