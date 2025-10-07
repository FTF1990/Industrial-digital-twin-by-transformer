"""
Industrial Digital Twin Models by Transformer

This package contains Transformer-based models for industrial sensor prediction.
"""

from .static_transformer import StaticSensorTransformer, SST
from .hybrid_transformer import HybridSensorTransformer, HST

# Backward compatibility aliases
CompactSensorTransformer = StaticSensorTransformer  # Legacy V1 name
HybridTemporalTransformer = HybridSensorTransformer  # Legacy V4 name

__all__ = [
    'StaticSensorTransformer',
    'HybridSensorTransformer',
    'SST',
    'HST',
    # Legacy names for backward compatibility
    'CompactSensorTransformer',
    'HybridTemporalTransformer'
]

__version__ = '2.0.0'
