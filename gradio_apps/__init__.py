"""
Inverse Optimization Module for Industrial Digital Twin

This module provides gradient-based inverse optimization capabilities for
control parameter tuning using trained digital twin models.

Core Components:
- InverseOptimizer: Gradient-based input optimization
- MultiObjectiveOptimizer: Pareto frontier generation
- KalmanCorrector: Real-time correction with Kalman filtering
- ConstraintManager: Input constraint handling
"""

from .config import OptimizationConfig, MultiObjectiveConfig, KalmanConfig
from .constraints import InputConstraint, ConstraintManager
from .inverse_optimizer import InverseOptimizer
from .multi_objective import MultiObjectiveOptimizer
from .kalman_filter import KalmanCorrector

__all__ = [
    'OptimizationConfig',
    'MultiObjectiveConfig',
    'KalmanConfig',
    'InputConstraint',
    'ConstraintManager',
    'InverseOptimizer',
    'MultiObjectiveOptimizer',
    'KalmanCorrector',
]

__version__ = '2.0.0'
