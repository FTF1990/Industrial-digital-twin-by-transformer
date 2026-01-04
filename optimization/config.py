"""
Configuration classes for inverse optimization
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class OptimizationConfig:
    """
    Configuration for inverse optimization

    Attributes:
        learning_rate: Learning rate for optimizer (default: 0.01)
        max_epochs: Maximum number of optimization iterations (default: 500)
        optimizer_type: Type of optimizer ('adam' or 'sgd') (default: 'adam')
        gradient_clip: Maximum gradient norm for clipping (default: 1.0)
        patience: Early stopping patience (default: 50)
        min_delta: Minimum change to qualify as improvement (default: 1e-6)
        use_gradient_normalization: Whether to normalize gradients (default: True)
        constraint_projection_freq: Frequency of constraint projection (default: 1)
        verbose: Print optimization progress (default: True)
        loss_type: Loss function type ('mse', 'mae', 'huber') (default: 'mse')
        lr_scheduler: Use learning rate scheduler (default: True)
        lr_scheduler_patience: LR scheduler patience (default: 20)
        lr_scheduler_factor: LR reduction factor (default: 0.5)
    """

    # Optimizer configuration
    learning_rate: float = 0.01
    max_epochs: int = 500
    optimizer_type: str = "adam"  # 'adam' or 'sgd'
    gradient_clip: float = 1.0

    # Early stopping configuration
    patience: int = 50
    min_delta: float = 1e-6

    # Numerical stability
    use_gradient_normalization: bool = True
    constraint_projection_freq: int = 1  # Project every N steps

    # Output
    verbose: bool = True

    # Loss function
    loss_type: str = "mse"  # 'mse', 'mae', 'huber'

    # Learning rate scheduler
    lr_scheduler: bool = True
    lr_scheduler_patience: int = 20
    lr_scheduler_factor: float = 0.5

    def __post_init__(self):
        """Validate configuration"""
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be positive")
        if self.optimizer_type not in ['adam', 'sgd']:
            raise ValueError("optimizer_type must be 'adam' or 'sgd'")
        if self.gradient_clip <= 0:
            raise ValueError("gradient_clip must be positive")
        if self.patience < 0:
            raise ValueError("patience must be non-negative")
        if self.loss_type not in ['mse', 'mae', 'huber']:
            raise ValueError("loss_type must be 'mse', 'mae', or 'huber'")


@dataclass
class MultiObjectiveConfig:
    """
    Configuration for multi-objective optimization

    Attributes:
        n_pareto_points: Number of points on Pareto frontier (default: 20)
        weight_min: Minimum weight value (default: 0.0)
        weight_max: Maximum weight value (default: 1.0)
        base_config: Base optimization configuration
    """

    n_pareto_points: int = 20
    weight_min: float = 0.0
    weight_max: float = 1.0
    base_config: Optional[OptimizationConfig] = None

    def __post_init__(self):
        """Validate and initialize"""
        if self.n_pareto_points < 2:
            raise ValueError("n_pareto_points must be at least 2")
        if not (0 <= self.weight_min <= 1):
            raise ValueError("weight_min must be in [0, 1]")
        if not (0 <= self.weight_max <= 1):
            raise ValueError("weight_max must be in [0, 1]")
        if self.weight_min >= self.weight_max:
            raise ValueError("weight_min must be less than weight_max")

        # Initialize base config if not provided
        if self.base_config is None:
            self.base_config = OptimizationConfig(verbose=False)


@dataclass
class KalmanConfig:
    """
    Configuration for Kalman filter correction

    Attributes:
        process_noise: Process noise covariance (Q matrix diagonal) (default: 0.01)
        measurement_noise: Measurement noise covariance (R matrix diagonal) (default: 0.1)
        initial_state_covariance: Initial state covariance (P0 diagonal) (default: 1.0)
        prediction_horizon: Number of steps to predict ahead (default: 1)
    """

    process_noise: float = 0.01
    measurement_noise: float = 0.1
    initial_state_covariance: float = 1.0
    prediction_horizon: int = 1

    def __post_init__(self):
        """Validate configuration"""
        if self.process_noise <= 0:
            raise ValueError("process_noise must be positive")
        if self.measurement_noise <= 0:
            raise ValueError("measurement_noise must be positive")
        if self.initial_state_covariance <= 0:
            raise ValueError("initial_state_covariance must be positive")
        if self.prediction_horizon < 1:
            raise ValueError("prediction_horizon must be at least 1")
