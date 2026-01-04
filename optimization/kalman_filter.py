"""
Kalman filter for real-time control correction

This module implements Kalman filtering to correct optimized control strategies
in real-time based on sensor feedback.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from sklearn.preprocessing import StandardScaler

try:
    from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
    from filterpy.common import Q_discrete_white_noise
    FILTERPY_AVAILABLE = True
except ImportError:
    FILTERPY_AVAILABLE = False
    print("Warning: filterpy not installed. Kalman filter functionality will be limited.")

from .config import KalmanConfig


class KalmanCorrector:
    """
    Kalman filter-based real-time control corrector

    Uses Unscented Kalman Filter (UKF) to handle the nonlinear neural network
    model as state transition function, correcting control inputs based on
    real-time sensor measurements.

    State vector: [optimizable_inputs, target_outputs]
    Measurement vector: [measured_target_outputs]
    """

    def __init__(
        self,
        model: torch.nn.Module,
        scaler_X: StandardScaler,
        scaler_y: StandardScaler,
        optimizable_input_indices: List[int],
        target_output_indices: List[int],
        fixed_input_values: Optional[np.ndarray] = None,
        device: str = 'cuda'
    ):
        """
        Initialize Kalman corrector

        Args:
            model: Trained neural network model
            scaler_X: Input scaler
            scaler_y: Output scaler
            optimizable_input_indices: Indices of inputs that can be optimized
            target_output_indices: Indices of outputs to track
            fixed_input_values: Values for fixed inputs (original scale)
            device: Device for computation
        """
        if not FILTERPY_AVAILABLE:
            raise ImportError(
                "filterpy is required for Kalman filtering. "
                "Install with: pip install filterpy"
            )

        self.model = model
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        self.optimizable_indices = optimizable_input_indices
        self.target_indices = target_output_indices
        self.fixed_values = fixed_input_values

        self.n_optimizable = len(optimizable_input_indices)
        self.n_targets = len(target_output_indices)
        self.n_inputs_total = scaler_X.n_features_in_

        # State dimension: optimizable inputs + target outputs
        self.dim_x = self.n_optimizable + self.n_targets

        # Measurement dimension: target outputs
        self.dim_z = self.n_targets

        # Move model to device
        self.model.to(self.device)
        self.model.eval()

        # Freeze model parameters
        for param in self.model.parameters():
            param.requires_grad = False

        self.ukf = None

    def _state_transition_function(self, x: np.ndarray, dt: float) -> np.ndarray:
        """
        State transition function using neural network model

        State vector x = [optimizable_inputs, target_outputs]

        Args:
            x: State vector
            dt: Time step (not used for static model)

        Returns:
            Next state vector
        """
        # Extract optimizable inputs from state
        optimizable_inputs = x[:self.n_optimizable]

        # Build full input vector
        full_inputs = np.zeros(self.n_inputs_total)

        if self.fixed_values is not None:
            full_inputs[:] = self.fixed_values

        full_inputs[self.optimizable_indices] = optimizable_inputs

        # Predict with model
        inputs_scaled = self.scaler_X.transform(full_inputs.reshape(1, -1))
        inputs_tensor = torch.from_numpy(inputs_scaled).float().to(self.device)

        with torch.no_grad():
            outputs_scaled = self.model(inputs_tensor)
            outputs = self.scaler_y.inverse_transform(
                outputs_scaled.cpu().numpy()
            )[0]

        # Extract target outputs
        target_outputs = outputs[self.target_indices]

        # New state: inputs remain same, outputs updated
        next_state = np.concatenate([optimizable_inputs, target_outputs])

        return next_state

    def _measurement_function(self, x: np.ndarray) -> np.ndarray:
        """
        Measurement function (extracts target outputs from state)

        Args:
            x: State vector

        Returns:
            Measurement vector (target outputs)
        """
        # Measurement is just the target outputs part of state
        return x[self.n_optimizable:]

    def initialize_filter(
        self,
        initial_state: np.ndarray,
        config: Optional[KalmanConfig] = None
    ):
        """
        Initialize Unscented Kalman Filter

        Args:
            initial_state: Initial state vector [optimizable_inputs, target_outputs]
            config: Kalman configuration
        """
        if config is None:
            config = KalmanConfig()

        # Create sigma points
        points = MerweScaledSigmaPoints(
            n=self.dim_x,
            alpha=0.1,
            beta=2.0,
            kappa=0.0
        )

        # Create UKF
        self.ukf = UnscentedKalmanFilter(
            dim_x=self.dim_x,
            dim_z=self.dim_z,
            dt=1.0,
            fx=self._state_transition_function,
            hx=self._measurement_function,
            points=points
        )

        # Set initial state
        self.ukf.x = initial_state.copy()

        # Set initial covariance
        self.ukf.P = np.eye(self.dim_x) * config.initial_state_covariance

        # Set process noise covariance Q
        self.ukf.Q = np.eye(self.dim_x) * config.process_noise

        # Set measurement noise covariance R
        self.ukf.R = np.eye(self.dim_z) * config.measurement_noise

    def predict(self) -> np.ndarray:
        """
        Prediction step

        Returns:
            Predicted state
        """
        if self.ukf is None:
            raise RuntimeError("Filter not initialized. Call initialize_filter() first.")

        self.ukf.predict()
        return self.ukf.x.copy()

    def update(self, measurement: np.ndarray):
        """
        Update step with measurement

        Args:
            measurement: Measured target outputs
        """
        if self.ukf is None:
            raise RuntimeError("Filter not initialized. Call initialize_filter() first.")

        self.ukf.update(measurement)

    def get_state(self) -> np.ndarray:
        """Get current state estimate"""
        if self.ukf is None:
            raise RuntimeError("Filter not initialized.")
        return self.ukf.x.copy()

    def get_corrected_inputs(self) -> np.ndarray:
        """Get corrected optimizable inputs from current state"""
        if self.ukf is None:
            raise RuntimeError("Filter not initialized.")
        return self.ukf.x[:self.n_optimizable].copy()

    def run_simulation(
        self,
        initial_state: np.ndarray,
        measurements: np.ndarray,
        config: Optional[KalmanConfig] = None
    ) -> Dict[str, Any]:
        """
        Run Kalman filter simulation over a sequence of measurements

        Args:
            initial_state: Initial state [optimizable_inputs, target_outputs]
            measurements: Sequence of measurements (n_steps, n_targets)
            config: Kalman configuration

        Returns:
            Dictionary containing:
                - 'states': State history (n_steps, dim_x)
                - 'predictions': Prediction history before update
                - 'corrected_inputs': Corrected input history
                - 'innovations': Measurement innovations (actual - predicted)
        """
        if config is None:
            config = KalmanConfig()

        # Initialize filter
        self.initialize_filter(initial_state, config)

        n_steps = len(measurements)

        # Storage
        states = np.zeros((n_steps, self.dim_x))
        predictions = np.zeros((n_steps, self.dim_x))
        corrected_inputs = np.zeros((n_steps, self.n_optimizable))
        innovations = np.zeros((n_steps, self.dim_z))

        for t in range(n_steps):
            # Prediction
            predicted_state = self.predict()
            predictions[t] = predicted_state

            # Update with measurement
            measurement = measurements[t]
            self.update(measurement)

            # Record state
            states[t] = self.get_state()
            corrected_inputs[t] = self.get_corrected_inputs()

            # Compute innovation
            predicted_measurement = self._measurement_function(predicted_state)
            innovations[t] = measurement - predicted_measurement

        return {
            'states': states,
            'predictions': predictions,
            'corrected_inputs': corrected_inputs,
            'innovations': innovations,
            'measurements': measurements
        }

    def compute_correction_metrics(
        self,
        simulation_results: Dict[str, Any],
        uncorrected_predictions: np.ndarray
    ) -> pd.DataFrame:
        """
        Compute metrics comparing corrected vs uncorrected predictions

        Args:
            simulation_results: Results from run_simulation()
            uncorrected_predictions: Model predictions without Kalman correction

        Returns:
            DataFrame with correction metrics
        """
        measurements = simulation_results['measurements']
        corrected_predictions = simulation_results['states'][:, self.n_optimizable:]

        metrics = []

        for i in range(self.n_targets):
            # Uncorrected errors
            uncorrected_error = measurements[:, i] - uncorrected_predictions[:, i]
            uncorrected_mae = np.mean(np.abs(uncorrected_error))
            uncorrected_rmse = np.sqrt(np.mean(uncorrected_error**2))

            # Corrected errors
            corrected_error = measurements[:, i] - corrected_predictions[:, i]
            corrected_mae = np.mean(np.abs(corrected_error))
            corrected_rmse = np.sqrt(np.mean(corrected_error**2))

            # Improvement
            mae_improvement = (uncorrected_mae - corrected_mae) / uncorrected_mae * 100
            rmse_improvement = (uncorrected_rmse - corrected_rmse) / uncorrected_rmse * 100

            metrics.append({
                'Target_Index': self.target_indices[i],
                'Uncorrected_MAE': uncorrected_mae,
                'Corrected_MAE': corrected_mae,
                'MAE_Improvement_%': mae_improvement,
                'Uncorrected_RMSE': uncorrected_rmse,
                'Corrected_RMSE': corrected_rmse,
                'RMSE_Improvement_%': rmse_improvement
            })

        return pd.DataFrame(metrics)

    @staticmethod
    def plot_correction_results(
        simulation_results: Dict[str, Any],
        uncorrected_predictions: np.ndarray,
        target_names: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (15, 10)
    ):
        """
        Plot Kalman correction results

        Args:
            simulation_results: Results from run_simulation()
            uncorrected_predictions: Model predictions without correction
            target_names: Names of target signals
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        import matplotlib.pyplot as plt

        measurements = simulation_results['measurements']
        corrected_predictions = simulation_results['states'][:, -measurements.shape[1]:]
        n_targets = measurements.shape[1]
        n_steps = len(measurements)

        if target_names is None:
            target_names = [f'Target_{i}' for i in range(n_targets)]

        # Create subplots
        fig, axes = plt.subplots(n_targets, 1, figsize=figsize)
        if n_targets == 1:
            axes = [axes]

        time_steps = np.arange(n_steps)

        for i in range(n_targets):
            ax = axes[i]

            # Plot three lines: actual, uncorrected, corrected
            ax.plot(
                time_steps, measurements[:, i],
                'k-', label='Actual (Measured)', linewidth=2, alpha=0.7
            )
            ax.plot(
                time_steps, uncorrected_predictions[:, i],
                'r--', label='Uncorrected Prediction', linewidth=1.5, alpha=0.6
            )
            ax.plot(
                time_steps, corrected_predictions[:, i],
                'g-', label='Kalman Corrected', linewidth=1.5, alpha=0.8
            )

            ax.set_xlabel('Time Step')
            ax.set_ylabel('Value')
            ax.set_title(f'{target_names[i]} - Kalman Correction')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig
