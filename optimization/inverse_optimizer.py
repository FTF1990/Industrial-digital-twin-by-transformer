"""
Gradient-based inverse optimizer for digital twin models

This module implements neural network gradient-based inverse optimization,
where the model parameters are frozen and input boundary conditions are
optimized to achieve target outputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable, Any
from sklearn.preprocessing import StandardScaler
import time

from .config import OptimizationConfig
from .constraints import ConstraintManager, InputConstraint


class InverseOptimizer:
    """
    Gradient-based inverse optimizer

    This class performs inverse optimization by:
    1. Freezing trained model parameters (model as digital twin)
    2. Setting input boundary conditions as optimizable variables
    3. Using gradient descent to minimize difference between predictions and targets
    4. Projecting inputs to satisfy constraints at each iteration

    The optimization uses standard PyTorch optimizers (Adam/SGD) but optimizes
    inputs instead of model parameters.
    """

    def __init__(
        self,
        model: nn.Module,
        scaler_X: StandardScaler,
        scaler_y: StandardScaler,
        device: str = 'cuda'
    ):
        """
        Initialize inverse optimizer

        Args:
            model: Trained model (will be frozen)
            scaler_X: Input feature scaler
            scaler_y: Output target scaler
            device: Device for computation ('cuda' or 'cpu')
        """
        self.model = model
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Move model to device and freeze parameters
        self.model.to(self.device)
        self.model.eval()
        self._freeze_model()

        # Get input/output dimensions
        self.n_inputs = scaler_X.n_features_in_
        self.n_outputs = scaler_y.n_features_in_

    def _freeze_model(self):
        """Freeze all model parameters to prevent updates"""
        for param in self.model.parameters():
            param.requires_grad = False

    def _compute_loss(
        self,
        predictions: torch.Tensor,
        targets: Dict[str, Dict[str, float]],
        loss_type: str = 'mse'
    ) -> torch.Tensor:
        """
        Compute optimization loss

        Args:
            predictions: Model predictions (scaled)
            targets: Target specifications {signal_idx: {'value': float, 'weight': float}}
            loss_type: Loss function type ('mse', 'mae', 'huber')

        Returns:
            Loss value
        """
        total_loss = torch.tensor(0.0, device=self.device)

        for signal_idx, target_spec in targets.items():
            target_value = target_spec['value']  # Already in scaled space
            weight = target_spec.get('weight', 1.0)

            pred = predictions[signal_idx]
            target = torch.tensor(target_value, dtype=torch.float32, device=self.device)

            # Compute loss based on type
            if loss_type == 'mse':
                loss = F.mse_loss(pred, target)
            elif loss_type == 'mae':
                loss = F.l1_loss(pred, target)
            elif loss_type == 'huber':
                loss = F.smooth_l1_loss(pred, target)
            else:
                raise ValueError(f"Unknown loss_type: {loss_type}")

            total_loss = total_loss + weight * loss

        return total_loss

    def optimize(
        self,
        targets: Dict[int, Dict[str, float]],
        constraint_manager: ConstraintManager,
        initial_inputs: Optional[np.ndarray] = None,
        config: Optional[OptimizationConfig] = None,
        callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Run inverse optimization

        Args:
            targets: Target specifications {signal_idx: {'value': val, 'bias': bias, 'weight': w}}
                    - 'value': Target value in original scale
                    - 'bias': Relative change (e.g., -0.10 for -10%), optional
                    - 'weight': Objective weight, optional (default 1.0)
            constraint_manager: Constraint manager for inputs
            initial_inputs: Initial input values (original scale), optional
            config: Optimization configuration, optional
            callback: Callback function(epoch, loss, inputs), optional

        Returns:
            Dictionary containing:
                - 'optimized_inputs': Final optimized inputs (original scale)
                - 'predictions': Final predictions (original scale)
                - 'loss_history': Loss values over iterations
                - 'input_history': Input values over iterations
                - 'converged': Whether optimization converged
                - 'num_epochs': Number of epochs run
                - 'final_loss': Final loss value
        """
        if config is None:
            config = OptimizationConfig()

        # Get initial inputs
        if initial_inputs is None:
            initial_inputs = constraint_manager.get_initial_values()
        else:
            initial_inputs = np.array(initial_inputs)

        # Validate initial inputs
        if not constraint_manager.validate_values(initial_inputs):
            initial_inputs = constraint_manager.project(
                torch.from_numpy(initial_inputs).float()
            ).numpy()

        # Scale inputs
        initial_inputs_scaled = self.scaler_X.transform(initial_inputs.reshape(1, -1))[0]

        # Convert to tensor with gradient
        optimizable_inputs = torch.tensor(
            initial_inputs_scaled,
            dtype=torch.float32,
            requires_grad=True,
            device=self.device
        )

        # Process targets: convert values to scaled space
        targets_scaled = {}
        for signal_idx, target_spec in targets.items():
            # Create dummy array with target value at signal_idx
            dummy = np.zeros((1, self.n_outputs))

            if 'value' in target_spec:
                # Direct target value
                target_value_orig = target_spec['value']
            elif 'bias' in target_spec:
                # Relative change from current prediction
                # First predict current output
                with torch.no_grad():
                    current_pred_scaled = self.model(optimizable_inputs.unsqueeze(0))[0]
                    current_pred_orig = self.scaler_y.inverse_transform(
                        current_pred_scaled.cpu().numpy().reshape(1, -1)
                    )[0]

                # Apply bias
                bias = target_spec['bias']
                target_value_orig = current_pred_orig[signal_idx] * (1 + bias)
            else:
                raise ValueError(f"Target {signal_idx} must have 'value' or 'bias'")

            # Scale target value
            dummy[0, signal_idx] = target_value_orig
            target_value_scaled = self.scaler_y.transform(dummy)[0, signal_idx]

            targets_scaled[signal_idx] = {
                'value': target_value_scaled,
                'weight': target_spec.get('weight', 1.0)
            }

        # Setup optimizer
        if config.optimizer_type == 'adam':
            optimizer = torch.optim.Adam([optimizable_inputs], lr=config.learning_rate)
        elif config.optimizer_type == 'sgd':
            optimizer = torch.optim.SGD(
                [optimizable_inputs], lr=config.learning_rate, momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer_type: {config.optimizer_type}")

        # Setup learning rate scheduler
        if config.lr_scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=config.lr_scheduler_factor,
                patience=config.lr_scheduler_patience,
                verbose=config.verbose
            )

        # Tracking variables
        loss_history = []
        input_history = []
        best_loss = float('inf')
        patience_counter = 0
        converged = False

        # Optimization loop
        start_time = time.time()

        for epoch in range(config.max_epochs):
            optimizer.zero_grad()

            # Forward pass
            predictions_scaled = self.model(optimizable_inputs.unsqueeze(0))[0]

            # Compute loss
            loss = self._compute_loss(
                predictions_scaled, targets_scaled, config.loss_type
            )

            # Backward pass
            loss.backward()

            # Gradient clipping
            if config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_([optimizable_inputs], config.gradient_clip)

            # Gradient normalization (optional)
            if config.use_gradient_normalization and optimizable_inputs.grad is not None:
                grad_norm = torch.norm(optimizable_inputs.grad)
                if grad_norm > 0:
                    optimizable_inputs.grad = optimizable_inputs.grad / grad_norm

            # Update inputs
            optimizer.step()

            # Project to constraints
            if epoch % config.constraint_projection_freq == 0:
                with torch.no_grad():
                    # Convert to original scale for projection
                    inputs_orig = self.scaler_X.inverse_transform(
                        optimizable_inputs.cpu().numpy().reshape(1, -1)
                    )[0]

                    # Project
                    inputs_projected = constraint_manager.project(
                        torch.from_numpy(inputs_orig).float()
                    ).numpy()

                    # Convert back to scaled space
                    inputs_scaled = self.scaler_X.transform(
                        inputs_projected.reshape(1, -1)
                    )[0]

                    optimizable_inputs.data = torch.from_numpy(inputs_scaled).float().to(
                        self.device
                    )

            # Record history
            loss_value = loss.item()
            loss_history.append(loss_value)

            inputs_orig = self.scaler_X.inverse_transform(
                optimizable_inputs.detach().cpu().numpy().reshape(1, -1)
            )[0]
            input_history.append(inputs_orig.copy())

            # Update learning rate scheduler
            if config.lr_scheduler:
                scheduler.step(loss_value)

            # Early stopping check
            if loss_value < best_loss - config.min_delta:
                best_loss = loss_value
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= config.patience:
                converged = True
                if config.verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break

            # Callback
            if callback is not None:
                callback(epoch, loss_value, inputs_orig)

            # Print progress
            if config.verbose and (epoch % 50 == 0 or epoch == config.max_epochs - 1):
                print(f"Epoch {epoch:4d} | Loss: {loss_value:.6f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Final prediction
        with torch.no_grad():
            final_predictions_scaled = self.model(optimizable_inputs.unsqueeze(0))[0]
            final_predictions = self.scaler_y.inverse_transform(
                final_predictions_scaled.cpu().numpy().reshape(1, -1)
            )[0]

        # Final inputs
        final_inputs = self.scaler_X.inverse_transform(
            optimizable_inputs.detach().cpu().numpy().reshape(1, -1)
        )[0]

        elapsed_time = time.time() - start_time

        if config.verbose:
            print(f"\nOptimization completed in {elapsed_time:.2f}s ({len(loss_history)} epochs)")
            print(f"Final loss: {loss_history[-1]:.6f}")

        return {
            'optimized_inputs': final_inputs,
            'predictions': final_predictions,
            'loss_history': loss_history,
            'input_history': np.array(input_history),
            'converged': converged,
            'num_epochs': len(loss_history),
            'final_loss': loss_history[-1],
            'elapsed_time': elapsed_time,
            'targets': targets,
            'config': config
        }

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """
        Make predictions with the model

        Args:
            inputs: Input values (original scale)

        Returns:
            Predictions (original scale)
        """
        inputs_scaled = self.scaler_X.transform(inputs.reshape(1, -1))
        inputs_tensor = torch.from_numpy(inputs_scaled).float().to(self.device)

        with torch.no_grad():
            predictions_scaled = self.model(inputs_tensor)
            predictions = self.scaler_y.inverse_transform(
                predictions_scaled.cpu().numpy()
            )

        return predictions[0]
