"""
Inference module for trained models
"""

import torch
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt


class ModelInference:
    """
    Inference engine for trained digital twin models

    This class handles model predictions, evaluation, and visualization
    for both V1 and V4 models.
    """

    def __init__(self, model, scaler_X, scaler_y, device='cuda'):
        """
        Initialize the inference engine

        Args:
            model (nn.Module): Trained model (V1 or V4)
            scaler_X (StandardScaler): Scaler for input features
            scaler_y (StandardScaler): Scaler for target values
            device (str): Device to use for inference. Default: 'cuda'
        """
        self.model = model
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

    def predict(self, X):
        """
        Make predictions on input data

        Args:
            X (np.ndarray): Input features (unscaled)

        Returns:
            np.ndarray: Predictions in original scale
        """
        # Scale input
        X_scaled = self.scaler_X.transform(X)

        # Predict
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            y_pred_scaled = self.model(X_tensor).cpu().numpy()

        # Inverse transform
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)

        return y_pred

    def evaluate(self, X, y_true, target_signal_names=None):
        """
        Evaluate model performance

        Args:
            X (np.ndarray): Input features (unscaled)
            y_true (np.ndarray): True target values (unscaled)
            target_signal_names (list, optional): Names of target signals

        Returns:
            dict: Evaluation metrics for each target signal
        """
        y_pred = self.predict(X)

        metrics = {}
        n_targets = y_true.shape[1]

        for i in range(n_targets):
            signal_name = target_signal_names[i] if target_signal_names else f"Signal_{i}"

            r2 = r2_score(y_true[:, i], y_pred[:, i])
            rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
            mae = mean_absolute_error(y_true[:, i], y_pred[:, i])

            metrics[signal_name] = {
                'R2': r2,
                'RMSE': rmse,
                'MAE': mae
            }

        # Overall metrics
        metrics['Overall'] = {
            'R2': np.mean([m['R2'] for m in metrics.values() if m != metrics.get('Overall')]),
            'RMSE': np.mean([m['RMSE'] for m in metrics.values() if m != metrics.get('Overall')]),
            'MAE': np.mean([m['MAE'] for m in metrics.values() if m != metrics.get('Overall')])
        }

        return metrics

    def plot_predictions(self, X, y_true, signal_indices=None,
                        target_signal_names=None, start_idx=0, end_idx=None):
        """
        Plot predictions vs actual values

        Args:
            X (np.ndarray): Input features (unscaled)
            y_true (np.ndarray): True target values (unscaled)
            signal_indices (list, optional): Indices of signals to plot
            target_signal_names (list, optional): Names of target signals
            start_idx (int): Start index for plotting. Default: 0
            end_idx (int, optional): End index for plotting

        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        y_pred = self.predict(X)

        if end_idx is None:
            end_idx = len(y_true)

        y_true_slice = y_true[start_idx:end_idx]
        y_pred_slice = y_pred[start_idx:end_idx]

        # Determine which signals to plot
        if signal_indices is None:
            signal_indices = range(min(3, y_true.shape[1]))  # Plot first 3 by default

        n_signals = len(signal_indices)
        fig, axes = plt.subplots(n_signals, 3, figsize=(18, 5*n_signals))

        if n_signals == 1:
            axes = axes.reshape(1, -1)

        for i, sig_idx in enumerate(signal_indices):
            signal_name = target_signal_names[sig_idx] if target_signal_names else f"Signal_{sig_idx}"

            y_true_sig = y_true_slice[:, sig_idx]
            y_pred_sig = y_pred_slice[:, sig_idx]
            residuals = y_true_sig - y_pred_sig

            # Time series plot
            ax1 = axes[i, 0]
            ax1.plot(range(len(y_true_sig)), y_true_sig, label='Actual', linewidth=2, alpha=0.8)
            ax1.plot(range(len(y_pred_sig)), y_pred_sig, label='Predicted', linewidth=2, alpha=0.8)
            ax1.set_title(f'{signal_name}\nPrediction vs Actual')
            ax1.set_xlabel('Time Step')
            ax1.set_ylabel('Value')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Residuals plot
            ax2 = axes[i, 1]
            ax2.plot(range(len(residuals)), residuals, color='red', linewidth=1.5, alpha=0.7)
            ax2.axhline(y=0, color='black', linestyle='--', linewidth=2)
            ax2.fill_between(range(len(residuals)), residuals, alpha=0.3, color='red')
            ax2.set_title(f'{signal_name}\nResiduals')
            ax2.set_xlabel('Time Step')
            ax2.set_ylabel('Residual (Actual - Predicted)')
            ax2.grid(True, alpha=0.3)

            # Scatter plot
            ax3 = axes[i, 2]
            ax3.scatter(y_true_sig, y_pred_sig, alpha=0.6, s=20)
            min_val = min(y_true_sig.min(), y_pred_sig.min())
            max_val = max(y_true_sig.max(), y_pred_sig.max())
            ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
            r2 = r2_score(y_true_sig, y_pred_sig)
            ax3.set_title(f'{signal_name}\nAccuracy (R²={r2:.3f})')
            ax3.set_xlabel('Actual Value')
            ax3.set_ylabel('Predicted Value')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def print_metrics(self, metrics):
        """
        Print evaluation metrics in a formatted way

        Args:
            metrics (dict): Metrics dictionary from evaluate()
        """
        print("=" * 60)
        print("Model Evaluation Metrics")
        print("=" * 60)

        for signal_name, metric in metrics.items():
            if signal_name == 'Overall':
                print("\n" + "=" * 60)
                print("OVERALL PERFORMANCE")
                print("=" * 60)

            print(f"\n{signal_name}:")
            print(f"  R² Score:  {metric['R2']:.4f}")
            print(f"  RMSE:      {metric['RMSE']:.4f}")
            print(f"  MAE:       {metric['MAE']:.4f}")

        print("\n" + "=" * 60)
