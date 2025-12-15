"""
Ensemble Model Builder - Generate ensemble models from Stage1 and Residual Boost
"""
import os
import sys
import json
import numpy as np
import torch
from typing import Dict, Any, List, Tuple
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Add parent directory to import models
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from models.residual_tft import batch_inference, compute_r2_safe
except ImportError:
    print("   Warning: Could not import from models.residual_tft")


class EnsembleBuilder:
    """Build ensemble models using Delta R² strategy"""

    @staticmethod
    def create_ensemble_from_data(
        ensemble_name: str,
        stage1_model_info: Dict[str, Any],
        residual_boost_model_info: Dict[str, Any],
        evaluation_data: np.ndarray,  # Shape: (N, num_boundary + num_target)
        boundary_signals: List[str],
        target_signals: List[str],
        delta_r2_threshold: float = 0.05,
        device: torch.device = None,
        save_config: bool = True,
        save_dir: str = "../saved_models/ensemble"
    ) -> Dict[str, Any]:
        """
        Create ensemble model from new evaluation data using Delta R² strategy

        Process:
        1. Stage1 inference on boundary data ’ base predictions
        2. Residual Boost inference on boundary data ’ residual predictions
        3. Calculate R² for each signal (Stage1 vs Ensemble)
        4. Select signals based on Delta R² threshold
        5. Save ensemble config

        Args:
            ensemble_name: Name for the ensemble
            stage1_model_info: Stage1 model information
            residual_boost_model_info: Residual Boost model information
            evaluation_data: Evaluation data array with boundary + target columns
            boundary_signals: List of boundary signal names
            target_signals: List of target signal names
            delta_r2_threshold: Delta R² threshold for signal selection
            device: PyTorch device
            save_config: Whether to save config file
            save_dir: Directory to save config

        Returns:
            ensemble_info: Dictionary with ensemble configuration and metrics
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print("=" * 80)
        print("<¯ Creating Ensemble Model (Delta R² Strategy)")
        print("=" * 80)

        # Extract data
        num_boundary = len(boundary_signals)
        num_target = len(target_signals)

        X_eval = evaluation_data[:, :num_boundary]  # Boundary signals
        y_true = evaluation_data[:, num_boundary:num_boundary + num_target]  # Target signals

        print(f"\n=Ê Data shapes:")
        print(f"   Boundary data (X): {X_eval.shape}")
        print(f"   Target data (y_true): {y_true.shape}")

        # Stage1 prediction
        print(f"\n=. Running Stage1 inference...")
        y_pred_stage1 = batch_inference(
            stage1_model_info['model'],
            X_eval,
            stage1_model_info['scalers']['X'],
            stage1_model_info['scalers']['y'],
            device,
            batch_size=512,
            model_name="Stage1"
        )

        # Residual Boost prediction
        print(f"\n=€ Running Residual Boost inference...")
        y_residual_pred = batch_inference(
            residual_boost_model_info['model'],
            X_eval,
            residual_boost_model_info['scalers']['X'],
            residual_boost_model_info['scalers']['y'],
            device,
            batch_size=512,
            model_name="Residual Boost"
        )

        # Calculate ensemble predictions
        y_pred_ensemble = y_pred_stage1 + y_residual_pred

        # Compute Delta R² for each signal
        signal_analysis = EnsembleBuilder._compute_delta_r2(
            y_true,
            y_pred_stage1,
            y_pred_ensemble,
            target_signals,
            delta_r2_threshold
        )

        # Count signals
        num_use_boost = sum(1 for item in signal_analysis if item['use_boost'])
        num_use_stage1_only = len(target_signals) - num_use_boost

        # Print signal analysis
        print(f"\n<¯ Signal Delta R² Analysis:")
        print(f"{'Signal name':<40} {'Stage1 R²':>12} {'Ensemble R²':>12} {'Delta R²':>12} {'Selection':>15}")
        print("-" * 95)

        for item in signal_analysis:
            choice = "Stage1+Boost" if item['use_boost'] else "Stage1 Only"
            print(f"{item['signal']:<40} {item['r2_stage1']:>12.4f} {item['r2_ensemble']:>12.4f} "
                  f"{item['delta_r2']:>12.4f} {choice:>15}")

        print("-" * 95)
        print(f" Using Stage1 + Residual Boost: {num_use_boost} signals")
        print(f" Using Stage1 only: {num_use_stage1_only} signals")

        # Generate final ensemble predictions
        y_ensemble_final = y_pred_stage1.copy()
        for i, item in enumerate(signal_analysis):
            if item['use_boost']:
                y_ensemble_final[:, i] = y_pred_stage1[:, i] + y_residual_pred[:, i]

        # Calculate overall metrics
        metrics = EnsembleBuilder._compute_overall_metrics(
            y_true, y_pred_stage1, y_ensemble_final
        )

        print(f"\n=È Overall Performance:")
        print(f"{'Metric':<15} {'Stage1':>15} {'Ensemble':>15} {'Improvement':>15}")
        print("-" * 65)
        print(f"{'MAE':<15} {metrics['stage1']['mae']:>15.6f} {metrics['ensemble']['mae']:>15.6f} {metrics['improvement']['mae_pct']:>14.2f}%")
        print(f"{'RMSE':<15} {metrics['stage1']['rmse']:>15.6f} {metrics['ensemble']['rmse']:>15.6f} {metrics['improvement']['rmse_pct']:>14.2f}%")
        print(f"{'R²':<15} {metrics['stage1']['r2']:>15.4f} {metrics['ensemble']['r2']:>15.4f} {metrics['improvement']['r2_pct']:>14.2f}%")

        # Build ensemble info
        ensemble_info = {
            'ensemble_name': ensemble_name,
            'stage1_model_name': stage1_model_info.get('model_name', 'unknown'),
            'residual_boost_model_name': residual_boost_model_info.get('model_name', 'unknown'),
            'delta_r2_threshold': float(delta_r2_threshold),
            'signal_analysis': signal_analysis,
            'num_use_boost': int(num_use_boost),
            'num_use_stage1_only': int(num_use_stage1_only),
            'metrics': metrics,
            'signals': {
                'boundary': boundary_signals,
                'target': target_signals
            },
            'predictions': {
                'y_true': y_true,
                'y_pred_stage1': y_pred_stage1,
                'y_pred_ensemble': y_ensemble_final,
                'y_residual_pred': y_residual_pred
            },
            'created_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # Save config
        if save_config:
            config_path = EnsembleBuilder._save_ensemble_config(
                ensemble_name, ensemble_info, save_dir
            )
            ensemble_info['config_path'] = config_path

        print(f"\n Ensemble model created: {ensemble_name}")
        return ensemble_info

    @staticmethod
    def update_ensemble_threshold(
        ensemble_name: str,
        ensemble_info: Dict[str, Any],
        new_threshold: float,
        save_config: bool = True,
        save_dir: str = "../saved_models/ensemble"
    ) -> Dict[str, Any]:
        """
        Update ensemble Delta R² threshold and regenerate signal selection

        Args:
            ensemble_name: Ensemble name
            ensemble_info: Existing ensemble info (must contain predictions)
            new_threshold: New Delta R² threshold
            save_config: Whether to save updated config
            save_dir: Directory to save config

        Returns:
            Updated ensemble_info
        """
        print(f"\n= Updating ensemble threshold: {ensemble_info['delta_r2_threshold']:.3f} ’ {new_threshold:.3f}")

        # Get stored predictions
        y_true = ensemble_info['predictions']['y_true']
        y_pred_stage1 = ensemble_info['predictions']['y_pred_stage1']
        y_residual_pred = ensemble_info['predictions']['y_residual_pred']
        y_pred_ensemble = y_pred_stage1 + y_residual_pred

        target_signals = ensemble_info['signals']['target']

        # Recompute signal analysis with new threshold
        signal_analysis = EnsembleBuilder._compute_delta_r2(
            y_true,
            y_pred_stage1,
            y_pred_ensemble,
            target_signals,
            new_threshold
        )

        num_use_boost = sum(1 for item in signal_analysis if item['use_boost'])
        num_use_stage1_only = len(target_signals) - num_use_boost

        # Generate new ensemble predictions
        y_ensemble_final = y_pred_stage1.copy()
        for i, item in enumerate(signal_analysis):
            if item['use_boost']:
                y_ensemble_final[:, i] = y_pred_stage1[:, i] + y_residual_pred[:, i]

        # Recompute metrics
        metrics = EnsembleBuilder._compute_overall_metrics(
            y_true, y_pred_stage1, y_ensemble_final
        )

        # Update ensemble info
        ensemble_info['delta_r2_threshold'] = float(new_threshold)
        ensemble_info['signal_analysis'] = signal_analysis
        ensemble_info['num_use_boost'] = int(num_use_boost)
        ensemble_info['num_use_stage1_only'] = int(num_use_stage1_only)
        ensemble_info['metrics'] = metrics
        ensemble_info['predictions']['y_pred_ensemble'] = y_ensemble_final
        ensemble_info['updated_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        print(f" Updated: {num_use_boost} signals use Residual Boost, {num_use_stage1_only} use Stage1 only")

        # Save updated config
        if save_config:
            config_path = EnsembleBuilder._save_ensemble_config(
                ensemble_name, ensemble_info, save_dir
            )
            ensemble_info['config_path'] = config_path

        return ensemble_info

    @staticmethod
    def _compute_delta_r2(
        y_true: np.ndarray,
        y_pred_stage1: np.ndarray,
        y_pred_ensemble: np.ndarray,
        signal_names: List[str],
        delta_r2_threshold: float
    ) -> List[Dict]:
        """Compute Delta R² for each signal"""
        signal_analysis = []

        for i, signal in enumerate(signal_names):
            y_true_sig = y_true[:, i].reshape(-1, 1)
            y_pred_stage1_sig = y_pred_stage1[:, i].reshape(-1, 1)
            y_pred_ensemble_sig = y_pred_ensemble[:, i].reshape(-1, 1)

            # Calculate R²
            r2_stage1, _ = compute_r2_safe(y_true_sig, y_pred_stage1_sig, method='per_output_mean')
            r2_ensemble, _ = compute_r2_safe(y_true_sig, y_pred_ensemble_sig, method='per_output_mean')

            # Calculate Delta R²
            delta_r2 = r2_ensemble - r2_stage1

            # Determine whether to use Residual Boost
            use_boost = delta_r2 > delta_r2_threshold

            signal_analysis.append({
                'signal': signal,
                'r2_stage1': float(r2_stage1),
                'r2_ensemble': float(r2_ensemble),
                'delta_r2': float(delta_r2),
                'use_boost': bool(use_boost)
            })

        return signal_analysis

    @staticmethod
    def _compute_overall_metrics(
        y_true: np.ndarray,
        y_pred_stage1: np.ndarray,
        y_pred_ensemble: np.ndarray
    ) -> Dict[str, Any]:
        """Compute overall performance metrics"""
        # Stage1 metrics
        mae_stage1 = mean_absolute_error(y_true, y_pred_stage1)
        rmse_stage1 = np.sqrt(mean_squared_error(y_true, y_pred_stage1))
        r2_stage1, _ = compute_r2_safe(y_true, y_pred_stage1, method='per_output_mean')

        # Ensemble metrics
        mae_ensemble = mean_absolute_error(y_true, y_pred_ensemble)
        rmse_ensemble = np.sqrt(mean_squared_error(y_true, y_pred_ensemble))
        r2_ensemble, _ = compute_r2_safe(y_true, y_pred_ensemble, method='per_output_mean')

        # Improvements
        improvement_mae = (mae_stage1 - mae_ensemble) / mae_stage1 * 100 if mae_stage1 > 0 else 0
        improvement_rmse = (rmse_stage1 - rmse_ensemble) / rmse_stage1 * 100 if rmse_stage1 > 0 else 0
        improvement_r2 = (r2_ensemble - r2_stage1) / (1 - r2_stage1) * 100 if r2_stage1 < 1 else 0

        return {
            'stage1': {
                'mae': float(mae_stage1),
                'rmse': float(rmse_stage1),
                'r2': float(r2_stage1)
            },
            'ensemble': {
                'mae': float(mae_ensemble),
                'rmse': float(rmse_ensemble),
                'r2': float(r2_ensemble)
            },
            'improvement': {
                'mae_pct': float(improvement_mae),
                'rmse_pct': float(improvement_rmse),
                'r2_pct': float(improvement_r2)
            }
        }

    @staticmethod
    def _save_ensemble_config(
        ensemble_name: str,
        ensemble_info: Dict[str, Any],
        save_dir: str
    ) -> str:
        """Save ensemble configuration to JSON"""
        os.makedirs(save_dir, exist_ok=True)

        config_path = os.path.join(save_dir, f"{ensemble_name}_config.json")

        # Prepare config (exclude large arrays)
        save_config = {
            'ensemble_name': ensemble_info['ensemble_name'],
            'stage1_model_name': ensemble_info['stage1_model_name'],
            'residual_boost_model_name': ensemble_info['residual_boost_model_name'],
            'delta_r2_threshold': ensemble_info['delta_r2_threshold'],
            'signal_analysis': ensemble_info['signal_analysis'],
            'num_use_boost': ensemble_info['num_use_boost'],
            'num_use_stage1_only': ensemble_info['num_use_stage1_only'],
            'metrics': ensemble_info['metrics'],
            'signals': ensemble_info['signals'],
            'created_time': ensemble_info['created_time'],
            'updated_time': ensemble_info.get('updated_time', ensemble_info['created_time'])
        }

        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(save_config, f, indent=2, ensure_ascii=False)

        print(f"=¾ Ensemble config saved: {config_path}")
        return config_path
