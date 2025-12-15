"""
Prediction Engine for Ensemble Inference
"""
import os
import sys
import numpy as np
import torch
from typing import Dict, Any, List, Optional
from datetime import datetime

# Add parent directory to import models
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from models.residual_tft import batch_inference
except ImportError:
    print("   Warning: Could not import from models.residual_tft")

from ..utils.file_handler import save_predictions_csv, generate_output_filename


class Predictor:
    """Batch prediction engine using ensemble models"""

    @staticmethod
    def batch_predict(
        ensemble_info: Dict[str, Any],
        stage1_model_info: Dict[str, Any],
        residual_boost_model_info: Dict[str, Any],
        input_data: np.ndarray,  # Shape: (N, num_boundary)
        output_dir: str,
        manual_boost_signals: Optional[Dict[str, bool]] = None,
        device: torch.device = None,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Batch prediction using ensemble model

        Process:
        1. Load boundary data
        2. Stage1 inference ’ base predictions
        3. Residual Boost inference ’ residual predictions
        4. Apply ensemble configuration (or manual override)
        5. Save results to CSV

        Args:
            ensemble_info: Ensemble configuration
            stage1_model_info: Stage1 model information
            residual_boost_model_info: Residual Boost model information
            input_data: Input boundary data (N_samples, N_boundary)
            output_dir: Output directory for results
            manual_boost_signals: Optional manual override {'signal_name': use_boost}
            device: PyTorch device
            include_metadata: Whether to save metadata file

        Returns:
            result_info: Dictionary with prediction results and metadata
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print("=" * 80)
        print("=€ Running Batch Inference")
        print("=" * 80)

        ensemble_name = ensemble_info['ensemble_name']
        target_signals = ensemble_info['signals']['target']
        signal_analysis = ensemble_info['signal_analysis']

        print(f"\n=Ê Input data shape: {input_data.shape}")
        print(f"   Ensemble: {ensemble_name}")
        print(f"   Number of samples: {input_data.shape[0]}")
        print(f"   Number of target signals: {len(target_signals)}")

        # Stage1 prediction
        print(f"\n=. Running Stage1 inference...")
        y_pred_stage1 = batch_inference(
            stage1_model_info['model'],
            input_data,
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
            input_data,
            residual_boost_model_info['scalers']['X'],
            residual_boost_model_info['scalers']['y'],
            device,
            batch_size=512,
            model_name="Residual Boost"
        )

        # Determine which signals use Residual Boost
        signals_used_boost = []
        y_final = y_pred_stage1.copy()

        if manual_boost_signals is not None:
            # Use manual override
            print(f"\n™  Using manual signal selection...")
            for i, signal in enumerate(target_signals):
                if manual_boost_signals.get(signal, False):
                    y_final[:, i] = y_pred_stage1[:, i] + y_residual_pred[:, i]
                    signals_used_boost.append(signal)
                    print(f"    {signal}: Stage1 + Residual Boost")
                else:
                    print(f"   ª {signal}: Stage1 only")
        else:
            # Use ensemble configuration
            print(f"\n™  Using ensemble configuration (Delta R² threshold = {ensemble_info['delta_r2_threshold']:.3f})...")
            for i, item in enumerate(signal_analysis):
                if item['use_boost']:
                    y_final[:, i] = y_pred_stage1[:, i] + y_residual_pred[:, i]
                    signals_used_boost.append(item['signal'])
                    print(f"    {item['signal']}: Stage1 + Residual Boost (”R²={item['delta_r2']:.4f})")
                else:
                    print(f"   ª {item['signal']}: Stage1 only (”R²={item['delta_r2']:.4f})")

        print(f"\n=È Summary: {len(signals_used_boost)}/{len(target_signals)} signals use Residual Boost")

        # Generate output filename
        output_path = generate_output_filename(
            ensemble_name, output_dir, prefix="predictions"
        )

        # Prepare metadata
        metadata = None
        if include_metadata:
            metadata = {
                'ensemble_name': ensemble_name,
                'stage1_model': stage1_model_info.get('model_name', 'unknown'),
                'residual_boost_model': residual_boost_model_info.get('model_name', 'unknown'),
                'delta_r2_threshold': ensemble_info['delta_r2_threshold'],
                'num_samples': input_data.shape[0],
                'num_signals': len(target_signals),
                'signals_used_boost': ', '.join(signals_used_boost),
                'num_signals_used_boost': len(signals_used_boost),
                'prediction_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

        # Save predictions
        save_predictions_csv(
            y_final,
            target_signals,
            output_path,
            metadata=metadata
        )

        # Build result info
        result_info = {
            'ensemble_name': ensemble_name,
            'output_path': output_path,
            'num_samples': int(input_data.shape[0]),
            'num_signals': len(target_signals),
            'signals_used_boost': signals_used_boost,
            'num_signals_used_boost': len(signals_used_boost),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'predictions': y_final.tolist() if y_final.shape[0] <= 100 else None  # Include if small
        }

        print(f"\n Batch inference completed!")
        print(f"   Output: {output_path}")
        return result_info

    @staticmethod
    def predict_with_ensemble(
        X: np.ndarray,
        stage1_model_info: Dict[str, Any],
        residual_boost_model_info: Dict[str, Any],
        boost_signals_mask: List[bool],
        device: torch.device
    ) -> np.ndarray:
        """
        Perform ensemble prediction with signal-level boost control

        Args:
            X: Input boundary data
            stage1_model_info: Stage1 model info
            residual_boost_model_info: Residual Boost model info
            boost_signals_mask: Boolean mask for which signals use boost
            device: PyTorch device

        Returns:
            predictions: Final ensemble predictions
        """
        # Stage1 prediction
        y_pred_stage1 = batch_inference(
            stage1_model_info['model'],
            X,
            stage1_model_info['scalers']['X'],
            stage1_model_info['scalers']['y'],
            device,
            batch_size=512,
            model_name="Stage1"
        )

        # Residual Boost prediction
        y_residual_pred = batch_inference(
            residual_boost_model_info['model'],
            X,
            residual_boost_model_info['scalers']['X'],
            residual_boost_model_info['scalers']['y'],
            device,
            batch_size=512,
            model_name="Residual Boost"
        )

        # Apply boost mask
        y_final = y_pred_stage1.copy()
        for i, use_boost in enumerate(boost_signals_mask):
            if use_boost:
                y_final[:, i] = y_pred_stage1[:, i] + y_residual_pred[:, i]

        return y_final
