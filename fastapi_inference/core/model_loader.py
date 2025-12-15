"""
Model Loading and Management
"""
import os
import sys
import json
import pickle
import torch
from typing import Dict, Any, Tuple
from datetime import datetime

# Add parent directory to path to import models
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from models.static_transformer import StaticSensorTransformer
    from models.residual_tft import GroupedMultiTargetTFT
except ImportError:
    print("   Warning: Could not import models from models/ directory")
    print("   Make sure you run this from the repo root")


class ModelLoader:
    """Model loading and management"""

    @staticmethod
    def load_stage1_model(
        inference_config_path: str,
        device: torch.device
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Load Stage1 SST model from inference config

        Args:
            inference_config_path: Path to inference config JSON
            device: PyTorch device

        Returns:
            Tuple of (model_name, model_info dict)
        """
        try:
            # Read config
            if not os.path.exists(inference_config_path):
                raise FileNotFoundError(f"Config file not found: {inference_config_path}")

            with open(inference_config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            model_name = config.get('model_name', os.path.basename(inference_config_path).replace('_inference.json', ''))
            model_path = config['model_path']
            scaler_path = config['scaler_path']

            # Check if files exist
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

            # Load model checkpoint
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            model_config = checkpoint['model_config']

            # Verify model type
            if model_config['type'] != 'StaticSensorTransformer':
                raise ValueError(f"Expected Stage1 model type 'StaticSensorTransformer', got '{model_config['type']}'")

            # Rebuild model
            model = StaticSensorTransformer(
                num_boundary_sensors=len(model_config['boundary_signals']),
                num_target_sensors=len(model_config['target_signals']),
                d_model=model_config['config']['d_model'],
                nhead=model_config['config']['nhead'],
                num_layers=model_config['config']['num_layers'],
                dropout=model_config['config']['dropout']
            )

            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            model.eval()

            # Load scalers
            with open(scaler_path, 'rb') as f:
                scalers = pickle.load(f)

            # Build model info
            model_info = {
                'model': model,
                'scalers': scalers,
                'config': model_config['config'],
                'boundary_signals': model_config['boundary_signals'],
                'target_signals': model_config['target_signals'],
                'model_type': 'stage1',
                'model_path': model_path,
                'scaler_path': scaler_path,
                'config_path': inference_config_path,
                'loaded_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            print(f" Stage1 model loaded: {model_name}")
            print(f"   Boundary signals: {len(model_info['boundary_signals'])}")
            print(f"   Target signals: {len(model_info['target_signals'])}")

            return model_name, model_info

        except Exception as e:
            print(f"L Failed to load Stage1 model: {e}")
            raise

    @staticmethod
    def load_residual_boost_model(
        inference_config_path: str,
        device: torch.device
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Load Residual Boost (TFT) model from inference config

        Args:
            inference_config_path: Path to inference config JSON
            device: PyTorch device

        Returns:
            Tuple of (model_name, model_info dict)
        """
        try:
            # Read config
            if not os.path.exists(inference_config_path):
                raise FileNotFoundError(f"Config file not found: {inference_config_path}")

            with open(inference_config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            model_name = config.get('model_name', os.path.basename(inference_config_path).replace('_inference.json', ''))
            model_path = config['model_path']
            scaler_path = config['scaler_path']

            # Check if files exist
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

            # Load model checkpoint
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            model_config = checkpoint['model_config']

            # Rebuild TFT model
            tft_model = GroupedMultiTargetTFT(
                num_targets=model_config['num_targets'],
                num_external_factors=model_config['num_external_factors'],
                d_model=config['architecture']['d_model'],
                nhead=config['architecture']['nhead'],
                num_encoder_layers=config['architecture']['num_encoder_layers'],
                num_decoder_layers=config['architecture']['num_decoder_layers'],
                dropout=config['architecture']['dropout'],
                use_grouping=config['architecture'].get('use_grouping', False),
                signal_groups=config['architecture'].get('signal_groups', None)
            )

            tft_model.load_state_dict(checkpoint['model_state_dict'])
            tft_model.to(device)
            tft_model.eval()

            # Load scalers
            with open(scaler_path, 'rb') as f:
                scalers = pickle.load(f)

            # Build model info
            model_info = {
                'model': tft_model,
                'scalers': scalers,
                'config': config['architecture'],
                'boundary_signals': config['signals']['boundary_signals'],
                'target_signals': config['signals']['target_signals'],
                'residual_signals': config['signals']['residual_signals'],
                'model_type': 'residual_boost',
                'base_model_name': config['data_config']['base_model_name'],
                'encoder_length': config['data_config']['encoder_length'],
                'future_horizon': config['data_config']['future_horizon'],
                'model_path': model_path,
                'scaler_path': scaler_path,
                'config_path': inference_config_path,
                'loaded_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            print(f" Residual Boost model loaded: {model_name}")
            print(f"   Base model: {model_info['base_model_name']}")
            print(f"   Boundary signals: {len(model_info['boundary_signals'])}")
            print(f"   Target signals: {len(model_info['target_signals'])}")

            return model_name, model_info

        except Exception as e:
            print(f"L Failed to load Residual Boost model: {e}")
            raise
