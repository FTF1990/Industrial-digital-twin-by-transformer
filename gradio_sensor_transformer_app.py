#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Gradio Interface for Industrial Digital Twin with Residual Boost Training
Complete Residual Boost Training System - Enhanced Gradio Interface
Features:
1. Stage1 (SST) model training - Static Sensor Transformer for baseline predictions
2. Residual extraction - Extract prediction errors from Stage1 model
3. Stage2 residual model training - Train on residuals to correct Stage1 errors
4. Intelligent R² threshold selection - Automatically decide which signals need Stage2
5. Ensemble model generation - Optimal combination of Stage1 and Stage2 predictions
6. Comprehensive visualization - Individual prediction vs actual comparison for all signals
7. CSV export - Save detailed inference results with per-signal R² scores
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, List, Tuple, Any, Optional
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
import warnings
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import matplotlib
import platform
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================================
# TFT model save function (added at top of file)

def save_tft_model_with_config(
        model_name: str,
        tft_model: nn.Module,
        config: Dict[str, Any],
        scalers: Dict[str, StandardScaler],
        residual_data_key: str,
        residual_info: Dict[str, Any],
        history: Dict[str, List[float]]
) -> Tuple[str, str, str]:
    """
    Save TFT model, config and scalers

    Args:
        model_name: TFTModel name
        tft_model: Trained TFT model
        config: Training config
        scalers: Data scalers
        residual_data_key: Residual data key
        residual_info: Residual data info
        history: Training history

    Returns:
        model_path: Model weight file path
        scaler_path: ScalerFile path
        inference_config_path: Inference configFile path
    """
    model_dir = "saved_models/tft_models"
    os.makedirs(model_dir, exist_ok=True)

    # 1. Save model weights
    model_path = os.path.join(model_dir, f"{model_name}.pth")
    torch.save({
        'model_state_dict': tft_model.state_dict(),
        'model_config': {
            'num_targets': tft_model.num_targets,
            'num_external_factors': tft_model.num_external_factors,
            'd_model': tft_model.d_model,
            'use_grouping': tft_model.use_grouping,
            'signal_groups': tft_model.signal_groups if hasattr(tft_model, 'signal_groups') else None
        },
        'training_config': config,
        'training_history': history,
        'residual_data_key': residual_data_key,
        'created_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }, model_path)

    # 2. Save scalers
    scaler_path = os.path.join(model_dir, f"{model_name}_scalers.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scalers, f)

    # 3. Save inference config JSON (most important)
    inference_config_path = os.path.join(model_dir, f"{model_name}_inference.json")
    inference_config = {
        'model_name': model_name,
        'model_type': 'ResidualTFT',
        'model_path': model_path,
        'scaler_path': scaler_path,

        # TFT model architecture
        'architecture': {
            'd_model': config['d_model'],
            'nhead': config['nhead'],
            'num_encoder_layers': config['num_encoder_layers'],
            'num_decoder_layers': config['num_decoder_layers'],
            'dropout': config['dropout'],
            'use_grouping': config.get('use_grouping', False),
            'signal_groups': config.get('signal_groups', None)
        },

        # Data config
        'data_config': {
            'encoder_length': config['encoder_length'],
            'future_horizon': residual_info['future_horizon'],
            'residual_data_key': residual_data_key,
            'base_model_name': residual_info['base_model_name'],
            'num_targets': len(residual_info['target_signals']),
            'num_external_factors': len(residual_info['boundary_signals'])
        },

        # Signal info
        'signals': {
            'boundary_signals': residual_info['boundary_signals'],
            'target_signals': residual_info['target_signals'],
            'residual_signals': residual_info['residual_signals']
        },

        # Training info
        'training_info': {
            'epochs_trained': len(history['train_losses']),
            'best_val_loss': min(history['val_losses']),
            'batch_size': config['batch_size'],
            'learning_rate': config['lr']
        },

        'created_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    with open(inference_config_path, 'w', encoding='utf-8') as f:
        json.dump(inference_config, f, indent=2, ensure_ascii=False)

    print(f"✅ TFTModel saved:")
    print(f"   📦 Model weights: {model_path}")
    print(f"   📊 Scalers: {scaler_path}")
    print(f"   📄 Inference config: {inference_config_path}")

    return model_path, scaler_path, inference_config_path


# ============================================================================
# TFT model load function

def load_tft_model_from_config(config_file_path: str, device: torch.device) -> Tuple[str, str]:
    """
    Load TFT model from inference config file

    Args:
        config_file_path: Inference configJSONFile path
        device: PyTorch device

    Returns:
        model_name: Model name
        status_msg: Load status message
    """
    try:
        # Read config
        with open(config_file_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        model_name = config['model_name']
        model_path = config['model_path']
        scaler_path = config['scaler_path']

        # Check if files exist
        if not os.path.exists(model_path):
            return None, f"❌ Model file does not exist: {model_path}"
        if not os.path.exists(scaler_path):
            return None, f"❌ Scaler file does not exist: {scaler_path}"

        # Load model
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

        # Save to global state
        global_state['residual_models'][model_name] = {
            'model': tft_model,
            'config': config['architecture'],
            'history': checkpoint.get('training_history', {'train_losses': [], 'val_losses': []}),
            'residual_data_key': config['data_config']['residual_data_key'],
            'info': {
                'base_model_name': config['data_config']['base_model_name'],
                'target_signals': config['signals']['target_signals'],
                'boundary_signals': config['signals']['boundary_signals'],
                'residual_signals': config['signals']['residual_signals'],
                'model_type': 'StaticSensorTransformer',  # Inherited from base model
                'future_horizon': config['data_config']['future_horizon']
            },
            'encoder_length': config['data_config']['encoder_length'],
            'future_horizon': config['data_config']['future_horizon']
        }

        global_state['residual_scalers'][model_name] = scalers

        # BuildStatus message
        status_msg = f"✅ TFT model loaded successfully!\n\n"
        status_msg += f"📌 Model name: {model_name}\n"
        status_msg += f"📊 Base model: {config['data_config']['base_model_name']}\n"
        status_msg += f"🎯 Number of target signals: {config['data_config']['num_targets']}\n"
        status_msg += f"📈 Number of boundary signals: {config['data_config']['num_external_factors']}\n"
        status_msg += f"📏 Historical window length: {config['data_config']['encoder_length']}\n"
        status_msg += f"🔮 Future prediction horizon: {config['data_config']['future_horizon']}\n"
        status_msg += f"⚙️ Model dimensions: {config['architecture']['d_model']}\n"
        status_msg += f"🕒 Created at: {config['created_time']}\n"

        if 'training_info' in config:
            ti = config['training_info']
            status_msg += f"\n📚 Training info:\n"
            status_msg += f"   - Training epochs: {ti['epochs_trained']}\n"
            status_msg += f"   - Best validation loss: {ti['best_val_loss']:.6f}\n"
            status_msg += f"   - Batch size: {ti['batch_size']}\n"
            status_msg += f"   - Learning rate: {ti['learning_rate']}\n"

        print(status_msg)
        return model_name, status_msg

    except Exception as e:
        error_msg = f"❌ TFT model loading failed:\n{str(e)}\n\n{traceback.format_exc()}"
        print(error_msg)
        return None, error_msg


# ============================================================================
# Configure Chinese font

def configure_chinese_font():
    """Configure matplotlib for Chinese font display"""
    import matplotlib
    import platform

    system = platform.system()
    if system == 'Darwin':  # macOS
        matplotlib.rc('font', family='Arial Unicode MS')
    elif system == 'Windows':
        matplotlib.rc('font', family='SimHei')
    else:  # Linux
        matplotlib.rc('font', family='DejaVu Sans')

    matplotlib.rcParams['axes.unicode_minus'] = False
    sns.set_style("whitegrid")


# ============================================================================
# Import modules

try:
    import gradio as gr

    print("✅ Gradio import successful")
except ImportError:
    print("❌ Please install gradio: pip install gradio")
    sys.exit(1)

# Trying to import local modules
try:
    from models.static_transformer import StaticSensorTransformer
    from models.residual_tft import (
        GroupedMultiTargetTFT,
        ResidualExtractor,
        train_residual_tft,
        prepare_residual_sequence_data,
        compute_r2_safe,
        compute_residuals_correctly,
        batch_inference,
        inference_with_boosting,
        compute_per_signal_metrics,
        clear_gpu_memory,
        print_gpu_memory
    )
    from models.utils import apply_ifd_smoothing

    print("✅ Local modules imported successfully")
except ImportError as e:
    print(f"⚠️ Local modules import failed: {e}")
    print("Trying to use relative imports...")

    try:
        from static_transformer import StaticSensorTransformer
        from residual_tft import (
            GroupedMultiTargetTFT,
            ResidualExtractor,
            train_residual_tft,
            prepare_residual_sequence_data,
            compute_r2_safe,
            compute_residuals_correctly,
            batch_inference,
            inference_with_boosting,
            compute_per_signal_metrics,
            clear_gpu_memory,
            print_gpu_memory
        )
        from utils import apply_ifd_smoothing

        print("✅ Relative import successful")
    except ImportError as e2:
        print(f"❌ Relative import also failed: {e2}")
        print("Will use inline definitions...")


# Setup device with enhanced GPU detection
def setup_device():
    """Setup computing device with GPU detection and configuration"""
    configure_chinese_font()
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"GPU detected successfully: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        return device
    else:
        device = torch.device('cpu')
        print("GPU not available, using CPU training")
        return device


device = setup_device()


def load_saved_models():
    """Load saved models from file system"""
    model_dir = "saved_models"
    if not os.path.exists(model_dir):
        return

    print(f"Loading saved models from {model_dir}...")

    for filename in os.listdir(model_dir):
        if filename.endswith('.pth') and not filename.endswith('_scalers.pkl'):
            model_name = filename[:-4]
            model_path = os.path.join(model_dir, filename)
            scaler_path = os.path.join(model_dir, f"{model_name}_scalers.pkl")

            try:
                checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                model_config = checkpoint['model_config']

                if model_config['type'] == 'StaticSensorTransformer':
                    model = StaticSensorTransformer(
                        num_boundary_sensors=len(model_config['boundary_signals']),
                        num_target_sensors=len(model_config['target_signals']),
                        d_model=model_config['config']['d_model'],
                        nhead=model_config['config']['nhead'],
                        num_layers=model_config['config']['num_layers'],
                        dropout=model_config['config']['dropout']
                    )
                else:
                    continue

                model.load_state_dict(checkpoint['model_state_dict'])
                model.to(device)
                model.eval()

                scalers = {}
                if os.path.exists(scaler_path):
                    with open(scaler_path, 'rb') as f:
                        scalers = pickle.load(f)

                global_state['trained_models'][model_name] = {
                    'model': model,
                    'type': model_config['type'],
                    'boundary_signals': model_config['boundary_signals'],
                    'target_signals': model_config['target_signals'],
                    'config': model_config['config'],
                    'model_path': model_path,
                    'scaler_path': scaler_path
                }

                global_state['scalers'][model_name] = scalers
                print(f"  Loading model: {model_name}")

            except Exception as e:
                print(f"  Model loading failed {model_name}: {e}")
                continue

    print(f"Model loading complete, loaded {len(global_state['trained_models'])} models")


# Global state management
global_state = {
    'df': None,
    'trained_models': {},
    'scalers': {},
    'residual_data': {},
    'residual_models': {},
    'residual_scalers': {},
    'final_predictions': {},
    'training_logs': {},
    'model_configs': {},
    'all_signals': [],
    # New: Stage2 Boost model storage
    'stage2_models': {},  # Stage2 residual model
    'stage2_scalers': {},  # Stage2 Scalers
    'ensemble_models': {},  # Ensemble inference model (SST + Stage2)
    'sundial_models': {},  # Sundial time series prediction model
    # Training control flags
    'stop_training_tab2': False,  # Flag to stop Tab2 training
    'stop_training_tab4': False,  # Flag to stop Tab4 training
}


# ============================================================================
# Colab Auto-load Support
# ============================================================================
def autoload_colab_data():
    """
    Automatically load pre-defined data from Colab environment

    This function checks for pre-saved CSV files and automatically loads them
    into global_state, making them immediately available in Tab1.

    Supports:
    - Standard predefined paths
    - Environment variable: COLAB_DATA_PATH
    - Wildcard matching in data/ folder
    - Google Drive mounted paths
    """
    import glob

    # Priority 1: Environment variable
    env_path = os.environ.get('COLAB_DATA_PATH')
    if env_path and os.path.exists(env_path):
        preload_paths = [env_path]
    else:
        # Priority 2: Standard predefined paths
        preload_paths = [
            'data/colab_preloaded_data.csv',
            'data/test_data.csv',
            '/content/colab_data.csv',
            # Add more common names
            'data/leap_data.csv',
            'data/sensor_data.csv',
            'data/training_data.csv',
            # Google Drive paths
            '/content/drive/MyDrive/data.csv',
            '/content/drive/MyDrive/colab_data.csv',
        ]

        # Priority 3: Wildcard search in data/ folder
        if os.path.exists('data'):
            csv_files = glob.glob('data/*.csv')
            if csv_files:
                # Add all CSV files in data/ folder
                preload_paths.extend(csv_files)

    for preload_path in preload_paths:
        if os.path.exists(preload_path):
            try:
                df_auto = pd.read_csv(preload_path)

                # Validate: must have at least 2 columns
                if df_auto.shape[1] < 2:
                    print(f"⚠️ [Colab Auto-load] Skipping {preload_path}: too few columns")
                    continue

                # Validate: must have at least 10 rows
                if df_auto.shape[0] < 10:
                    print(f"⚠️ [Colab Auto-load] Skipping {preload_path}: too few rows")
                    continue

                global_state['df'] = df_auto
                global_state['data_loaded'] = True

                print("=" * 80)
                print("✅✅✅ [Colab Auto-load] Data successfully loaded into Tab1! ✅✅✅")
                print(f"📊 Data shape: {df_auto.shape}")
                print(f"📋 Columns: {list(df_auto.columns)[:10]}")  # Show first 10 columns
                if df_auto.shape[1] > 10:
                    print(f"    ... and {df_auto.shape[1] - 10} more columns")
                print(f"📁 Source: {preload_path}")
                print("=" * 80)

                return df_auto
            except Exception as e:
                print(f"⚠️ [Colab Auto-load] Failed to load {preload_path}: {e}")
                continue

    return None

# Auto-load disabled - user can manually select files in Tab1
# To enable auto-load, uncomment the line below:
# autoload_colab_data()


load_saved_models()

plt.style.use('default')
sns.set_palette("husl")

print("=" * 80)
print("Industrial Digital Twin with Residual Boost - Enhanced Interface")
print("=" * 80)
print(f"Using device: {device}")
print(f"PyTorch version: {torch.__version__}")
print("=" * 80)


# ============================================================================
# Model loading and inference config management

def save_inference_config(model_name, model_type, model_path, scaler_path,
                          boundary_signals, target_signals, config_dict):
    """
    Save inference config file - for direct model loading for inference later

    Args:
        model_name: Model name
        model_type: Model type
        model_path: Model weight file path
        scaler_path: ScalerFile path
        boundary_signals: Boundary signal list
        target_signals: Target signal list
        config_dict: Model architecture config
    """
    inference_config = {
        'model_name': model_name,
        'model_type': model_type,
        'model_path': model_path,
        'scaler_path': scaler_path,
        'boundary_signals': boundary_signals,
        'target_signals': target_signals,
        'architecture': config_dict,
        'created_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    model_dir = os.path.dirname(model_path)
    config_path = os.path.join(model_dir, f"{model_name}_inference.json")

    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(inference_config, f, indent=2, ensure_ascii=False)

    print(f"✅ Inference configsaved: {config_path}")
    return config_path


def load_model_from_inference_config(config_file_path, device):
    """
    Load model from inference config file

    Args:
        config_file_path: Inference configJSONFile path
        device: PyTorch device

    Returns:
        model_name: Model name
        success_msg: Success message
    """
    try:
        with open(config_file_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        model_name = config['model_name']
        model_type = config['model_type']
        model_path = config['model_path']
        scaler_path = config['scaler_path']

        # Check if files exist
        if not os.path.exists(model_path):
            return None, f"❌ Model file does not exist: {model_path}"
        if not os.path.exists(scaler_path):
            return None, f"❌ Scaler file does not exist: {scaler_path}"

        # Load model
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        arch = config['architecture']

        if model_type == 'StaticSensorTransformer':
            model = StaticSensorTransformer(
                num_boundary_sensors=len(config['boundary_signals']),
                num_target_sensors=len(config['target_signals']),
                d_model=arch['d_model'],
                nhead=arch['nhead'],
                num_layers=arch['num_layers'],
                dropout=arch['dropout']
            )
        else:
            return None, f"❌ Unsupported model type: {model_type}"

        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

        # Load scalers
        with open(scaler_path, 'rb') as f:
            scalers = pickle.load(f)

        # Save to global state
        global_state['trained_models'][model_name] = {
            'model': model,
            'type': model_type,
            'boundary_signals': config['boundary_signals'],
            'target_signals': config['target_signals'],
            'config': arch,
            'model_path': model_path,
            'scaler_path': scaler_path
        }

        global_state['scalers'][model_name] = scalers

        success_msg = f"✅ Model loaded successfully!\n\n"
        success_msg += f"📌 Model name: {model_name}\n"
        success_msg += f"📊 Model type: {model_type}\n"
        success_msg += f"🎯 Number of boundary signals: {len(config['boundary_signals'])}\n"
        success_msg += f"📈 Number of target signals: {len(config['target_signals'])}\n"
        success_msg += f"⚙️ Model parameters: d_model={arch['d_model']}, nhead={arch['nhead']}, layers={arch['num_layers']}\n"
        success_msg += f"🕒 Created at: {config['created_time']}\n"

        print(success_msg)
        return model_name, success_msg

    except Exception as e:
        error_msg = f"❌ Model loading failed:\n{str(e)}\n\n{traceback.format_exc()}"
        print(error_msg)
        return None, error_msg


# ============================================================================
# Stage2 Boost model definition and training functions

def train_stage2_boost_model(
        residual_data_key: str,
        config: Dict[str, Any],
        progress=None
) -> Tuple[str, Dict[str, Any]]:
    """
    Train Stage2 Boost residual model

    Args:
        residual_data_key: Residual data key
        config: Training config
        progress: Gradio progress object for real-time updates

    Returns:
        status_msg: Training status message
        results: Training result dictionary
    """
    try:
        if residual_data_key not in global_state['residual_data']:
            return "❌ Residual data does not exist！", {}

        log_msg = []
        log_msg.append("=" * 80)
        log_msg.append("🚀 Starting training Stage2 Boost residual model")
        log_msg.append("=" * 80)

        # Get residual data
        residuals_df = global_state['residual_data'][residual_data_key]['data']
        residual_info = global_state['residual_data'][residual_data_key]['info']

        boundary_signals = residual_info['boundary_signals']
        target_signals = residual_info['target_signals']
        residual_signals = residual_info['residual_signals']

        # Use Stage1's data split ratios (CRITICAL: ensures same test set as Stage1)
        data_split_info = residual_info.get('data_split', {})
        test_size = data_split_info.get('test_size', config.get('test_size', 0.15))
        val_size = data_split_info.get('val_size', config.get('val_size', 0.15))

        if 'data_split' in residual_info:
            log_msg.append(f"\n✓ Using Stage1's data split ratios (test={test_size:.2f}, val={val_size:.2f})")
        else:
            log_msg.append(f"\n⚠️  Warning: Stage1 data split info not found, using config values")

        log_msg.append(f"\n📊 Data info:")
        log_msg.append(f"  Residual data: {residual_data_key}")
        log_msg.append(f"  Number of boundary signals: {len(boundary_signals)}")
        log_msg.append(f"  Number of target signals: {len(target_signals)}")
        log_msg.append(f"  Data length: {len(residuals_df)}")

        # Prepare training data
        X = residuals_df[boundary_signals].values
        y_residual = residuals_df[residual_signals].values

        # Data split (using Stage1's ratios)
        train_size = int(len(X) * (1 - test_size - val_size))
        val_size_samples = int(len(X) * val_size)

        X_train = X[:train_size]
        X_val = X[train_size:train_size + val_size_samples]
        X_test = X[train_size + val_size_samples:]

        y_train = y_residual[:train_size]
        y_val = y_residual[train_size:train_size + val_size_samples]
        y_test = y_residual[train_size + val_size_samples:]

        log_msg.append(f"\n🔀 Data split:")
        log_msg.append(f"  Training set: {len(X_train)} ({len(X_train) / len(X) * 100:.1f}%)")
        log_msg.append(f"  Validation set: {len(X_val)} ({len(X_val) / len(X) * 100:.1f}%)")
        log_msg.append(f"  Test set: {len(X_test)} ({len(X_test) / len(X) * 100:.1f}%)")

        # Data standardization
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_train_scaled = scaler_X.fit_transform(X_train)
        X_val_scaled = scaler_X.transform(X_val)
        X_test_scaled = scaler_X.transform(X_test)

        y_train_scaled = scaler_y.fit_transform(y_train)
        y_val_scaled = scaler_y.transform(y_val)
        y_test_scaled = scaler_y.transform(y_test)

        # Create DataLoader
        train_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_train_scaled),
            torch.FloatTensor(y_train_scaled)
        )
        val_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_val_scaled),
            torch.FloatTensor(y_val_scaled)
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False
        )

        # Initialize Stage2 model (using SST architecture)
        log_msg.append(f"\n🏗️ Initializing Stage2 residual model:")
        log_msg.append(f"  Architecture: StaticSensorTransformer")
        log_msg.append(f"  d_model: {config['d_model']}")
        log_msg.append(f"  nhead: {config['nhead']}")
        log_msg.append(f"  num_layers: {config['num_layers']}")

        stage2_model = StaticSensorTransformer(
            num_boundary_sensors=len(boundary_signals),
            num_target_sensors=len(target_signals),
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_layers=config['num_layers'],
            dropout=config['dropout']
        ).to(device)

        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(
            stage2_model.parameters(),
            lr=config['lr'],
            weight_decay=config.get('weight_decay', 1e-5)
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.get('scheduler_factor', 0.7),
            patience=config.get('scheduler_patience', 15)
        )
        log_msg.append(f"📊 Learning rate scheduler: ReduceLROnPlateau (factor={config.get('scheduler_factor', 0.7)}, patience={config.get('scheduler_patience', 15)})")

        criterion = nn.MSELoss()

        # Mixed precision training
        scaler = GradScaler()

        # Training loop
        log_msg.append(f"\n🎯 Starting training (mixed precision, total epochs: {config['epochs']})")

        history = {
            'train_losses': [],
            'val_losses': [],
            'train_r2': [],
            'val_r2': [],
            'train_mae': [],
            'val_mae': []
        }

        best_val_loss = float('inf')
        patience_counter = 0
        early_stop_patience = config.get('early_stop_patience', 25)

        for epoch in range(config['epochs']):
            # Training phase with mixed precision
            stage2_model.train()
            train_loss = 0.0
            train_preds = []
            train_targets = []

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                optimizer.zero_grad()

                # Mixed precision forward pass
                with autocast():
                    outputs = stage2_model(batch_X)
                    loss = criterion(outputs, batch_y)

                # Mixed precision backward pass
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)

                # Gradient clipping
                if config.get('grad_clip', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(stage2_model.parameters(), config['grad_clip'])

                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()
                train_preds.append(outputs.detach().cpu().numpy())
                train_targets.append(batch_y.detach().cpu().numpy())

            train_loss /= len(train_loader)
            train_preds = np.vstack(train_preds)
            train_targets = np.vstack(train_targets)
            train_r2, _ = compute_r2_safe(train_targets, train_preds, method='per_output_mean')
            train_mae = mean_absolute_error(train_targets, train_preds)

            # Validation phase with mixed precision
            stage2_model.eval()
            val_loss = 0.0
            val_preds = []
            val_targets = []

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    with autocast():
                        outputs = stage2_model(batch_X)
                        loss = criterion(outputs, batch_y)

                    val_loss += loss.item()
                    val_preds.append(outputs.cpu().numpy())
                    val_targets.append(batch_y.cpu().numpy())

            val_loss /= len(val_loader)
            val_preds = np.vstack(val_preds)
            val_targets = np.vstack(val_targets)
            val_r2, _ = compute_r2_safe(val_targets, val_preds, method='per_output_mean')
            val_mae = mean_absolute_error(val_targets, val_preds)

            # Record history
            history['train_losses'].append(train_loss)
            history['val_losses'].append(val_loss)
            history['train_r2'].append(train_r2)
            history['val_r2'].append(val_r2)
            history['train_mae'].append(train_mae)
            history['val_mae'].append(val_mae)

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                # Save best model
                best_model_state = stage2_model.state_dict().copy()
            else:
                patience_counter += 1

            # Progress output (enhanced version)
            if (epoch + 1) % max(1, config['epochs'] // 20) == 0 or epoch == 0 or epoch == config['epochs'] - 1:
                # Get current learning rate
                current_lr = optimizer.param_groups[0]['lr']

                # Calculate RMSE
                train_rmse = np.sqrt(train_loss)
                val_rmse = np.sqrt(val_loss)

                msg = f"\nEpoch {epoch + 1}/{config['epochs']}"
                msg += f"\n  📉 Train: Loss={train_loss:.4f}, RMSE={train_rmse:.4f}, MAE={train_mae:.4f}, R²={train_r2:.4f}"
                msg += f"\n  📊 Val:   Loss={val_loss:.4f}, RMSE={val_rmse:.4f}, MAE={val_mae:.4f}, R²={val_r2:.4f}"
                msg += f"\n  🎯 Val/Train Ratio: {val_loss/train_loss:.2f}x"
                msg += f"\n  📚 LR: {current_lr:.2e}"
                log_msg.append(msg)

                # Update progress bar with current status
                if progress:
                    progress((epoch + 1) / config['epochs'], desc=f"Epoch {epoch+1}/{config['epochs']} - Val R²: {val_r2:.4f}")

            # Early stopping
            if patience_counter >= early_stop_patience:
                log_msg.append(f"\n⏸️ Early stopping triggered (Epoch {epoch + 1})")
                break

        # Load best model
        stage2_model.load_state_dict(best_model_state)

        # Test set evaluation with batch inference
        y_test_pred = batch_inference(
            stage2_model, X_test, scaler_X, scaler_y, device,
            batch_size=config['batch_size'], model_name="Stage2"
        )

        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_r2, _ = compute_r2_safe(y_test, y_test_pred, method='per_output_mean')

        # Training history summary
        log_msg.append(f"\n📈 Training history summary ({len(history['train_losses'])} epochs):")
        log_msg.append(f"  Best validation loss: {best_val_loss:.4f} (Epoch {np.argmin(history['val_losses']) + 1})")
        log_msg.append(f"  Best validation R²: {max(history['val_r2']):.4f} (Epoch {np.argmax(history['val_r2']) + 1})")
        log_msg.append(f"  Best validation MAE: {min(history['val_mae']):.4f} (Epoch {np.argmin(history['val_mae']) + 1})")
        log_msg.append(f"  Final training loss: {history['train_losses'][-1]:.4f}")
        log_msg.append(f"  Final validation loss: {history['val_losses'][-1]:.4f}")

        log_msg.append(f"\n📊 Test set performance:")
        log_msg.append(f"  MAE: {test_mae:.6f}")
        log_msg.append(f"  RMSE: {test_rmse:.6f}")
        log_msg.append(f"  R²: {test_r2:.4f}")

        # Save model
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = f"Stage2_Boost_{residual_data_key}_{timestamp}"

        model_dir = "saved_models/stage2_boost"
        os.makedirs(model_dir, exist_ok=True)

        model_path = os.path.join(model_dir, f"{model_name}.pth")
        torch.save({
            'model_state_dict': stage2_model.state_dict(),
            'config': config,
            'history': history,
            'residual_data_key': residual_data_key,
            'boundary_signals': boundary_signals,
            'target_signals': target_signals,
            'residual_signals': residual_signals,
            'test_metrics': {
                'mae': test_mae,
                'rmse': test_rmse,
                'r2': test_r2
            }
        }, model_path)

        # Save scalers
        scaler_path = os.path.join(model_dir, f"{model_name}_scalers.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump({'X': scaler_X, 'y': scaler_y}, f)

        # Save to global state
        global_state['stage2_models'][model_name] = {
            'model': stage2_model,
            'config': config,
            'history': history,
            'residual_data_key': residual_data_key,
            'boundary_signals': boundary_signals,
            'target_signals': target_signals,
            'residual_signals': residual_signals,
            'model_path': model_path,
            'scaler_path': scaler_path,
            'test_metrics': {
                'mae': test_mae,
                'rmse': test_rmse,
                'r2': test_r2
            }
        }

        global_state['stage2_scalers'][model_name] = {'X': scaler_X, 'y': scaler_y}

        log_msg.append(f"\n✅ Stage2 model training completed and saved:")
        log_msg.append(f"  Model name: {model_name}")
        log_msg.append(f"  Model path: {model_path}")
        log_msg.append(f"  Scaler path: {scaler_path}")

        results = {
            'model_name': model_name,
            'history': history,
            'test_metrics': {
                'mae': test_mae,
                'rmse': test_rmse,
                'r2': test_r2
            }
        }

        return "\n".join(log_msg), results

    except Exception as e:
        error_msg = f"❌ Stage2 model training failed:\n{str(e)}\n\n{traceback.format_exc()}"
        print(error_msg)
        return error_msg, {}


def train_stage2_boost_model_generator(residual_data_key: str, config: Dict[str, Any], progress=None):
    """
    Generator version of train_stage2_boost_model for real-time log updates

    Yields:
        Current log message string after each progress update
    """
    try:
        if residual_data_key not in global_state['residual_data']:
            yield "❌ Residual data does not exist！"
            return

        log_msg = []
        log_msg.append("=" * 80)
        log_msg.append("🚀 Starting training Stage2 Boost residual model")
        log_msg.append("=" * 80)

        # Get residual data
        residuals_df = global_state['residual_data'][residual_data_key]['data']
        residual_info = global_state['residual_data'][residual_data_key]['info']

        boundary_signals = residual_info['boundary_signals']
        target_signals = residual_info['target_signals']
        residual_signals = residual_info['residual_signals']

        log_msg.append(f"\n📊 Data info:")
        log_msg.append(f"  Residual data: {residual_data_key}")
        log_msg.append(f"  Number of boundary signals: {len(boundary_signals)}")
        log_msg.append(f"  Number of target signals: {len(target_signals)}")
        log_msg.append(f"  Data length: {len(residuals_df)}")

        yield "\n".join(log_msg)

        # Prepare training data
        X = residuals_df[boundary_signals].values
        y_residual = residuals_df[residual_signals].values

        # IMPORTANT: Use RANDOM split instead of sequential split
        # This prevents distribution mismatch between train/val/test sets
        # Sequential split causes issues because:
        #   - Early data (Stage1 train set) has small residuals
        #   - Late data (Stage1 test set) has large residuals
        #   - This leads to train/test distribution mismatch

        from sklearn.model_selection import train_test_split

        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y_residual,
            test_size=config['test_size'],
            random_state=42,
            shuffle=True
        )

        # Second split: separate train and validation sets
        val_ratio = config['val_size'] / (1 - config['test_size'])
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio,
            random_state=42,
            shuffle=True
        )

        log_msg.append(f"\n🔀 Data split (random split to avoid distribution mismatch):")
        log_msg.append(f"  Training set: {len(X_train)} ({len(X_train) / len(X) * 100:.1f}%)")
        log_msg.append(f"  Validation set: {len(X_val)} ({len(X_val) / len(X) * 100:.1f}%)")
        log_msg.append(f"  Test set: {len(X_test)} ({len(X_test) / len(X) * 100:.1f}%)")
        log_msg.append(f"  💡 Use random shuffle to ensure consistent distribution across sets")

        # Data standardization
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_train_scaled = scaler_X.fit_transform(X_train)
        X_val_scaled = scaler_X.transform(X_val)
        X_test_scaled = scaler_X.transform(X_test)

        y_train_scaled = scaler_y.fit_transform(y_train)
        y_val_scaled = scaler_y.transform(y_val)
        y_test_scaled = scaler_y.transform(y_test)

        # Create DataLoader
        train_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_train_scaled),
            torch.FloatTensor(y_train_scaled)
        )
        val_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_val_scaled),
            torch.FloatTensor(y_val_scaled)
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=config['batch_size'], shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=config['batch_size'], shuffle=False
        )

        # Initialize Stage2 model
        log_msg.append(f"\n🏗️ Initializing Stage2 residual model:")
        log_msg.append(f"  Architecture: StaticSensorTransformer")
        log_msg.append(f"  d_model: {config['d_model']}")
        log_msg.append(f"  nhead: {config['nhead']}")
        log_msg.append(f"  num_layers: {config['num_layers']}")

        stage2_model = StaticSensorTransformer(
            num_boundary_sensors=len(boundary_signals),
            num_target_sensors=len(target_signals),
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_layers=config['num_layers'],
            dropout=config['dropout']
        ).to(device)

        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(
            stage2_model.parameters(),
            lr=config['lr'],
            weight_decay=config.get('weight_decay', 1e-5)
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min',
            factor=config.get('scheduler_factor', 0.7),
            patience=config.get('scheduler_patience', 15)
        )
        log_msg.append(f"📊 Learning rate scheduler: ReduceLROnPlateau")

        criterion = nn.MSELoss()
        scaler_amp = GradScaler()

        # Training loop
        log_msg.append(f"\n🎯 Starting training (mixed precision, total epochs: {config['epochs']})")
        yield "\n".join(log_msg)

        history = {
            'train_losses': [], 'val_losses': [],
            'train_r2': [], 'val_r2': [],
            'train_mae': [], 'val_mae': []
        }

        best_val_loss = float('inf')
        patience_counter = 0
        early_stop_patience = config.get('early_stop_patience', 25)

        for epoch in range(config['epochs']):
            # Check if training should be stopped
            if global_state['stop_training_tab4']:
                log_msg.append(f"\n⚠️  Training stopped at Epoch {epoch+1}/{config['epochs']} by user")
                global_state['stop_training_tab4'] = False  # Reset flag
                yield "\n".join(log_msg)
                break

            # Training phase
            stage2_model.train()
            train_loss = 0.0
            train_preds = []
            train_targets = []

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()

                with autocast():
                    outputs = stage2_model(batch_X)
                    loss = criterion(outputs, batch_y)

                scaler_amp.scale(loss).backward()
                scaler_amp.unscale_(optimizer)

                if config.get('grad_clip', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(stage2_model.parameters(), config['grad_clip'])

                scaler_amp.step(optimizer)
                scaler_amp.update()

                train_loss += loss.item()
                train_preds.append(outputs.detach().cpu().numpy())
                train_targets.append(batch_y.detach().cpu().numpy())

            train_loss /= len(train_loader)
            train_preds = np.vstack(train_preds)
            train_targets = np.vstack(train_targets)
            train_r2, _ = compute_r2_safe(train_targets, train_preds, method='per_output_mean')
            train_mae = mean_absolute_error(train_targets, train_preds)

            # Validation phase
            stage2_model.eval()
            val_loss = 0.0
            val_preds = []
            val_targets = []

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    with autocast():
                        outputs = stage2_model(batch_X)
                        loss = criterion(outputs, batch_y)

                    val_loss += loss.item()
                    val_preds.append(outputs.cpu().numpy())
                    val_targets.append(batch_y.cpu().numpy())

            val_loss /= len(val_loader)
            val_preds = np.vstack(val_preds)
            val_targets = np.vstack(val_targets)
            val_r2, _ = compute_r2_safe(val_targets, val_preds, method='per_output_mean')
            val_mae = mean_absolute_error(val_targets, val_preds)

            # Record history
            history['train_losses'].append(train_loss)
            history['val_losses'].append(val_loss)
            history['train_r2'].append(train_r2)
            history['val_r2'].append(val_r2)
            history['train_mae'].append(train_mae)
            history['val_mae'].append(val_mae)

            scheduler.step(val_loss)

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = stage2_model.state_dict().copy()
            else:
                patience_counter += 1

            # Progress output - yield updates periodically
            if (epoch + 1) % max(1, config['epochs'] // 20) == 0 or epoch == 0 or epoch == config['epochs'] - 1:
                current_lr = optimizer.param_groups[0]['lr']
                train_rmse = np.sqrt(train_loss)
                val_rmse = np.sqrt(val_loss)

                msg = f"\nEpoch {epoch + 1}/{config['epochs']}"
                msg += f"\n  📉 Train: Loss={train_loss:.4f}, RMSE={train_rmse:.4f}, MAE={train_mae:.4f}, R²={train_r2:.4f}"
                msg += f"\n  📊 Val:   Loss={val_loss:.4f}, RMSE={val_rmse:.4f}, MAE={val_mae:.4f}, R²={val_r2:.4f}"
                msg += f"\n  🎯 Val/Train Ratio: {val_loss/train_loss:.2f}x"
                msg += f"\n  📚 LR: {current_lr:.2e}"
                log_msg.append(msg)

                if progress:
                    progress((epoch + 1) / config['epochs'], desc=f"Epoch {epoch+1}/{config['epochs']} - Val R²: {val_r2:.4f}")

                # Yield current log state
                yield "\n".join(log_msg)

            # Early stopping
            if patience_counter >= early_stop_patience:
                log_msg.append(f"\n⏸️ Early stopping triggered (Epoch {epoch + 1})")
                yield "\n".join(log_msg)
                break

        # Load best model
        stage2_model.load_state_dict(best_model_state)

        # Test set evaluation
        y_test_pred = batch_inference(
            stage2_model, X_test, scaler_X, scaler_y, device,
            batch_size=config['batch_size'], model_name="Stage2"
        )

        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_r2, _ = compute_r2_safe(y_test, y_test_pred, method='per_output_mean')

        # Training history summary
        log_msg.append(f"\n📈 Training history summary ({len(history['train_losses'])} epochs):")
        log_msg.append(f"  Best validation loss: {best_val_loss:.4f} (Epoch {np.argmin(history['val_losses']) + 1})")
        log_msg.append(f"  Best validation R²: {max(history['val_r2']):.4f} (Epoch {np.argmax(history['val_r2']) + 1})")
        log_msg.append(f"  Best validation MAE: {min(history['val_mae']):.4f} (Epoch {np.argmin(history['val_mae']) + 1})")

        log_msg.append(f"\n📊 Test set performance:")
        log_msg.append(f"  MAE: {test_mae:.6f}")
        log_msg.append(f"  RMSE: {test_rmse:.6f}")
        log_msg.append(f"  R²: {test_r2:.4f}")

        # Save model
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = f"Stage2_Boost_{residual_data_key}_{timestamp}"

        model_dir = "saved_models/stage2_boost"
        os.makedirs(model_dir, exist_ok=True)

        model_path = os.path.join(model_dir, f"{model_name}.pth")
        torch.save({
            'model_state_dict': stage2_model.state_dict(),
            'config': config,
            'history': history,
            'residual_data_key': residual_data_key,
            'boundary_signals': boundary_signals,
            'target_signals': target_signals,
            'residual_signals': residual_signals,
            'test_metrics': {'mae': test_mae, 'rmse': test_rmse, 'r2': test_r2}
        }, model_path)

        # Save scalers
        scaler_path = os.path.join(model_dir, f"{model_name}_scalers.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump({'X': scaler_X, 'y': scaler_y}, f)

        # Save inference config JSON
        inference_config_path = os.path.join(model_dir, f"{model_name}_inference.json")
        inference_config = {
            'model_name': model_name,
            'model_type': 'Stage2_ResidualTransformer',
            'model_path': model_path,
            'scaler_path': scaler_path,

            # Model architecture
            'architecture': {
                'd_model': config['d_model'],
                'nhead': config['nhead'],
                'num_layers': config['num_layers'],
                'dropout': config['dropout']
            },

            # Data config
            'data_config': {
                'residual_data_key': residual_data_key,
                'num_boundary_sensors': len(boundary_signals),
                'num_target_sensors': len(target_signals)
            },

            # Signal info
            'signals': {
                'boundary_signals': boundary_signals,
                'target_signals': target_signals,
                'residual_signals': residual_signals
            },

            # Training info
            'training_info': {
                'epochs_trained': len(history['train_losses']),
                'best_val_loss': min(history['val_losses']),
                'final_test_mae': test_mae,
                'final_test_rmse': test_rmse,
                'final_test_r2': test_r2,
                'batch_size': config['batch_size'],
                'learning_rate': config['lr']
            },

            'created_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        with open(inference_config_path, 'w', encoding='utf-8') as f:
            json.dump(inference_config, f, indent=2, ensure_ascii=False)

        # Save to global state
        global_state['stage2_models'][model_name] = {
            'model': stage2_model,
            'config': config,
            'history': history,
            'residual_data_key': residual_data_key,
            'boundary_signals': boundary_signals,
            'target_signals': target_signals,
            'residual_signals': residual_signals,
            'model_path': model_path,
            'scaler_path': scaler_path,
            'test_metrics': {'mae': test_mae, 'rmse': test_rmse, 'r2': test_r2}
        }

        global_state['stage2_scalers'][model_name] = {'X': scaler_X, 'y': scaler_y}

        log_msg.append(f"\n✅ Stage2 model training completed and saved:")
        log_msg.append(f"  Model name: {model_name}")
        log_msg.append(f"  Model path: {model_path}")
        log_msg.append(f"  Scaler path: {scaler_path}")
        log_msg.append(f"  Inference config: {inference_config_path}")

        yield "\n".join(log_msg)

    except Exception as e:
        error_msg = f"❌ Stage2 model training failed:\n{str(e)}\n\n{traceback.format_exc()}"
        print(error_msg)
        yield error_msg


def create_ensemble_visualization(ensemble_info: Dict[str, Any]):
    """
    Create visualization charts for ensemble model

    Shows ALL output signals with individual prediction vs actual comparison
    and R² scores for each signal
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        # Set font to avoid Chinese character issues
        plt.rcParams['font.family'] = 'DejaVu Sans'

        signal_analysis = ensemble_info['signal_analysis']
        target_signals = ensemble_info['signals']['target']
        num_signals = len(signal_analysis)

        signals = [item['signal'] for item in signal_analysis]
        r2_stage1 = [item['r2_stage1'] for item in signal_analysis]
        r2_ensemble = [item['r2_ensemble'] for item in signal_analysis]

        # Get prediction data
        predictions = ensemble_info.get('predictions', {})
        y_true = predictions.get('y_true')
        y_pred_base = predictions.get('y_pred_base')
        y_pred_ensemble = predictions.get('y_pred_ensemble')

        # Calculate layout: 1 summary row + rows for all individual signals (4 signals per row)
        signals_per_row = 4
        num_signal_rows = (num_signals + signals_per_row - 1) // signals_per_row
        total_rows = 1 + num_signal_rows

        # Create figure with dynamic height
        fig_height = 5 + num_signal_rows * 4
        fig = plt.figure(figsize=(24, fig_height))

        gs = fig.add_gridspec(total_rows, signals_per_row, hspace=0.35, wspace=0.3)

        fig.suptitle(f'Ensemble Model Analysis - {ensemble_info["name"]} (All {num_signals} Signals)',
                     fontsize=16, fontweight='bold', y=0.995)

        # Row 0: Overall R² comparison for all signals
        ax_summary = fig.add_subplot(gs[0, :])
        x = np.arange(len(signals))
        width = 0.35

        bars1 = ax_summary.bar(x - width/2, r2_stage1, width, label='Stage1', alpha=0.8, color='skyblue')
        bars2 = ax_summary.bar(x + width/2, r2_ensemble, width, label='Ensemble', alpha=0.8, color='orange')

        ax_summary.set_xlabel('Signals', fontsize=11)
        ax_summary.set_ylabel('R² Score', fontsize=11)
        ax_summary.set_title('R² Comparison for All Signals', fontsize=13, fontweight='bold')
        ax_summary.set_xticks(x)
        ax_summary.set_xticklabels(signals, rotation=45, ha='right', fontsize=9)
        ax_summary.legend(fontsize=10)
        ax_summary.grid(axis='y', alpha=0.3)
        ax_summary.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

        # Add R² value annotations on bars
        for idx, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            height1 = bar1.get_height()
            height2 = bar2.get_height()
            ax_summary.text(bar1.get_x() + bar1.get_width()/2., height1,
                          f'{r2_stage1[idx]:.3f}',
                          ha='center', va='bottom', fontsize=7, rotation=0)
            ax_summary.text(bar2.get_x() + bar2.get_width()/2., height2,
                          f'{r2_ensemble[idx]:.3f}',
                          ha='center', va='bottom', fontsize=7, rotation=0)

        # Rows 1+: Individual signal prediction plots (ALL signals)
        if y_true is not None and y_pred_base is not None and y_pred_ensemble is not None:
            plot_samples = min(300, len(y_true))

            for idx in range(num_signals):
                row = 1 + idx // signals_per_row
                col = idx % signals_per_row

                ax = fig.add_subplot(gs[row, col])

                signal_name = signals[idx] if idx < len(signals) else f'Signal {idx+1}'

                # Plot predictions vs true values
                ax.plot(y_true[:plot_samples, idx], label='True', alpha=0.8, linewidth=1.5, color='green')
                ax.plot(y_pred_base[:plot_samples, idx], label='Stage1', alpha=0.7, linewidth=1.2, color='skyblue')
                ax.plot(y_pred_ensemble[:plot_samples, idx], label='Ensemble', alpha=0.7, linewidth=1.2, color='orange')

                # Add R² scores in title
                ax.set_title(f'{signal_name}\nStage1 R²={r2_stage1[idx]:.4f}, Ensemble R²={r2_ensemble[idx]:.4f}',
                           fontsize=9, fontweight='bold')
                ax.legend(fontsize=7, loc='best')
                ax.set_xlabel('Sample Index', fontsize=8)
                ax.set_ylabel('Value', fontsize=8)
                ax.grid(alpha=0.3)
                ax.tick_params(labelsize=7)

        plt.tight_layout()
        return fig

    except Exception as e:
        print(f"Visualization generation failed: {e}")
        traceback.print_exc()
        return None


def compute_signal_r2_and_select_threshold(
        base_model_name: str,
        stage2_model_name: str,
        delta_r2_threshold: float = 0.05
) -> Tuple[str, Dict[str, Any], Any]:
    """
    Generate Ensemble Inference Model using Delta R² strategy (evaluate on test set only)

    New logic：
    1. Use test set data from Stage2 training
    2. Calculate Delta R² = R²_ensemble - R²_stage1 for each signal
    3. If Delta R² > threshold, Stage2 has significant improvement, use Stage1+Stage2
    4. Otherwise use only Stage1 prediction

    Args:
        base_model_name: Base SST model name
        stage2_model_name: Stage2 residual model name
        delta_r2_threshold: Delta R² threshold (default 0.05, i.e., 5% improvement)

    Returns:
        status_msg: Status information
        ensemble_info: Ensemble model information
        fig: Visualization chart
    """
    try:
        log_msg = []
        log_msg.append("=" * 80)
        log_msg.append("🎯 Generate Ensemble Inference Model (Delta R² Strategy)")
        log_msg.append("=" * 80)

        # Check if models exist
        if base_model_name not in global_state['trained_models']:
            return f"❌ Base model {base_model_name} does not exist！", {}, None

        if stage2_model_name not in global_state['stage2_models']:
            return f"❌ Stage2 model {stage2_model_name} does not exist！", {}, None

        # Get models
        base_model_info = global_state['trained_models'][base_model_name]
        stage2_model_info = global_state['stage2_models'][stage2_model_name]

        base_model = base_model_info['model']
        stage2_model = stage2_model_info['model']
        stage2_config = stage2_model_info['config']

        # Get residual data
        residual_data_key = stage2_model_info['residual_data_key']

        # If original residual data does not exist, try to use any available residual data
        if residual_data_key not in global_state['residual_data']:
            available_residual_keys = list(global_state['residual_data'].keys())
            if not available_residual_keys:
                return f"❌ No available residual data! Please generate residual data in Tab3 first。", {}, None

            # Use first available residual data
            residual_data_key = available_residual_keys[0]
            log_msg.append(f"\n⚠️  Original residual data not found, using: {residual_data_key}")

        residuals_df = global_state['residual_data'][residual_data_key]['data']
        residual_info = global_state['residual_data'][residual_data_key]['info']

        boundary_signals = residual_info['boundary_signals']
        target_signals = residual_info['target_signals']

        # Validate signal matching
        stage2_boundary = stage2_model_info.get('boundary_signals', boundary_signals)
        stage2_target = stage2_model_info.get('target_signals', target_signals)

        if set(stage2_boundary) != set(boundary_signals):
            log_msg.append(f"\n⚠️  Warning：Stage2 model's boundary signals were not matched with Residual data")
        if set(stage2_target) != set(target_signals):
            log_msg.append(f"\n⚠️  Warning：Stage2 model's target signals were not matched with Residual data")

        log_msg.append(f"\n📊 Model information:")
        log_msg.append(f"  Base model: {base_model_name}")
        log_msg.append(f"  Stage2 model: {stage2_model_name}")
        log_msg.append(f"  Number of target signals: {len(target_signals)}")
        log_msg.append(f"  Delta R² Threshold: {delta_r2_threshold:.3f} ({delta_r2_threshold*100:.1f}%)")

        # CRITICAL: Use Stage1's data split (from residual_info) to ensure same test set
        data_split_info = residual_info.get('data_split', {})
        test_size = data_split_info.get('test_size', 0.15)
        val_size = data_split_info.get('val_size', 0.15)

        total_size = len(residuals_df)
        train_size = int(total_size * (1 - test_size - val_size))
        val_size_actual = int(total_size * val_size)
        test_start_idx = train_size + val_size_actual

        log_msg.append(f"\n🔀 Data split (using Stage1's split ratios for fair comparison):")
        log_msg.append(f"  Total data: {total_size}")
        log_msg.append(f"  Training set: {train_size} ({train_size/total_size*100:.1f}%)")
        log_msg.append(f"  Validation set: {val_size_actual} ({val_size_actual/total_size*100:.1f}%)")
        log_msg.append(f"  Test set: {total_size - test_start_idx} ({(total_size - test_start_idx)/total_size*100:.1f}%)")
        log_msg.append(f"  ✓ Same test set as Stage1 for R² comparability")

        # Extract test set data
        y_true_cols = [f"{sig}_true" for sig in target_signals]
        y_pred_cols = [f"{sig}_pred" for sig in target_signals]

        y_true_test = residuals_df[y_true_cols].iloc[test_start_idx:].values
        y_pred_stage1_test = residuals_df[y_pred_cols].iloc[test_start_idx:].values
        X_test = residuals_df[boundary_signals].iloc[test_start_idx:].values

        # Use Stage2 model to generate residual values fo test set
        y_residual_pred_test = batch_inference(
            stage2_model,
            X_test,
            global_state['stage2_scalers'][stage2_model_name]['X'],
            global_state['stage2_scalers'][stage2_model_name]['y'],
            device,
            batch_size=512,
            model_name="Stage2"
        )

        # Calculate R² score for each signal
        signal_analysis = []

        for i, signal in enumerate(target_signals):
            y_true_sig = y_true_test[:, i]
            y_pred_stage1_sig = y_pred_stage1_test[:, i]
            y_pred_ensemble_sig = y_pred_stage1_sig + y_residual_pred_test[:, i]

            # Calculate Stage1 R²
            r2_stage1, _ = compute_r2_safe(
                y_true_sig.reshape(-1, 1),
                y_pred_stage1_sig.reshape(-1, 1),
                method='per_output_mean'
            )

            # Calculate Ensemble R²
            r2_ensemble, _ = compute_r2_safe(
                y_true_sig.reshape(-1, 1),
                y_pred_ensemble_sig.reshape(-1, 1),
                method='per_output_mean'
            )

            # Calculate Delta R²
            delta_r2 = r2_ensemble - r2_stage1

            # Determine whether to use Stage2
            use_stage2 = delta_r2 > delta_r2_threshold

            signal_analysis.append({
                'signal': signal,
                'r2_stage1': float(r2_stage1),
                'r2_ensemble': float(r2_ensemble),
                'delta_r2': float(delta_r2),
                'use_stage2': bool(use_stage2)
            })

        # Count signals using Stage2
        num_use_stage2 = sum(1 for item in signal_analysis if item['use_stage2'])
        num_use_stage1_only = len(target_signals) - num_use_stage2

        log_msg.append(f"\n🎯 Signal Delta R² Analysis:")
        log_msg.append(f"{'Signal name':<30} {'Stage1 R²':>12} {'Ensemble R²':>12} {'Delta R²':>12} {'Selection':>10}")
        log_msg.append("-" * 80)

        for item in signal_analysis:
            choice = "Stage1+2" if item['use_stage2'] else "Stage1"
            log_msg.append(
                f"{item['signal']:<30} {item['r2_stage1']:>12.4f} {item['r2_ensemble']:>12.4f} "
                f"{item['delta_r2']:>12.4f} {choice:>10}"
            )

        log_msg.append("-" * 80)
        log_msg.append(f"Using Stage1+Stage2: {num_use_stage2} signals")
        log_msg.append(f"Using Stage1 only: {num_use_stage1_only} signals")

        # Generate final ensemble prediction (on test set)
        y_ensemble_test = y_pred_stage1_test.copy()
        for i, item in enumerate(signal_analysis):
            if item['use_stage2']:
                y_ensemble_test[:, i] = y_pred_stage1_test[:, i] + y_residual_pred_test[:, i]

        # Calculate overall performance
        mae_stage1 = mean_absolute_error(y_true_test, y_pred_stage1_test)
        mae_ensemble = mean_absolute_error(y_true_test, y_ensemble_test)
        rmse_stage1 = np.sqrt(mean_squared_error(y_true_test, y_pred_stage1_test))
        rmse_ensemble = np.sqrt(mean_squared_error(y_true_test, y_ensemble_test))
        r2_stage1, _ = compute_r2_safe(y_true_test, y_pred_stage1_test, method='per_output_mean')
        r2_ensemble, _ = compute_r2_safe(y_true_test, y_ensemble_test, method='per_output_mean')

        improvement_mae = (mae_stage1 - mae_ensemble) / mae_stage1 * 100 if mae_stage1 > 0 else 0
        improvement_rmse = (rmse_stage1 - rmse_ensemble) / rmse_stage1 * 100 if rmse_stage1 > 0 else 0
        improvement_r2 = (r2_ensemble - r2_stage1) / (1 - r2_stage1) * 100 if r2_stage1 < 1 else 0

        log_msg.append(f"\n📈 Overall Performance Comparison (Test set):")
        log_msg.append(f"{'metrics':<15} {'Stage1':>15} {'Ensemble':>15} {'improvement':>15}")
        log_msg.append("-" * 65)
        log_msg.append(f"{'MAE':<15} {mae_stage1:>15.6f} {mae_ensemble:>15.6f} {improvement_mae:>14.2f}%")
        log_msg.append(f"{'RMSE':<15} {rmse_stage1:>15.6f} {rmse_ensemble:>15.6f} {improvement_rmse:>14.2f}%")
        log_msg.append(f"{'R²':<15} {r2_stage1:>15.4f} {r2_ensemble:>15.4f} {improvement_r2:>14.2f}%")

        # Save ensemble model information
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        ensemble_name = f"Ensemble_{base_model_name}_{timestamp}"

        ensemble_info = {
            'name': ensemble_name,
            'base_model_name': base_model_name,
            'stage2_model_name': stage2_model_name,
            'delta_r2_threshold': float(delta_r2_threshold),
            'signal_analysis': signal_analysis,
            'num_use_stage2': int(num_use_stage2),
            'num_use_stage1_only': int(num_use_stage1_only),
            'metrics': {
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
            },
            'predictions': {
                'y_true': y_true_test,
                'y_pred_base': y_pred_stage1_test,  # Use y_pred_base to match tab6 reading
                'y_pred_ensemble': y_ensemble_test,
                'y_residual_pred': y_residual_pred_test
            },
            'signals': {
                'boundary': boundary_signals,
                'target': target_signals
            },
            'data_split': {
                'test_size': float(test_size),
                'val_size': float(val_size),
                'test_start_idx': int(test_start_idx),
                'test_samples': int(len(y_true_test))
            },
            'created_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        global_state['ensemble_models'][ensemble_name] = ensemble_info

        # Save config file
        ensemble_dir = "saved_models/ensemble"
        os.makedirs(ensemble_dir, exist_ok=True)

        config_path = os.path.join(ensemble_dir, f"{ensemble_name}_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            # Save config (exclude large arrays, ensure all types are JSON serializable)
            save_config = {
                'name': ensemble_name,
                'base_model_name': base_model_name,
                'stage2_model_name': stage2_model_name,
                'delta_r2_threshold': float(delta_r2_threshold),
                'signal_analysis': signal_analysis,  # Converted to Python native types
                'num_use_stage2': int(num_use_stage2),
                'num_use_stage1_only': int(num_use_stage1_only),
                'metrics': ensemble_info['metrics'],  # Converted
                'signals': ensemble_info['signals'],
                'data_split': ensemble_info['data_split'],
                'created_time': ensemble_info['created_time']
            }
            json.dump(save_config, f, indent=2, ensure_ascii=False)

        # Generate summary CSV file
        csv_path = os.path.join(ensemble_dir, f"{ensemble_name}_summary.csv")
        summary_data = []
        for item in signal_analysis:
            summary_data.append({
                'Signal name': item['signal'],
                'Stage1_R2': item['r2_stage1'],
                'Ensemble_R2': item['r2_ensemble'],
                'Delta_R2': item['delta_r2'],
                'R2 Improvement (%)': item['delta_r2'] * 100,
                'Selection_model': 'Stage1+Stage2' if item['use_stage2'] else 'Stage1',
                'Use Stage2': 'Yes' if item['use_stage2'] else 'No'
            })

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

        log_msg.append(f"\n✅ Ensemble Inference Model generated:")
        log_msg.append(f"  Model name: {ensemble_name}")
        log_msg.append(f"  Config path: {config_path}")
        log_msg.append(f"  summaryCSV: {csv_path}")

        # Generate visualization chart
        fig = create_ensemble_visualization(ensemble_info)

        return "\n".join(log_msg), ensemble_info, fig

    except Exception as e:
        error_msg = f"❌ Ensemble model generation failed:\n{str(e)}\n\n{traceback.format_exc()}"
        print(error_msg)
        return error_msg, {}, None


# ============================================================================
# Data loading functions

def load_data_from_csv(file_obj):
    """Load data from CSV file"""
    try:
        df = pd.read_csv(file_obj.name)

        # If there are unnamed columns, set as index
        if 'Unnamed: 0' in df.columns:
            df = df.set_index('Unnamed: 0')
            df.index.name = 'index'
        elif df.index.name is None:
            df.index.name = 'index'

        global_state['df'] = df
        global_state['all_signals'] = list(df.columns)

        status = f"✅ Data loaded successfully!\n\n"
        status += f"📊 Data dimensions: {df.shape}\n"
        status += f"📈 Number of samples: {len(df):,}\n"
        status += f"🎯 Number of features: {len(df.columns)}\n\n"
        status += f"First 5 columns: {', '.join(df.columns[:5].tolist())}"

        signals_display = f"Available Signals ({len(df.columns)}):\n" + ", ".join(df.columns.tolist())

        # Data preview (first 100 rows)
        preview_df = df.head(100)

        return status, preview_df, signals_display

    except Exception as e:
        error_msg = f"❌ Data loading failed: {str(e)}"
        return error_msg, None, ""


def get_available_csv_files():
    """
    Get list of available CSV files in data/ folder

    Returns:
        List of CSV file paths (safe - never raises exceptions)
    """
    try:
        import glob

        csv_files = []

        # Search in data/ folder
        if os.path.exists('data'):
            csv_files.extend(glob.glob('data/*.csv'))

        # Search in current directory
        csv_files.extend(glob.glob('*.csv'))

        # Sort by modification time (newest first)
        csv_files = sorted(csv_files, key=lambda x: os.path.getmtime(x) if os.path.exists(x) else 0, reverse=True)

        return csv_files if csv_files else []

    except Exception as e:
        print(f"⚠️ Error in get_available_csv_files: {e}")
        return []  # Return empty list on error


def load_csv_from_path(csv_path):
    """
    Load CSV file from a given path

    Args:
        csv_path: Path to CSV file

    Returns:
        status: Status message
        preview_df: Data preview (first 100 rows)
        signals: Available signals
    """
    if not csv_path or csv_path == "(no CSV files found)":
        return "❌ Please select a valid CSV file", None, ""

    if not os.path.exists(csv_path):
        return f"❌ File does not exist: {csv_path}", None, ""

    try:
        df = pd.read_csv(csv_path)

        # If there are unnamed columns, set as index
        if 'Unnamed: 0' in df.columns:
            df = df.set_index('Unnamed: 0')
            df.index.name = 'index'
        elif df.index.name is None:
            df.index.name = 'index'

        global_state['df'] = df
        global_state['all_signals'] = list(df.columns)

        status = f"✅ Data loaded successfully!\n\n"
        status += f"📁 file: {csv_path}\n"
        status += f"📊 Data dimensions: {df.shape}\n"
        status += f"📈 Number of samples: {len(df):,}\n"
        status += f"🎯 Number of features: {len(df.columns)}\n\n"
        status += f"First 5 columns: {', '.join(df.columns[:5].tolist())}"

        signals_display = f"Available Signals ({len(df.columns)}):\n" + ", ".join(df.columns.tolist())

        # Data preview (first 100 rows)
        preview_df = df.head(100)

        return status, preview_df, signals_display

    except Exception as e:
        error_msg = f"❌ Data loading failed: {str(e)}"
        return error_msg, None, ""


def check_preloaded_data():
    """
    Check if data was pre-loaded (from Colab) and return its status

    Returns:
        status: Status message
        preview_df: Data preview (first 100 rows)
        signals_display: Available signals
    """
    if global_state.get('df') is not None:
        df = global_state['df']

        status = f"✅ [Preloaded] Data loaded!\n\n"
        status += f"📊 Data dimensions: {df.shape}\n"
        status += f"📈 Number of samples: {len(df):,}\n"
        status += f"🎯 Number of features: {len(df.columns)}\n\n"
        status += f"Column names: {', '.join(df.columns[:5].tolist())}"
        if len(df.columns) > 5:
            status += f"... (total{len(df.columns)}columns)"

        signals_display = f"Available Signals ({len(df.columns)}):\n" + ", ".join(df.columns.tolist())

        # Data preview (first 100 rows)
        preview_df = df.head(100)

        return status, preview_df, signals_display
    else:
        return "⚠️ Data was not loaded", None, ""


def load_signals_config_from_json(json_file):
    """
    Load boundary and target signals configuration from JSON file

    Args:
        json_file: Gradio File object or file path

    Returns:
        boundary_signals: List of boundary signal names
        target_signals: List of target signal names
        status_message: Status message
    """
    try:
        # Handle Gradio File object or direct path
        if hasattr(json_file, 'name'):
            file_path = json_file.name
        else:
            file_path = json_file

        if not file_path:
            return [], [], "❌ Please Upload JSON Config File"

        if not os.path.exists(file_path):
            return [], [], f"❌ File does not exist: {file_path}"

        # Load JSON
        with open(file_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        boundary_signals = config.get('boundary_signals', [])
        target_signals = config.get('target_signals', [])

        if not boundary_signals or not target_signals:
            return [], [], "❌ JSON file format error, missing 'boundary_signals' or 'target_signals'"

        status = f"✅ JSON config loaded successfully!\n\n"
        status += f"📥 Number of boundary signals: {len(boundary_signals)}\n"
        status += f"📤 Number of target signals: {len(target_signals)}\n"
        status += f"📁 file: {os.path.basename(file_path)}"

        return boundary_signals, target_signals, status

    except json.JSONDecodeError as e:
        return [], [], f"❌ JSON parsing failed: {str(e)}"
    except Exception as e:
        return [], [], f"❌ Loading failed: {str(e)}"


def get_available_json_configs():
    """
    Get list of available JSON config files in data/ folder

    Returns:
        List of JSON file paths
    """
    try:
        import glob

        json_files = []

        # Search in data/ folder
        if os.path.exists('data'):
            json_files.extend(glob.glob('data/*.json'))

        # Search in current directory
        json_files.extend(glob.glob('*.json'))

        # Sort by modification time (newest first)
        json_files = sorted(json_files, key=lambda x: os.path.getmtime(x) if os.path.exists(x) else 0, reverse=True)

        return json_files if json_files else []

    except Exception as e:
        print(f"⚠️ Error in get_available_json_configs: {e}")
        return []


def create_sample_data():
    """Create sample data"""
    try:
        np.random.seed(42)
        n_samples = 10000
        n_boundary = 10
        n_target = 5

        # Generate correlated sensor data
        X = np.random.randn(n_samples, n_boundary)
        y = X[:, :n_target] + 0.5 * X[:, :n_target] ** 2 + 0.1 * np.random.randn(n_samples, n_target)

        boundary_cols = [f"boundary_{i + 1}" for i in range(n_boundary)]
        target_cols = [f"target_{i + 1}" for i in range(n_target)]

        df = pd.DataFrame(
            np.column_stack([X, y]),
            columns=boundary_cols + target_cols
        )

        df.index.name = 'index'
        global_state['df'] = df
        global_state['all_signals'] = list(df.columns)

        status = f"✅ Sample data created successfully!\n\n"
        status += f"📊 Data dimensions: {df.shape}\n"
        status += f"📈 Number of samples: {len(df):,}\n"
        status += f"🎯 Boundary signals: {n_boundary}\n"
        status += f"🎯 Target signals: {n_target}\n\n"
        status += "💡 Tip: Sample data simulates nonlinear relationships between sensors"

        signals_display = f"Available Signals ({len(df.columns)}):\n" + ", ".join(df.columns.tolist())

        # Data preview (first 100 rows)
        preview_df = df.head(100)

        return status, preview_df, signals_display

    except Exception as e:
        error_msg = f"❌ Sample data creation failed: {str(e)}"
        return error_msg, None, ""


# ============================================================================
# SST model training functions

def train_base_model_ui(
        boundary_signals, target_signals, model_type,
        epochs, batch_size, lr,
        d_model, nhead, num_layers, dropout,
        test_size, val_size,
        weight_decay, scheduler_patience, scheduler_factor, grad_clip_norm,
        temporal_signals=None, apply_smoothing=False,
        progress=gr.Progress()
):
    """UI function for training base model"""
    try:
        if global_state['df'] is None:
            return "❌ Please load data first！"

        if not boundary_signals or not target_signals:
            return "❌ Please select boundary signals and target signals！"

        log_messages = []
        log_messages.append("=" * 80)
        log_messages.append(f"🚀 Starting training {model_type}")
        log_messages.append("=" * 80)
        log_messages.append(f"\n📊 Training config:")
        log_messages.append(f"  Model type: {model_type}")
        log_messages.append(f"  Number of boundary signals: {len(boundary_signals)}")
        log_messages.append(f"  Number of target signals: {len(target_signals)}")
        log_messages.append(f"  Training epochs: {epochs}")
        log_messages.append(f"  Batch size: {batch_size}")
        log_messages.append(f"  Learning rate: {lr}")

        df = global_state['df']

        # Prepare data
        X = df[boundary_signals].values
        y = df[target_signals].values

        # Apply IFD smoothing (if needed)
        if apply_smoothing and temporal_signals:
            log_messages.append(f"\n🔧 Applying IFD smoothing...")
            # Apply smoothing to the full y array for specified temporal signals
            y_smoothed = apply_ifd_smoothing(
                y_data=y,
                target_sensors=target_signals,
                ifd_sensor_names=temporal_signals,
                window_length=15,
                polyorder=3
            )
            # Update y with smoothed values
            y = y_smoothed
            # Update df with smoothed target signals
            for i, sig in enumerate(target_signals):
                df[sig] = y[:, i]
            log_messages.append(f"  Applied to {len(temporal_signals)} temporal signals with smoothing")

        # Data split
        train_size = int(len(X) * (1 - test_size - val_size))
        val_size_samples = int(len(X) * val_size)

        X_train = X[:train_size]
        X_val = X[train_size:train_size + val_size_samples]
        X_test = X[train_size + val_size_samples:]

        y_train = y[:train_size]
        y_val = y[train_size:train_size + val_size_samples]
        y_test = y[train_size + val_size_samples:]

        log_messages.append(f"\n🔀 Data split:")
        log_messages.append(f"  Training set: {len(X_train)} ({len(X_train) / len(X) * 100:.1f}%)")
        log_messages.append(f"  Validation set: {len(X_val)} ({len(X_val) / len(X) * 100:.1f}%)")
        log_messages.append(f"  Test set: {len(X_test)} ({len(X_test) / len(X) * 100:.1f}%)")

        # Data standardization
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_train_scaled = scaler_X.fit_transform(X_train)
        X_val_scaled = scaler_X.transform(X_val)
        X_test_scaled = scaler_X.transform(X_test)

        y_train_scaled = scaler_y.fit_transform(y_train)
        y_val_scaled = scaler_y.transform(y_val)
        y_test_scaled = scaler_y.transform(y_test)

        # Create DataLoader
        train_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_train_scaled),
            torch.FloatTensor(y_train_scaled)
        )
        val_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_val_scaled),
            torch.FloatTensor(y_val_scaled)
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )

        # Initialize model
        log_messages.append(f"\n🏗️ Initializing model: {model_type}")

        if model_type == 'StaticSensorTransformer':
            model = StaticSensorTransformer(
                num_boundary_sensors=len(boundary_signals),
                num_target_sensors=len(target_signals),
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dropout=dropout
            ).to(device)
        else:
            return f"❌ Unsupported model type: {model_type}"

        # Optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience
        )
        log_messages.append(f"📊 Optimizer: AdamW (lr={lr:.2e}, weight_decay={weight_decay:.2e})")
        log_messages.append(f"📊 Learning rate scheduler: ReduceLROnPlateau (factor={scheduler_factor}, patience={scheduler_patience})")
        log_messages.append(f"✂️ Gradient Clipping: {grad_clip_norm}")
        criterion = nn.MSELoss()

        # Mixed precision training
        scaler = GradScaler()

        # Training loop
        log_messages.append(f"\n🎯 Starting training (mixed precision)...")
        history = {
            'train_losses': [],
            'val_losses': [],
            'train_r2': [],
            'val_r2': [],
            'train_mae': [],
            'val_mae': []
        }
        best_val_loss = float('inf')
        patience_counter = 0
        early_stop_patience = 25

        for epoch in range(epochs):
            # Check if training should be stopped
            if global_state['stop_training_tab2']:
                log_messages.append(f"\n⚠️  Training stopped at Epoch {epoch+1}/{epochs} by user")
                global_state['stop_training_tab2'] = False  # Reset flag
                break

            # Training with mixed precision
            model.train()
            train_loss = 0.0
            train_preds = []
            train_targets = []
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()

                # Mixed precision forward pass
                with autocast():
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)

                # Mixed precision backward pass
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()
                train_preds.append(outputs.detach().cpu().numpy())
                train_targets.append(batch_y.detach().cpu().numpy())

            train_loss /= len(train_loader)

            # Calculate training metrics
            train_preds_arr = np.vstack(train_preds)
            train_targets_arr = np.vstack(train_targets)

            # Inverse transform to original space for metrics
            train_preds_orig = scaler_y.inverse_transform(train_preds_arr)
            train_targets_orig = scaler_y.inverse_transform(train_targets_arr)
            train_r2, _ = compute_r2_safe(train_targets_orig, train_preds_orig, method='per_output_mean')
            train_mae = mean_absolute_error(train_targets_orig, train_preds_orig)

            # Validation with mixed precision
            model.eval()
            val_loss = 0.0
            val_preds = []
            val_targets = []
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    with autocast():
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    val_preds.append(outputs.cpu().numpy())
                    val_targets.append(batch_y.cpu().numpy())

            val_loss /= len(val_loader)

            # Calculate validation metrics
            val_preds_arr = np.vstack(val_preds)
            val_targets_arr = np.vstack(val_targets)

            # Inverse transform to original space for metrics
            val_preds_orig = scaler_y.inverse_transform(val_preds_arr)
            val_targets_orig = scaler_y.inverse_transform(val_targets_arr)
            val_r2, _ = compute_r2_safe(val_targets_orig, val_preds_orig, method='per_output_mean')
            val_mae = mean_absolute_error(val_targets_orig, val_preds_orig)

            history['train_losses'].append(train_loss)
            history['val_losses'].append(val_loss)
            history['train_r2'].append(train_r2)
            history['val_r2'].append(val_r2)
            history['train_mae'].append(train_mae)
            history['val_mae'].append(val_mae)

            scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1

            # Progress display (Enhanced - show MAE, RMSE, R2 and learning rate)
            if (epoch + 1) % max(1, epochs // 20) == 0 or epoch == 0:
                # Get current learning rate
                current_lr = optimizer.param_groups[0]['lr']

                # Calculate RMSE (more intuitive)
                train_rmse = np.sqrt(train_loss)
                val_rmse = np.sqrt(val_loss)

                msg = f"\nEpoch {epoch + 1}/{epochs}"
                msg += f"\n  📉 Train: Loss={train_loss:.4f}, RMSE={train_rmse:.4f}, MAE={train_mae:.4f}, R²={train_r2:.4f}"
                msg += f"\n  📊 Val:   Loss={val_loss:.4f}, RMSE={val_rmse:.4f}, MAE={val_mae:.4f}, R²={val_r2:.4f}"
                msg += f"\n  🎯 Val/Train Ratio: {val_loss/train_loss:.2f}x"
                msg += f"\n  📚 LR: {current_lr:.2e}"

                log_messages.append(msg)
                progress((epoch + 1) / epochs, desc=f"Epoch {epoch+1}/{epochs} - Val R²: {val_r2:.4f}")

            if patience_counter >= early_stop_patience:
                log_messages.append(f"\n⏸️ Early stopping triggered (Epoch {epoch + 1})")
                break

        # Load best model
        model.load_state_dict(best_model_state)

        # Test set evaluation with mixed precision and batch inference
        y_test_pred = batch_inference(
            model, X_test, scaler_X, scaler_y, device,
            batch_size=batch_size, model_name=model_type
        )

        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_r2, _ = compute_r2_safe(y_test, y_test_pred, method='per_output_mean')

        # Training history summary
        log_messages.append(f"\n📈 Training history summary ({len(history['train_losses'])} epochs):")
        log_messages.append(f"  Best validation loss: {best_val_loss:.4f} (Epoch {np.argmin(history['val_losses']) + 1})")
        log_messages.append(f"  Best validation R²: {max(history['val_r2']):.4f} (Epoch {np.argmax(history['val_r2']) + 1})")
        log_messages.append(f"  Best validation MAE: {min(history['val_mae']):.4f} (Epoch {np.argmin(history['val_mae']) + 1})")
        log_messages.append(f"  Final training loss: {history['train_losses'][-1]:.4f}")
        log_messages.append(f"  Final validation loss: {history['val_losses'][-1]:.4f}")

        log_messages.append(f"\n📊 Test set performance:")
        log_messages.append(f"  MAE: {test_mae:.6f}")
        log_messages.append(f"  RMSE: {test_rmse:.6f}")
        log_messages.append(f"  R²: {test_r2:.4f}")

        # Save model
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = f"{model_type}_{timestamp}"

        model_dir = "saved_models"
        os.makedirs(model_dir, exist_ok=True)

        model_path = os.path.join(model_dir, f"{model_name}.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': {
                'type': model_type,
                'boundary_signals': boundary_signals,
                'target_signals': target_signals,
                'config': {
                    'd_model': d_model,
                    'nhead': nhead,
                    'num_layers': num_layers,
                    'dropout': dropout,
                    'batch_size': batch_size
                },
                'data_split': {
                    'test_size': float(test_size),
                    'val_size': float(val_size),
                    'total_samples': len(X),
                    'train_samples': len(X_train),
                    'val_samples': len(X_val),
                    'test_samples': len(X_test)
                }
            },
            'training_history': history
        }, model_path)

        # Save scalers
        scaler_path = os.path.join(model_dir, f"{model_name}_scalers.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump({'X': scaler_X, 'y': scaler_y}, f)

        # Save inference config
        save_inference_config(
            model_name, model_type, model_path, scaler_path,
            boundary_signals, target_signals,
            {
                'd_model': d_model,
                'nhead': nhead,
                'num_layers': num_layers,
                'dropout': dropout
            }
        )

        # Save to global state
        global_state['trained_models'][model_name] = {
            'model': model,
            'type': model_type,
            'boundary_signals': boundary_signals,
            'target_signals': target_signals,
            'config': {
                'd_model': d_model,
                'nhead': nhead,
                'num_layers': num_layers,
                'dropout': dropout,
                'batch_size': batch_size
            },
            'model_path': model_path,
            'scaler_path': scaler_path
        }

        global_state['scalers'][model_name] = {'X': scaler_X, 'y': scaler_y}

        log_messages.append(f"\n✅ Model training completed and saved:")
        log_messages.append(f"  Model name: {model_name}")
        log_messages.append(f"  Model path: {model_path}")
        log_messages.append(f"  Scaler path: {scaler_path}")

        return "\n".join(log_messages)

    except Exception as e:
        error_msg = f"❌ Training failed:\n{str(e)}\n\n{traceback.format_exc()}"
        print(error_msg)
        return error_msg


def train_base_model_ui_generator(
        boundary_signals, target_signals, model_type,
        epochs, batch_size, lr,
        d_model, nhead, num_layers, dropout,
        test_size, val_size,
        weight_decay, scheduler_patience, scheduler_factor, grad_clip_norm,
        temporal_signals=None, apply_smoothing=False,
        progress=gr.Progress()
):
    """Generator version for real-time log updates during training"""
    try:
        if global_state['df'] is None:
            yield "❌ Please load data first！"
            return

        if not boundary_signals or not target_signals:
            yield "❌ Please select boundary signals and target signals！"
            return

        accumulated_log = ""

        def log_and_yield(message):
            """Helper function to accumulate and yield log messages"""
            nonlocal accumulated_log
            accumulated_log += message + "\n"
            return accumulated_log

        yield log_and_yield("=" * 80)
        yield log_and_yield(f"🚀 Starting training {model_type}")
        yield log_and_yield("=" * 80)
        yield log_and_yield(f"\n📊 Training config:")
        yield log_and_yield(f"  Model type: {model_type}")
        yield log_and_yield(f"  Number of boundary signals: {len(boundary_signals)}")
        yield log_and_yield(f"  Number of target signals: {len(target_signals)}")
        yield log_and_yield(f"  Training epochs: {epochs}")
        yield log_and_yield(f"  Batch size: {batch_size}")
        yield log_and_yield(f"  Learning rate: {lr}")

        df = global_state['df']

        # Prepare data
        X = df[boundary_signals].values
        y = df[target_signals].values

        # Apply IFD smoothing (if needed)
        if apply_smoothing and temporal_signals:
            yield log_and_yield(f"\n🔧 Applying IFD smoothing...")
            # Apply smoothing to the full y array for specified temporal signals
            y_smoothed = apply_ifd_smoothing(
                y_data=y,
                target_sensors=target_signals,
                ifd_sensor_names=temporal_signals,
                window_length=15,
                polyorder=3
            )
            # Update y with smoothed values
            y = y_smoothed
            # Update df with smoothed target signals
            for i, sig in enumerate(target_signals):
                df[sig] = y[:, i]
            yield log_and_yield(f"  Applied to {len(temporal_signals)} temporal signals with smoothing")

        # Data split
        train_size = int(len(X) * (1 - test_size - val_size))
        val_size_samples = int(len(X) * val_size)

        X_train = X[:train_size]
        X_val = X[train_size:train_size + val_size_samples]
        X_test = X[train_size + val_size_samples:]

        y_train = y[:train_size]
        y_val = y[train_size:train_size + val_size_samples]
        y_test = y[train_size + val_size_samples:]

        yield log_and_yield(f"\n🔀 Data split:")
        yield log_and_yield(f"  Training set: {len(X_train)} ({len(X_train) / len(X) * 100:.1f}%)")
        yield log_and_yield(f"  Validation set: {len(X_val)} ({len(X_val) / len(X) * 100:.1f}%)")
        yield log_and_yield(f"  Test set: {len(X_test)} ({len(X_test) / len(X) * 100:.1f}%)")

        # Data standardization
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_train_scaled = scaler_X.fit_transform(X_train)
        X_val_scaled = scaler_X.transform(X_val)
        X_test_scaled = scaler_X.transform(X_test)

        y_train_scaled = scaler_y.fit_transform(y_train)
        y_val_scaled = scaler_y.transform(y_val)
        y_test_scaled = scaler_y.transform(y_test)

        # Create DataLoader
        train_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_train_scaled),
            torch.FloatTensor(y_train_scaled)
        )
        val_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_val_scaled),
            torch.FloatTensor(y_val_scaled)
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )

        # Initialize model
        yield log_and_yield(f"\n🏗️ Initializing model: {model_type}")

        if model_type == 'StaticSensorTransformer':
            model = StaticSensorTransformer(
                num_boundary_sensors=len(boundary_signals),
                num_target_sensors=len(target_signals),
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dropout=dropout
            ).to(device)
        else:
            yield log_and_yield(f"❌ Unsupported model type: {model_type}")
            return

        # Optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience
        )
        yield log_and_yield(f"📊 Optimizer: AdamW (lr={lr:.2e}, weight_decay={weight_decay:.2e})")
        yield log_and_yield(f"📊 Learning rate scheduler: ReduceLROnPlateau (factor={scheduler_factor}, patience={scheduler_patience})")
        yield log_and_yield(f"✂️ Gradient Clipping: {grad_clip_norm}")
        criterion = nn.MSELoss()

        # Mixed precision training
        scaler = GradScaler()

        # Training loop
        yield log_and_yield(f"\n🎯 Starting training (mixed precision)...")
        history = {
            'train_losses': [],
            'val_losses': [],
            'train_r2': [],
            'val_r2': [],
            'train_mae': [],
            'val_mae': []
        }
        best_val_loss = float('inf')
        patience_counter = 0
        early_stop_patience = 25

        for epoch in range(epochs):
            # Check if training should be stopped
            if global_state['stop_training_tab2']:
                yield log_and_yield(f"\n⚠️  Training stopped at Epoch {epoch+1}/{epochs} by user")
                global_state['stop_training_tab2'] = False  # Reset flag
                break

            # Training with mixed precision
            model.train()
            train_loss = 0.0
            train_preds = []
            train_targets = []
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()

                # Mixed precision forward pass
                with autocast():
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)

                # Mixed precision backward pass
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()
                train_preds.append(outputs.detach().cpu().numpy())
                train_targets.append(batch_y.detach().cpu().numpy())

            train_loss /= len(train_loader)

            # Calculate training metrics
            train_preds_arr = np.vstack(train_preds)
            train_targets_arr = np.vstack(train_targets)

            # Inverse transform to original space for metrics
            train_preds_orig = scaler_y.inverse_transform(train_preds_arr)
            train_targets_orig = scaler_y.inverse_transform(train_targets_arr)
            train_r2, _ = compute_r2_safe(train_targets_orig, train_preds_orig, method='per_output_mean')
            train_mae = mean_absolute_error(train_targets_orig, train_preds_orig)

            # Validation with mixed precision
            model.eval()
            val_loss = 0.0
            val_preds = []
            val_targets = []
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    with autocast():
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    val_preds.append(outputs.cpu().numpy())
                    val_targets.append(batch_y.cpu().numpy())

            val_loss /= len(val_loader)

            # Calculate validation metrics
            val_preds_arr = np.vstack(val_preds)
            val_targets_arr = np.vstack(val_targets)

            # Inverse transform to original space for metrics
            val_preds_orig = scaler_y.inverse_transform(val_preds_arr)
            val_targets_orig = scaler_y.inverse_transform(val_targets_arr)
            val_r2, _ = compute_r2_safe(val_targets_orig, val_preds_orig, method='per_output_mean')
            val_mae = mean_absolute_error(val_targets_orig, val_preds_orig)

            history['train_losses'].append(train_loss)
            history['val_losses'].append(val_loss)
            history['train_r2'].append(train_r2)
            history['val_r2'].append(val_r2)
            history['train_mae'].append(train_mae)
            history['val_mae'].append(val_mae)

            scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1

            # Progress display (Enhanced - show MAE, RMSE, R2 and learning rate)
            if (epoch + 1) % max(1, epochs // 20) == 0 or epoch == 0:
                # Get current learning rate
                current_lr = optimizer.param_groups[0]['lr']

                # Calculate RMSE (more intuitive)
                train_rmse = np.sqrt(train_loss)
                val_rmse = np.sqrt(val_loss)

                msg = f"\nEpoch {epoch + 1}/{epochs}"
                msg += f"\n  📉 Train: Loss={train_loss:.4f}, RMSE={train_rmse:.4f}, MAE={train_mae:.4f}, R²={train_r2:.4f}"
                msg += f"\n  📊 Val:   Loss={val_loss:.4f}, RMSE={val_rmse:.4f}, MAE={val_mae:.4f}, R²={val_r2:.4f}"
                msg += f"\n  🎯 Val/Train Ratio: {val_loss/train_loss:.2f}x"
                msg += f"\n  📚 LR: {current_lr:.2e}"

                yield log_and_yield(msg)
                progress((epoch + 1) / epochs, desc=f"Epoch {epoch+1}/{epochs} - Val R²: {val_r2:.4f}")

            if patience_counter >= early_stop_patience:
                yield log_and_yield(f"\n⏸️ Early stopping triggered (Epoch {epoch + 1})")
                break

        # Load best model
        model.load_state_dict(best_model_state)

        # Test set evaluation with mixed precision and batch inference
        y_test_pred = batch_inference(
            model, X_test, scaler_X, scaler_y, device,
            batch_size=batch_size, model_name=model_type
        )

        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_r2, _ = compute_r2_safe(y_test, y_test_pred, method='per_output_mean')

        # Training history summary
        yield log_and_yield(f"\n📈 Training history summary ({len(history['train_losses'])} epochs):")
        yield log_and_yield(f"  Best validation loss: {best_val_loss:.4f} (Epoch {np.argmin(history['val_losses']) + 1})")
        yield log_and_yield(f"  Best validation R²: {max(history['val_r2']):.4f} (Epoch {np.argmax(history['val_r2']) + 1})")
        yield log_and_yield(f"  Best validation MAE: {min(history['val_mae']):.4f} (Epoch {np.argmin(history['val_mae']) + 1})")
        yield log_and_yield(f"  Final training loss: {history['train_losses'][-1]:.4f}")
        yield log_and_yield(f"  Final validation loss: {history['val_losses'][-1]:.4f}")

        yield log_and_yield(f"\n📊 Test set performance:")
        yield log_and_yield(f"  MAE: {test_mae:.6f}")
        yield log_and_yield(f"  RMSE: {test_rmse:.6f}")
        yield log_and_yield(f"  R²: {test_r2:.4f}")

        # Save model
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = f"{model_type}_{timestamp}"

        model_dir = "saved_models"
        os.makedirs(model_dir, exist_ok=True)

        model_path = os.path.join(model_dir, f"{model_name}.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': {
                'type': model_type,
                'boundary_signals': boundary_signals,
                'target_signals': target_signals,
                'config': {
                    'd_model': d_model,
                    'nhead': nhead,
                    'num_layers': num_layers,
                    'dropout': dropout,
                    'batch_size': batch_size
                },
                'data_split': {
                    'test_size': float(test_size),
                    'val_size': float(val_size),
                    'total_samples': len(X),
                    'train_samples': len(X_train),
                    'val_samples': len(X_val),
                    'test_samples': len(X_test)
                }
            },
            'training_history': history
        }, model_path)

        # Save scalers
        scaler_path = os.path.join(model_dir, f"{model_name}_scalers.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump({'X': scaler_X, 'y': scaler_y}, f)

        # Save inference config
        save_inference_config(
            model_name, model_type, model_path, scaler_path,
            boundary_signals, target_signals,
            {
                'd_model': d_model,
                'nhead': nhead,
                'num_layers': num_layers,
                'dropout': dropout
            }
        )

        # Save to global state
        global_state['trained_models'][model_name] = {
            'model': model,
            'type': model_type,
            'boundary_signals': boundary_signals,
            'target_signals': target_signals,
            'config': {
                'd_model': d_model,
                'nhead': nhead,
                'num_layers': num_layers,
                'dropout': dropout,
                'batch_size': batch_size
            },
            'model_path': model_path,
            'scaler_path': scaler_path
        }

        global_state['scalers'][model_name] = {'X': scaler_X, 'y': scaler_y}

        yield log_and_yield(f"\n✅ Model training completed and saved:")
        yield log_and_yield(f"  Model name: {model_name}")
        yield log_and_yield(f"  Model path: {model_path}")
        yield log_and_yield(f"  Scaler path: {scaler_path}")

    except Exception as e:
        error_msg = f"❌ Training failed:\n{str(e)}\n\n{traceback.format_exc()}"
        print(error_msg)
        yield error_msg


# ============================================================================
# Residual extraction functions

def get_inference_config_files():
    """
    Get list of inference config JSON files in saved_models folder

    Returns:
        List of inference config file paths
    """
    try:
        import glob

        config_files = []

        # Search in saved_models folder and subdirectories
        if os.path.exists('saved_models'):
            config_files.extend(glob.glob('saved_models/**/*_inference.json', recursive=True))

        # Sort by modification time (newest first)
        config_files = sorted(config_files, key=lambda x: os.path.getmtime(x) if os.path.exists(x) else 0, reverse=True)

        return config_files if config_files else []

    except Exception as e:
        print(f"⚠️ Error in get_inference_config_files: {e}")
        return []


def get_scalers_files():
    """
    Get list of scaler pkl files in saved_models folder

    Returns:
        List of scaler file paths
    """
    try:
        import glob

        scaler_files = []

        # Search in saved_models folder and subdirectories
        if os.path.exists('saved_models'):
            scaler_files.extend(glob.glob('saved_models/**/*_scalers.pkl', recursive=True))

        # Sort by modification time (newest first)
        scaler_files = sorted(scaler_files, key=lambda x: os.path.getmtime(x) if os.path.exists(x) else 0, reverse=True)

        return scaler_files if scaler_files else []

    except Exception as e:
        print(f"⚠️ Error in get_scalers_files: {e}")
        return []


def get_model_files():
    """
    Get list of model pth files in saved_models folder

    Returns:
        List of model file paths
    """
    try:
        import glob

        model_files = []

        # Search in saved_models folder and subdirectories
        if os.path.exists('saved_models'):
            # Get all .pth files, excluding scalers
            all_pth_files = glob.glob('saved_models/**/*.pth', recursive=True)
            # Filter out files that are not model files (e.g., optimizer states)
            model_files = [f for f in all_pth_files if not f.endswith('_scalers.pth')]

        # Sort by modification time (newest first)
        model_files = sorted(model_files, key=lambda x: os.path.getmtime(x) if os.path.exists(x) else 0, reverse=True)

        return model_files if model_files else []

    except Exception as e:
        print(f"⚠️ Error in get_model_files: {e}")
        return []


def load_model_from_inference_config_path(config_path):
    """
    Load model from inference config file path

    Args:
        config_path: Path to inference config JSON file

    Returns:
        model_name: Loaded model name
        status: Status message
    """
    try:
        if not config_path:
            return None, "❌ Please select Inference config file"

        if not os.path.exists(config_path):
            return None, f"❌ file does not exist: {config_path}"

        # Load config
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        model_name = config.get('model_name')
        if not model_name:
            return None, "❌ Config file missing model_name"

        status = f"✅ Config loaded successfully!\n\n"
        status += f"📁 Config file: {os.path.basename(config_path)}\n"
        status += f"🤖 Model name: {model_name}\n"
        status += f"📥 Number of boundary signals: {len(config.get('boundary_signals', []))}\n"
        status += f"📤 Number of target signals: {len(config.get('target_signals', []))}"

        return model_name, status

    except json.JSONDecodeError as e:
        return None, f"❌ JSON parsing failed: {str(e)}"
    except Exception as e:
        return None, f"❌ Loading failed: {str(e)}"


def load_scalers_from_path(scaler_path, model_name):
    """
    Load scalers from a pickle file path for a specific model

    Args:
        scaler_path: Path to scaler pickle file
        model_name: Model name to associate the scalers with

    Returns:
        status_msg: Status message
    """
    try:
        if not scaler_path:
            return "❌ Please select a scalers file！"

        if not model_name:
            return "❌ Please select a model first！"

        if not os.path.exists(scaler_path):
            return f"❌ File does not exist: {scaler_path}"

        # Load scalers from file
        with open(scaler_path, 'rb') as f:
            scalers = pickle.load(f)

        # Save to global state
        if 'manual_scalers' not in global_state:
            global_state['manual_scalers'] = {}

        global_state['manual_scalers'][model_name] = scalers

        success_msg = f"✅ Scalers loaded successfully!\n\n"
        success_msg += f"📌 Model name: {model_name}\n"
        success_msg += f"📊 ScalersContains: {list(scalers.keys())}\n"

        # Verify scalers have required keys
        if 'X' in scalers and 'y' in scalers:
            success_msg += f"✓ Contains required X and y scalers\n"
        else:
            success_msg += f"⚠️ Warning: scalers may be missing X or y keys\n"

        print(success_msg)
        return success_msg

    except Exception as e:
        error_msg = f"❌ ScalersLoading failed:\n{str(e)}\n\n{traceback.format_exc()}"
        print(error_msg)
        return error_msg


def load_model_from_path(model_path):
    """
    Load SST model from a .pth file path

    Args:
        model_path: Path to model .pth file

    Returns:
        model_name: Loaded model name
        status_msg: Status message
    """
    try:
        if not model_path:
            return None, "❌ Please select a model file！"

        if not os.path.exists(model_path):
            return None, f"❌ File does not exist: {model_path}"

        # Extract model name from path
        model_name = os.path.splitext(os.path.basename(model_path))[0]

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        if 'model_config' not in checkpoint:
            return None, f"❌ Model file format error: missing model_config"

        model_config = checkpoint['model_config']

        if model_config.get('type') != 'StaticSensorTransformer':
            return None, f"❌ Unsupported model type: {model_config.get('type')}"

        boundary_signals = model_config['boundary_signals']
        target_signals = model_config['target_signals']
        config = model_config['config']

        # Create model
        model = StaticSensorTransformer(
            num_boundary_sensors=len(boundary_signals),
            num_target_sensors=len(target_signals),
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_layers=config['num_layers'],
            dropout=config['dropout']
        ).to(device)

        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # Try to load scalers from checkpoint
        scalers = None
        scaler_source = "Not loaded"
        if 'scalers' in checkpoint:
            scalers = checkpoint['scalers']
            scaler_source = "Loaded from checkpoint"
            # Also save to manual_scalers for consistency
            if 'manual_scalers' not in global_state:
                global_state['manual_scalers'] = {}
            global_state['manual_scalers'][model_name] = scalers

        # Save to global state
        global_state['trained_models'][model_name] = {
            'model': model,
            'type': model_config['type'],
            'boundary_signals': boundary_signals,
            'target_signals': target_signals,
            'config': config,
            'model_path': model_path,
            'scaler_path': model_path.replace('.pth', '_scalers.pkl')
        }

        success_msg = f"✅ SSTModel loaded successfully!\n\n"
        success_msg += f"📌 Model name: {model_name}\n"
        success_msg += f"📊 Model type: {model_config['type']}\n"
        success_msg += f"🎯 Number of boundary signals: {len(boundary_signals)}\n"
        success_msg += f"📈 Number of target signals: {len(target_signals)}\n"
        success_msg += f"⚙️ Model parameters: d_model={config['d_model']}, nhead={config['nhead']}, layers={config['num_layers']}\n"
        success_msg += f"📊 Scalers status: {scaler_source}\n"

        if scalers is None:
            success_msg += f"\n⚠️ Tip: This model checkpoint does not contain scalers\n"
            success_msg += f"   If you need to extract residuals, please manually load from the Load Scalers File section below\n"

        print(success_msg)
        return model_name, success_msg

    except Exception as e:
        error_msg = f"❌ Model loading failed:\n{str(e)}\n\n{traceback.format_exc()}"
        print(error_msg)
        return None, error_msg


def load_model_from_path_ui(model_path):
    """
    UI wrapper for load_model_from_path to properly update dropdown

    Args:
        model_path: Path to model file

    Returns:
        tuple: (dropdown_update, status_msg)
    """
    model_name, status_msg = load_model_from_path(model_path)

    if model_name:
        # Update dropdown choices and set the value
        return gr.update(choices=get_available_models(), value=model_name), status_msg
    else:
        # Just return status without updating dropdown
        return gr.update(), status_msg


def load_model_from_inference_config_path_ui(config_path):
    """
    UI wrapper for load_model_from_inference_config_path to properly update dropdown

    Args:
        config_path: Path to inference config file

    Returns:
        tuple: (dropdown_update, status_msg)
    """
    model_name, status_msg = load_model_from_inference_config_path(config_path)

    if model_name:
        # Update dropdown choices and set the value
        return gr.update(choices=get_available_models(), value=model_name), status_msg
    else:
        # Just return status without updating dropdown
        return gr.update(), status_msg


def extract_residuals_ui(model_name):
    """UI function for residual extraction - full dataset inference"""
    try:
        if not model_name:
            return "❌ Please select a model！", None

        if global_state['df'] is None:
            return "❌ Please load data first！", None

        log_msg = []
        log_msg.append("=" * 80)
        log_msg.append("📊 Starting residual extraction（full dataset）")
        log_msg.append("=" * 80)

        df = global_state['df']
        log_msg.append(f"\n📈 Dataset size: {len(df):,} records")

        # Load model
        model_path = os.path.join("saved_models", f"{model_name}.pth")
        if not os.path.exists(model_path):
            return f"❌ Model file does not exist: {model_path}", None

        log_msg.append(f"\n📥 Loading model: {model_name}")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        model_config = checkpoint['model_config']
        boundary_signals = model_config['boundary_signals']
        target_signals = model_config['target_signals']

        log_msg.append(f"  Number of boundary signals: {len(boundary_signals)}")
        log_msg.append(f"  Number of target signals: {len(target_signals)}")

        # Check if signals exist in dataframe
        missing_boundary = [s for s in boundary_signals if s not in df.columns]
        missing_target = [s for s in target_signals if s not in df.columns]

        if missing_boundary:
            return f"❌ Missing boundary signals in dataset: {missing_boundary}", None
        if missing_target:
            return f"❌ Missing target signals in dataset: {missing_target}", None

        # Prepare data
        X = df[boundary_signals].values
        y = df[target_signals].values

        # Load scalers - try checkpoint first, then manual_scalers
        scalers = None
        if 'scalers' in checkpoint:
            scalers = checkpoint['scalers']
            log_msg.append(f"  ✓ Loaded scalers from checkpoint")
        elif 'manual_scalers' in global_state and model_name in global_state['manual_scalers']:
            scalers = global_state['manual_scalers'][model_name]
            log_msg.append(f"  ✓ Loaded from manually loaded scalers")
        else:
            error_msg = "❌ Scalers not found！\n\n"
            error_msg += f"No scalers in checkpoint and scalers not manually loaded。\n\n"
            error_msg += "💡 Solution:\n"
            error_msg += "1. Upload corresponding scalers.pkl file in the Load Scalers File section below\n"
            error_msg += f"2. Filename should be like: {model_name}_scalers.pkl\n"
            error_msg += "3. After clicking the 📥 Load Scalers button, click the 🔬 Extract Residuals button again\n"
            return error_msg, None

        scaler_X = scalers['X']
        scaler_y = scalers['y']

        # Load model
        config = model_config['config']
        model = StaticSensorTransformer(
            num_boundary_sensors=len(boundary_signals),
            num_target_sensors=len(target_signals),
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_layers=config['num_layers'],
            dropout=config['dropout']
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        log_msg.append(f"\n🔄 Starting inference...")

        # Batch inference
        from models.residual_tft import batch_inference
        y_pred = batch_inference(
            model, X, scaler_X, scaler_y, device,
            batch_size=512, model_name="SST"
        )

        # Calculate residuals in original space
        residuals = y - y_pred

        # Calculate metrics
        from models.residual_tft import compute_r2_safe
        r2, per_signal_r2 = compute_r2_safe(y, y_pred, method='per_output_mean')
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))

        log_msg.append(f"\n📊 Inference completed:")
        log_msg.append(f"  MAE: {mae:.6f}")
        log_msg.append(f"  RMSE: {rmse:.6f}")
        log_msg.append(f"  R²: {r2:.4f}")

        # Create residuals dataframe
        residual_cols = [f"{sig}_residual" for sig in target_signals]
        pred_cols = [f"{sig}_pred" for sig in target_signals]
        true_cols = [f"{sig}_true" for sig in target_signals]

        residuals_data = {}

        # Add boundary signals (input features) - needed for Stage2 training
        for i, sig in enumerate(boundary_signals):
            residuals_data[sig] = X[:, i]

        # Add residuals, predictions, and true values
        for i, sig in enumerate(target_signals):
            residuals_data[f"{sig}_residual"] = residuals[:, i]
            residuals_data[f"{sig}_pred"] = y_pred[:, i]
            residuals_data[f"{sig}_true"] = y[:, i]

        residuals_df = pd.DataFrame(residuals_data)

        # Save residual data
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        residual_key = f"{model_name}_{timestamp}"

        # Extract data_split info from model checkpoint
        data_split_info = model_config.get('data_split', {
            'test_size': 0.15,  # Default fallback
            'val_size': 0.15,
            'total_samples': len(residuals_df)
        })

        info = {
            'base_model_name': model_name,
            'boundary_signals': boundary_signals,
            'target_signals': target_signals,
            'residual_signals': residual_cols,
            'data_split': data_split_info,  # Save data split info for Stage2
            'metrics': {
                'mae': float(mae),
                'rmse': float(rmse),
                'r2': float(r2),
                'per_signal_r2': per_signal_r2.tolist() if isinstance(per_signal_r2, np.ndarray) else per_signal_r2
            }
        }

        global_state['residual_data'][residual_key] = {
            'data': residuals_df,
            'info': info
        }

        log_msg.append(f"\n✅ Residual extraction completed:")
        log_msg.append(f"  Residual data ID: {residual_key}")
        log_msg.append(f"  Data shape: {residuals_df.shape}")

        # Export residuals to CSV
        residuals_export_dir = "saved_models/residuals_data"
        os.makedirs(residuals_export_dir, exist_ok=True)

        csv_filename = os.path.join(residuals_export_dir, f"{residual_key}_residuals.csv")
        residuals_df.to_csv(csv_filename, index=False)

        log_msg.append(f"\n💾 Residual data exported:")
        log_msg.append(f"  CSV file: {csv_filename}")
        log_msg.append(f"  Contains: actual values, predictions, and residuals")

        # Create residual visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Residual analysis - {model_name}', fontsize=16)

        # Residual distribution
        residual_cols = info['residual_signals']
        all_residuals = residuals_df[residual_cols].values.flatten()

        axes[0, 0].hist(all_residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].set_title('Residual distribution')
        axes[0, 0].set_xlabel('residuals')
        axes[0, 0].set_ylabel('Frequency')

        # Residual Sequence
        axes[0, 1].plot(residuals_df[residual_cols[0]].values[:1000])
        axes[0, 1].set_title(f'Residual Sequence ({residual_cols[0]})')
        axes[0, 1].set_xlabel('Index')
        axes[0, 1].set_ylabel('Residuals')

        # Residual Statistics
        residual_stats = residuals_df[residual_cols].describe()
        axes[1, 0].axis('off')
        stats_text = "Residual Statistics:\n"
        stats_text += f"Mean: {residual_stats.loc['mean'].mean():.6f}\n"
        stats_text += f"Std: {residual_stats.loc['std'].mean():.6f}\n"
        stats_text += f"Min: {residual_stats.loc['min'].min():.6f}\n"
        stats_text += f"Max: {residual_stats.loc['max'].max():.6f}\n"
        axes[1, 0].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center')

        # Prediction vs Actual
        true_cols = [f"{sig}_true" for sig in info['target_signals']]
        pred_cols = [f"{sig}_pred" for sig in info['target_signals']]

        y_true = residuals_df[true_cols].values[:1000, 0]
        y_pred = residuals_df[pred_cols].values[:1000, 0]

        axes[1, 1].plot(y_true, label='True', alpha=0.7)
        axes[1, 1].plot(y_pred, label='Predicted', alpha=0.7)
        axes[1, 1].set_title('Predicted vs Actual (First 1000 samples)')
        axes[1, 1].legend()

        plt.tight_layout()

        return "\n".join(log_msg), fig

    except Exception as e:
        error_msg = f"❌ Residual extraction failed:\n{str(e)}\n\n{traceback.format_exc()}"
        print(error_msg)
        return error_msg, None


# ============================================================================
# Helper functions

def get_available_models():
    """Get list of available trained models"""
    return list(global_state['trained_models'].keys())


def get_residual_data_keys():
    """Get list of available residual data keys"""
    return list(global_state['residual_data'].keys())


def get_stage2_model_keys():
    """Get list of available Stage2 models"""
    return list(global_state['stage2_models'].keys())


def get_stage2_inference_config_files():
    """
    Get list of Stage2 inference config JSON files in saved_models/stage2_boost folder

    Returns:
        List of Stage2 inference config file paths
    """
    try:
        import glob

        config_files = []

        # Search in saved_models/stage2_boost folder
        if os.path.exists('saved_models/stage2_boost'):
            config_files.extend(glob.glob('saved_models/stage2_boost/*_inference.json', recursive=False))

        # Sort by modification time (newest first)
        config_files = sorted(config_files, key=lambda x: os.path.getmtime(x) if os.path.exists(x) else 0, reverse=True)

        return config_files if config_files else []

    except Exception as e:
        print(f"⚠️ Error in get_stage2_inference_config_files: {e}")
        return []


def get_stage2_model_files():
    """
    Get list of Stage2 model .pth files in saved_models/stage2_boost folder

    Returns:
        List of Stage2 model file paths
    """
    try:
        import glob

        model_files = []

        # Search in saved_models/stage2_boost folder
        if os.path.exists('saved_models/stage2_boost'):
            model_files.extend(glob.glob('saved_models/stage2_boost/*.pth', recursive=False))

        # Sort by modification time (newest first)
        model_files = sorted(model_files, key=lambda x: os.path.getmtime(x) if os.path.exists(x) else 0, reverse=True)

        return model_files if model_files else []

    except Exception as e:
        print(f"⚠️ Error in get_stage2_model_files: {e}")
        return []


def get_stage2_scalers_files():
    """
    Get list of Stage2 scaler .pkl files in saved_models/stage2_boost folder

    Returns:
        List of Stage2 scaler file paths
    """
    try:
        import glob

        scaler_files = []

        # Search in saved_models/stage2_boost folder
        if os.path.exists('saved_models/stage2_boost'):
            scaler_files.extend(glob.glob('saved_models/stage2_boost/*_scalers.pkl', recursive=False))

        # Sort by modification time (newest first)
        scaler_files = sorted(scaler_files, key=lambda x: os.path.getmtime(x) if os.path.exists(x) else 0, reverse=True)

        return scaler_files if scaler_files else []

    except Exception as e:
        print(f"⚠️ Error in get_stage2_scalers_files: {e}")
        return []


def load_stage2_scalers(scaler_path, stage2_model_key):
    """
    Manually load scalers for a Stage2 model

    Args:
        scaler_path: Path to scaler .pkl file
        stage2_model_key: Key of the Stage2 model in global_state

    Returns:
        status_message
    """
    try:
        if not scaler_path or not os.path.exists(scaler_path):
            return "❌ Please select a valid scaler file！"

        if not stage2_model_key:
            return "❌ Please select a Stage2 model first!"

        # Check if model exists
        if stage2_model_key not in global_state['stage2_models']:
            return f"❌ Model {stage2_model_key} does not exist! Please load Stage2 model first。"

        # Load scalers
        with open(scaler_path, 'rb') as f:
            scalers = pickle.load(f)

        # Store in global state
        global_state['stage2_scalers'][stage2_model_key] = scalers

        status_msg = f"✅ Successfully loaded scalers！\n\n"
        status_msg += f"Scaler path: {scaler_path}\n"
        status_msg += f"Associated model: {stage2_model_key}\n"
        status_msg += f"Scaler type: {type(scalers)}\n"
        if isinstance(scalers, dict):
            status_msg += f"Contains keys: {list(scalers.keys())}\n"

        return status_msg

    except Exception as e:
        return f"❌ Loading failed:\n{str(e)}\n\n{traceback.format_exc()}"


def load_stage2_from_inference_config(config_path):
    """
    Load Stage2 model from inference config JSON file

    Args:
        config_path: Path to inference config JSON file

    Returns:
        tuple: (model_key, status_message)
    """
    try:
        if not config_path or not os.path.exists(config_path):
            return None, "❌ Please select a valid inference config file！"

        with open(config_path, 'r') as f:
            config = json.load(f)

        model_name = config.get('model_name', '')
        model_path = config.get('model_path', '')
        scaler_path = config.get('scaler_path', '')

        if not model_name or not model_path:
            return None, "❌ Config file format error: missing model_name or model_path！"

        # Load model checkpoint
        if not os.path.exists(model_path):
            return None, f"❌ Model file does not exist: {model_path}"

        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        # Load scalers
        scalers = None
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scalers = pickle.load(f)
        elif 'scalers' in checkpoint:
            scalers = checkpoint['scalers']
        else:
            return None, "❌ Scalers not found! Please ensure config file contains scaler_path or model checkpoint contains scalers。"

        # Get model architecture from inference config JSON
        architecture = config.get('architecture', {})
        signals_info = config.get('signals', {})
        residual_data_key = config.get('data_config', {}).get('residual_data_key', 'unknown')

        # Get boundary and target signals from inference config JSON
        boundary_signals = signals_info.get('boundary_signals', [])
        target_signals = signals_info.get('target_signals', [])

        # Fallback: try to get from checkpoint if not in JSON
        if not boundary_signals or not target_signals:
            boundary_signals = checkpoint.get('boundary_signals', [])
            target_signals = checkpoint.get('target_signals', [])

        if not boundary_signals or not target_signals:
            return None, "❌ Both config file and checkpoint are missing boundary_signals or target_signals！"

        # Get training config from checkpoint for storing
        training_config = checkpoint.get('config', {})

        # Initialize model using architecture from inference config
        stage2_model = StaticSensorTransformer(
            num_boundary_sensors=len(boundary_signals),
            num_target_sensors=len(target_signals),
            d_model=architecture.get('d_model', 128),
            nhead=architecture.get('nhead', 8),
            num_layers=architecture.get('num_layers', 4),
            dropout=architecture.get('dropout', 0.15)
        ).to(device)

        # Load state dict
        stage2_model.load_state_dict(checkpoint['model_state_dict'])
        stage2_model.eval()

        # Store in global state with a unique key
        model_key = f"stage2_{model_name}"
        global_state['stage2_models'][model_key] = {
            'model': stage2_model,
            'config': training_config,
            'boundary_signals': boundary_signals,
            'target_signals': target_signals,
            'residual_data_key': residual_data_key,
            'model_path': model_path
        }
        global_state['stage2_scalers'][model_key] = scalers

        status_msg = f"✅ Successfully loaded Stage2 model！\n\n"
        status_msg += f"Model name: {model_name}\n"
        status_msg += f"Model key: {model_key}\n"
        status_msg += f"Model path: {model_path}\n"
        status_msg += f"Number of boundary signals: {len(boundary_signals)}\n"
        status_msg += f"Number of target signals: {len(target_signals)}\n"
        status_msg += f"Residual data key: {residual_data_key}\n\n"
        status_msg += f"Please select 'Stage2 model' : {model_key}"

        return model_key, status_msg

    except Exception as e:
        return None, f"❌ Loading failed:\n{str(e)}\n\n{traceback.format_exc()}"


def load_stage2_from_model_file(model_path):
    """
    Load Stage2 model from .pth model file

    Args:
        model_path: Path to model .pth file

    Returns:
        tuple: (model_key, status_message)
    """
    try:
        if not model_path or not os.path.exists(model_path):
            return None, "❌ Please select model file！"

        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        # Extract configurations
        # Training saves 'config' not 'model_config' at top level
        config = checkpoint.get('config', {})
        residual_data_key = checkpoint.get('residual_data_key', 'unknown')

        # Get signals - these are saved at top level in checkpoint, not in config
        boundary_signals = checkpoint.get('boundary_signals', [])
        target_signals = checkpoint.get('target_signals', [])

        if not boundary_signals or not target_signals:
            return None, f"❌ Model file missing boundary_signals or target_signals！\n\nCheckpoint keys: {list(checkpoint.keys())}"

        # Load scalers - try checkpoint first, then external file
        scalers = None
        if 'scalers' in checkpoint:
            scalers = checkpoint['scalers']
        else:
            # Try to find corresponding scaler file
            scaler_path = model_path.replace('.pth', '_scalers.pkl')
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    scalers = pickle.load(f)

        if not scalers:
            return None, "❌ Scalers not found! Please ensure model checkpoint contains scalers or corresponding *_scalers.pkl file exists。"

        # Initialize model - use config from training
        stage2_model = StaticSensorTransformer(
            num_boundary_sensors=len(boundary_signals),
            num_target_sensors=len(target_signals),
            d_model=config.get('d_model', 128),
            nhead=config.get('nhead', 8),
            num_layers=config.get('num_layers', 4),
            dropout=config.get('dropout', 0.15)
        ).to(device)

        # Load state dict
        stage2_model.load_state_dict(checkpoint['model_state_dict'])
        stage2_model.eval()

        # Extract model name from path
        model_name = os.path.basename(model_path).replace('.pth', '')
        model_key = f"stage2_{model_name}"

        # Store in global state
        global_state['stage2_models'][model_key] = {
            'model': stage2_model,
            'config': config,
            'boundary_signals': boundary_signals,
            'target_signals': target_signals,
            'residual_data_key': residual_data_key,
            'model_path': model_path
        }
        global_state['stage2_scalers'][model_key] = scalers

        status_msg = f"✅ Successfully loaded Stage2 model！\n\n"
        status_msg += f"Model name: {model_name}\n"
        status_msg += f"Model key: {model_key}\n"
        status_msg += f"Model path: {model_path}\n"
        status_msg += f"Number of boundary signals: {len(boundary_signals)}\n"
        status_msg += f"Number of target signals: {len(target_signals)}\n"
        status_msg += f"Residual data key: {residual_data_key}\n\n"
        status_msg += f"Please select 'Stage2 model: {model_key}"

        return model_key, status_msg

    except Exception as e:
        return None, f"❌ Loading failed:\n{str(e)}\n\n{traceback.format_exc()}"


def get_ensemble_model_keys():
    """Get list of available ensemble models"""
    return list(global_state['ensemble_models'].keys())


# ============================================================================
# Gradio interface creation

def create_unified_interface():
    """Create unified Gradio interface"""

    with gr.Blocks(title="Industrial Digital Twin Residual Boost Training System", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🏭 Industrial Digital Twin Residual Boost Training System
        ### Enhanced Residual Boost Training with Stage2 Model

        **New Features:**
        - ✨ Stage2 residual modelTraining
        - 🎯 Smart R² threshold selection for Ensemble Inference Model
        - 📊 Secondary inference comparison (Ensemble model vs Pure SST model)
        - 🔮 Sundial time series model predicting future residuals
        """)

        with gr.Tabs():
            # Tab 1: Data Loading
            with gr.Tab("📂 Data loading", elem_id="data_loading"):
                gr.Markdown("## Select, Upload or Create Data")

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 📁 Select Existing CSV File")
                        csv_file_selector = gr.Dropdown(
                            choices=[],  # Empty initially, populated on page load
                            label="Select CSV file from data folder",
                            info="Click Refresh to load available CSV files"
                        )
                        with gr.Row():
                            select_csv_btn = gr.Button("📂 Load Selected File", variant="primary", size="lg")
                            refresh_csv_btn = gr.Button("🔄 Refresh List", size="sm")

                        gr.Markdown("### 📤 Or Upload CSV File")
                        data_file = gr.File(label="Upload CSV File", file_types=['.csv'])
                        upload_btn = gr.Button("📥 Load Uploaded File", variant="secondary", size="lg")

                        gr.Markdown("### 🎲 Or Create Sample Data")
                        sample_btn = gr.Button("🎲 Create Sample Data", size="lg")

                    with gr.Column(scale=1):
                        data_status = gr.Textbox(label="Data Status", lines=10, interactive=False)
                        signals_display = gr.Textbox(label="Available Signals", lines=10, interactive=False)

                # Data preview table
                with gr.Row():
                    data_preview = gr.Dataframe(
                        label="📊 Data Preview (first 100 rows)",
                        interactive=False,
                        wrap=True
                    )

            # Tab 2: SST Model Training
            with gr.Tab("🎯 SST Model Training", elem_id="sst_training"):
                gr.Markdown("## Train Static Sensor Transformer (SST)")

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 🎛️ Signal Selection")

                        # JSON config loading
                        with gr.Accordion("📁 Load Signal Config from JSON", open=False):
                            json_config_selector = gr.Dropdown(
                                choices=get_available_json_configs(),
                                label="Select JSON config from data folder",
                                info="Or manually upload JSON file"
                            )
                            with gr.Row():
                                load_json_btn = gr.Button("📂 Load Config", size="sm", variant="secondary")
                                refresh_json_btn = gr.Button("🔄 Refresh", size="sm")
                            json_upload = gr.File(
                                label="Upload JSON Config File",
                                file_types=['.json'],
                                type="filepath"
                            )
                            json_status = gr.Textbox(
                                label="Config Loading Status",
                                lines=3,
                                interactive=False
                            )

                        boundary_signals_static = gr.Dropdown(
                            choices=[], label="Boundary Signals (Input)", multiselect=True
                        )
                        target_signals_static = gr.Dropdown(
                            choices=[], label="Target Signals (Output)", multiselect=True
                        )

                        gr.Markdown("### 🏗️ Model Architecture")
                        with gr.Row():
                            d_model_static = gr.Slider(32, 1280, 256, 32, label="Model Dimension")
                            nhead_static = gr.Slider(2, 80, 16, 2, label="Number of Attention Heads")
                        with gr.Row():
                            num_layers_static = gr.Slider(1, 30, 6, 1, label="Number of Transformer Layers")
                            dropout_static = gr.Slider(0, 0.5, 0.1, 0.05, label="Dropout Rate")

                        gr.Markdown("### 🎯 Training Parameters")
                        with gr.Row():
                            epochs_static = gr.Slider(10, 250, 50, 10, label="Training Epochs")
                            batch_size_static = gr.Slider(16, 2560, 512, 16, label="Batch size")
                        with gr.Row():
                            lr_static = gr.Number(value=0.00003, label="Learning rate")
                            weight_decay_static = gr.Number(value=1e-5, label="Weight Decay")

                        gr.Markdown("### ⚙️ Optimizer Settings")
                        with gr.Row():
                            grad_clip_norm_static = gr.Slider(0.1, 5.0, 1.0, 0.1, label="Gradient Clipping")
                            scheduler_patience_static = gr.Slider(1, 15, 8, 1, label="Learning Rate Scheduler Patience")
                        scheduler_factor_static = gr.Slider(0.1, 0.9, 0.5, 0.1, label="Learning Rate Decay Factor")

                        gr.Markdown("### 🔀 Data split")
                        with gr.Row():
                            test_size_static = gr.Slider(0.1, 0.3, 0.15, 0.05, label="Test Set Ratio")
                            val_size_static = gr.Slider(0.1, 0.3, 0.15, 0.05, label="Validation Set Ratio")

                        train_btn_static = gr.Button("▶️ Start Training SST", variant="primary", size="lg")
                        stop_btn_tab2 = gr.Button("⏹️ Stop Training", variant="stop", size="lg")

                    with gr.Column(scale=1):
                        gr.Markdown("### 📊 Training Log")
                        training_log_static = gr.Textbox(
                            label="Training Progress",
                            lines=30,
                            autoscroll=True,
                            interactive=False
                        )

            # Tab 3: Residual Extraction
            with gr.Tab("🔬 Residual Extraction", elem_id="residual_extraction"):
                gr.Markdown("## Extract residuals from trained SST model")
                gr.Markdown("Inference on entire dataset to generate residuals for Stage2 training")

                with gr.Row():
                    with gr.Column(scale=1):
                        model_selector = gr.Dropdown(
                            choices=get_available_models(),
                            label="Select SST Model"
                        )
                        refresh_models_btn = gr.Button("🔄 Refresh Model List", size="sm")

                        gr.Markdown("### 📤 Inference Config File (Optional)")
                        gr.Markdown("Optionally select saved inference config to load model")

                        inference_config_selector = gr.Dropdown(
                            choices=get_inference_config_files(),
                            label="Select inference config from saved_models folder",
                            info="Select *_inference.json file"
                        )
                        with gr.Row():
                            load_inference_btn = gr.Button("📥 Load Config", size="sm", variant="secondary")
                            refresh_inference_btn = gr.Button("🔄 Refresh Config List", size="sm")

                        inference_load_status = gr.Textbox(label="Config Loading Status", lines=3, interactive=False)

                        gr.Markdown("### 🤖 Load SST Model File (Optional)")
                        gr.Markdown("Select .pth model file from saved_models folder to load directly")
                        model_file_selector = gr.Dropdown(
                            choices=get_model_files(),
                            label="Select model file from saved_models folder",
                            info="Select *.pth file"
                        )
                        with gr.Row():
                            load_model_file_btn = gr.Button("📥 Loading model", size="sm", variant="secondary")
                            refresh_model_files_btn = gr.Button("🔄 Refresh Model List", size="sm")
                        model_load_status = gr.Textbox(label="Model Loading Status", lines=3, interactive=False)

                        gr.Markdown("### 📊 Load Scalers File (Optional)")
                        gr.Markdown("If model checkpoint does not contain scalers, select from saved_models folder")
                        scalers_file_selector = gr.Dropdown(
                            choices=get_scalers_files(),
                            label="Select scalers file from saved_models folder",
                            info="Select *_scalers.pkl file"
                        )
                        with gr.Row():
                            load_scalers_btn = gr.Button("📥 Load Scalers", size="sm", variant="secondary")
                            refresh_scalers_btn = gr.Button("🔄 Refresh Scalers List", size="sm")
                        scalers_load_status = gr.Textbox(label="ScalersLoading status", lines=3, interactive=False)

                        extract_btn = gr.Button("🔬 Extract Residuals（full dataset）", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        residual_status = gr.Textbox(label="Residual Extraction Status", lines=20, interactive=False)
                        residual_plot = gr.Plot(label="Residual Visualization")

                # Event binding
                refresh_models_btn.click(
                    fn=lambda: gr.update(choices=get_available_models()),
                    outputs=[model_selector]
                )

                # Refresh inference config list
                refresh_inference_btn.click(
                    fn=lambda: gr.update(choices=get_inference_config_files()),
                    outputs=[inference_config_selector]
                )

                # Load inference config from selector
                load_inference_btn.click(
                    fn=load_model_from_inference_config_path_ui,
                    inputs=[inference_config_selector],
                    outputs=[model_selector, inference_load_status]
                )

                # Refresh model file list
                refresh_model_files_btn.click(
                    fn=lambda: gr.update(choices=get_model_files()),
                    outputs=[model_file_selector]
                )

                # Load model file
                load_model_file_btn.click(
                    fn=load_model_from_path_ui,
                    inputs=[model_file_selector],
                    outputs=[model_selector, model_load_status]
                )

                # Refresh scalers file list
                refresh_scalers_btn.click(
                    fn=lambda: gr.update(choices=get_scalers_files()),
                    outputs=[scalers_file_selector]
                )

                # Load scalers file
                load_scalers_btn.click(
                    fn=load_scalers_from_path,
                    inputs=[scalers_file_selector, model_selector],
                    outputs=[scalers_load_status]
                )

                # Extract residuals (full dataset)
                extract_btn.click(
                    fn=extract_residuals_ui,
                    inputs=[model_selector],
                    outputs=[residual_status, residual_plot]
                )

            # Tab 4: Stage2 BoostTraining
            with gr.Tab("🚀 Stage2 BoostTraining", elem_id="stage2_training"):
                gr.Markdown("## Train Stage2residual model")
                gr.Markdown("Train Stage2 model based on extracted residuals to further improve prediction accuracy")

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 📊 Data Selection")
                        residual_data_selector_stage2 = gr.Dropdown(
                            choices=get_residual_data_keys(),
                            label="Select Residual Data"
                        )
                        refresh_residual_btn_stage2 = gr.Button("🔄 Refresh", size="sm")

                        gr.Markdown("### 🏗️ Model Architecture")
                        with gr.Row():
                            d_model_stage2 = gr.Slider(32, 640, 256, 32, label="Model Dimension")
                            nhead_stage2 = gr.Slider(2, 40, 16, 2, label="Number of Attention Heads")
                        with gr.Row():
                            num_layers_stage2 = gr.Slider(1, 20, 6, 1, label="Number of Transformer Layers")
                            dropout_stage2 = gr.Slider(0, 0.5, 0.15, 0.05, label="Dropout Rate")

                        gr.Markdown("### 🎯 Training Parameters")
                        with gr.Row():
                            epochs_stage2 = gr.Slider(10, 400, 80, 10, label="Training Epochs")
                            batch_size_stage2 = gr.Slider(16, 2560, 512, 16, label="Batch size")
                        with gr.Row():
                            lr_stage2 = gr.Number(value=0.000001, label="Learning rate")
                            weight_decay_stage2 = gr.Number(value=5e-6, label="Weight Decay")

                        gr.Markdown("### ⚙️ Optimizer Settings")
                        with gr.Row():
                            grad_clip_stage2 = gr.Slider(0.1, 2.5, 0.5, 0.1, label="Gradient Clipping")
                            scheduler_patience_stage2 = gr.Slider(1, 75, 10, 1, label="Learning Rate Scheduler Patience")
                        scheduler_factor_stage2 = gr.Slider(0.1, 0.9, 0.7, 0.1, label="Learning Rate Decay Factor")

                        gr.Markdown("### 🔀 Data split")
                        with gr.Row():
                            test_size_stage2 = gr.Slider(0.1, 0.3, 0.15, 0.05, label="Test Set Ratio")
                            val_size_stage2 = gr.Slider(0.1, 0.3, 0.15, 0.05, label="Validation Set Ratio")

                        train_stage2_btn = gr.Button("🚀 Start Training Stage2", variant="primary", size="lg")
                        stop_btn_tab4 = gr.Button("⏹️ Stop Training", variant="stop", size="lg")

                    with gr.Column(scale=1):
                        gr.Markdown("### 📊 Training Log")
                        stage2_training_log = gr.Textbox(
                            label="Training Progress",
                            lines=30,
                            autoscroll=True,
                            interactive=False
                        )

                # Stage2Training
                def train_stage2_ui_generator(residual_data_key, d_model, nhead, num_layers, dropout,
                                             epochs, batch_size, lr, weight_decay, grad_clip,
                                             scheduler_patience, scheduler_factor,
                                             test_size, val_size, progress=gr.Progress()):
                    """Generator function for real-time log updates"""

                    config = {
                        'd_model': int(d_model),
                        'nhead': int(nhead),
                        'num_layers': int(num_layers),
                        'dropout': float(dropout),
                        'epochs': int(epochs),
                        'batch_size': int(batch_size),
                        'lr': float(lr),
                        'weight_decay': float(weight_decay),
                        'grad_clip': float(grad_clip),
                        'scheduler_patience': int(scheduler_patience),
                        'scheduler_factor': float(scheduler_factor),
                        'test_size': float(test_size),
                        'val_size': float(val_size),
                        'early_stop_patience': 25
                    }

                    # Yield initial message
                    yield "🚀 Initializing Stage2 training...\n"

                    # Import here to avoid circular dependency
                    import time

                    # Call training function and yield intermediate results
                    try:
                        # Run the training in a way that allows yielding
                        # This is a workaround since train_stage2_boost_model is not a generator
                        final_status = ""
                        for update in train_stage2_boost_model_generator(residual_data_key, config, progress):
                            yield update
                            final_status = update
                    except Exception as e:
                        yield f"❌ Training failed:\n{str(e)}\n\n{traceback.format_exc()}"

                refresh_residual_btn_stage2.click(
                    fn=lambda: gr.update(choices=get_residual_data_keys()),
                    outputs=[residual_data_selector_stage2]
                )

                train_stage2_btn.click(
                    fn=train_stage2_ui_generator,
                    inputs=[
                        residual_data_selector_stage2,
                        d_model_stage2, nhead_stage2, num_layers_stage2, dropout_stage2,
                        epochs_stage2, batch_size_stage2, lr_stage2,
                        weight_decay_stage2, grad_clip_stage2,
                        scheduler_patience_stage2, scheduler_factor_stage2,
                        test_size_stage2, val_size_stage2
                    ],
                    outputs=[stage2_training_log]
                )

            # Tab 5: Ensemble Inference Model Generation
            with gr.Tab("🎯 Ensemble Inference Model", elem_id="ensemble_model"):
                gr.Markdown("## Generate Ensemble Inference Model (Delta R² Strategy)")
                gr.Markdown("""
                **Optimized Strategy**：
                - Use Test set data to evaluate Delta R² = R²_ensemble - R²_stage1 for each signal
                - Only apply Stage2 correction to signals with Delta R² > threshold (indicating Stage2 can improve performance)
                - Other signals use only Stage1 prediction
                - Automatically generate analysis report and CSV summary for all signals
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 🔧 Model Selection")
                        base_model_selector = gr.Dropdown(
                            choices=get_available_models(),
                            label="Select Base SST Model"
                        )
                        stage2_model_selector = gr.Dropdown(
                            choices=get_stage2_model_keys(),
                            label="Select Stage2 Model"
                        )
                        refresh_ensemble_btn = gr.Button("🔄 Refresh", size="sm")

                        gr.Markdown("### 📤 Load Stage2 Model (Optional)")
                        gr.Markdown("Load pre-trained Stage2 model from saved_models/stage2_boost folder")

                        stage2_inference_config_selector = gr.Dropdown(
                            choices=get_stage2_inference_config_files(),
                            label="Select Stage2 Inference Config File",
                            info="Select *_inference.json file"
                        )
                        with gr.Row():
                            load_stage2_inference_btn = gr.Button("📥 Load Config", size="sm", variant="secondary")
                            refresh_stage2_inference_btn = gr.Button("🔄 Refresh Config", size="sm")

                        stage2_model_file_selector = gr.Dropdown(
                            choices=get_stage2_model_files(),
                            label="Select Stage2 Model File",
                            info="Select *.pth file"
                        )
                        with gr.Row():
                            load_stage2_model_btn = gr.Button("📥 Loading model", size="sm", variant="secondary")
                            refresh_stage2_model_btn = gr.Button("🔄 Refresh Models", size="sm")

                        gr.Markdown("### 📊 Load Stage2 Scalers (Optional)")
                        gr.Markdown("If model loading fails due to missing scalers, manually load here")
                        stage2_scalers_selector = gr.Dropdown(
                            choices=get_stage2_scalers_files(),
                            label="Select Stage2 Scalers File",
                            info="Select *_scalers.pkl file"
                        )
                        with gr.Row():
                            load_stage2_scalers_btn = gr.Button("📥 Load Scalers", size="sm", variant="secondary")
                            refresh_stage2_scalers_btn = gr.Button("🔄 RefreshScalers", size="sm")

                        stage2_load_status = gr.Textbox(label="Stage2Loading status", lines=5, interactive=False)

                        gr.Markdown("### 🎚️ Delta R² Threshold Settings")
                        delta_r2_threshold_slider = gr.Slider(
                            0.0, 0.5, 0.05, 0.01,
                            label="Delta R² Threshold",
                            info="Only for signals with Delta R² > Threshold will be selected in Ensemble model（0.05 = 5% R2 boost）"
                        )

                        generate_ensemble_btn = gr.Button("🎯 Generate Ensemble model", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        ensemble_status = gr.Textbox(
                            label="Generation status",
                            lines=30,
                            autoscroll=True,
                            interactive=False
                        )

                # Add Visualization
                with gr.Row():
                    ensemble_visualization = gr.Plot(
                        label="Ensemble Model Analysis Visualization",
                        show_label=True
                    )

                def generate_ensemble_ui(base_model_name, stage2_model_name, delta_r2_threshold):
                    if not base_model_name or not stage2_model_name:
                        return "❌ Please select Base model and Stage2 model！", None

                    status_msg, ensemble_info, fig = compute_signal_r2_and_select_threshold(
                        base_model_name, stage2_model_name, delta_r2_threshold
                    )
                    return status_msg, fig

                def load_stage2_inference_ui(config_path):
                    """UI wrapper for loading Stage2 from inference config"""
                    model_key, status_msg = load_stage2_from_inference_config(config_path)
                    if model_key:
                        return gr.update(choices=get_stage2_model_keys(), value=model_key), status_msg
                    else:
                        return gr.update(), status_msg

                def load_stage2_model_ui(model_path):
                    """UI wrapper for loading Stage2 from model file"""
                    model_key, status_msg = load_stage2_from_model_file(model_path)
                    if model_key:
                        return gr.update(choices=get_stage2_model_keys(), value=model_key), status_msg
                    else:
                        return gr.update(), status_msg

                # Event bindings for Stage2 model loading
                refresh_stage2_inference_btn.click(
                    fn=lambda: gr.update(choices=get_stage2_inference_config_files()),
                    outputs=[stage2_inference_config_selector]
                )

                load_stage2_inference_btn.click(
                    fn=load_stage2_inference_ui,
                    inputs=[stage2_inference_config_selector],
                    outputs=[stage2_model_selector, stage2_load_status]
                )

                refresh_stage2_model_btn.click(
                    fn=lambda: gr.update(choices=get_stage2_model_files()),
                    outputs=[stage2_model_file_selector]
                )

                load_stage2_model_btn.click(
                    fn=load_stage2_model_ui,
                    inputs=[stage2_model_file_selector],
                    outputs=[stage2_model_selector, stage2_load_status]
                )

                refresh_stage2_scalers_btn.click(
                    fn=lambda: gr.update(choices=get_stage2_scalers_files()),
                    outputs=[stage2_scalers_selector]
                )

                load_stage2_scalers_btn.click(
                    fn=load_stage2_scalers,
                    inputs=[stage2_scalers_selector, stage2_model_selector],
                    outputs=[stage2_load_status]
                )

                refresh_ensemble_btn.click(
                    fn=lambda: (gr.update(choices=get_available_models()),
                                gr.update(choices=get_stage2_model_keys())),
                    outputs=[base_model_selector, stage2_model_selector]
                )

                generate_ensemble_btn.click(
                    fn=generate_ensemble_ui,
                    inputs=[base_model_selector, stage2_model_selector, delta_r2_threshold_slider],
                    outputs=[ensemble_status, ensemble_visualization]
                )

            # Tab 6: Reinference Comparison
            with gr.Tab("📊 Reinference Comparison", elem_id="reinference_comparison"):
                gr.Markdown("## Reinference Comparison")
                gr.Markdown("Selectionindexrange，Comparing the performance of Ensemble model and stage1 SST model")

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 🎯 Model Selection")
                        ensemble_selector_reinf = gr.Dropdown(
                            choices=get_ensemble_model_keys(),
                            label="SelectionEnsemble model"
                        )
                        refresh_reinf_btn = gr.Button("🔄 Refresh", size="sm")

                        gr.Markdown("### 📏 IndexrangeSelection")
                        with gr.Row():
                            reinf_start_idx = gr.Number(value=0, label="StartIndex", precision=0)
                            reinf_end_idx = gr.Number(value=1000, label="EndIndex", precision=0)

                        compare_reinf_btn = gr.Button("📊 Run inferencing", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        reinf_status = gr.Textbox(
                            label="Comparison results",
                            lines=20,
                            autoscroll=True,
                            interactive=False
                        )
                        reinf_plot = gr.Plot(label="Performance comparisonVisualization")

                def compare_reinference_ui(ensemble_name, start_idx, end_idx):
                    """
                    Compare reinference results with visualization of ALL signals
                    and export CSV with predictions, actual values, and R² scores
                    """
                    if not ensemble_name:
                        return "❌ please select ensemble model！", None

                    if ensemble_name not in global_state['ensemble_models']:
                        return "❌ Ensemble modeldoes not exist！", None

                    try:
                        ensemble_info = global_state['ensemble_models'][ensemble_name]

                        # Get prediction data
                        y_true = ensemble_info['predictions']['y_true']
                        y_pred_base = ensemble_info['predictions']['y_pred_base']
                        y_pred_ensemble = ensemble_info['predictions']['y_pred_ensemble']

                        # Get signal names
                        signal_names = ensemble_info.get('signals', {}).get('target', [])
                        num_signals = y_true.shape[1]
                        if not signal_names or len(signal_names) != num_signals:
                            signal_names = [f'Signal_{i+1}' for i in range(num_signals)]

                        # Slicing
                        start_idx = max(0, int(start_idx))
                        end_idx = min(len(y_true), int(end_idx))

                        y_true_seg = y_true[start_idx:end_idx]
                        y_pred_base_seg = y_pred_base[start_idx:end_idx]
                        y_pred_ensemble_seg = y_pred_ensemble[start_idx:end_idx]

                        # Calculate overall performance
                        mae_base = mean_absolute_error(y_true_seg, y_pred_base_seg)
                        mae_ensemble = mean_absolute_error(y_true_seg, y_pred_ensemble_seg)
                        rmse_base = np.sqrt(mean_squared_error(y_true_seg, y_pred_base_seg))
                        rmse_ensemble = np.sqrt(mean_squared_error(y_true_seg, y_pred_ensemble_seg))
                        r2_base_overall, _ = compute_r2_safe(y_true_seg, y_pred_base_seg, method='per_output_mean')
                        r2_ensemble_overall, _ = compute_r2_safe(y_true_seg, y_pred_ensemble_seg, method='per_output_mean')

                        # Calculate R² for each signal
                        r2_base_per_signal = []
                        r2_ensemble_per_signal = []
                        for i in range(num_signals):
                            r2_base_i, _ = compute_r2_safe(
                                y_true_seg[:, i:i+1],
                                y_pred_base_seg[:, i:i+1],
                                method='per_output_mean'
                            )
                            r2_ensemble_i, _ = compute_r2_safe(
                                y_true_seg[:, i:i+1],
                                y_pred_ensemble_seg[:, i:i+1],
                                method='per_output_mean'
                            )
                            r2_base_per_signal.append(r2_base_i)
                            r2_ensemble_per_signal.append(r2_ensemble_i)

                        improvement_mae = (mae_base - mae_ensemble) / mae_base * 100 if mae_base != 0 else 0
                        improvement_rmse = (rmse_base - rmse_ensemble) / rmse_base * 100 if rmse_base != 0 else 0

                        status = f"📊 Second inferecing results comparison\n"
                        status += f"=" * 60 + "\n\n"
                        status += f"📏 Indexrange: [{start_idx}, {end_idx})\n"
                        status += f"📈 Number of samples: {len(y_true_seg):,}\n"
                        status += f"🎯 Number of output signals: {num_signals}\n\n"

                        status += f"total Performance comparison:\n"
                        status += f"{'metrics':<15} {'Stage1':>15} {'Ensemble':>15} {'improvement':>12}\n"
                        status += "-" * 60 + "\n"
                        status += f"{'MAE':<15} {mae_base:>15.6f} {mae_ensemble:>15.6f} {improvement_mae:>11.2f}%\n"
                        status += f"{'RMSE':<15} {rmse_base:>15.6f} {rmse_ensemble:>15.6f} {improvement_rmse:>11.2f}%\n"
                        status += f"{'R²':<15} {r2_base_overall:>15.4f} {r2_ensemble_overall:>15.4f}\n\n"

                        # Export CSV with all signal predictions and R² scores
                        os.makedirs('saved_models/reinference_results', exist_ok=True)
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        csv_filename = f'saved_models/reinference_results/{ensemble_name}_idx{start_idx}-{end_idx}_{timestamp}.csv'

                        # Build CSV data
                        csv_data = {'sample_index': np.arange(start_idx, end_idx)}

                        for i, signal_name in enumerate(signal_names):
                            csv_data[f'{signal_name}_true'] = y_true_seg[:, i]
                            csv_data[f'{signal_name}_stage1_pred'] = y_pred_base_seg[:, i]
                            csv_data[f'{signal_name}_ensemble_pred'] = y_pred_ensemble_seg[:, i]

                        df_export = pd.DataFrame(csv_data)
                        df_export.to_csv(csv_filename, index=False)

                        # Also save R² scores summary
                        r2_summary_filename = f'saved_models/reinference_results/{ensemble_name}_R2_summary_{timestamp}.csv'
                        r2_summary_data = {
                            'signal_name': signal_names,
                            'r2_stage1': r2_base_per_signal,
                            'r2_ensemble': r2_ensemble_per_signal,
                            'delta_r2': [r2_ensemble_per_signal[i] - r2_base_per_signal[i] for i in range(num_signals)]
                        }
                        df_r2_summary = pd.DataFrame(r2_summary_data)
                        df_r2_summary.to_csv(r2_summary_filename, index=False)

                        status += f"✅ CSV saved:\n"
                        status += f"   📁 Prediction data: {csv_filename}\n"
                        status += f"   📁 R²summary: {r2_summary_filename}\n\n"

                        status += f"results of each signalR²:\n"
                        status += f"{'signals':<20} {'Stage1 R²':>12} {'Ensemble R²':>12} {'Delta R²':>12}\n"
                        status += "-" * 60 + "\n"
                        for i, signal_name in enumerate(signal_names):
                            delta_r2 = r2_ensemble_per_signal[i] - r2_base_per_signal[i]
                            status += f"{signal_name:<20} {r2_base_per_signal[i]:>12.4f} {r2_ensemble_per_signal[i]:>12.4f} {delta_r2:>12.4f}\n"

                        # Create visualization - show ALL signals
                        # Layout: 1 summary row + rows for individual signals (4 per row)
                        signals_per_row = 4
                        num_signal_rows = (num_signals + signals_per_row - 1) // signals_per_row
                        total_rows = 1 + num_signal_rows

                        fig_height = 5 + num_signal_rows * 4
                        fig = plt.figure(figsize=(24, fig_height))
                        gs = fig.add_gridspec(total_rows, signals_per_row, hspace=0.35, wspace=0.3)

                        # Set font to avoid encoding issues
                        plt.rcParams['font.family'] = 'DejaVu Sans'

                        fig.suptitle(f'Reinference Comparison - {ensemble_name} (All {num_signals} Signals)',
                                   fontsize=16, fontweight='bold', y=0.995)

                        # Row 0: R² comparison for all signals
                        ax_summary = fig.add_subplot(gs[0, :])
                        x = np.arange(num_signals)
                        width = 0.35

                        bars1 = ax_summary.bar(x - width/2, r2_base_per_signal, width,
                                             label='Stage1', alpha=0.8, color='skyblue')
                        bars2 = ax_summary.bar(x + width/2, r2_ensemble_per_signal, width,
                                             label='Ensemble', alpha=0.8, color='orange')

                        ax_summary.set_xlabel('Signals', fontsize=11)
                        ax_summary.set_ylabel('R² Score', fontsize=11)
                        ax_summary.set_title('R² Comparison for All Signals', fontsize=13, fontweight='bold')
                        ax_summary.set_xticks(x)
                        ax_summary.set_xticklabels(signal_names, rotation=45, ha='right', fontsize=9)
                        ax_summary.legend(fontsize=10)
                        ax_summary.grid(axis='y', alpha=0.3)
                        ax_summary.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

                        # Add R² annotations
                        for idx, (bar1, bar2) in enumerate(zip(bars1, bars2)):
                            height1 = bar1.get_height()
                            height2 = bar2.get_height()
                            ax_summary.text(bar1.get_x() + bar1.get_width()/2., height1,
                                          f'{r2_base_per_signal[idx]:.3f}',
                                          ha='center', va='bottom', fontsize=7)
                            ax_summary.text(bar2.get_x() + bar2.get_width()/2., height2,
                                          f'{r2_ensemble_per_signal[idx]:.3f}',
                                          ha='center', va='bottom', fontsize=7)

                        # Rows 1+: Individual signal prediction plots (ALL signals)
                        plot_samples = min(300, len(y_true_seg))

                        for idx in range(num_signals):
                            row = 1 + idx // signals_per_row
                            col = idx % signals_per_row

                            ax = fig.add_subplot(gs[row, col])

                            signal_name = signal_names[idx]

                            # Plot predictions vs true values
                            ax.plot(y_true_seg[:plot_samples, idx], label='True',
                                  alpha=0.8, linewidth=1.5, color='green')
                            ax.plot(y_pred_base_seg[:plot_samples, idx], label='Stage1',
                                  alpha=0.7, linewidth=1.2, color='skyblue')
                            ax.plot(y_pred_ensemble_seg[:plot_samples, idx], label='Ensemble',
                                  alpha=0.7, linewidth=1.2, color='orange')

                            # Add R² scores in title
                            ax.set_title(f'{signal_name}\nStage1 R²={r2_base_per_signal[idx]:.4f}, Ensemble R²={r2_ensemble_per_signal[idx]:.4f}',
                                       fontsize=9, fontweight='bold')
                            ax.legend(fontsize=7, loc='best')
                            ax.set_xlabel('Sample Index', fontsize=8)
                            ax.set_ylabel('Value', fontsize=8)
                            ax.grid(alpha=0.3)
                            ax.tick_params(labelsize=7)

                        plt.tight_layout()

                        return status, fig

                    except Exception as e:
                        error_msg = f"❌ inferecing failed:\n{str(e)}\n\n{traceback.format_exc()}"
                        return error_msg, None

                refresh_reinf_btn.click(
                    fn=lambda: gr.update(choices=get_ensemble_model_keys()),
                    outputs=[ensemble_selector_reinf]
                )

                compare_reinf_btn.click(
                    fn=compare_reinference_ui,
                    inputs=[ensemble_selector_reinf, reinf_start_idx, reinf_end_idx],
                    outputs=[reinf_status, reinf_plot]
                )

        # Footer info
        gr.Markdown("""
        ---
        ## 📖 Usage Flow

        ### Complete Process
        1️⃣ **Data Loading** → Upload CSV or Create sample data
        2️⃣ **SST Model Training** → Training Static Sensor Mapping Transformer
        3️⃣ **Residual Extraction** → Extract prediction residuals from the SST model
        4️⃣ **Stage2 Training** → Training Stage2 residual model
        5️⃣ **Generate Ensemble model** → Intelligent $R^2$ Threshold Selection, generate Ensemble Inference Model
        6️⃣ **Reinference Comparison** → Compare the performance improvement of the Ensemble model versus the SST model

        **🎯 Innovation Points**:
        - ✨ Stage2 Boost Architecture: Targeted improvement for low $R^2$ signals
        - 🎯 Intelligent Threshold Selection: Automatically decide which signals require Stage2
        - 📊 Ensemble Inference Model: Optimal combination of SST and Stage2
        - 📈 Full Signal Visualization: Independent comparative analysis for every output signal
        """)

        # Auto refresh dropdowns on page load
        # Initial load: populate dropdowns and check for pre-loaded data
        def initial_load():
            """Load initial state including pre-loaded data from Colab"""
            # Get dropdown choices
            models = get_available_models()
            residual_keys = get_residual_data_keys()
            stage2_keys = get_stage2_model_keys()
            ensemble_keys = get_ensemble_model_keys()

            # Get available CSV files (safe - won't break interface)
            csv_files = get_available_csv_files()

            # Get available JSON config files
            json_configs = get_available_json_configs()

            # Get available inference config files
            inference_configs = get_inference_config_files()

            # Check for pre-loaded data (but don't auto-load)
            status, preview_df, signals = check_preloaded_data()

            # Get column choices if data exists
            if global_state.get('df') is not None:
                cols = list(global_state['df'].columns)
            else:
                cols = []

            return (
                gr.update(choices=models),
                gr.update(choices=residual_keys),
                gr.update(choices=stage2_keys),
                gr.update(choices=ensemble_keys),
                gr.update(choices=csv_files),  # Populate CSV file selector
                gr.update(choices=json_configs),  # Populate JSON config selector
                gr.update(choices=inference_configs),  # Populate inference config selector
                status, signals, preview_df,
                gr.update(choices=cols), gr.update(choices=cols)
            )

        demo.load(
            fn=initial_load,
            outputs=[
                model_selector, residual_data_selector_stage2,
                stage2_model_selector, ensemble_selector_reinf,
                csv_file_selector,  # CSV file selector
                json_config_selector,  # JSON config selector (Tab2)
                inference_config_selector,  # Inference config selector (Tab3)
                data_status, signals_display, data_preview,
                boundary_signals_static, target_signals_static
            ]
        )

        # Data loading events
        def load_data_and_update(file_obj):
            status, preview_df, signals = load_data_from_csv(file_obj)
            if preview_df is not None:
                cols = list(global_state['df'].columns)
                return (
                    status, signals, preview_df,
                    gr.update(choices=cols), gr.update(choices=cols)
                )
            return (
                status, signals, None,
                gr.update(choices=[]), gr.update(choices=[])
            )

        def create_sample_and_update():
            status, preview_df, signals = create_sample_data()
            if preview_df is not None:
                cols = list(global_state['df'].columns)
                return (
                    status, signals, preview_df,
                    gr.update(choices=cols), gr.update(choices=cols)
                )
            return (
                status, signals, None,
                gr.update(choices=[]), gr.update(choices=[])
            )

        # CSV file selector event - load from data/ folder
        def load_from_selector_and_update(csv_path):
            status, preview_df, signals = load_csv_from_path(csv_path)
            if preview_df is not None:
                cols = list(global_state['df'].columns)
                return (
                    status, signals, preview_df,
                    gr.update(choices=cols), gr.update(choices=cols)
                )
            return (
                status, signals, None,
                gr.update(choices=[]), gr.update(choices=[])
            )

        select_csv_btn.click(
            fn=load_from_selector_and_update,
            inputs=[csv_file_selector],
            outputs=[
                data_status, signals_display, data_preview,
                boundary_signals_static, target_signals_static
            ]
        )

        # Refresh CSV file list
        refresh_csv_btn.click(
            fn=lambda: gr.update(choices=get_available_csv_files()),
            outputs=[csv_file_selector]
        )

        # Upload button event - load from uploaded file
        upload_btn.click(
            fn=load_data_and_update,
            inputs=[data_file],
            outputs=[
                data_status, signals_display, data_preview,
                boundary_signals_static, target_signals_static
            ]
        )

        # Sample button event - create sample data
        sample_btn.click(
            fn=create_sample_and_update,
            outputs=[
                data_status, signals_display, data_preview,
                boundary_signals_static, target_signals_static
            ]
        )

        # JSON config loading event
        def load_json_from_selector(json_path):
            """Load JSON config from dropdown selector"""
            if not json_path:
                return gr.update(), gr.update(), "⚠️ please select json config file"
            boundary, target, status = load_signals_config_from_json(json_path)
            return gr.update(value=boundary), gr.update(value=target), status

        def load_json_from_upload(json_file):
            """Load JSON config from uploaded file"""
            if not json_file:
                return gr.update(), gr.update(), "⚠️ please upload json config file"
            boundary, target, status = load_signals_config_from_json(json_file)
            return gr.update(value=boundary), gr.update(value=target), status

        # Load JSON from selector
        load_json_btn.click(
            fn=load_json_from_selector,
            inputs=[json_config_selector],
            outputs=[boundary_signals_static, target_signals_static, json_status]
        )

        # Load JSON from uploaded file
        json_upload.change(
            fn=load_json_from_upload,
            inputs=[json_upload],
            outputs=[boundary_signals_static, target_signals_static, json_status]
        )

        # Refresh JSON file list
        refresh_json_btn.click(
            fn=lambda: gr.update(choices=get_available_json_configs()),
            outputs=[json_config_selector]
        )

        # Training button binding
        train_btn_static.click(
            fn=train_base_model_ui_generator,
            inputs=[
                boundary_signals_static, target_signals_static,
                gr.Textbox(value="StaticSensorTransformer", visible=False),
                epochs_static, batch_size_static, lr_static,
                d_model_static, nhead_static, num_layers_static, dropout_static,
                test_size_static, val_size_static,
                weight_decay_static, scheduler_patience_static, scheduler_factor_static, grad_clip_norm_static,
                gr.State(value=None), gr.State(value=False)
            ],
            outputs=[training_log_static]
        )

        # Stop button binding - Tab2
        def stop_training_tab2():
            global_state['stop_training_tab2'] = True
            return "⚠️  Stop after this epoch..."

        stop_btn_tab2.click(
            fn=stop_training_tab2,
            outputs=[training_log_static]
        )

        # Stop button binding - Tab4
        def stop_training_tab4():
            global_state['stop_training_tab4'] = True
            return "⚠️  Stop after this epoch..."

        stop_btn_tab4.click(
            fn=stop_training_tab4,
            outputs=[stage2_training_log]
        )

    return demo


# ============================================================================
# Launch application

if __name__ == "__main__":
    import sys

    print("Starting Industrial Digital Twin Residual Boost Training System...")
    print("="*80)

    # Create necessary directories
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("saved_models/stage2_boost", exist_ok=True)
    os.makedirs("saved_models/ensemble", exist_ok=True)
    os.makedirs("saved_models/reinference_results", exist_ok=True)
    os.makedirs("saved_models/residuals_data", exist_ok=True)
    print("✅ Created necessary model save directories")

    # Check if running in Colab
    try:
        import google.colab
        IN_COLAB = True
        print("✅ colab confirmed")
    except:
        IN_COLAB = False
        print("✅ local confirmed")

    demo = create_unified_interface()
    print("✅ UI built")
    print("="*80)

    if IN_COLAB:
        # Colab environment - use share=True for public URL
        print("\n🌐 Start gradio in colab...")
        print("📝 note：Gradio will generate a public link")
        demo.launch(
            share=True,
            debug=True,
            show_error=True,
            inline=False  # Use separate window
        )
    else:
        # Local environment - try multiple ports
        print("\n🌐 Run gradio locally...")
        for port in range(7860, 7870):
            try:
                print(f"try {port}...")
                demo.launch(
                    server_name="127.0.0.1",
                    server_port=port,
                    share=False,
                    debug=True,
                    show_error=True,
                    quiet=False
                )
                print(f"✅ Service Started！")
                print(f"🔗 Address: http://localhost:{port}")
                print("="*80)
                break
            except OSError:
                print(f"⚠️  port {port} was not available，try next...")
                continue
        else:
            print("❌ no available port (7860-7869)")
