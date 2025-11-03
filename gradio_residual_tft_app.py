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
        inference_config_path: 推理配置File path
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

    print(f"✅ TFT模型已保存:")
    print(f"   📦 模型权重: {model_path}")
    print(f"   📊 Scalers: {scaler_path}")
    print(f"   📄 推理配置: {inference_config_path}")

    return model_path, scaler_path, inference_config_path


# ============================================================================
# TFT model load function

def load_tft_model_from_config(config_file_path: str, device: torch.device) -> Tuple[str, str]:
    """
    Load TFT model from inference config file

    Args:
        config_file_path: 推理配置JSONFile path
        device: PyTorch device

    Returns:
        model_name: Model name
        status_msg: 加载Status message
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
            return None, f"❌ 模型文件不存在: {model_path}"
        if not os.path.exists(scaler_path):
            return None, f"❌ Scaler文件不存在: {scaler_path}"

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
                'model_type': 'StaticSensorTransformer',  # 从base model继承
                'future_horizon': config['data_config']['future_horizon']
            },
            'encoder_length': config['data_config']['encoder_length'],
            'future_horizon': config['data_config']['future_horizon']
        }

        global_state['residual_scalers'][model_name] = scalers

        # 构建Status message
        status_msg = f"✅ TFT模型加载成功!\n\n"
        status_msg += f"📌 Model name: {model_name}\n"
        status_msg += f"📊 基础模型: {config['data_config']['base_model_name']}\n"
        status_msg += f"🎯 目标信号数: {config['data_config']['num_targets']}\n"
        status_msg += f"📈 边界信号数: {config['data_config']['num_external_factors']}\n"
        status_msg += f"📏 历史窗口长度: {config['data_config']['encoder_length']}\n"
        status_msg += f"🔮 未来预测长度: {config['data_config']['future_horizon']}\n"
        status_msg += f"⚙️ 模型维度: {config['architecture']['d_model']}\n"
        status_msg += f"🕒 创建时间: {config['created_time']}\n"

        if 'training_info' in config:
            ti = config['training_info']
            status_msg += f"\n📚 Training info:\n"
            status_msg += f"   - Training轮数: {ti['epochs_trained']}\n"
            status_msg += f"   - 最佳Validation损失: {ti['best_val_loss']:.6f}\n"
            status_msg += f"   - 批大小: {ti['batch_size']}\n"
            status_msg += f"   - 学习率: {ti['learning_rate']}\n"

        print(status_msg)
        return model_name, status_msg

    except Exception as e:
        error_msg = f"❌ TFT模型加载失败:\n{str(e)}\n\n{traceback.format_exc()}"
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

    print("✅ Gradio导入成功")
except ImportError:
    print("❌ 请安装gradio: pip install gradio")
    sys.exit(1)

# 尝试导入本地模块
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

    print("✅ 本地模块导入成功")
except ImportError as e:
    print(f"⚠️ 本地模块导入失败: {e}")
    print("尝试使用相对导入...")

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

        print("✅ 相对导入成功")
    except ImportError as e2:
        print(f"❌ 相对导入也失败: {e2}")
        print("将使用内联定义...")


# Setup device with enhanced GPU detection
def setup_device():
    """Setup computing device with GPU detection and configuration"""
    configure_chinese_font()
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"GPU检测成功: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA版本: {torch.version.cuda}")
        print(f"  GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        return device
    else:
        device = torch.device('cpu')
        print("GPU不可用，使用CPU训练")
        return device


device = setup_device()


def load_saved_models():
    """Load saved models from file system"""
    model_dir = "saved_models"
    if not os.path.exists(model_dir):
        return

    print(f"正在加载已保存的模型从 {model_dir}...")

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
                print(f"  加载模型: {model_name}")

            except Exception as e:
                print(f"  加载模型失败 {model_name}: {e}")
                continue

    print(f"模型加载完成，共加载 {len(global_state['trained_models'])} 个模型")


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

    print(f"✅ 推理配置已保存: {config_path}")
    return config_path


def load_model_from_inference_config(config_file_path, device):
    """
    从推理配置文件Load model

    Args:
        config_file_path: 推理配置JSONFile path
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
            return None, f"❌ 模型文件不存在: {model_path}"
        if not os.path.exists(scaler_path):
            return None, f"❌ Scaler文件不存在: {scaler_path}"

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
            return None, f"❌ 不支持的Model type: {model_type}"

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

        success_msg = f"✅ 模型加载成功!\n\n"
        success_msg += f"📌 Model name: {model_name}\n"
        success_msg += f"📊 Model type: {model_type}\n"
        success_msg += f"🎯 边界信号数: {len(config['boundary_signals'])}\n"
        success_msg += f"📈 目标信号数: {len(config['target_signals'])}\n"
        success_msg += f"⚙️ 模型参数: d_model={arch['d_model']}, nhead={arch['nhead']}, layers={arch['num_layers']}\n"
        success_msg += f"🕒 创建时间: {config['created_time']}\n"

        print(success_msg)
        return model_name, success_msg

    except Exception as e:
        error_msg = f"❌ 模型加载失败:\n{str(e)}\n\n{traceback.format_exc()}"
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
            return "❌ 残差数据不存在！", {}

        log_msg = []
        log_msg.append("=" * 80)
        log_msg.append("🚀 开始训练 Stage2 Boost 残差模型")
        log_msg.append("=" * 80)

        # Get residual data
        residuals_df = global_state['residual_data'][residual_data_key]['data']
        residual_info = global_state['residual_data'][residual_data_key]['info']

        boundary_signals = residual_info['boundary_signals']
        target_signals = residual_info['target_signals']
        residual_signals = residual_info['residual_signals']

        log_msg.append(f"\n📊 数据信息:")
        log_msg.append(f"  残差数据: {residual_data_key}")
        log_msg.append(f"  边界信号数: {len(boundary_signals)}")
        log_msg.append(f"  目标信号数: {len(target_signals)}")
        log_msg.append(f"  数据长度: {len(residuals_df)}")

        # Prepare training data
        X = residuals_df[boundary_signals].values
        y_residual = residuals_df[residual_signals].values

        # Data split
        train_size = int(len(X) * (1 - config['test_size'] - config['val_size']))
        val_size = int(len(X) * config['val_size'])

        X_train = X[:train_size]
        X_val = X[train_size:train_size + val_size]
        X_test = X[train_size + val_size:]

        y_train = y_residual[:train_size]
        y_val = y_residual[train_size:train_size + val_size]
        y_test = y_residual[train_size + val_size:]

        log_msg.append(f"\n🔀 数据分割:")
        log_msg.append(f"  训练集: {len(X_train)} ({len(X_train) / len(X) * 100:.1f}%)")
        log_msg.append(f"  验证集: {len(X_val)} ({len(X_val) / len(X) * 100:.1f}%)")
        log_msg.append(f"  测试集: {len(X_test)} ({len(X_test) / len(X) * 100:.1f}%)")

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
        log_msg.append(f"\n🏗️ 初始化Stage2残差模型:")
        log_msg.append(f"  架构: StaticSensorTransformer")
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
        log_msg.append(f"📊 学习率调度器: ReduceLROnPlateau (factor={config.get('scheduler_factor', 0.7)}, patience={config.get('scheduler_patience', 15)})")

        criterion = nn.MSELoss()

        # Mixed precision training
        scaler = GradScaler()

        # Training loop
        log_msg.append(f"\n🎯 开始训练 (混合精度, 总轮数: {config['epochs']})")

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

            # Progress output (增强版)
            if (epoch + 1) % max(1, config['epochs'] // 20) == 0 or epoch == 0 or epoch == config['epochs'] - 1:
                # 获取当前学习率
                current_lr = optimizer.param_groups[0]['lr']

                # 计算RMSE
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
                log_msg.append(f"\n⏸️ 早停触发 (Epoch {epoch + 1})")
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
        log_msg.append(f"\n📈 训练历史总结 ({len(history['train_losses'])} epochs):")
        log_msg.append(f"  最佳验证Loss: {best_val_loss:.4f} (Epoch {np.argmin(history['val_losses']) + 1})")
        log_msg.append(f"  最佳验证R²: {max(history['val_r2']):.4f} (Epoch {np.argmax(history['val_r2']) + 1})")
        log_msg.append(f"  最佳验证MAE: {min(history['val_mae']):.4f} (Epoch {np.argmin(history['val_mae']) + 1})")
        log_msg.append(f"  最终训练Loss: {history['train_losses'][-1]:.4f}")
        log_msg.append(f"  最终验证Loss: {history['val_losses'][-1]:.4f}")

        log_msg.append(f"\n📊 测试集性能:")
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

        log_msg.append(f"\n✅ Stage2模型训练完成并保存:")
        log_msg.append(f"  模型名称: {model_name}")
        log_msg.append(f"  模型路径: {model_path}")
        log_msg.append(f"  Scaler路径: {scaler_path}")

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
        error_msg = f"❌ Stage2模型训练失败:\n{str(e)}\n\n{traceback.format_exc()}"
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
            yield "❌ 残差数据不存在！"
            return

        log_msg = []
        log_msg.append("=" * 80)
        log_msg.append("🚀 开始训练 Stage2 Boost 残差模型")
        log_msg.append("=" * 80)

        # Get residual data
        residuals_df = global_state['residual_data'][residual_data_key]['data']
        residual_info = global_state['residual_data'][residual_data_key]['info']

        boundary_signals = residual_info['boundary_signals']
        target_signals = residual_info['target_signals']
        residual_signals = residual_info['residual_signals']

        log_msg.append(f"\n📊 数据信息:")
        log_msg.append(f"  残差数据: {residual_data_key}")
        log_msg.append(f"  边界信号数: {len(boundary_signals)}")
        log_msg.append(f"  目标信号数: {len(target_signals)}")
        log_msg.append(f"  数据长度: {len(residuals_df)}")

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

        log_msg.append(f"\n🔀 数据分割 (随机分割，避免分布不一致):")
        log_msg.append(f"  训练集: {len(X_train)} ({len(X_train) / len(X) * 100:.1f}%)")
        log_msg.append(f"  验证集: {len(X_val)} ({len(X_val) / len(X) * 100:.1f}%)")
        log_msg.append(f"  测试集: {len(X_test)} ({len(X_test) / len(X) * 100:.1f}%)")
        log_msg.append(f"  💡 使用随机shuffle确保各集合分布一致")

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
        log_msg.append(f"\n🏗️ 初始化Stage2残差模型:")
        log_msg.append(f"  架构: StaticSensorTransformer")
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
        log_msg.append(f"📊 学习率调度器: ReduceLROnPlateau")

        criterion = nn.MSELoss()
        scaler_amp = GradScaler()

        # Training loop
        log_msg.append(f"\n🎯 开始训练 (混合精度, 总轮数: {config['epochs']})")
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
                log_msg.append(f"\n⚠️  训练在 Epoch {epoch+1}/{config['epochs']} 时被用户中止")
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
                log_msg.append(f"\n⏸️ 早停触发 (Epoch {epoch + 1})")
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
        log_msg.append(f"\n📈 训练历史总结 ({len(history['train_losses'])} epochs):")
        log_msg.append(f"  最佳验证Loss: {best_val_loss:.4f} (Epoch {np.argmin(history['val_losses']) + 1})")
        log_msg.append(f"  最佳验证R²: {max(history['val_r2']):.4f} (Epoch {np.argmax(history['val_r2']) + 1})")
        log_msg.append(f"  最佳验证MAE: {min(history['val_mae']):.4f} (Epoch {np.argmin(history['val_mae']) + 1})")

        log_msg.append(f"\n📊 测试集性能:")
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

        log_msg.append(f"\n✅ Stage2模型训练完成并保存:")
        log_msg.append(f"  模型名称: {model_name}")
        log_msg.append(f"  模型路径: {model_path}")
        log_msg.append(f"  Scaler路径: {scaler_path}")
        log_msg.append(f"  推理配置: {inference_config_path}")

        yield "\n".join(log_msg)

    except Exception as e:
        error_msg = f"❌ Stage2模型训练失败:\n{str(e)}\n\n{traceback.format_exc()}"
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
    使用Delta R²策略生成综合推理模型 (仅在测试集上评估)

    新逻辑：
    1. 使用Stage2训练的测试集数据
    2. 计算每个信号的 Delta R² = R²_ensemble - R²_stage1
    3. 如果 Delta R² > 阈值，说明Stage2有显著提升，使用Stage1+Stage2
    4. 否则只使用Stage1预测

    Args:
        base_model_name: 基础SST模型名称
        stage2_model_name: Stage2残差模型名称
        delta_r2_threshold: Delta R²阈值 (默认0.05，即5%提升)

    Returns:
        status_msg: 状态信息
        ensemble_info: 综合模型信息
        fig: 可视化图表
    """
    try:
        log_msg = []
        log_msg.append("=" * 80)
        log_msg.append("🎯 生成综合推理模型 (Delta R² 策略)")
        log_msg.append("=" * 80)

        # Check if models exist
        if base_model_name not in global_state['trained_models']:
            return f"❌ 基础模型 {base_model_name} 不存在！", {}, None

        if stage2_model_name not in global_state['stage2_models']:
            return f"❌ Stage2模型 {stage2_model_name} 不存在！", {}, None

        # Get models
        base_model_info = global_state['trained_models'][base_model_name]
        stage2_model_info = global_state['stage2_models'][stage2_model_name]

        base_model = base_model_info['model']
        stage2_model = stage2_model_info['model']
        stage2_config = stage2_model_info['config']

        # Get residual data
        residual_data_key = stage2_model_info['residual_data_key']

        # 如果原始残差数据不存在，尝试使用任何可用的残差数据
        if residual_data_key not in global_state['residual_data']:
            available_residual_keys = list(global_state['residual_data'].keys())
            if not available_residual_keys:
                return f"❌ 没有可用的残差数据！请先在 Tab3 中生成残差数据。", {}, None

            # 使用第一个可用的残差数据
            residual_data_key = available_residual_keys[0]
            log_msg.append(f"\n⚠️  原始残差数据不存在，使用: {residual_data_key}")

        residuals_df = global_state['residual_data'][residual_data_key]['data']
        residual_info = global_state['residual_data'][residual_data_key]['info']

        boundary_signals = residual_info['boundary_signals']
        target_signals = residual_info['target_signals']

        # 验证信号匹配
        stage2_boundary = stage2_model_info.get('boundary_signals', boundary_signals)
        stage2_target = stage2_model_info.get('target_signals', target_signals)

        if set(stage2_boundary) != set(boundary_signals):
            log_msg.append(f"\n⚠️  警告：Stage2模型的边界信号与残差数据不完全匹配")
        if set(stage2_target) != set(target_signals):
            log_msg.append(f"\n⚠️  警告：Stage2模型的目标信号与残差数据不完全匹配")

        log_msg.append(f"\n📊 模型信息:")
        log_msg.append(f"  基础模型: {base_model_name}")
        log_msg.append(f"  Stage2模型: {stage2_model_name}")
        log_msg.append(f"  目标信号数: {len(target_signals)}")
        log_msg.append(f"  Delta R²阈值: {delta_r2_threshold:.3f} ({delta_r2_threshold*100:.1f}%)")

        # 使用Stage2训练时相同的数据分割获取测试集
        test_size = stage2_config.get('test_size', 0.2)
        val_size = stage2_config.get('val_size', 0.1)

        total_size = len(residuals_df)
        train_size = int(total_size * (1 - test_size - val_size))
        val_size_actual = int(total_size * val_size)
        test_start_idx = train_size + val_size_actual

        log_msg.append(f"\n🔀 数据分割 (使用测试集评估):")
        log_msg.append(f"  总数据: {total_size}")
        log_msg.append(f"  训练集: {train_size} ({train_size/total_size*100:.1f}%)")
        log_msg.append(f"  验证集: {val_size_actual} ({val_size_actual/total_size*100:.1f}%)")
        log_msg.append(f"  测试集: {total_size - test_start_idx} ({(total_size - test_start_idx)/total_size*100:.1f}%)")

        # 提取测试集数据
        y_true_cols = [f"{sig}_true" for sig in target_signals]
        y_pred_cols = [f"{sig}_pred" for sig in target_signals]

        y_true_test = residuals_df[y_true_cols].iloc[test_start_idx:].values
        y_pred_stage1_test = residuals_df[y_pred_cols].iloc[test_start_idx:].values
        X_test = residuals_df[boundary_signals].iloc[test_start_idx:].values

        # 使用Stage2模型预测测试集残差
        y_residual_pred_test = batch_inference(
            stage2_model,
            X_test,
            global_state['stage2_scalers'][stage2_model_name]['X'],
            global_state['stage2_scalers'][stage2_model_name]['y'],
            device,
            batch_size=512,
            model_name="Stage2"
        )

        # 计算每个信号的R²分数
        signal_analysis = []

        for i, signal in enumerate(target_signals):
            y_true_sig = y_true_test[:, i]
            y_pred_stage1_sig = y_pred_stage1_test[:, i]
            y_pred_ensemble_sig = y_pred_stage1_sig + y_residual_pred_test[:, i]

            # 计算Stage1的R²
            r2_stage1, _ = compute_r2_safe(
                y_true_sig.reshape(-1, 1),
                y_pred_stage1_sig.reshape(-1, 1),
                method='per_output_mean'
            )

            # 计算Ensemble的R²
            r2_ensemble, _ = compute_r2_safe(
                y_true_sig.reshape(-1, 1),
                y_pred_ensemble_sig.reshape(-1, 1),
                method='per_output_mean'
            )

            # 计算Delta R²
            delta_r2 = r2_ensemble - r2_stage1

            # 判断是否使用Stage2
            use_stage2 = delta_r2 > delta_r2_threshold

            signal_analysis.append({
                'signal': signal,
                'r2_stage1': float(r2_stage1),
                'r2_ensemble': float(r2_ensemble),
                'delta_r2': float(delta_r2),
                'use_stage2': bool(use_stage2)
            })

        # 统计使用Stage2的信号数
        num_use_stage2 = sum(1 for item in signal_analysis if item['use_stage2'])
        num_use_stage1_only = len(target_signals) - num_use_stage2

        log_msg.append(f"\n🎯 信号Delta R²分析:")
        log_msg.append(f"{'信号名称':<30} {'Stage1 R²':>12} {'Ensemble R²':>12} {'Delta R²':>12} {'选择':>10}")
        log_msg.append("-" * 80)

        for item in signal_analysis:
            choice = "Stage1+2" if item['use_stage2'] else "Stage1"
            log_msg.append(
                f"{item['signal']:<30} {item['r2_stage1']:>12.4f} {item['r2_ensemble']:>12.4f} "
                f"{item['delta_r2']:>12.4f} {choice:>10}"
            )

        log_msg.append("-" * 80)
        log_msg.append(f"使用Stage1+Stage2: {num_use_stage2} 个信号")
        log_msg.append(f"仅使用Stage1: {num_use_stage1_only} 个信号")

        # 生成最终综合预测（在测试集上）
        y_ensemble_test = y_pred_stage1_test.copy()
        for i, item in enumerate(signal_analysis):
            if item['use_stage2']:
                y_ensemble_test[:, i] = y_pred_stage1_test[:, i] + y_residual_pred_test[:, i]

        # 计算整体性能
        mae_stage1 = mean_absolute_error(y_true_test, y_pred_stage1_test)
        mae_ensemble = mean_absolute_error(y_true_test, y_ensemble_test)
        rmse_stage1 = np.sqrt(mean_squared_error(y_true_test, y_pred_stage1_test))
        rmse_ensemble = np.sqrt(mean_squared_error(y_true_test, y_ensemble_test))
        r2_stage1, _ = compute_r2_safe(y_true_test, y_pred_stage1_test, method='per_output_mean')
        r2_ensemble, _ = compute_r2_safe(y_true_test, y_ensemble_test, method='per_output_mean')

        improvement_mae = (mae_stage1 - mae_ensemble) / mae_stage1 * 100 if mae_stage1 > 0 else 0
        improvement_rmse = (rmse_stage1 - rmse_ensemble) / rmse_stage1 * 100 if rmse_stage1 > 0 else 0
        improvement_r2 = (r2_ensemble - r2_stage1) / (1 - r2_stage1) * 100 if r2_stage1 < 1 else 0

        log_msg.append(f"\n📈 整体性能对比 (测试集):")
        log_msg.append(f"{'指标':<15} {'Stage1':>15} {'Ensemble':>15} {'改进':>15}")
        log_msg.append("-" * 65)
        log_msg.append(f"{'MAE':<15} {mae_stage1:>15.6f} {mae_ensemble:>15.6f} {improvement_mae:>14.2f}%")
        log_msg.append(f"{'RMSE':<15} {rmse_stage1:>15.6f} {rmse_ensemble:>15.6f} {improvement_rmse:>14.2f}%")
        log_msg.append(f"{'R²':<15} {r2_stage1:>15.4f} {r2_ensemble:>15.4f} {improvement_r2:>14.2f}%")

        # 保存综合模型信息
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
                'y_pred_base': y_pred_stage1_test,  # 使用y_pred_base以匹配tab6的读取
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

        # 保存配置文件
        ensemble_dir = "saved_models/ensemble"
        os.makedirs(ensemble_dir, exist_ok=True)

        config_path = os.path.join(ensemble_dir, f"{ensemble_name}_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            # 保存配置（排除大数组，确保所有类型可JSON序列化）
            save_config = {
                'name': ensemble_name,
                'base_model_name': base_model_name,
                'stage2_model_name': stage2_model_name,
                'delta_r2_threshold': float(delta_r2_threshold),
                'signal_analysis': signal_analysis,  # 已转换为Python原生类型
                'num_use_stage2': int(num_use_stage2),
                'num_use_stage1_only': int(num_use_stage1_only),
                'metrics': ensemble_info['metrics'],  # 已转换
                'signals': ensemble_info['signals'],
                'data_split': ensemble_info['data_split'],
                'created_time': ensemble_info['created_time']
            }
            json.dump(save_config, f, indent=2, ensure_ascii=False)

        # 生成汇总CSV文件
        csv_path = os.path.join(ensemble_dir, f"{ensemble_name}_summary.csv")
        summary_data = []
        for item in signal_analysis:
            summary_data.append({
                '信号名称': item['signal'],
                'Stage1_R2': item['r2_stage1'],
                'Ensemble_R2': item['r2_ensemble'],
                'Delta_R2': item['delta_r2'],
                'R2提升(%)': item['delta_r2'] * 100,
                '选择模型': 'Stage1+Stage2' if item['use_stage2'] else 'Stage1',
                '是否使用Stage2': 'Yes' if item['use_stage2'] else 'No'
            })

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

        log_msg.append(f"\n✅ 综合推理模型已生成:")
        log_msg.append(f"  模型名称: {ensemble_name}")
        log_msg.append(f"  配置路径: {config_path}")
        log_msg.append(f"  汇总CSV: {csv_path}")

        # 生成可视化图表
        fig = create_ensemble_visualization(ensemble_info)

        return "\n".join(log_msg), ensemble_info, fig

    except Exception as e:
        error_msg = f"❌ 综合模型生成失败:\n{str(e)}\n\n{traceback.format_exc()}"
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

        status = f"✅ 数据加载成功!\n\n"
        status += f"📊 数据维度: {df.shape}\n"
        status += f"📈 样本数: {len(df):,}\n"
        status += f"🎯 特征数: {len(df.columns)}\n\n"
        status += f"前5列: {', '.join(df.columns[:5].tolist())}"

        signals_display = f"可用信号 ({len(df.columns)}个):\n" + ", ".join(df.columns.tolist())

        # Data preview (first 100 rows)
        preview_df = df.head(100)

        return status, preview_df, signals_display

    except Exception as e:
        error_msg = f"❌ 数据加载失败: {str(e)}"
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
        return "❌ 请选择有效的CSV文件", None, ""

    if not os.path.exists(csv_path):
        return f"❌ 文件不存在: {csv_path}", None, ""

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

        status = f"✅ 数据加载成功!\n\n"
        status += f"📁 文件: {csv_path}\n"
        status += f"📊 数据维度: {df.shape}\n"
        status += f"📈 样本数: {len(df):,}\n"
        status += f"🎯 特征数: {len(df.columns)}\n\n"
        status += f"前5列: {', '.join(df.columns[:5].tolist())}"

        signals_display = f"可用信号 ({len(df.columns)}个):\n" + ", ".join(df.columns.tolist())

        # Data preview (first 100 rows)
        preview_df = df.head(100)

        return status, preview_df, signals_display

    except Exception as e:
        error_msg = f"❌ 数据加载失败: {str(e)}"
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

        status = f"✅ [预加载] 数据已加载!\n\n"
        status += f"📊 数据维度: {df.shape}\n"
        status += f"📈 样本数: {len(df):,}\n"
        status += f"🎯 特征数: {len(df.columns)}\n\n"
        status += f"列名: {', '.join(df.columns[:5].tolist())}"
        if len(df.columns) > 5:
            status += f"... (共{len(df.columns)}列)"

        signals_display = f"可用信号 ({len(df.columns)}个):\n" + ", ".join(df.columns.tolist())

        # Data preview (first 100 rows)
        preview_df = df.head(100)

        return status, preview_df, signals_display
    else:
        return "⚠️ 尚未加载数据，请选择CSV文件、上传文件或创建示例数据", None, ""


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
            return [], [], "❌ 请上传JSON配置文件"

        if not os.path.exists(file_path):
            return [], [], f"❌ 文件不存在: {file_path}"

        # Load JSON
        with open(file_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        boundary_signals = config.get('boundary_signals', [])
        target_signals = config.get('target_signals', [])

        if not boundary_signals or not target_signals:
            return [], [], "❌ JSON文件格式错误，缺少 'boundary_signals' 或 'target_signals'"

        status = f"✅ JSON配置加载成功!\n\n"
        status += f"📥 边界信号数: {len(boundary_signals)}\n"
        status += f"📤 目标信号数: {len(target_signals)}\n"
        status += f"📁 文件: {os.path.basename(file_path)}"

        return boundary_signals, target_signals, status

    except json.JSONDecodeError as e:
        return [], [], f"❌ JSON解析失败: {str(e)}"
    except Exception as e:
        return [], [], f"❌ 加载失败: {str(e)}"


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

        status = f"✅ 示例数据创建成功!\n\n"
        status += f"📊 数据维度: {df.shape}\n"
        status += f"📈 样本数: {len(df):,}\n"
        status += f"🎯 边界信号: {n_boundary}个\n"
        status += f"🎯 目标信号: {n_target}个\n\n"
        status += "💡 提示: 示例数据模拟了传感器之间的非线性关系"

        signals_display = f"可用信号 ({len(df.columns)}个):\n" + ", ".join(df.columns.tolist())

        # Data preview (first 100 rows)
        preview_df = df.head(100)

        return status, preview_df, signals_display

    except Exception as e:
        error_msg = f"❌ 示例数据创建失败: {str(e)}"
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
            return "❌ 请先加载数据！"

        if not boundary_signals or not target_signals:
            return "❌ 请选择边界信号和目标信号！"

        log_messages = []
        log_messages.append("=" * 80)
        log_messages.append(f"🚀 开始训练 {model_type}")
        log_messages.append("=" * 80)
        log_messages.append(f"\n📊 训练配置:")
        log_messages.append(f"  模型类型: {model_type}")
        log_messages.append(f"  边界信号数: {len(boundary_signals)}")
        log_messages.append(f"  目标信号数: {len(target_signals)}")
        log_messages.append(f"  训练轮数: {epochs}")
        log_messages.append(f"  批大小: {batch_size}")
        log_messages.append(f"  学习率: {lr}")

        df = global_state['df']

        # Prepare data
        X = df[boundary_signals].values
        y = df[target_signals].values

        # Apply IFD smoothing (if needed)
        if apply_smoothing and temporal_signals:
            log_messages.append(f"\n🔧 应用IFD平滑...")
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
            log_messages.append(f"  已对 {len(temporal_signals)} 个时序信号应用平滑")

        # Data split
        train_size = int(len(X) * (1 - test_size - val_size))
        val_size_samples = int(len(X) * val_size)

        X_train = X[:train_size]
        X_val = X[train_size:train_size + val_size_samples]
        X_test = X[train_size + val_size_samples:]

        y_train = y[:train_size]
        y_val = y[train_size:train_size + val_size_samples]
        y_test = y[train_size + val_size_samples:]

        log_messages.append(f"\n🔀 数据分割:")
        log_messages.append(f"  训练集: {len(X_train)} ({len(X_train) / len(X) * 100:.1f}%)")
        log_messages.append(f"  验证集: {len(X_val)} ({len(X_val) / len(X) * 100:.1f}%)")
        log_messages.append(f"  测试集: {len(X_test)} ({len(X_test) / len(X) * 100:.1f}%)")

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
        log_messages.append(f"\n🏗️ 初始化模型: {model_type}")

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
            return f"❌ 不支持的模型类型: {model_type}"

        # Optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience
        )
        log_messages.append(f"📊 优化器: AdamW (lr={lr:.2e}, weight_decay={weight_decay:.2e})")
        log_messages.append(f"📊 学习率调度器: ReduceLROnPlateau (factor={scheduler_factor}, patience={scheduler_patience})")
        log_messages.append(f"✂️ 梯度裁剪: {grad_clip_norm}")
        criterion = nn.MSELoss()

        # Mixed precision training
        scaler = GradScaler()

        # Training loop
        log_messages.append(f"\n🎯 开始训练 (混合精度)...")
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
                log_messages.append(f"\n⚠️  训练在 Epoch {epoch+1}/{epochs} 时被用户中止")
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

            # 进度显示 (增强版 - 显示MAE, RMSE, R2和学习率)
            if (epoch + 1) % max(1, epochs // 20) == 0 or epoch == 0:
                # 获取当前学习率
                current_lr = optimizer.param_groups[0]['lr']

                # 计算RMSE (更直观)
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
                log_messages.append(f"\n⏸️ 早停触发 (Epoch {epoch + 1})")
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
        log_messages.append(f"\n📈 训练历史总结 ({len(history['train_losses'])} epochs):")
        log_messages.append(f"  最佳验证Loss: {best_val_loss:.4f} (Epoch {np.argmin(history['val_losses']) + 1})")
        log_messages.append(f"  最佳验证R²: {max(history['val_r2']):.4f} (Epoch {np.argmax(history['val_r2']) + 1})")
        log_messages.append(f"  最佳验证MAE: {min(history['val_mae']):.4f} (Epoch {np.argmin(history['val_mae']) + 1})")
        log_messages.append(f"  最终训练Loss: {history['train_losses'][-1]:.4f}")
        log_messages.append(f"  最终验证Loss: {history['val_losses'][-1]:.4f}")

        log_messages.append(f"\n📊 测试集性能:")
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
                }
            },
            'training_history': history
        }, model_path)

        # Save scalers
        scaler_path = os.path.join(model_dir, f"{model_name}_scalers.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump({'X': scaler_X, 'y': scaler_y}, f)

        # 保存推理配置
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

        log_messages.append(f"\n✅ 模型训练完成并保存:")
        log_messages.append(f"  模型名称: {model_name}")
        log_messages.append(f"  模型路径: {model_path}")
        log_messages.append(f"  Scaler路径: {scaler_path}")

        return "\n".join(log_messages)

    except Exception as e:
        error_msg = f"❌ 训练失败:\n{str(e)}\n\n{traceback.format_exc()}"
        print(error_msg)
        return error_msg


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
            return None, "❌ 请选择推理配置文件"

        if not os.path.exists(config_path):
            return None, f"❌ 文件不存在: {config_path}"

        # Load config
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        model_name = config.get('model_name')
        if not model_name:
            return None, "❌ 配置文件中缺少 model_name"

        status = f"✅ 配置加载成功!\n\n"
        status += f"📁 配置文件: {os.path.basename(config_path)}\n"
        status += f"🤖 模型名称: {model_name}\n"
        status += f"📥 边界信号数: {len(config.get('boundary_signals', []))}\n"
        status += f"📤 目标信号数: {len(config.get('target_signals', []))}"

        return model_name, status

    except json.JSONDecodeError as e:
        return None, f"❌ JSON解析失败: {str(e)}"
    except Exception as e:
        return None, f"❌ 加载失败: {str(e)}"


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
            return "❌ 请选择scalers文件！"

        if not model_name:
            return "❌ 请先选择模型！"

        if not os.path.exists(scaler_path):
            return f"❌ 文件不存在: {scaler_path}"

        # Load scalers from file
        with open(scaler_path, 'rb') as f:
            scalers = pickle.load(f)

        # Save to global state
        if 'manual_scalers' not in global_state:
            global_state['manual_scalers'] = {}

        global_state['manual_scalers'][model_name] = scalers

        success_msg = f"✅ Scalers加载成功!\n\n"
        success_msg += f"📌 Model name: {model_name}\n"
        success_msg += f"📊 Scalers包含: {list(scalers.keys())}\n"

        # Verify scalers have required keys
        if 'X' in scalers and 'y' in scalers:
            success_msg += f"✓ 包含必需的X和y scalers\n"
        else:
            success_msg += f"⚠️ 警告: scalers可能缺少X或y键\n"

        print(success_msg)
        return success_msg

    except Exception as e:
        error_msg = f"❌ Scalers加载失败:\n{str(e)}\n\n{traceback.format_exc()}"
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
            return None, "❌ 请选择模型文件！"

        if not os.path.exists(model_path):
            return None, f"❌ 文件不存在: {model_path}"

        # Extract model name from path
        model_name = os.path.splitext(os.path.basename(model_path))[0]

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        if 'model_config' not in checkpoint:
            return None, f"❌ 模型文件格式错误: 缺少model_config"

        model_config = checkpoint['model_config']

        if model_config.get('type') != 'StaticSensorTransformer':
            return None, f"❌ 不支持的模型类型: {model_config.get('type')}"

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
        scaler_source = "未加载"
        if 'scalers' in checkpoint:
            scalers = checkpoint['scalers']
            scaler_source = "从checkpoint加载"
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

        success_msg = f"✅ SST模型加载成功!\n\n"
        success_msg += f"📌 Model name: {model_name}\n"
        success_msg += f"📊 Model type: {model_config['type']}\n"
        success_msg += f"🎯 边界信号数: {len(boundary_signals)}\n"
        success_msg += f"📈 目标信号数: {len(target_signals)}\n"
        success_msg += f"⚙️ 模型参数: d_model={config['d_model']}, nhead={config['nhead']}, layers={config['num_layers']}\n"
        success_msg += f"📊 Scalers状态: {scaler_source}\n"

        if scalers is None:
            success_msg += f"\n⚠️ 提示: 该模型checkpoint中不包含scalers\n"
            success_msg += f"   如需提取残差，请从下方'加载Scalers文件'区域手动加载\n"

        print(success_msg)
        return model_name, success_msg

    except Exception as e:
        error_msg = f"❌ 模型加载失败:\n{str(e)}\n\n{traceback.format_exc()}"
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
            return "❌ 请选择模型！", None

        if global_state['df'] is None:
            return "❌ 请先加载数据！", None

        log_msg = []
        log_msg.append("=" * 80)
        log_msg.append("📊 开始提取残差（全数据集）")
        log_msg.append("=" * 80)

        df = global_state['df']
        log_msg.append(f"\n📈 数据集大小: {len(df):,} 条")

        # Load model
        model_path = os.path.join("saved_models", f"{model_name}.pth")
        if not os.path.exists(model_path):
            return f"❌ 模型文件不存在: {model_path}", None

        log_msg.append(f"\n📥 加载模型: {model_name}")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        model_config = checkpoint['model_config']
        boundary_signals = model_config['boundary_signals']
        target_signals = model_config['target_signals']

        log_msg.append(f"  边界信号数: {len(boundary_signals)}")
        log_msg.append(f"  目标信号数: {len(target_signals)}")

        # Check if signals exist in dataframe
        missing_boundary = [s for s in boundary_signals if s not in df.columns]
        missing_target = [s for s in target_signals if s not in df.columns]

        if missing_boundary:
            return f"❌ 数据集中缺少边界信号: {missing_boundary}", None
        if missing_target:
            return f"❌ 数据集中缺少目标信号: {missing_target}", None

        # Prepare data
        X = df[boundary_signals].values
        y = df[target_signals].values

        # Load scalers - try checkpoint first, then manual_scalers
        scalers = None
        if 'scalers' in checkpoint:
            scalers = checkpoint['scalers']
            log_msg.append(f"  ✓ 从checkpoint加载scalers")
        elif 'manual_scalers' in global_state and model_name in global_state['manual_scalers']:
            scalers = global_state['manual_scalers'][model_name]
            log_msg.append(f"  ✓ 从手动加载的scalers加载")
        else:
            error_msg = "❌ 未找到scalers！\n\n"
            error_msg += f"checkpoint中无scalers，且未手动加载scalers。\n\n"
            error_msg += "💡 解决方法:\n"
            error_msg += "1. 在下方'加载Scalers文件'区域上传对应的scalers.pkl文件\n"
            error_msg += f"2. 文件名应该类似: {model_name}_scalers.pkl\n"
            error_msg += "3. 点击'📥 加载Scalers'按钮后，再次点击'🔬 提取残差'按钮\n"
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

        log_msg.append(f"\n🔄 开始推理...")

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

        log_msg.append(f"\n📊 推理完成:")
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

        info = {
            'base_model_name': model_name,
            'boundary_signals': boundary_signals,
            'target_signals': target_signals,
            'residual_signals': residual_cols,
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

        log_msg.append(f"\n✅ 残差提取完成:")
        log_msg.append(f"  残差数据标识: {residual_key}")
        log_msg.append(f"  数据形状: {residuals_df.shape}")

        # Create residual visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'残差分析 - {model_name}', fontsize=16)

        # 残差分布
        residual_cols = info['residual_signals']
        all_residuals = residuals_df[residual_cols].values.flatten()

        axes[0, 0].hist(all_residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].set_title('残差分布')
        axes[0, 0].set_xlabel('残差值')
        axes[0, 0].set_ylabel('频数')

        # 残差序列
        axes[0, 1].plot(residuals_df[residual_cols[0]].values[:1000])
        axes[0, 1].set_title(f'残差序列 ({residual_cols[0]})')
        axes[0, 1].set_xlabel('Index')
        axes[0, 1].set_ylabel('残差')

        # 残差统计
        residual_stats = residuals_df[residual_cols].describe()
        axes[1, 0].axis('off')
        stats_text = "残差统计:\n"
        stats_text += f"Mean: {residual_stats.loc['mean'].mean():.6f}\n"
        stats_text += f"Std: {residual_stats.loc['std'].mean():.6f}\n"
        stats_text += f"Min: {residual_stats.loc['min'].min():.6f}\n"
        stats_text += f"Max: {residual_stats.loc['max'].max():.6f}\n"
        axes[1, 0].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center')

        # 预测vs真实
        true_cols = [f"{sig}_true" for sig in info['target_signals']]
        pred_cols = [f"{sig}_pred" for sig in info['target_signals']]

        y_true = residuals_df[true_cols].values[:1000, 0]
        y_pred = residuals_df[pred_cols].values[:1000, 0]

        axes[1, 1].plot(y_true, label='True', alpha=0.7)
        axes[1, 1].plot(y_pred, label='Predicted', alpha=0.7)
        axes[1, 1].set_title('预测 vs 真实 (前1000个样本)')
        axes[1, 1].legend()

        plt.tight_layout()

        return "\n".join(log_msg), fig

    except Exception as e:
        error_msg = f"❌ 残差提取失败:\n{str(e)}\n\n{traceback.format_exc()}"
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
            return "❌ 请选择有效的Scaler文件！"

        if not stage2_model_key:
            return "❌ 请先选择一个Stage2模型！"

        # Check if model exists
        if stage2_model_key not in global_state['stage2_models']:
            return f"❌ 模型 {stage2_model_key} 不存在！请先加载Stage2模型。"

        # Load scalers
        with open(scaler_path, 'rb') as f:
            scalers = pickle.load(f)

        # Store in global state
        global_state['stage2_scalers'][stage2_model_key] = scalers

        status_msg = f"✅ 成功加载 Scalers！\n\n"
        status_msg += f"Scaler路径: {scaler_path}\n"
        status_msg += f"关联模型: {stage2_model_key}\n"
        status_msg += f"Scaler类型: {type(scalers)}\n"
        if isinstance(scalers, dict):
            status_msg += f"包含的键: {list(scalers.keys())}\n"

        return status_msg

    except Exception as e:
        return f"❌ 加载失败:\n{str(e)}\n\n{traceback.format_exc()}"


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
            return None, "❌ 请选择有效的推理配置文件！"

        with open(config_path, 'r') as f:
            config = json.load(f)

        model_name = config.get('model_name', '')
        model_path = config.get('model_path', '')
        scaler_path = config.get('scaler_path', '')

        if not model_name or not model_path:
            return None, "❌ 配置文件格式错误：缺少 model_name 或 model_path！"

        # Load model checkpoint
        if not os.path.exists(model_path):
            return None, f"❌ 模型文件不存在: {model_path}"

        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        # Load scalers
        scalers = None
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scalers = pickle.load(f)
        elif 'scalers' in checkpoint:
            scalers = checkpoint['scalers']
        else:
            return None, "❌ 未找到 scalers！请确保配置文件中包含 scaler_path 或模型 checkpoint 中包含 scalers。"

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
            return None, "❌ 配置文件和checkpoint中都缺少 boundary_signals 或 target_signals！"

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

        status_msg = f"✅ 成功加载 Stage2 模型！\n\n"
        status_msg += f"模型名称: {model_name}\n"
        status_msg += f"模型键: {model_key}\n"
        status_msg += f"模型路径: {model_path}\n"
        status_msg += f"边界信号数: {len(boundary_signals)}\n"
        status_msg += f"目标信号数: {len(target_signals)}\n"
        status_msg += f"残差数据键: {residual_data_key}\n\n"
        status_msg += f"请在下方的 'Stage2模型' 下拉框中选择: {model_key}"

        return model_key, status_msg

    except Exception as e:
        return None, f"❌ 加载失败:\n{str(e)}\n\n{traceback.format_exc()}"


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
            return None, "❌ 请选择有效的模型文件！"

        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        # Extract configurations
        # Training saves 'config' not 'model_config' at top level
        config = checkpoint.get('config', {})
        residual_data_key = checkpoint.get('residual_data_key', 'unknown')

        # Get signals - these are saved at top level in checkpoint, not in config
        boundary_signals = checkpoint.get('boundary_signals', [])
        target_signals = checkpoint.get('target_signals', [])

        if not boundary_signals or not target_signals:
            return None, f"❌ 模型文件中缺少 boundary_signals 或 target_signals！\n\nCheckpoint keys: {list(checkpoint.keys())}"

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
            return None, "❌ 未找到 scalers！请确保模型 checkpoint 中包含 scalers 或存在对应的 *_scalers.pkl 文件。"

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

        status_msg = f"✅ 成功加载 Stage2 模型！\n\n"
        status_msg += f"模型名称: {model_name}\n"
        status_msg += f"模型键: {model_key}\n"
        status_msg += f"模型路径: {model_path}\n"
        status_msg += f"边界信号数: {len(boundary_signals)}\n"
        status_msg += f"目标信号数: {len(target_signals)}\n"
        status_msg += f"残差数据键: {residual_data_key}\n\n"
        status_msg += f"请在下方的 'Stage2模型' 下拉框中选择: {model_key}"

        return model_key, status_msg

    except Exception as e:
        return None, f"❌ 加载失败:\n{str(e)}\n\n{traceback.format_exc()}"


def get_ensemble_model_keys():
    """Get list of available ensemble models"""
    return list(global_state['ensemble_models'].keys())


# ============================================================================
# Gradio interface creation

def create_unified_interface():
    """Create unified Gradio interface"""

    with gr.Blocks(title="工业数字孪生残差Boost训练系统", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🏭 工业数字孪生残差BoostTraining系统
        ### Enhanced Residual Boost Training with Stage2 Model

        **新功能:**
        - ✨ Stage2 residual modelTraining
        - 🎯 智能R²阈值选择生成综合推理模型
        - 📊 Secondary inference comparison (Ensemble model vs Pure SST model)
        - 🔮 Sundial time series model predicting future residuals
        """)

        with gr.Tabs():
            # Tab 1: 数据加载
            with gr.Tab("📂 数据加载", elem_id="data_loading"):
                gr.Markdown("## 选择、上传或创建数据")

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 📁 选择已有CSV文件")
                        csv_file_selector = gr.Dropdown(
                            choices=[],  # Empty initially, populated on page load
                            label="选择data文件夹下的CSV文件",
                            info="点击'刷新列表'来加载可用的CSV文件"
                        )
                        with gr.Row():
                            select_csv_btn = gr.Button("📂 加载选中文件", variant="primary", size="lg")
                            refresh_csv_btn = gr.Button("🔄 刷新列表", size="sm")

                        gr.Markdown("### 📤 或上传CSV文件")
                        data_file = gr.File(label="上传CSV文件", file_types=['.csv'])
                        upload_btn = gr.Button("📥 加载上传文件", variant="secondary", size="lg")

                        gr.Markdown("### 🎲 或创建示例数据")
                        sample_btn = gr.Button("🎲 创建示例数据", size="lg")

                    with gr.Column(scale=1):
                        data_status = gr.Textbox(label="数据状态", lines=10, interactive=False)
                        signals_display = gr.Textbox(label="可用信号", lines=10, interactive=False)

                # Data preview table
                with gr.Row():
                    data_preview = gr.Dataframe(
                        label="📊 数据预览 (前100行)",
                        interactive=False,
                        wrap=True
                    )

            # Tab 2: SST模型Training
            with gr.Tab("🎯 SST模型Training", elem_id="sst_training"):
                gr.Markdown("## 训练静态传感器映射Transformer (SST)")

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 🎛️ 信号选择")

                        # JSON配置加载
                        with gr.Accordion("📁 从JSON加载信号配置", open=False):
                            json_config_selector = gr.Dropdown(
                                choices=get_available_json_configs(),
                                label="选择data文件夹下的JSON配置",
                                info="或手动上传JSON文件"
                            )
                            with gr.Row():
                                load_json_btn = gr.Button("📂 加载配置", size="sm", variant="secondary")
                                refresh_json_btn = gr.Button("🔄 刷新", size="sm")
                            json_upload = gr.File(
                                label="上传JSON配置文件",
                                file_types=['.json'],
                                type="filepath"
                            )
                            json_status = gr.Textbox(
                                label="配置加载状态",
                                lines=3,
                                interactive=False
                            )

                        boundary_signals_static = gr.Dropdown(
                            choices=[], label="边界信号 (输入)", multiselect=True
                        )
                        target_signals_static = gr.Dropdown(
                            choices=[], label="目标信号 (输出)", multiselect=True
                        )

                        gr.Markdown("### 🏗️ 模型架构")
                        with gr.Row():
                            d_model_static = gr.Slider(32, 1280, 256, 32, label="模型维度")
                            nhead_static = gr.Slider(2, 80, 16, 2, label="注意力头数")
                        with gr.Row():
                            num_layers_static = gr.Slider(1, 30, 6, 1, label="Transformer层数")
                            dropout_static = gr.Slider(0, 0.5, 0.1, 0.05, label="Dropout率")

                        gr.Markdown("### 🎯 训练参数")
                        with gr.Row():
                            epochs_static = gr.Slider(10, 250, 50, 10, label="训练轮数")
                            batch_size_static = gr.Slider(16, 2560, 512, 16, label="批大小")
                        with gr.Row():
                            lr_static = gr.Number(value=0.0001, label="学习率")
                            weight_decay_static = gr.Number(value=1e-5, label="权重衰减")

                        gr.Markdown("### ⚙️ 优化器设置")
                        with gr.Row():
                            grad_clip_norm_static = gr.Slider(0.1, 5.0, 1.0, 0.1, label="梯度裁剪")
                            scheduler_patience_static = gr.Slider(1, 15, 3, 1, label="学习率调度耐心值")
                        scheduler_factor_static = gr.Slider(0.1, 0.9, 0.5, 0.1, label="学习率衰减因子")

                        gr.Markdown("### 🔀 数据分割")
                        with gr.Row():
                            test_size_static = gr.Slider(0.1, 0.3, 0.15, 0.05, label="测试集比例")
                            val_size_static = gr.Slider(0.1, 0.3, 0.15, 0.05, label="验证集比例")

                        train_btn_static = gr.Button("▶️ 开始TrainingSST", variant="primary", size="lg")
                        stop_btn_tab2 = gr.Button("⏹️ 停止训练", variant="stop", size="lg")

                    with gr.Column(scale=1):
                        gr.Markdown("### 📊 训练日志")
                        training_log_static = gr.Textbox(
                            label="训练进度",
                            lines=30,
                            autoscroll=True,
                            interactive=False
                        )

            # Tab 3: 残差提取
            with gr.Tab("🔬 残差提取", elem_id="residual_extraction"):
                gr.Markdown("## 从训练好的SST模型提取残差")
                gr.Markdown("对整个数据集进行推理，生成残差用于Stage2训练")

                with gr.Row():
                    with gr.Column(scale=1):
                        model_selector = gr.Dropdown(
                            choices=get_available_models(),
                            label="选择SST模型"
                        )
                        refresh_models_btn = gr.Button("🔄 刷新模型列表", size="sm")

                        gr.Markdown("### 📤 推理配置文件（可选）")
                        gr.Markdown("可选择已保存的推理配置文件来加载模型")

                        inference_config_selector = gr.Dropdown(
                            choices=get_inference_config_files(),
                            label="选择saved_models文件夹下的推理配置",
                            info="选择 *_inference.json 文件"
                        )
                        with gr.Row():
                            load_inference_btn = gr.Button("📥 加载配置", size="sm", variant="secondary")
                            refresh_inference_btn = gr.Button("🔄 刷新配置列表", size="sm")

                        inference_load_status = gr.Textbox(label="配置加载状态", lines=3, interactive=False)

                        gr.Markdown("### 🤖 加载SST模型文件（可选）")
                        gr.Markdown("从saved_models文件夹选择.pth模型文件直接加载")
                        model_file_selector = gr.Dropdown(
                            choices=get_model_files(),
                            label="选择saved_models文件夹下的模型文件",
                            info="选择 *.pth 文件"
                        )
                        with gr.Row():
                            load_model_file_btn = gr.Button("📥 加载模型", size="sm", variant="secondary")
                            refresh_model_files_btn = gr.Button("🔄 刷新模型列表", size="sm")
                        model_load_status = gr.Textbox(label="模型加载状态", lines=3, interactive=False)

                        gr.Markdown("### 📊 加载Scalers文件（可选）")
                        gr.Markdown("如果模型checkpoint中不包含scalers，从saved_models文件夹选择")
                        scalers_file_selector = gr.Dropdown(
                            choices=get_scalers_files(),
                            label="选择saved_models文件夹下的Scalers文件",
                            info="选择 *_scalers.pkl 文件"
                        )
                        with gr.Row():
                            load_scalers_btn = gr.Button("📥 加载Scalers", size="sm", variant="secondary")
                            refresh_scalers_btn = gr.Button("🔄 刷新Scalers列表", size="sm")
                        scalers_load_status = gr.Textbox(label="Scalers加载状态", lines=3, interactive=False)

                        extract_btn = gr.Button("🔬 提取残差（全数据集）", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        residual_status = gr.Textbox(label="残差提取状态", lines=20, interactive=False)
                        residual_plot = gr.Plot(label="残差可视化")

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
                gr.Markdown("## 训练Stage2残差模型")
                gr.Markdown("基于提取的残差训练Stage2模型，进一步提升预测精度")

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 📊 数据选择")
                        residual_data_selector_stage2 = gr.Dropdown(
                            choices=get_residual_data_keys(),
                            label="选择残差数据"
                        )
                        refresh_residual_btn_stage2 = gr.Button("🔄 刷新", size="sm")

                        gr.Markdown("### 🏗️ 模型架构")
                        with gr.Row():
                            d_model_stage2 = gr.Slider(32, 640, 128, 32, label="模型维度")
                            nhead_stage2 = gr.Slider(2, 40, 8, 2, label="注意力头数")
                        with gr.Row():
                            num_layers_stage2 = gr.Slider(1, 20, 4, 1, label="Transformer层数")
                            dropout_stage2 = gr.Slider(0, 0.5, 0.15, 0.05, label="Dropout率")

                        gr.Markdown("### 🎯 训练参数")
                        with gr.Row():
                            epochs_stage2 = gr.Slider(10, 400, 80, 10, label="训练轮数")
                            batch_size_stage2 = gr.Slider(16, 2560, 512, 16, label="批大小")
                        with gr.Row():
                            lr_stage2 = gr.Number(value=0.0001, label="学习率")
                            weight_decay_stage2 = gr.Number(value=5e-6, label="权重衰减")

                        gr.Markdown("### ⚙️ 优化器设置")
                        with gr.Row():
                            grad_clip_stage2 = gr.Slider(0.1, 2.5, 0.5, 0.1, label="梯度裁剪")
                            scheduler_patience_stage2 = gr.Slider(1, 75, 15, 1, label="学习率调度耐心值")
                        scheduler_factor_stage2 = gr.Slider(0.1, 0.9, 0.7, 0.1, label="学习率衰减因子")

                        gr.Markdown("### 🔀 数据分割")
                        with gr.Row():
                            test_size_stage2 = gr.Slider(0.1, 0.3, 0.15, 0.05, label="测试集比例")
                            val_size_stage2 = gr.Slider(0.1, 0.3, 0.15, 0.05, label="验证集比例")

                        train_stage2_btn = gr.Button("🚀 开始TrainingStage2", variant="primary", size="lg")
                        stop_btn_tab4 = gr.Button("⏹️ 停止训练", variant="stop", size="lg")

                    with gr.Column(scale=1):
                        gr.Markdown("### 📊 训练日志")
                        stage2_training_log = gr.Textbox(
                            label="训练进度",
                            lines=30,
                            autoscroll=True,
                            interactive=False
                        )

                # Stage2Training函数
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
                    yield "🚀 正在初始化Stage2训练...\n"

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
                        yield f"❌ 训练失败:\n{str(e)}\n\n{traceback.format_exc()}"

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

            # Tab 5: 综合推理模型生成
            with gr.Tab("🎯 综合推理模型", elem_id="ensemble_model"):
                gr.Markdown("## 生成综合推理模型 (Delta R² 策略)")
                gr.Markdown("""
                **优化后的策略**：
                - 使用测试集数据评估每个信号的Delta R² = R²_ensemble - R²_stage1
                - 只对Delta R² > 阈值的信号应用Stage2修正（说明Stage2确实能提升性能）
                - 其余信号仅使用Stage1预测
                - 自动生成所有信号的分析报告和CSV汇总文件
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 🔧 模型选择")
                        base_model_selector = gr.Dropdown(
                            choices=get_available_models(),
                            label="选择基础SST模型"
                        )
                        stage2_model_selector = gr.Dropdown(
                            choices=get_stage2_model_keys(),
                            label="选择Stage2模型"
                        )
                        refresh_ensemble_btn = gr.Button("🔄 刷新", size="sm")

                        gr.Markdown("### 📤 加载Stage2模型（可选）")
                        gr.Markdown("从saved_models/stage2_boost文件夹加载预训练的Stage2模型")

                        stage2_inference_config_selector = gr.Dropdown(
                            choices=get_stage2_inference_config_files(),
                            label="选择Stage2推理配置文件",
                            info="选择 *_inference.json 文件"
                        )
                        with gr.Row():
                            load_stage2_inference_btn = gr.Button("📥 加载配置", size="sm", variant="secondary")
                            refresh_stage2_inference_btn = gr.Button("🔄 刷新配置", size="sm")

                        stage2_model_file_selector = gr.Dropdown(
                            choices=get_stage2_model_files(),
                            label="选择Stage2模型文件",
                            info="选择 *.pth 文件"
                        )
                        with gr.Row():
                            load_stage2_model_btn = gr.Button("📥 加载模型", size="sm", variant="secondary")
                            refresh_stage2_model_btn = gr.Button("🔄 刷新模型", size="sm")

                        gr.Markdown("### 📊 加载Stage2 Scalers（可选）")
                        gr.Markdown("如果模型加载失败提示缺少scalers，可在此手动加载")
                        stage2_scalers_selector = gr.Dropdown(
                            choices=get_stage2_scalers_files(),
                            label="选择Stage2 Scalers文件",
                            info="选择 *_scalers.pkl 文件"
                        )
                        with gr.Row():
                            load_stage2_scalers_btn = gr.Button("📥 加载Scalers", size="sm", variant="secondary")
                            refresh_stage2_scalers_btn = gr.Button("🔄 刷新Scalers", size="sm")

                        stage2_load_status = gr.Textbox(label="Stage2加载状态", lines=5, interactive=False)

                        gr.Markdown("### 🎚️ Delta R²阈值设置")
                        delta_r2_threshold_slider = gr.Slider(
                            0.0, 0.5, 0.05, 0.01,
                            label="Delta R²阈值",
                            info="只对Delta R² > 阈值的信号应用Stage2修正（0.05 = 5%提升）"
                        )

                        generate_ensemble_btn = gr.Button("🎯 生成综合模型", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        ensemble_status = gr.Textbox(
                            label="生成状态",
                            lines=30,
                            autoscroll=True,
                            interactive=False
                        )

                # 添加可视化输出
                with gr.Row():
                    ensemble_visualization = gr.Plot(
                        label="综合模型分析可视化",
                        show_label=True
                    )

                def generate_ensemble_ui(base_model_name, stage2_model_name, delta_r2_threshold):
                    if not base_model_name or not stage2_model_name:
                        return "❌ 请选择基础模型和Stage2模型！", None

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

            # Tab 6: 二次推理比较
            with gr.Tab("📊 二次推理比较", elem_id="reinference_comparison"):
                gr.Markdown("## 二次推理比较")
                gr.Markdown("选择index范围，比较综合模型与纯SST模型的性能提升")

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 🎯 模型选择")
                        ensemble_selector_reinf = gr.Dropdown(
                            choices=get_ensemble_model_keys(),
                            label="选择综合模型"
                        )
                        refresh_reinf_btn = gr.Button("🔄 刷新", size="sm")

                        gr.Markdown("### 📏 Index范围选择")
                        with gr.Row():
                            reinf_start_idx = gr.Number(value=0, label="起始Index", precision=0)
                            reinf_end_idx = gr.Number(value=1000, label="结束Index", precision=0)

                        compare_reinf_btn = gr.Button("📊 执行比较", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        reinf_status = gr.Textbox(
                            label="比较结果",
                            lines=20,
                            autoscroll=True,
                            interactive=False
                        )
                        reinf_plot = gr.Plot(label="性能对比可视化")

                def compare_reinference_ui(ensemble_name, start_idx, end_idx):
                    """
                    Compare reinference results with visualization of ALL signals
                    and export CSV with predictions, actual values, and R² scores
                    """
                    if not ensemble_name:
                        return "❌ 请选择综合模型！", None

                    if ensemble_name not in global_state['ensemble_models']:
                        return "❌ 综合模型不存在！", None

                    try:
                        ensemble_info = global_state['ensemble_models'][ensemble_name]

                        # 获取预测数据
                        y_true = ensemble_info['predictions']['y_true']
                        y_pred_base = ensemble_info['predictions']['y_pred_base']
                        y_pred_ensemble = ensemble_info['predictions']['y_pred_ensemble']

                        # Get signal names
                        signal_names = ensemble_info.get('signals', {}).get('target', [])
                        num_signals = y_true.shape[1]
                        if not signal_names or len(signal_names) != num_signals:
                            signal_names = [f'Signal_{i+1}' for i in range(num_signals)]

                        # 切片
                        start_idx = max(0, int(start_idx))
                        end_idx = min(len(y_true), int(end_idx))

                        y_true_seg = y_true[start_idx:end_idx]
                        y_pred_base_seg = y_pred_base[start_idx:end_idx]
                        y_pred_ensemble_seg = y_pred_ensemble[start_idx:end_idx]

                        # 计算整体性能
                        mae_base = mean_absolute_error(y_true_seg, y_pred_base_seg)
                        mae_ensemble = mean_absolute_error(y_true_seg, y_pred_ensemble_seg)
                        rmse_base = np.sqrt(mean_squared_error(y_true_seg, y_pred_base_seg))
                        rmse_ensemble = np.sqrt(mean_squared_error(y_true_seg, y_pred_ensemble_seg))
                        r2_base_overall, _ = compute_r2_safe(y_true_seg, y_pred_base_seg, method='per_output_mean')
                        r2_ensemble_overall, _ = compute_r2_safe(y_true_seg, y_pred_ensemble_seg, method='per_output_mean')

                        # 计算每个信号的R²
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

                        status = f"📊 二次推理比较结果\n"
                        status += f"=" * 60 + "\n\n"
                        status += f"📏 Index范围: [{start_idx}, {end_idx})\n"
                        status += f"📈 样本数: {len(y_true_seg):,}\n"
                        status += f"🎯 输出信号数: {num_signals}\n\n"

                        status += f"整体性能对比:\n"
                        status += f"{'指标':<15} {'Stage1':>15} {'Ensemble':>15} {'改进':>12}\n"
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

                        status += f"✅ CSV结果已保存:\n"
                        status += f"   📁 预测数据: {csv_filename}\n"
                        status += f"   📁 R²汇总: {r2_summary_filename}\n\n"

                        status += f"各信号R²对比:\n"
                        status += f"{'信号':<20} {'Stage1 R²':>12} {'Ensemble R²':>12} {'Delta R²':>12}\n"
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
                        error_msg = f"❌ 比较失败:\n{str(e)}\n\n{traceback.format_exc()}"
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

            # # Tab 7: Sundial时序预测 (已注释 - 功能待开发)
            # with gr.Tab("🔮 Sundial残差预测", elem_id="sundial_forecast"):
            #     gr.Markdown("## Sundial时序模型预测未来残差")
            #     gr.Markdown("基于综合模型的最终残差，使用Sundial预测未来残差趋势")
            #
            #     with gr.Row():
            #         with gr.Column(scale=1):
            #             gr.Markdown("### 🎯 模型选择")
            #             ensemble_selector_sundial = gr.Dropdown(
            #                 choices=get_ensemble_model_keys(),
            #                 label="选择综合模型"
            #             )
            #             refresh_sundial_btn = gr.Button("🔄 刷新", size="sm")
            #
            #             gr.Markdown("### ⚙️ Sundial配置")
            #             sundial_forecast_horizon = gr.Slider(
            #                 10, 500, 100, 10,
            #                 label="预测步数",
            #                 info="预测未来多少个时间步的残差"
            #             )
            #
            #             with gr.Row():
            #                 sundial_d_model = gr.Slider(32, 256, 64, 32, label="模型维度")
            #                 sundial_nhead = gr.Slider(2, 8, 4, 2, label="注意力头数")
            #
            #             train_sundial_btn = gr.Button("🔮 TrainingSundial", variant="primary", size="lg")
            #             predict_sundial_btn = gr.Button("📈 预测未来残差", size="lg")
            #
            #         with gr.Column(scale=1):
            #             sundial_status = gr.Textbox(
            #                 label="状态信息",
            #                 lines=20,
            #                 autoscroll=True,
            #                 interactive=False
            #             )
            #             sundial_plot = gr.Plot(label="残差预测可视化")
            #
            #     def train_sundial_ui(ensemble_name, forecast_horizon, d_model, nhead):
            #         if not ensemble_name:
            #             return "❌ 请选择综合模型！", None
            #
            #         # 这里是SundialTraining的占位符，实际需要实现Sundial模型
            #         status = f"🔮 Sundial模型训练\n"
            #         status += f"=" * 60 + "\n\n"
            #         status += f"📊 综合模型: {ensemble_name}\n"
            #         status += f"📏 预测步数: {int(forecast_horizon)}\n"
            #         status += f"⚙️ 模型配置: d_model={int(d_model)}, nhead={int(nhead)}\n\n"
            #         status += f"⚠️ Sundial模型训练功能正在开发中...\n"
            #         status += f"💡 Sundial将基于时间序列特性预测未来残差趋势\n"
            #
            #         return status, None
            #
            #     def predict_sundial_ui(ensemble_name):
            #         if not ensemble_name:
            #             return "❌ 请选择综合模型！", None
            #
            #         status = f"📈 Sundial残差预测\n"
            #         status += f"=" * 60 + "\n\n"
            #         status += f"⚠️ Sundial预测功能正在开发中...\n"
            #
            #         return status, None
            #
            #     refresh_sundial_btn.click(
            #         fn=lambda: gr.update(choices=get_ensemble_model_keys()),
            #         outputs=[ensemble_selector_sundial]
            #     )
            #
            #     train_sundial_btn.click(
            #         fn=train_sundial_ui,
            #         inputs=[ensemble_selector_sundial, sundial_forecast_horizon,
            #                 sundial_d_model, sundial_nhead],
            #         outputs=[sundial_status, sundial_plot]
            #     )
            #
            #     predict_sundial_btn.click(
            #         fn=predict_sundial_ui,
            #         inputs=[ensemble_selector_sundial],
            #         outputs=[sundial_status, sundial_plot]
            #     )

        # Footer info
        gr.Markdown("""
        ---
        ## 📖 使用流程

        ### 完整流程
        1️⃣ **数据加载** → 上传CSV或Create sample data
        2️⃣ **SST模型Training** → Training静态传感器映射Transformer
        3️⃣ **残差提取** → 从SST模型提取预测残差
        4️⃣ **Stage2Training** → TrainingStage2 residual model
        5️⃣ **生成综合模型** → 智能R²阈值选择，生成综合推理模型
        6️⃣ **二次推理比较** → 对比综合模型与SST模型的性能提升

        **🎯 创新点**:
        - ✨ Stage2 Boost架构：针对性改进低R²信号
        - 🎯 智能阈值选择：自动决定哪些信号需要Stage2
        - 📊 综合推理模型：最优组合SST和Stage2
        - 📈 全信号可视化：每个输出信号独立对比分析
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

        # JSON配置加载事件
        def load_json_from_selector(json_path):
            """Load JSON config from dropdown selector"""
            if not json_path:
                return gr.update(), gr.update(), "⚠️ 请选择JSON配置文件"
            boundary, target, status = load_signals_config_from_json(json_path)
            return gr.update(value=boundary), gr.update(value=target), status

        def load_json_from_upload(json_file):
            """Load JSON config from uploaded file"""
            if not json_file:
                return gr.update(), gr.update(), "⚠️ 请上传JSON配置文件"
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

        # Training按钮绑定
        train_btn_static.click(
            fn=train_base_model_ui,
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

        # Stop按钮绑定 - Tab2
        def stop_training_tab2():
            global_state['stop_training_tab2'] = True
            return "⚠️  已发送停止信号，训练将在当前 epoch 结束后停止..."

        stop_btn_tab2.click(
            fn=stop_training_tab2,
            outputs=[training_log_static]
        )

        # Stop按钮绑定 - Tab4
        def stop_training_tab4():
            global_state['stop_training_tab4'] = True
            return "⚠️  已发送停止信号，训练将在当前 epoch 结束后停止..."

        stop_btn_tab4.click(
            fn=stop_training_tab4,
            outputs=[stage2_training_log]
        )

    return demo


# ============================================================================
# Launch application

if __name__ == "__main__":
    import sys

    print("启动工业数字孪生残差Boost训练系统...")
    print("="*80)

    # Create necessary directories
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("saved_models/stage2_boost", exist_ok=True)
    os.makedirs("saved_models/ensemble", exist_ok=True)
    os.makedirs("saved_models/reinference_results", exist_ok=True)
    print("✅ 已创建必要的模型保存目录")

    # Check if running in Colab
    try:
        import google.colab
        IN_COLAB = True
        print("✅ 检测到Colab环境")
    except:
        IN_COLAB = False
        print("✅ 本地环境")

    demo = create_unified_interface()
    print("✅ 界面创建完成")
    print("="*80)

    if IN_COLAB:
        # Colab environment - use share=True for public URL
        print("\n🌐 在Colab中启动Gradio...")
        print("📝 提示：Gradio将生成一个公网链接")
        demo.launch(
            share=True,
            debug=True,
            show_error=True,
            inline=False  # Use separate window
        )
    else:
        # Local environment - try multiple ports
        print("\n🌐 在本地环境中启动Gradio...")
        for port in range(7860, 7870):
            try:
                print(f"尝试端口 {port}...")
                demo.launch(
                    server_name="127.0.0.1",
                    server_port=port,
                    share=False,
                    debug=True,
                    show_error=True,
                    quiet=False
                )
                print(f"✅ 服务器启动成功！")
                print(f"🔗 访问地址: http://localhost:{port}")
                print("="*80)
                break
            except OSError:
                print(f"⚠️  端口 {port} 被占用，尝试下一个...")
                continue
        else:
            print("❌ 无法找到可用端口 (7860-7869)")
