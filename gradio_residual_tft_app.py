#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Gradio Interface for Industrial Digital Twin with Residual Boost Training
完整的残差Boost训练系统 - 增强Gradio界面
新增功能：
1. Stage2残差模型训练（基于SST生成的残差）
2. 智能R2阈值选择生成综合推理模型
3. 二次推理比较（综合模型 vs 纯SST模型）
4. Sundial时序模型预测未来残差
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
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
# TFT模型保存函数（在文件顶部添加）
# ============================================================================

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
    保存TFT模型、配置和scalers

    Args:
        model_name: TFT模型名称
        tft_model: 训练好的TFT模型
        config: 训练配置
        scalers: 数据标准化器
        residual_data_key: 残差数据标识
        residual_info: 残差数据信息
        history: 训练历史

    Returns:
        model_path: 模型权重文件路径
        scaler_path: Scaler文件路径
        inference_config_path: 推理配置文件路径
    """
    model_dir = "saved_models/tft_models"
    os.makedirs(model_dir, exist_ok=True)

    # 1. 保存模型权重
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

    # 2. 保存Scalers
    scaler_path = os.path.join(model_dir, f"{model_name}_scalers.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scalers, f)

    # 3. 保存推理配置JSON（最重要）
    inference_config_path = os.path.join(model_dir, f"{model_name}_inference.json")
    inference_config = {
        'model_name': model_name,
        'model_type': 'ResidualTFT',
        'model_path': model_path,
        'scaler_path': scaler_path,

        # TFT模型架构
        'architecture': {
            'd_model': config['d_model'],
            'nhead': config['nhead'],
            'num_encoder_layers': config['num_encoder_layers'],
            'num_decoder_layers': config['num_decoder_layers'],
            'dropout': config['dropout'],
            'use_grouping': config.get('use_grouping', False),
            'signal_groups': config.get('signal_groups', None)
        },

        # 数据配置
        'data_config': {
            'encoder_length': config['encoder_length'],
            'future_horizon': residual_info['future_horizon'],
            'residual_data_key': residual_data_key,
            'base_model_name': residual_info['base_model_name'],
            'num_targets': len(residual_info['target_signals']),
            'num_external_factors': len(residual_info['boundary_signals'])
        },

        # 信号信息
        'signals': {
            'boundary_signals': residual_info['boundary_signals'],
            'target_signals': residual_info['target_signals'],
            'residual_signals': residual_info['residual_signals']
        },

        # 训练信息
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
# TFT模型加载函数
# ============================================================================

def load_tft_model_from_config(config_file_path: str, device: torch.device) -> Tuple[str, str]:
    """
    从推理配置文件加载TFT模型

    Args:
        config_file_path: 推理配置JSON文件路径
        device: PyTorch设备

    Returns:
        model_name: 模型名称
        status_msg: 加载状态消息
    """
    try:
        # 读取配置
        with open(config_file_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        model_name = config['model_name']
        model_path = config['model_path']
        scaler_path = config['scaler_path']

        # 检查文件是否存在
        if not os.path.exists(model_path):
            return None, f"❌ 模型文件不存在: {model_path}"
        if not os.path.exists(scaler_path):
            return None, f"❌ Scaler文件不存在: {scaler_path}"

        # 加载模型
        checkpoint = torch.load(model_path, map_location=device)
        model_config = checkpoint['model_config']

        # 重建TFT模型
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

        # 加载Scalers
        with open(scaler_path, 'rb') as f:
            scalers = pickle.load(f)

        # 保存到全局状态
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

        # 构建状态消息
        status_msg = f"✅ TFT模型加载成功!\n\n"
        status_msg += f"📌 模型名称: {model_name}\n"
        status_msg += f"📊 基础模型: {config['data_config']['base_model_name']}\n"
        status_msg += f"🎯 目标信号数: {config['data_config']['num_targets']}\n"
        status_msg += f"📈 边界信号数: {config['data_config']['num_external_factors']}\n"
        status_msg += f"📏 历史窗口长度: {config['data_config']['encoder_length']}\n"
        status_msg += f"🔮 未来预测长度: {config['data_config']['future_horizon']}\n"
        status_msg += f"⚙️ 模型维度: {config['architecture']['d_model']}\n"
        status_msg += f"🕒 创建时间: {config['created_time']}\n"

        if 'training_info' in config:
            ti = config['training_info']
            status_msg += f"\n📚 训练信息:\n"
            status_msg += f"   - 训练轮数: {ti['epochs_trained']}\n"
            status_msg += f"   - 最佳验证损失: {ti['best_val_loss']:.6f}\n"
            status_msg += f"   - 批大小: {ti['batch_size']}\n"
            status_msg += f"   - 学习率: {ti['learning_rate']}\n"

        print(status_msg)
        return model_name, status_msg

    except Exception as e:
        error_msg = f"❌ TFT模型加载失败:\n{str(e)}\n\n{traceback.format_exc()}"
        print(error_msg)
        return None, error_msg


# ============================================================================
# 配置中文字体
# ============================================================================

def configure_chinese_font():
    """配置matplotlib支持中文显示"""
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
# 导入模块
# ============================================================================

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
        prepare_residual_sequence_data
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
            prepare_residual_sequence_data
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
    """从文件系统加载已保存的模型"""
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
                checkpoint = torch.load(model_path, map_location=device)
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
    # 新增：Stage2 Boost模型存储
    'stage2_models': {},  # Stage2残差模型
    'stage2_scalers': {},  # Stage2 Scalers
    'ensemble_models': {},  # 综合推理模型 (SST + Stage2)
    'sundial_models': {},  # Sundial时序预测模型
}

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
# 模型加载和推理配置管理
# ============================================================================

def save_inference_config(model_name, model_type, model_path, scaler_path,
                          boundary_signals, target_signals, config_dict):
    """
    保存推理配置文件 - 用于后续直接加载模型进行推理

    Args:
        model_name: 模型名称
        model_type: 模型类型
        model_path: 模型权重文件路径
        scaler_path: Scaler文件路径
        boundary_signals: 边界信号列表
        target_signals: 目标信号列表
        config_dict: 模型架构配置
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
    从推理配置文件加载模型

    Args:
        config_file_path: 推理配置JSON文件路径
        device: PyTorch设备

    Returns:
        model_name: 模型名称
        success_msg: 成功消息
    """
    try:
        with open(config_file_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        model_name = config['model_name']
        model_type = config['model_type']
        model_path = config['model_path']
        scaler_path = config['scaler_path']

        # 检查文件是否存在
        if not os.path.exists(model_path):
            return None, f"❌ 模型文件不存在: {model_path}"
        if not os.path.exists(scaler_path):
            return None, f"❌ Scaler文件不存在: {scaler_path}"

        # 加载模型
        checkpoint = torch.load(model_path, map_location=device)
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
            return None, f"❌ 不支持的模型类型: {model_type}"

        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

        # 加载scalers
        with open(scaler_path, 'rb') as f:
            scalers = pickle.load(f)

        # 保存到全局状态
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
        success_msg += f"📌 模型名称: {model_name}\n"
        success_msg += f"📊 模型类型: {model_type}\n"
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
# Stage2 Boost模型定义和训练函数
# ============================================================================

def train_stage2_boost_model(
        residual_data_key: str,
        config: Dict[str, Any],
        progress_callback=None
) -> Tuple[str, Dict[str, Any]]:
    """
    训练Stage2 Boost残差模型

    Args:
        residual_data_key: 残差数据标识
        config: 训练配置
        progress_callback: 进度回调函数

    Returns:
        status_msg: 训练状态消息
        results: 训练结果字典
    """
    try:
        if residual_data_key not in global_state['residual_data']:
            return "❌ 残差数据不存在！", {}

        log_msg = []
        log_msg.append("=" * 80)
        log_msg.append("🚀 开始训练 Stage2 Boost 残差模型")
        log_msg.append("=" * 80)

        # 获取残差数据
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

        # 准备训练数据
        X = residuals_df[boundary_signals].values
        y_residual = residuals_df[residual_signals].values

        # 数据分割
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

        # 数据标准化
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_train_scaled = scaler_X.fit_transform(X_train)
        X_val_scaled = scaler_X.transform(X_val)
        X_test_scaled = scaler_X.transform(X_test)

        y_train_scaled = scaler_y.fit_transform(y_train)
        y_val_scaled = scaler_y.transform(y_val)
        y_test_scaled = scaler_y.transform(y_test)

        # 创建DataLoader
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

        # 初始化Stage2模型（使用SST架构）
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

        # 优化器和调度器
        optimizer = torch.optim.AdamW(
            stage2_model.parameters(),
            lr=config['lr'],
            weight_decay=config.get('weight_decay', 1e-5)
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=config.get('scheduler_patience', 10),
            verbose=True
        )

        criterion = nn.MSELoss()

        # 训练循环
        log_msg.append(f"\n🎯 开始训练 (总轮数: {config['epochs']})")

        history = {
            'train_losses': [],
            'val_losses': [],
            'train_r2': [],
            'val_r2': []
        }

        best_val_loss = float('inf')
        patience_counter = 0
        early_stop_patience = config.get('early_stop_patience', 25)

        for epoch in range(config['epochs']):
            # 训练阶段
            stage2_model.train()
            train_loss = 0.0
            train_preds = []
            train_targets = []

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                optimizer.zero_grad()
                outputs = stage2_model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()

                # 梯度裁剪
                if config.get('grad_clip', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(stage2_model.parameters(), config['grad_clip'])

                optimizer.step()

                train_loss += loss.item()
                train_preds.append(outputs.detach().cpu().numpy())
                train_targets.append(batch_y.detach().cpu().numpy())

            train_loss /= len(train_loader)
            train_preds = np.vstack(train_preds)
            train_targets = np.vstack(train_targets)
            train_r2 = r2_score(train_targets, train_preds)

            # 验证阶段
            stage2_model.eval()
            val_loss = 0.0
            val_preds = []
            val_targets = []

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = stage2_model(batch_X)
                    loss = criterion(outputs, batch_y)

                    val_loss += loss.item()
                    val_preds.append(outputs.cpu().numpy())
                    val_targets.append(batch_y.cpu().numpy())

            val_loss /= len(val_loader)
            val_preds = np.vstack(val_preds)
            val_targets = np.vstack(val_targets)
            val_r2 = r2_score(val_targets, val_preds)

            # 记录历史
            history['train_losses'].append(train_loss)
            history['val_losses'].append(val_loss)
            history['train_r2'].append(train_r2)
            history['val_r2'].append(val_r2)

            # 学习率调度
            scheduler.step(val_loss)

            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                # 保存最佳模型
                best_model_state = stage2_model.state_dict().copy()
            else:
                patience_counter += 1

            # 进度输出
            if (epoch + 1) % max(1, config['epochs'] // 20) == 0 or epoch == 0 or epoch == config['epochs'] - 1:
                msg = f"Epoch {epoch + 1}/{config['epochs']} - "
                msg += f"Train Loss: {train_loss:.6f}, Train R²: {train_r2:.4f} | "
                msg += f"Val Loss: {val_loss:.6f}, Val R²: {val_r2:.4f}"
                log_msg.append(msg)

                if progress_callback:
                    progress_callback("\n".join(log_msg))

            # 早停
            if patience_counter >= early_stop_patience:
                log_msg.append(f"\n⏸️ 早停触发 (Epoch {epoch + 1})")
                break

        # 加载最佳模型
        stage2_model.load_state_dict(best_model_state)

        # 测试集评估
        stage2_model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
            y_test_pred_scaled = stage2_model(X_test_tensor).cpu().numpy()

        y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled)

        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_r2 = r2_score(y_test, y_test_pred)

        log_msg.append(f"\n📊 测试集性能:")
        log_msg.append(f"  MAE: {test_mae:.6f}")
        log_msg.append(f"  RMSE: {test_rmse:.6f}")
        log_msg.append(f"  R²: {test_r2:.4f}")

        # 保存模型
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

        # 保存Scalers
        scaler_path = os.path.join(model_dir, f"{model_name}_scalers.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump({'X': scaler_X, 'y': scaler_y}, f)

        # 保存到全局状态
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


def compute_signal_r2_and_select_threshold(
        base_model_name: str,
        stage2_model_name: str,
        r2_threshold: float = 0.4
) -> Tuple[str, Dict[str, Any]]:
    """
    计算每个信号的R²分数，智能选择R²阈值，生成综合推理模型

    Args:
        base_model_name: 基础SST模型名称
        stage2_model_name: Stage2残差模型名称
        r2_threshold: R²阈值（只对R² < 阈值的信号应用Stage2）

    Returns:
        status_msg: 状态消息
        ensemble_info: 综合模型信息
    """
    try:
        log_msg = []
        log_msg.append("=" * 80)
        log_msg.append("🎯 计算信号R²分数并生成综合推理模型")
        log_msg.append("=" * 80)

        # 检查模型是否存在
        if base_model_name not in global_state['trained_models']:
            return f"❌ 基础模型 {base_model_name} 不存在！", {}

        if stage2_model_name not in global_state['stage2_models']:
            return f"❌ Stage2模型 {stage2_model_name} 不存在！", {}

        # 获取模型
        base_model_info = global_state['trained_models'][base_model_name]
        stage2_model_info = global_state['stage2_models'][stage2_model_name]

        base_model = base_model_info['model']
        stage2_model = stage2_model_info['model']

        # 获取残差数据
        residual_data_key = stage2_model_info['residual_data_key']
        if residual_data_key not in global_state['residual_data']:
            return f"❌ 残差数据 {residual_data_key} 不存在！", {}

        residuals_df = global_state['residual_data'][residual_data_key]['data']
        residual_info = global_state['residual_data'][residual_data_key]['info']

        boundary_signals = residual_info['boundary_signals']
        target_signals = residual_info['target_signals']

        log_msg.append(f"\n📊 数据信息:")
        log_msg.append(f"  基础模型: {base_model_name}")
        log_msg.append(f"  Stage2模型: {stage2_model_name}")
        log_msg.append(f"  目标信号数: {len(target_signals)}")

        # 获取真实值和预测值
        y_true_cols = [f"{sig}_true" for sig in target_signals]
        y_pred_cols = [f"{sig}_pred" for sig in target_signals]

        y_true = residuals_df[y_true_cols].values
        y_pred_base = residuals_df[y_pred_cols].values

        # Stage2预测残差
        X = residuals_df[boundary_signals].values
        X_scaled = global_state['stage2_scalers'][stage2_model_name]['X'].transform(X)

        stage2_model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(device)
            y_residual_pred_scaled = stage2_model(X_tensor).cpu().numpy()

        y_residual_pred = global_state['stage2_scalers'][stage2_model_name]['y'].inverse_transform(
            y_residual_pred_scaled
        )

        # 计算每个信号的R²
        signal_r2_scores = []
        for i, signal in enumerate(target_signals):
            r2 = r2_score(y_true[:, i], y_pred_base[:, i])
            signal_r2_scores.append({
                'signal': signal,
                'r2': r2,
                'apply_stage2': r2 < r2_threshold
            })

        log_msg.append(f"\n🎯 信号R²分析 (阈值: {r2_threshold}):")
        log_msg.append(f"{'信号名称':<30} {'R²分数':>10} {'应用Stage2':>12}")
        log_msg.append("-" * 55)

        num_signals_use_stage2 = 0
        for item in signal_r2_scores:
            log_msg.append(
                f"{item['signal']:<30} {item['r2']:>10.4f} {'✓' if item['apply_stage2'] else '✗':>12}"
            )
            if item['apply_stage2']:
                num_signals_use_stage2 += 1

        log_msg.append("-" * 55)
        log_msg.append(f"需要Stage2的信号数: {num_signals_use_stage2} / {len(target_signals)}")

        # 生成综合预测（选择性应用Stage2）
        y_ensemble = y_pred_base.copy()
        for i, item in enumerate(signal_r2_scores):
            if item['apply_stage2']:
                # 应用Stage2修正
                y_ensemble[:, i] = y_pred_base[:, i] + y_residual_pred[:, i]

        # 计算综合模型性能
        mae_base = mean_absolute_error(y_true, y_pred_base)
        mae_ensemble = mean_absolute_error(y_true, y_ensemble)
        rmse_base = np.sqrt(mean_squared_error(y_true, y_pred_base))
        rmse_ensemble = np.sqrt(mean_squared_error(y_true, y_ensemble))
        r2_base = r2_score(y_true, y_pred_base)
        r2_ensemble = r2_score(y_true, y_ensemble)

        improvement_mae = (mae_base - mae_ensemble) / mae_base * 100
        improvement_rmse = (rmse_base - rmse_ensemble) / rmse_base * 100
        improvement_r2 = (r2_ensemble - r2_base) / (1 - r2_base) * 100

        log_msg.append(f"\n📈 性能对比:")
        log_msg.append(f"{'指标':<15} {'基础模型(SST)':>18} {'综合模型':>18} {'改进':>15}")
        log_msg.append("-" * 70)
        log_msg.append(f"{'MAE':<15} {mae_base:>18.6f} {mae_ensemble:>18.6f} {improvement_mae:>14.2f}%")
        log_msg.append(f"{'RMSE':<15} {rmse_base:>18.6f} {rmse_ensemble:>18.6f} {improvement_rmse:>14.2f}%")
        log_msg.append(f"{'R²':<15} {r2_base:>18.4f} {r2_ensemble:>18.4f} {improvement_r2:>14.2f}%")

        # 保存综合模型信息
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        ensemble_name = f"Ensemble_{base_model_name}_{stage2_model_name}_{timestamp}"

        ensemble_info = {
            'name': ensemble_name,
            'base_model_name': base_model_name,
            'stage2_model_name': stage2_model_name,
            'r2_threshold': r2_threshold,
            'signal_r2_scores': signal_r2_scores,
            'num_signals_use_stage2': num_signals_use_stage2,
            'metrics': {
                'base': {'mae': mae_base, 'rmse': rmse_base, 'r2': r2_base},
                'ensemble': {'mae': mae_ensemble, 'rmse': rmse_ensemble, 'r2': r2_ensemble},
                'improvement': {
                    'mae_pct': improvement_mae,
                    'rmse_pct': improvement_rmse,
                    'r2_pct': improvement_r2
                }
            },
            'predictions': {
                'y_true': y_true,
                'y_pred_base': y_pred_base,
                'y_pred_ensemble': y_ensemble,
                'y_residual_pred': y_residual_pred
            },
            'signals': {
                'boundary': boundary_signals,
                'target': target_signals
            },
            'created_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        global_state['ensemble_models'][ensemble_name] = ensemble_info

        # 保存综合模型配置
        ensemble_dir = "saved_models/ensemble"
        os.makedirs(ensemble_dir, exist_ok=True)

        config_path = os.path.join(ensemble_dir, f"{ensemble_name}_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            # 保存配置（不包含大数组）
            save_config = {
                'name': ensemble_name,
                'base_model_name': base_model_name,
                'stage2_model_name': stage2_model_name,
                'r2_threshold': r2_threshold,
                'signal_r2_scores': signal_r2_scores,
                'num_signals_use_stage2': num_signals_use_stage2,
                'metrics': ensemble_info['metrics'],
                'signals': ensemble_info['signals'],
                'created_time': ensemble_info['created_time']
            }
            json.dump(save_config, f, indent=2, ensure_ascii=False)

        log_msg.append(f"\n✅ 综合推理模型已生成:")
        log_msg.append(f"  模型名称: {ensemble_name}")
        log_msg.append(f"  配置路径: {config_path}")

        return "\n".join(log_msg), ensemble_info

    except Exception as e:
        error_msg = f"❌ 综合模型生成失败:\n{str(e)}\n\n{traceback.format_exc()}"
        print(error_msg)
        return error_msg, {}


# ============================================================================
# 数据加载函数
# ============================================================================

def load_data_from_csv(file_obj):
    """从CSV文件加载数据"""
    try:
        df = pd.read_csv(file_obj.name)

        # 如果有unnamed列，设为index
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

        return status, df, signals_display

    except Exception as e:
        error_msg = f"❌ 数据加载失败: {str(e)}"
        return error_msg, None, ""


def create_sample_data():
    """创建示例数据"""
    try:
        np.random.seed(42)
        n_samples = 10000
        n_boundary = 10
        n_target = 5

        # 生成相关的传感器数据
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

        return status, df, signals_display

    except Exception as e:
        error_msg = f"❌ 示例数据创建失败: {str(e)}"
        return error_msg, None, ""


# ============================================================================
# SST模型训练函数
# ============================================================================

def train_base_model_ui(
        boundary_signals, target_signals, model_type,
        epochs, batch_size, lr,
        d_model, nhead, num_layers, dropout,
        test_size, val_size,
        temporal_signals=None, apply_smoothing=False,
        progress=gr.Progress()
):
    """训练基础模型的UI函数"""
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

        # 准备数据
        X = df[boundary_signals].values
        y = df[target_signals].values

        # 应用IFD平滑（如果需要）
        if apply_smoothing and temporal_signals:
            log_messages.append(f"\n🔧 应用IFD平滑...")
            for sig in temporal_signals:
                if sig in df.columns:
                    df[sig] = apply_ifd_smoothing(df[sig].values)

        # 数据分割
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

        # 数据标准化
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_train_scaled = scaler_X.fit_transform(X_train)
        X_val_scaled = scaler_X.transform(X_val)
        X_test_scaled = scaler_X.transform(X_test)

        y_train_scaled = scaler_y.fit_transform(y_train)
        y_val_scaled = scaler_y.transform(y_val)
        y_test_scaled = scaler_y.transform(y_test)

        # 创建DataLoader
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

        # 初始化模型
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

        # 优化器
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        criterion = nn.MSELoss()

        # 训练循环
        log_messages.append(f"\n🎯 开始训练...")
        history = {'train_losses': [], 'val_losses': []}
        best_val_loss = float('inf')
        patience_counter = 0
        early_stop_patience = 25

        for epoch in range(epochs):
            # 训练
            model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            # 验证
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()

            val_loss /= len(val_loader)

            history['train_losses'].append(train_loss)
            history['val_losses'].append(val_loss)

            scheduler.step(val_loss)

            # 早停
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1

            # 进度显示
            if (epoch + 1) % max(1, epochs // 20) == 0 or epoch == 0:
                msg = f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
                log_messages.append(msg)
                progress((epoch + 1) / epochs, desc=msg)

            if patience_counter >= early_stop_patience:
                log_messages.append(f"\n⏸️ 早停触发 (Epoch {epoch + 1})")
                break

        # 加载最佳模型
        model.load_state_dict(best_model_state)

        # 测试集评估
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
            y_test_pred_scaled = model(X_test_tensor).cpu().numpy()

        y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled)

        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_r2 = r2_score(y_test, y_test_pred)

        log_messages.append(f"\n📊 测试集性能:")
        log_messages.append(f"  MAE: {test_mae:.6f}")
        log_messages.append(f"  RMSE: {test_rmse:.6f}")
        log_messages.append(f"  R²: {test_r2:.4f}")

        # 保存模型
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

        # 保存Scalers
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

        # 保存到全局状态
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
# 残差提取函数
# ============================================================================

def extract_residuals_ui(model_name, future_horizon, use_segment, start_index, end_index):
    """残差提取UI函数"""
    try:
        if not model_name:
            return "❌ 请选择模型！", None

        if global_state['df'] is None:
            return "❌ 请先加载数据！", None

        log_msg = []
        log_msg.append("=" * 80)
        log_msg.append("📊 开始提取残差")
        log_msg.append("=" * 80)

        df = global_state['df']

        # 片段选择
        if use_segment:
            start_idx = max(0, int(start_index))
            end_idx = min(len(df), int(end_index))
            df_segment = df.iloc[start_idx:end_idx].copy()
            log_msg.append(f"\n✂️ 使用数据片段: index [{start_idx}, {end_idx})")
            log_msg.append(f"   片段长度: {len(df_segment):,}")
        else:
            df_segment = df.copy()
            log_msg.append(f"\n📈 使用全部数据: {len(df_segment):,} 条")

        # 使用ResidualExtractor提取残差
        from residual_tft import ResidualExtractor

        residuals_df, info = ResidualExtractor.extract_residuals_from_trained_models(
            model_name, df_segment, global_state, device
        )

        if residuals_df.empty:
            return "❌ 残差提取失败！", None

        # 保存残差数据
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        residual_key = f"{model_name}_{timestamp}_h{future_horizon}"

        info['future_horizon'] = future_horizon
        info['base_model_name'] = model_name

        global_state['residual_data'][residual_key] = {
            'data': residuals_df,
            'info': info
        }

        log_msg.append(f"\n✅ 残差提取完成:")
        log_msg.append(f"  残差数据标识: {residual_key}")
        log_msg.append(f"  数据形状: {residuals_df.shape}")

        # 创建残差可视化
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
# 辅助函数
# ============================================================================

def get_available_models():
    """获取可用的训练模型列表"""
    return list(global_state['trained_models'].keys())


def get_residual_data_keys():
    """获取可用的残差数据标识列表"""
    return list(global_state['residual_data'].keys())


def get_stage2_model_keys():
    """获取可用的Stage2模型列表"""
    return list(global_state['stage2_models'].keys())


def get_ensemble_model_keys():
    """获取可用的综合模型列表"""
    return list(global_state['ensemble_models'].keys())


# ============================================================================
# Gradio界面创建
# ============================================================================

def create_unified_interface():
    """创建统一的Gradio界面"""

    with gr.Blocks(title="工业数字孪生残差Boost训练系统", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🏭 工业数字孪生残差Boost训练系统
        ### Enhanced Residual Boost Training with Stage2 Model

        **新功能:**
        - ✨ Stage2残差模型训练
        - 🎯 智能R²阈值选择生成综合推理模型
        - 📊 二次推理比较（综合模型 vs 纯SST模型）
        - 🔮 Sundial时序模型预测未来残差
        """)

        with gr.Tabs():
            # Tab 1: 数据加载
            with gr.Tab("📂 数据加载", elem_id="data_loading"):
                gr.Markdown("## 上传数据或创建示例数据")

                with gr.Row():
                    with gr.Column(scale=1):
                        data_file = gr.File(label="上传CSV文件", file_types=['.csv'])
                        load_btn = gr.Button("📥 加载数据", variant="primary", size="lg")
                        sample_btn = gr.Button("🎲 创建示例数据", size="lg")

                    with gr.Column(scale=1):
                        data_status = gr.Textbox(label="数据状态", lines=10, interactive=False)
                        signals_display = gr.Textbox(label="可用信号", lines=10, interactive=False)

            # Tab 2: SST模型训练
            with gr.Tab("🎯 SST模型训练", elem_id="sst_training"):
                gr.Markdown("## 训练静态传感器映射Transformer (SST)")

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 🎛️ 信号选择")
                        boundary_signals_static = gr.Dropdown(
                            choices=[], label="边界信号 (输入)", multiselect=True
                        )
                        target_signals_static = gr.Dropdown(
                            choices=[], label="目标信号 (输出)", multiselect=True
                        )

                        gr.Markdown("### 🏗️ 模型架构")
                        with gr.Row():
                            d_model_static = gr.Slider(32, 512, 128, 32, label="模型维度")
                            nhead_static = gr.Slider(2, 16, 8, 2, label="注意力头数")
                        num_layers_static = gr.Slider(1, 8, 3, 1, label="Transformer层数")

                        gr.Markdown("### 🎯 训练参数")
                        with gr.Row():
                            epochs_static = gr.Slider(10, 500, 100, 10, label="训练轮数")
                            batch_size_static = gr.Slider(16, 256, 64, 16, label="批大小")
                        with gr.Row():
                            lr_static = gr.Number(value=0.001, label="学习率")
                            dropout_static = gr.Slider(0, 0.5, 0.1, 0.05, label="Dropout率")

                        gr.Markdown("### 🔀 数据分割")
                        with gr.Row():
                            test_size_static = gr.Slider(0.1, 0.3, 0.15, 0.05, label="测试集比例")
                            val_size_static = gr.Slider(0.1, 0.3, 0.15, 0.05, label="验证集比例")

                        train_btn_static = gr.Button("▶️ 开始训练SST", variant="primary", size="lg")

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

                with gr.Row():
                    with gr.Column(scale=1):
                        model_selector = gr.Dropdown(
                            choices=get_available_models(),
                            label="选择SST模型"
                        )
                        refresh_models_btn = gr.Button("🔄 刷新", size="sm")

                        future_horizon = gr.Slider(
                            1, 100, 10, 1,
                            label="未来预测长度",
                            info="用于后续TFT训练的未来步数"
                        )

                        gr.Markdown("### ✂️ 数据片段选择（可选）")
                        use_segment_checkbox = gr.Checkbox(
                            label="使用数据片段",
                            value=False
                        )
                        with gr.Row():
                            start_index_input = gr.Number(value=0, label="起始索引", precision=0)
                            end_index_input = gr.Number(value=10000, label="结束索引", precision=0)

                        with gr.Row():
                            preset_10k_btn = gr.Button("0-10K", size="sm")
                            preset_50k_btn = gr.Button("0-50K", size="sm")
                            preset_100k_btn = gr.Button("0-100K", size="sm")
                            preset_200k_btn = gr.Button("0-200K", size="sm")

                        extract_btn = gr.Button("🔬 提取残差", variant="primary", size="lg")

                        gr.Markdown("### 📤 加载推理配置")
                        inference_config_file = gr.File(
                            label="上传推理配置文件 (*_inference.json)",
                            file_types=['.json']
                        )
                        load_inference_btn = gr.Button("📥 加载配置", size="sm")
                        inference_load_status = gr.Textbox(label="加载状态", lines=3, interactive=False)

                    with gr.Column(scale=1):
                        residual_status = gr.Textbox(label="残差提取状态", lines=15, interactive=False)
                        residual_plot = gr.Plot(label="残差可视化")

                # 事件绑定
                def set_range_preset(start, end):
                    return start, end, True

                preset_10k_btn.click(
                    fn=lambda: set_range_preset(0, 10000),
                    outputs=[start_index_input, end_index_input, use_segment_checkbox]
                )
                preset_50k_btn.click(
                    fn=lambda: set_range_preset(0, 50000),
                    outputs=[start_index_input, end_index_input, use_segment_checkbox]
                )
                preset_100k_btn.click(
                    fn=lambda: set_range_preset(0, 100000),
                    outputs=[start_index_input, end_index_input, use_segment_checkbox]
                )
                preset_200k_btn.click(
                    fn=lambda: set_range_preset(0, 200000),
                    outputs=[start_index_input, end_index_input, use_segment_checkbox]
                )

                refresh_models_btn.click(
                    fn=lambda: gr.update(choices=get_available_models()),
                    outputs=[model_selector]
                )

                extract_btn.click(
                    fn=extract_residuals_ui,
                    inputs=[
                        model_selector,
                        future_horizon,
                        use_segment_checkbox,
                        start_index_input,
                        end_index_input
                    ],
                    outputs=[residual_status, residual_plot]
                )

                load_inference_btn.click(
                    fn=lambda f: load_model_from_inference_config(f.name, device) if f else (None, "❌ 请上传文件"),
                    inputs=[inference_config_file],
                    outputs=[model_selector, inference_load_status]
                )

            # Tab 4: Stage2 Boost训练
            with gr.Tab("🚀 Stage2 Boost训练", elem_id="stage2_training"):
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
                            d_model_stage2 = gr.Slider(32, 256, 128, 32, label="模型维度")
                            nhead_stage2 = gr.Slider(2, 16, 8, 2, label="注意力头数")
                        num_layers_stage2 = gr.Slider(1, 8, 3, 1, label="Transformer层数")

                        gr.Markdown("### 🎯 训练参数")
                        with gr.Row():
                            epochs_stage2 = gr.Slider(10, 200, 100, 10, label="训练轮数")
                            batch_size_stage2 = gr.Slider(16, 128, 32, 16, label="批大小")
                        with gr.Row():
                            lr_stage2 = gr.Number(value=0.001, label="学习率")
                            dropout_stage2 = gr.Slider(0, 0.5, 0.1, 0.05, label="Dropout率")
                        with gr.Row():
                            weight_decay_stage2 = gr.Number(value=1e-5, label="权重衰减")
                            grad_clip_stage2 = gr.Slider(0.1, 5.0, 1.0, 0.1, label="梯度裁剪")

                        gr.Markdown("### 🔀 数据分割")
                        with gr.Row():
                            test_size_stage2 = gr.Slider(0.1, 0.3, 0.15, 0.05, label="测试集比例")
                            val_size_stage2 = gr.Slider(0.1, 0.3, 0.15, 0.05, label="验证集比例")

                        train_stage2_btn = gr.Button("🚀 开始训练Stage2", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        gr.Markdown("### 📊 训练日志")
                        stage2_training_log = gr.Textbox(
                            label="训练进度",
                            lines=30,
                            autoscroll=True,
                            interactive=False
                        )

                # Stage2训练函数
                def train_stage2_ui(residual_data_key, d_model, nhead, num_layers, dropout,
                                    epochs, batch_size, lr, weight_decay, grad_clip,
                                    test_size, val_size, progress=gr.Progress()):

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
                        'test_size': float(test_size),
                        'val_size': float(val_size),
                        'early_stop_patience': 25,
                        'scheduler_patience': 10
                    }

                    def progress_callback(msg):
                        progress(0.5, desc="训练中...")
                        return msg

                    status_msg, results = train_stage2_boost_model(
                        residual_data_key, config, progress_callback
                    )

                    return status_msg

                refresh_residual_btn_stage2.click(
                    fn=lambda: gr.update(choices=get_residual_data_keys()),
                    outputs=[residual_data_selector_stage2]
                )

                train_stage2_btn.click(
                    fn=train_stage2_ui,
                    inputs=[
                        residual_data_selector_stage2,
                        d_model_stage2, nhead_stage2, num_layers_stage2, dropout_stage2,
                        epochs_stage2, batch_size_stage2, lr_stage2,
                        weight_decay_stage2, grad_clip_stage2,
                        test_size_stage2, val_size_stage2
                    ],
                    outputs=[stage2_training_log]
                )

            # Tab 5: 综合推理模型生成
            with gr.Tab("🎯 综合推理模型", elem_id="ensemble_model"):
                gr.Markdown("## 生成综合推理模型")
                gr.Markdown("智能选择R²阈值，组合SST模型和Stage2模型生成综合预测")

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

                        gr.Markdown("### 🎚️ R²阈值设置")
                        r2_threshold_slider = gr.Slider(
                            0.0, 1.0, 0.4, 0.05,
                            label="R²阈值",
                            info="只对R² < 阈值的信号应用Stage2修正"
                        )

                        generate_ensemble_btn = gr.Button("🎯 生成综合模型", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        ensemble_status = gr.Textbox(
                            label="生成状态",
                            lines=30,
                            autoscroll=True,
                            interactive=False
                        )

                def generate_ensemble_ui(base_model_name, stage2_model_name, r2_threshold):
                    if not base_model_name or not stage2_model_name:
                        return "❌ 请选择基础模型和Stage2模型！"

                    status_msg, ensemble_info = compute_signal_r2_and_select_threshold(
                        base_model_name, stage2_model_name, r2_threshold
                    )
                    return status_msg

                refresh_ensemble_btn.click(
                    fn=lambda: (gr.update(choices=get_available_models()),
                                gr.update(choices=get_stage2_model_keys())),
                    outputs=[base_model_selector, stage2_model_selector]
                )

                generate_ensemble_btn.click(
                    fn=generate_ensemble_ui,
                    inputs=[base_model_selector, stage2_model_selector, r2_threshold_slider],
                    outputs=[ensemble_status]
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

                        # 切片
                        start_idx = max(0, int(start_idx))
                        end_idx = min(len(y_true), int(end_idx))

                        y_true_seg = y_true[start_idx:end_idx]
                        y_pred_base_seg = y_pred_base[start_idx:end_idx]
                        y_pred_ensemble_seg = y_pred_ensemble[start_idx:end_idx]

                        # 计算性能
                        mae_base = mean_absolute_error(y_true_seg, y_pred_base_seg)
                        mae_ensemble = mean_absolute_error(y_true_seg, y_pred_ensemble_seg)
                        rmse_base = np.sqrt(mean_squared_error(y_true_seg, y_pred_base_seg))
                        rmse_ensemble = np.sqrt(mean_squared_error(y_true_seg, y_pred_ensemble_seg))
                        r2_base = r2_score(y_true_seg, y_pred_base_seg)
                        r2_ensemble = r2_score(y_true_seg, y_pred_ensemble_seg)

                        improvement_mae = (mae_base - mae_ensemble) / mae_base * 100
                        improvement_rmse = (rmse_base - rmse_ensemble) / rmse_base * 100

                        status = f"📊 二次推理比较结果\n"
                        status += f"=" * 60 + "\n\n"
                        status += f"📏 Index范围: [{start_idx}, {end_idx})\n"
                        status += f"📈 样本数: {len(y_true_seg):,}\n\n"

                        status += f"性能对比:\n"
                        status += f"{'指标':<15} {'SST模型':>15} {'综合模型':>15} {'改进':>12}\n"
                        status += "-" * 60 + "\n"
                        status += f"{'MAE':<15} {mae_base:>15.6f} {mae_ensemble:>15.6f} {improvement_mae:>11.2f}%\n"
                        status += f"{'RMSE':<15} {rmse_base:>15.6f} {rmse_ensemble:>15.6f} {improvement_rmse:>11.2f}%\n"
                        status += f"{'R²':<15} {r2_base:>15.4f} {r2_ensemble:>15.4f}\n"

                        # 创建可视化
                        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                        fig.suptitle(f'二次推理比较 - {ensemble_name}', fontsize=16)

                        # 预测对比（第一个信号）
                        plot_samples = min(500, len(y_true_seg))
                        axes[0, 0].plot(y_true_seg[:plot_samples, 0], label='True', alpha=0.7)
                        axes[0, 0].plot(y_pred_base_seg[:plot_samples, 0], label='SST', alpha=0.7)
                        axes[0, 0].plot(y_pred_ensemble_seg[:plot_samples, 0], label='Ensemble', alpha=0.7)
                        axes[0, 0].set_title('预测对比 (信号1)')
                        axes[0, 0].legend()
                        axes[0, 0].set_xlabel('Index')
                        axes[0, 0].set_ylabel('Value')

                        # 误差对比
                        error_base = np.abs(y_true_seg - y_pred_base_seg).mean(axis=1)
                        error_ensemble = np.abs(y_true_seg - y_pred_ensemble_seg).mean(axis=1)

                        axes[0, 1].plot(error_base[:plot_samples], label='SST Error', alpha=0.7)
                        axes[0, 1].plot(error_ensemble[:plot_samples], label='Ensemble Error', alpha=0.7)
                        axes[0, 1].set_title('平均绝对误差对比')
                        axes[0, 1].legend()
                        axes[0, 1].set_xlabel('Index')
                        axes[0, 1].set_ylabel('MAE')

                        # 误差分布
                        axes[1, 0].hist(error_base, bins=50, alpha=0.5, label='SST', edgecolor='black')
                        axes[1, 0].hist(error_ensemble, bins=50, alpha=0.5, label='Ensemble', edgecolor='black')
                        axes[1, 0].set_title('误差分布')
                        axes[1, 0].legend()
                        axes[1, 0].set_xlabel('Error')
                        axes[1, 0].set_ylabel('Frequency')

                        # 性能指标柱状图
                        metrics = ['MAE', 'RMSE', 'R²']
                        base_values = [mae_base, rmse_base, r2_base]
                        ensemble_values = [mae_ensemble, rmse_ensemble, r2_ensemble]

                        x = np.arange(len(metrics))
                        width = 0.35

                        axes[1, 1].bar(x - width / 2, base_values, width, label='SST', alpha=0.8)
                        axes[1, 1].bar(x + width / 2, ensemble_values, width, label='Ensemble', alpha=0.8)
                        axes[1, 1].set_title('性能指标对比')
                        axes[1, 1].set_xticks(x)
                        axes[1, 1].set_xticklabels(metrics)
                        axes[1, 1].legend()

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

            # Tab 7: Sundial时序预测
            with gr.Tab("🔮 Sundial残差预测", elem_id="sundial_forecast"):
                gr.Markdown("## Sundial时序模型预测未来残差")
                gr.Markdown("基于综合模型的最终残差，使用Sundial预测未来残差趋势")

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 🎯 模型选择")
                        ensemble_selector_sundial = gr.Dropdown(
                            choices=get_ensemble_model_keys(),
                            label="选择综合模型"
                        )
                        refresh_sundial_btn = gr.Button("🔄 刷新", size="sm")

                        gr.Markdown("### ⚙️ Sundial配置")
                        sundial_forecast_horizon = gr.Slider(
                            10, 500, 100, 10,
                            label="预测步数",
                            info="预测未来多少个时间步的残差"
                        )

                        with gr.Row():
                            sundial_d_model = gr.Slider(32, 256, 64, 32, label="模型维度")
                            sundial_nhead = gr.Slider(2, 8, 4, 2, label="注意力头数")

                        train_sundial_btn = gr.Button("🔮 训练Sundial", variant="primary", size="lg")
                        predict_sundial_btn = gr.Button("📈 预测未来残差", size="lg")

                    with gr.Column(scale=1):
                        sundial_status = gr.Textbox(
                            label="状态信息",
                            lines=20,
                            autoscroll=True,
                            interactive=False
                        )
                        sundial_plot = gr.Plot(label="残差预测可视化")

                def train_sundial_ui(ensemble_name, forecast_horizon, d_model, nhead):
                    if not ensemble_name:
                        return "❌ 请选择综合模型！", None

                    # 这里是Sundial训练的占位符，实际需要实现Sundial模型
                    status = f"🔮 Sundial模型训练\n"
                    status += f"=" * 60 + "\n\n"
                    status += f"📊 综合模型: {ensemble_name}\n"
                    status += f"📏 预测步数: {int(forecast_horizon)}\n"
                    status += f"⚙️ 模型配置: d_model={int(d_model)}, nhead={int(nhead)}\n\n"
                    status += f"⚠️ Sundial模型训练功能正在开发中...\n"
                    status += f"💡 Sundial将基于时间序列特性预测未来残差趋势\n"

                    return status, None

                def predict_sundial_ui(ensemble_name):
                    if not ensemble_name:
                        return "❌ 请选择综合模型！", None

                    status = f"📈 Sundial残差预测\n"
                    status += f"=" * 60 + "\n\n"
                    status += f"⚠️ Sundial预测功能正在开发中...\n"

                    return status, None

                refresh_sundial_btn.click(
                    fn=lambda: gr.update(choices=get_ensemble_model_keys()),
                    outputs=[ensemble_selector_sundial]
                )

                train_sundial_btn.click(
                    fn=train_sundial_ui,
                    inputs=[ensemble_selector_sundial, sundial_forecast_horizon,
                            sundial_d_model, sundial_nhead],
                    outputs=[sundial_status, sundial_plot]
                )

                predict_sundial_btn.click(
                    fn=predict_sundial_ui,
                    inputs=[ensemble_selector_sundial],
                    outputs=[sundial_status, sundial_plot]
                )

        # 页脚信息
        gr.Markdown("""
        ---
        ## 📖 使用流程

        ### 完整流程
        1️⃣ **数据加载** → 上传CSV或创建示例数据
        2️⃣ **SST模型训练** → 训练静态传感器映射Transformer
        3️⃣ **残差提取** → 从SST模型提取预测残差
        4️⃣ **Stage2训练** → 训练Stage2残差模型
        5️⃣ **生成综合模型** → 智能R²阈值选择，生成综合推理模型
        6️⃣ **二次推理比较** → 对比综合模型与SST模型的性能提升
        7️⃣ **Sundial预测** → 预测未来残差趋势

        **🎯 创新点**: 
        - ✨ Stage2 Boost架构：针对性改进低R²信号
        - 🎯 智能阈值选择：自动决定哪些信号需要Stage2
        - 📊 综合推理模型：最优组合SST和Stage2
        - 🔮 残差预测：Sundial预测未来残差趋势
        """)

        # 页面加载时自动刷新下拉框
        demo.load(
            fn=lambda: (
                gr.update(choices=get_available_models()),
                gr.update(choices=get_residual_data_keys()),
                gr.update(choices=get_stage2_model_keys()),
                gr.update(choices=get_ensemble_model_keys())
            ),
            outputs=[model_selector, residual_data_selector_stage2,
                     stage2_model_selector, ensemble_selector_reinf]
        )

        # 数据加载事件
        def load_data_and_update(file_obj):
            status, df, signals = load_data_from_csv(file_obj)
            if df is not None:
                cols = list(df.columns)
                return (
                    status, signals,
                    gr.update(choices=cols), gr.update(choices=cols)
                )
            return (
                status, signals,
                gr.update(choices=[]), gr.update(choices=[])
            )

        def create_sample_and_update():
            status, df, signals = create_sample_data()
            if df is not None:
                cols = list(df.columns)
                return (
                    status, signals,
                    gr.update(choices=cols), gr.update(choices=cols)
                )
            return (
                status, signals,
                gr.update(choices=[]), gr.update(choices=[])
            )

        load_btn.click(
            fn=load_data_and_update,
            inputs=[data_file],
            outputs=[
                data_status, signals_display,
                boundary_signals_static, target_signals_static
            ]
        )

        sample_btn.click(
            fn=create_sample_and_update,
            outputs=[
                data_status, signals_display,
                boundary_signals_static, target_signals_static
            ]
        )

        # 训练按钮绑定
        train_btn_static.click(
            fn=train_base_model_ui,
            inputs=[
                boundary_signals_static, target_signals_static,
                gr.Textbox(value="StaticSensorTransformer", visible=False),
                epochs_static, batch_size_static, lr_static,
                d_model_static, nhead_static, num_layers_static, dropout_static,
                test_size_static, val_size_static,
                gr.State(value=None), gr.State(value=False)
            ],
            outputs=[training_log_static]
        )

    return demo


# ============================================================================
# 启动应用
# ============================================================================

if __name__ == "__main__":
    print("启动工业数字孪生残差Boost训练系统...")
    demo = create_unified_interface()
    print("界面创建完成，启动服务器...")

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
            print(f"服务器启动成功！")
            print(f"访问地址: http://localhost:{port}")
            print("=" * 80)
            break
        except OSError:
            print(f"端口 {port} 被占用，尝试下一个...")
            continue
    else:
        print("无法找到可用端口 (7860-7869)")
