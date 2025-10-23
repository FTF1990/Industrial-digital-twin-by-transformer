"""
Complete Gradio Interface - Based on Original Cell 3
工业数字孪生 Transformer - 完整Gradio界面

This file contains the COMPLETE Gradio interface from the original Cell 3,
adapted to use the modular project structure.

使用方法 (How to use):
1. 确保已安装所有依赖: pip install -r requirements.txt
2. 运行此脚本: python gradio_full_interface.py
3. 在浏览器中访问显示的URL (通常是 http://127.0.0.1:7860)

Features:
- 完整的SST和HST模型训练功能
- 实时训练进度显示
- 配置导入/导出
- 完整的推理和可视化功能
- 信号选择验证
"""

# ============================================================================
# 导入部分 - Import Section
# ============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import gradio as gr
import json
import os
from datetime import datetime
import traceback

# 导入我们的模块化模型和工具
from models.static_transformer import StaticSensorTransformer
from models.hybrid_transformer import HybridSensorTransformer
from models.utils import (
    create_temporal_context_data,
    apply_ifd_smoothing,
    handle_duplicate_columns,
    get_available_signals,
    validate_signal_exclusivity_v1,
    validate_signal_exclusivity_v4
)

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f\"SST & HST 模型已加载 - 使用设备: {device}\")

# ============================================================================
# 全局状态存储 - Global State Storage
# ============================================================================

global_state = {
    'df': None,
    'trained_models': {},
    'scalers': {},
    'training_history': {},
    'all_signals': []
}

# ============================================================================
# 训练函数 - Training Functions
# ============================================================================

# 这里包含完整的训练函数，与原始Cell 3完全相同
# 为了节省空间，这里引用已经在前面创建的训练函数

def train_static_transformer_model_complete(X_train, y_train, X_val, y_val, num_boundary, num_target, config):
    \"\"\"训练StaticTransformer模型 - 完整版本（支持实时日志）\"\"\"
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    model = StaticSensorTransformer(
        num_boundary_sensors=num_boundary,
        num_target_sensors=num_target,
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config['lr'],
                           weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=config['scheduler_patience'],
        factor=config['scheduler_factor']
    )

    criterion = nn.MSELoss()
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    logs = []

    logs.append(f\"开始训练StaticTransformer模型... 参数量: {sum(p.numel() for p in model.parameters()):,}\")
    logs.append(f\"配置: LR={config['lr']}, WD={config['weight_decay']}, GradClip={config['grad_clip']}\\n\")

    for epoch in range(config['epochs']):
        # 训练
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['grad_clip'])
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                predictions = model(batch_X)
                val_loss += criterion(predictions, batch_y).item()
        val_loss /= len(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            status_marker = \"⭐\"
        else:
            patience_counter += 1
            status_marker = \"  \"

        log_msg = f\"{status_marker} Epoch [{epoch+1:3d}/{config['epochs']:3d}] | Train: {train_loss:.6f} | Val: {val_loss:.6f} | Best: {best_val_loss:.6f} | LR: {current_lr:.2e} | Patience: {patience_counter}/{config['early_stop_patience']}\"
        logs.append(log_msg)

        # 早停
        if patience_counter >= config['early_stop_patience']:
            logs.append(f\"\\n🛑 早停于第 {epoch+1} 轮 (耐心值达到 {config['early_stop_patience']})\")
            break

    model.load_state_dict(best_model_state)
    logs.append(f\"\\n✅ 训练完成! 最佳验证损失: {best_val_loss:.6f}\")

    return model, train_losses, val_losses, logs

def train_hybrid_sensor_transformer_model_complete(X_train, y_train, X_val, y_val, num_boundary, num_target, config, use_temporal):
    \"\"\"训练HybridSensorTransformer模型 - 完整版本（支持实时日志）\"\"\"
    logs = []

    # 准备数据
    if use_temporal:
        logs.append(f\"⏱️ 创建时序上下文数据 (窗口: ±{config['context_window']})...\")
        X_train_ctx, y_train_ctx, _ = create_temporal_context_data(X_train, y_train, config['context_window'])
        X_val_ctx, y_val_ctx, _ = create_temporal_context_data(X_val, y_val, config['context_window'])
        logs.append(f\"  • 时序数据: 训练{X_train_ctx.shape}, 验证{X_val_ctx.shape}\\n\")

        train_dataset = TensorDataset(torch.FloatTensor(X_train_ctx), torch.FloatTensor(y_train_ctx))
        val_dataset = TensorDataset(torch.FloatTensor(X_val_ctx), torch.FloatTensor(y_val_ctx))
    else:
        logs.append(\"📍 使用静态映射模式...\\n\")
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    model = HybridSensorTransformer(
        num_boundary_sensors=num_boundary,
        num_target_sensors=num_target,
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        use_temporal=use_temporal,
        context_window=config['context_window']
    ).to(device)

    # 手动应用gain初始化
    gain_value = config.get('gain', 0.1)
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if 'head' in name or 'fusion' in name:
                nn.init.xavier_uniform_(module.weight, gain=gain_value)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    logs.append(f\"🏗️ HybridSensorTransformer模型参数量: {sum(p.numel() for p in model.parameters()):,}\")
    logs.append(f\"⚙️ 配置: Gain={gain_value}, LR={config['lr']}, WD={config['weight_decay']}, GradClip={config['grad_clip']}\\n\")

    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=config['scheduler_patience'],
        factor=config['scheduler_factor']
    )

    criterion = nn.MSELoss()
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    for epoch in range(config['epochs']):
        # 训练
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['grad_clip'])
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                predictions = model(batch_X)
                val_loss += criterion(predictions, batch_y).item()
        val_loss /= len(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            status_marker = \"⭐\"
        else:
            patience_counter += 1
            status_marker = \"  \"

        log_msg = f\"{status_marker} Epoch [{epoch+1:3d}/{config['epochs']:3d}] | Train: {train_loss:.6f} | Val: {val_loss:.6f} | Best: {best_val_loss:.6f} | LR: {current_lr:.2e} | Patience: {patience_counter}/{config['early_stop_patience']}\"
        logs.append(log_msg)

        # 早停
        if patience_counter >= config['early_stop_patience']:
            logs.append(f\"\\n🛑 早停于第 {epoch+1} 轮 (耐心值达到 {config['early_stop_patience']})\")
            break

    model.load_state_dict(best_model_state)
    logs.append(f\"\\n✅ 训练完成! 最佳验证损失: {best_val_loss:.6f}\")

    return model, train_losses, val_losses, logs

# ============================================================================
# 配置导入导出函数 - Configuration Import/Export Functions
# ============================================================================

def export_config_static_transformer(boundary_signals, target_signals, test_size, val_size,
                   epochs, batch_size, lr, d_model, nhead, num_layers, dropout,
                   weight_decay, scheduler_patience, scheduler_factor,
                   grad_clip, early_stop_patience):
    """导出StaticTransformer模型配置为JSON"""
    config = {
        "model_type": "static_transformer",
        "signals": {
            "boundary": boundary_signals,
            "target": target_signals
        },
        "data_split": {
            "test_size": test_size,
            "val_size": val_size
        },
        "training": {
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "grad_clip": float(grad_clip),
            "early_stop_patience": int(early_stop_patience)
        },
        "model_architecture": {
            "d_model": int(d_model),
            "nhead": int(nhead),
            "num_layers": int(num_layers),
            "dropout": float(dropout)
        },
        "scheduler": {
            "patience": int(scheduler_patience),
            "factor": float(scheduler_factor)
        }
    }

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"config_static_transformer_{timestamp}.json"
    filepath = os.path.join('.', filename)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    return filepath, f"✅ 配置已导出到: {filename}"

def export_config_hybrid_sensor_transformer(boundary_signals, target_signals, temporal_signals,
                   test_size, val_size, epochs, batch_size, lr, d_model, nhead,
                   num_layers, dropout, context_window, apply_smoothing, gain,
                   weight_decay, scheduler_patience, scheduler_factor,
                   grad_clip, early_stop_patience):
    """导出HybridSensorTransformer模型配置为JSON"""
    config = {
        "model_type": "HybridSensorTransformer",
        "signals": {
            "boundary": boundary_signals,
            "target": target_signals,
            "temporal": temporal_signals
        },
        "data_split": {
            "test_size": test_size,
            "val_size": val_size
        },
        "training": {
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "grad_clip": float(grad_clip),
            "early_stop_patience": int(early_stop_patience)
        },
        "model_architecture": {
            "d_model": int(d_model),
            "nhead": int(nhead),
            "num_layers": int(num_layers),
            "dropout": float(dropout),
            "gain": float(gain)
        },
        "v4_specific": {
            "context_window": int(context_window),
            "apply_smoothing": bool(apply_smoothing)
        },
        "scheduler": {
            "patience": int(scheduler_patience),
            "factor": float(scheduler_factor)
        }
    }

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"config_hybrid_transformer_{timestamp}.json"
    filepath = os.path.join('.', filename)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    return filepath, f"✅ 配置已导出到: {filename}"

def import_config(config_file):
    """导入配置文件"""
    try:
        with open(config_file.name, 'r', encoding='utf-8') as f:
            config = json.load(f)

        model_type = config.get('model_type', 'static_transformer')

        if model_type in ['V1', 'static_transformer']:
            return (
                config['signals']['boundary'],
                config['signals']['target'],
                [],  # temporal_signals (V1不需要)
                config['data_split']['test_size'],
                config['data_split']['val_size'],
                config['training']['epochs'],
                config['training']['batch_size'],
                config['training']['lr'],
                config['model_architecture']['d_model'],
                config['model_architecture']['nhead'],
                config['model_architecture']['num_layers'],
                config['model_architecture']['dropout'],
                5,   # context_window (V1默认值)
                True,  # apply_smoothing (V1默认值)
                0.1,   # gain (V1默认值)
                config['training']['weight_decay'],
                config['scheduler']['patience'],
                config['scheduler']['factor'],
                config['training']['grad_clip'],
                config['training']['early_stop_patience'],
                f"✅ StaticTransformer配置加载成功！\n包含 {len(config['signals']['boundary'])} 个边界信号和 {len(config['signals']['target'])} 个目标信号"
            )
        else:  # V4/HybridSensorTransformer
            return (
                config['signals']['boundary'],
                config['signals']['target'],
                config['signals'].get('temporal', []),
                config['data_split']['test_size'],
                config['data_split']['val_size'],
                config['training']['epochs'],
                config['training']['batch_size'],
                config['training']['lr'],
                config['model_architecture']['d_model'],
                config['model_architecture']['nhead'],
                config['model_architecture']['num_layers'],
                config['model_architecture']['dropout'],
                config['v4_specific']['context_window'],
                config['v4_specific']['apply_smoothing'],
                config['model_architecture']['gain'],
                config['training']['weight_decay'],
                config['scheduler']['patience'],
                config['scheduler']['factor'],
                config['training']['grad_clip'],
                config['training']['early_stop_patience'],
                f"✅ HybridSensorTransformer配置加载成功！\n包含 {len(config['signals']['boundary'])} 个边界信号, {len(config['signals']['target'])} 个目标信号, {len(config['signals'].get('temporal', []))} 个时序信号"
            )
    except Exception as e:
        return ([], [], [], 0.2, 0.2, 100, 64, 0.001, 128, 8, 3, 0.1, 5,
                True, 0.1, 1e-5, 10, 0.5, 1.0, 25,
                f"❌ 配置加载失败: {str(e)}")

# ============================================================================
# 训练Tab回调函数 - Training Tab Callback Functions
# ============================================================================

def on_load_data(dataframe):
    """加载数据"""
    try:
        # 优先使用全局df
        if 'df' in globals():
            global_state['df'] = globals()['df'].copy()
        elif dataframe is not None:
            global_state['df'] = pd.read_csv(dataframe.name) if hasattr(dataframe, 'name') else dataframe
        else:
            return ("❌ 请上传CSV文件或确保df变量已加载", "",
                   gr.update(choices=[]), gr.update(choices=[]),
                   gr.update(choices=[]), gr.update(choices=[]))

        # 检查并处理重复列名
        original_shape = global_state['df'].shape
        global_state['df'], duplicates = handle_duplicate_columns(global_state['df'])

        signals = get_available_signals(global_state['df'])
        global_state['all_signals'] = signals  # 保存到全局状态

        # 构建状态消息
        status_msg = f"✓ 数据加载成功!\n形状: {global_state['df'].shape}\n可用信号: {len(signals)}个"

        if duplicates:
            status_msg += f"\n\n⚠️ 检测到重复列名 (已自动处理):"
            for col, count in list(duplicates.items())[:5]:
                status_msg += f"\n • {col}: 出现 {count+1} 次"
                status_msg += f" → 重命名为 {col}, {col}_#2"
                if count > 1:
                    status_msg += f", {col}_#3..."
            if len(duplicates) > 5:
                status_msg += f"\n   ... 还有 {len(duplicates)-5} 个重复项"
            status_msg += "\n\n💡 提示: 可以在下方信号列表中看到所有可用信号"

        # 构建信号列表显示
        signals_display = "=" * 60 + "\n"
        signals_display += f"可用信号总数: {len(signals)}\n"
        signals_display += "=" * 60 + "\n\n"
        for i, sig in enumerate(signals, 1):
            signals_display += f"{i:4d}. {sig}\n"
        signals_display += "\n" + "=" * 60

        return (
            status_msg,
            signals_display,
            gr.update(choices=signals, value=[]),
            gr.update(choices=signals, value=[]),
            gr.update(choices=signals, value=[]),
            gr.update(choices=signals, value=[])
        )
    except Exception as e:
        return (f"❌ 加载失败: {str(e)}", "",
               gr.update(choices=[]), gr.update(choices=[]),
               gr.update(choices=[]), gr.update(choices=[]))

def start_training_static_transformer(boundary_signals, target_signals, test_size, val_size,
                     epochs, batch_size, lr, d_model, nhead, num_layers, dropout,
                     weight_decay, scheduler_patience, scheduler_factor,
                     grad_clip, early_stop_patience):
    """开始训练StaticTransformer模型"""
    if global_state['df'] is None:
        yield "❌ 请先加载数据!"
        return

    if not boundary_signals:
        yield "❌ 请至少选择一个边界条件信号!"
        return

    if not target_signals:
        yield "❌ 请至少选择一个目标信号!"
        return

    # 验证信号互斥性
    is_valid, error_msg = validate_signal_exclusivity_v1(boundary_signals, target_signals)
    if not is_valid:
        yield error_msg
        return

    try:
        log_messages = []
        log_messages.append("=" * 80)
        log_messages.append(f"开始训练 StaticTransformer 模型")
        log_messages.append("=" * 80)

        # 准备数据
        log_messages.append("\n📊 准备数据...")
        df = global_state['df']
        X = df[boundary_signals].values
        y = df[target_signals].values
        log_messages.append(f" • 输入特征: {len(boundary_signals)}个")
        log_messages.append(f" • 输出目标: {len(target_signals)}个")
        log_messages.append(f" • 总样本数: {len(X):,}")
        yield '\n'.join(log_messages)

        # 标准化
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y)

        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled, test_size=test_size, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size, random_state=42
        )

        log_messages.append(f"\n📂 数据分割:")
        log_messages.append(f" • 训练集: {len(X_train):,} 样本 ({len(X_train)/len(X)*100:.1f}%)")
        log_messages.append(f" • 验证集: {len(X_val):,} 样本 ({len(X_val)/len(X)*100:.1f}%)")
        log_messages.append(f" • 测试集: {len(X_test):,} 样本 ({len(X_test)/len(X)*100:.1f}%)")
        yield '\n'.join(log_messages)

        # 配置
        config = {
            'epochs': int(epochs),
            'batch_size': int(batch_size),
            'lr': float(lr),
            'd_model': int(d_model),
            'nhead': int(nhead),
            'num_layers': int(num_layers),
            'dropout': float(dropout),
            'weight_decay': float(weight_decay),
            'scheduler_patience': int(scheduler_patience),
            'scheduler_factor': float(scheduler_factor),
            'grad_clip': float(grad_clip),
            'early_stop_patience': int(early_stop_patience)
        }

        log_messages.append("\n" + "=" * 80)
        log_messages.append("🚀 开始训练...")
        log_messages.append("=" * 80)
        yield '\n'.join(log_messages)

        # 训练 - 实时输出
        model, train_losses, val_losses, training_logs = train_static_transformer_model_complete(
            X_train, y_train, X_val, y_val,
            len(boundary_signals), len(target_signals), config
        )

        # 逐行输出训练日志
        for log_line in training_logs:
            log_messages.append(log_line)
            yield '\n'.join(log_messages)

        log_messages.append("\n" + "=" * 80)
        log_messages.append("🧪 评估测试集性能...")
        log_messages.append("=" * 80)
        yield '\n'.join(log_messages)

        # 测试集评估
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test).to(device)
            y_pred_scaled = model(X_test_tensor).cpu().numpy()
            y_pred = scaler_y.inverse_transform(y_pred_scaled)
            y_true = scaler_y.inverse_transform(y_test)

        # 计算指标
        metrics = {}
        for i, sensor in enumerate(target_signals):
            r2 = r2_score(y_true[:, i], y_pred[:, i])
            rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
            mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
            metrics[sensor] = {'R2': r2, 'RMSE': rmse, 'MAE': mae}

        avg_r2 = np.mean([m['R2'] for m in metrics.values()])
        avg_rmse = np.mean([m['RMSE'] for m in metrics.values()])

        log_messages.append(f"\n📈 测试集整体性能:")
        log_messages.append(f" • 平均 R²: {avg_r2:.4f}")
        log_messages.append(f" • 平均 RMSE: {avg_rmse:.4f}")

        # 显示前5个信号的详细指标
        log_messages.append(f"\n📊 前5个目标信号详细指标:")
        for i, (sensor, metric) in enumerate(list(metrics.items())[:5]):
            log_messages.append(f" {i+1}. {sensor[:50]}")
            log_messages.append(f" R²={metric['R2']:.4f}, RMSE={metric['RMSE']:.4f}, MAE={metric['MAE']:.4f}")
        if len(metrics) > 5:
            log_messages.append(f"   ... 还有 {len(metrics)-5} 个信号")
        yield '\n'.join(log_messages)

        # 保存模型
        model_name = f"StaticTransformer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        global_state['trained_models'][model_name] = {
            'model': model,
            'type': 'StaticTransformer',
            'boundary_signals': boundary_signals,
            'target_signals': target_signals,
            'temporal_signals': None,
            'config': config,
            'metrics': metrics,
            'use_temporal': False
        }
        global_state['scalers'][model_name] = {'X': scaler_X, 'y': scaler_y}
        global_state['training_history'][model_name] = {
            'train_losses': train_losses,
            'val_losses': val_losses
        }

        log_messages.append("\n" + "=" * 80)
        log_messages.append(f"✅ 训练完成并保存!")
        log_messages.append(f"📦 模型名称: {model_name}")
        log_messages.append("=" * 80)
        yield '\n'.join(log_messages)

    except Exception as e:
        error_msg = f"❌ 训练失败:\n{str(e)}\n\n详细错误:\n{traceback.format_exc()}"
        yield error_msg

def start_training_hybrid_sensor_transformer(boundary_signals, target_signals, temporal_signals,
                     test_size, val_size, epochs, batch_size, lr, d_model, nhead,
                     num_layers, dropout, context_window, apply_smoothing, gain,
                     weight_decay, scheduler_patience, scheduler_factor,
                     grad_clip, early_stop_patience):
    """开始训练HybridSensorTransformer模型"""
    if global_state['df'] is None:
        yield "❌ 请先加载数据!"
        return

    if not boundary_signals:
        yield "❌ 请至少选择一个边界条件信号!"
        return

    if not target_signals:
        yield "❌ 请至少选择一个目标信号!"
        return

    # 验证信号互斥性
    is_valid, error_msg = validate_signal_exclusivity_v4(boundary_signals, target_signals, temporal_signals)
    if not is_valid:
        yield error_msg
        return

    try:
        log_messages = []
        log_messages.append("=" * 80)
        log_messages.append(f"开始训练 HybridSensorTransformer 模型")
        log_messages.append("=" * 80)

        # 准备数据
        log_messages.append("\n📊 准备数据...")
        df = global_state['df']
        X = df[boundary_signals].values
        y = df[target_signals].values
        log_messages.append(f" • 输入特征: {len(boundary_signals)}个")
        log_messages.append(f" • 输出目标: {len(target_signals)}个")
        log_messages.append(f" • 总样本数: {len(X):,}")
        yield '\n'.join(log_messages)

        # V4特定: IFD平滑
        if apply_smoothing and temporal_signals:
            log_messages.append(f"\n🔧 应用IFD平滑滤波到 {len(temporal_signals)} 个信号...")
            y = apply_ifd_smoothing(y, target_signals, temporal_signals)
            yield '\n'.join(log_messages)

        # 标准化
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y)

        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled, test_size=test_size, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size, random_state=42
        )

        log_messages.append(f"\n📂 数据分割:")
        log_messages.append(f" • 训练集: {len(X_train):,} 样本 ({len(X_train)/len(X)*100:.1f}%)")
        log_messages.append(f" • 验证集: {len(X_val):,} 样本 ({len(X_val)/len(X)*100:.1f}%)")
        log_messages.append(f" • 测试集: {len(X_test):,} 样本 ({len(X_test)/len(X)*100:.1f}%)")
        yield '\n'.join(log_messages)

        # 配置
        config = {
            'epochs': int(epochs),
            'batch_size': int(batch_size),
            'lr': float(lr),
            'd_model': int(d_model),
            'nhead': int(nhead),
            'num_layers': int(num_layers),
            'dropout': float(dropout),
            'context_window': int(context_window),
            'gain': float(gain),
            'weight_decay': float(weight_decay),
            'scheduler_patience': int(scheduler_patience),
            'scheduler_factor': float(scheduler_factor),
            'grad_clip': float(grad_clip),
            'early_stop_patience': int(early_stop_patience)
        }

        log_messages.append("\n" + "=" * 80)
        log_messages.append("🚀 开始训练...")
        log_messages.append("=" * 80)
        yield '\n'.join(log_messages)

        # 训练 - 实时输出
        use_temporal = len(temporal_signals) > 0 if temporal_signals else False
        model, train_losses, val_losses, training_logs = train_v4_model_complete(
            X_train, y_train, X_val, y_val,
            len(boundary_signals), len(target_signals),
            config, use_temporal
        )

        # 逐行输出训练日志
        for log_line in training_logs:
            log_messages.append(log_line)
            yield '\n'.join(log_messages)

        log_messages.append("\n" + "=" * 80)
        log_messages.append("🧪 评估测试集性能...")
        log_messages.append("=" * 80)
        yield '\n'.join(log_messages)

        # 测试集评估
        model.eval()
        with torch.no_grad():
            if use_temporal:
                X_test_ctx, y_test_ctx, _ = create_temporal_context_data(
                    X_test, y_test, context_window
                )
                X_test_tensor = torch.FloatTensor(X_test_ctx).to(device)
                y_pred_scaled = model(X_test_tensor).cpu().numpy()
                y_test_eval = y_test_ctx
            else:
                X_test_tensor = torch.FloatTensor(X_test).to(device)
                y_pred_scaled = model(X_test_tensor).cpu().numpy()
                y_test_eval = y_test

            y_pred = scaler_y.inverse_transform(y_pred_scaled)
            y_true = scaler_y.inverse_transform(y_test_eval)

        # 计算指标
        metrics = {}
        for i, sensor in enumerate(target_signals):
            r2 = r2_score(y_true[:, i], y_pred[:, i])
            rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
            mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
            metrics[sensor] = {'R2': r2, 'RMSE': rmse, 'MAE': mae}

        avg_r2 = np.mean([m['R2'] for m in metrics.values()])
        avg_rmse = np.mean([m['RMSE'] for m in metrics.values()])

        log_messages.append(f"\n📈 测试集整体性能:")
        log_messages.append(f" • 平均 R²: {avg_r2:.4f}")
        log_messages.append(f" • 平均 RMSE: {avg_rmse:.4f}")

        # 显示前5个信号的详细指标
        log_messages.append(f"\n📊 前5个目标信号详细指标:")
        for i, (sensor, metric) in enumerate(list(metrics.items())[:5]):
            log_messages.append(f" {i+1}. {sensor[:50]}")
            log_messages.append(f" R²={metric['R2']:.4f}, RMSE={metric['RMSE']:.4f}, MAE={metric['MAE']:.4f}")
        if len(metrics) > 5:
            log_messages.append(f"   ... 还有 {len(metrics)-5} 个信号")
        yield '\n'.join(log_messages)

        # 保存模型
        model_name = f"HybridSensorTransformer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        global_state['trained_models'][model_name] = {
            'model': model,
            'type': 'HybridSensorTransformer',
            'boundary_signals': boundary_signals,
            'target_signals': target_signals,
            'temporal_signals': temporal_signals,
            'config': config,
            'metrics': metrics,
            'use_temporal': use_temporal
        }
        global_state['scalers'][model_name] = {'X': scaler_X, 'y': scaler_y}
        global_state['training_history'][model_name] = {
            'train_losses': train_losses,
            'val_losses': val_losses
        }

        log_messages.append("\n" + "=" * 80)
        log_messages.append(f"✅ 训练完成并保存!")
        log_messages.append(f"📦 模型名称: {model_name}")
        log_messages.append("=" * 80)
        yield '\n'.join(log_messages)

    except Exception as e:
        error_msg = f"❌ 训练失败:\n{str(e)}\n\n详细错误:\n{traceback.format_exc()}"
        yield error_msg

# ============================================================================
# 推理Tab回调函数 - Inference Tab Callback Functions
# ============================================================================

def get_model_list():
    """获取可用模型列表"""
    return list(global_state['trained_models'].keys())

def on_model_load(model_name):
    """加载模型"""
    if not model_name or model_name not in global_state['trained_models']:
        return "请选择有效的模型", gr.update(choices=[])

    model_info = global_state['trained_models'][model_name]
    target_signals = model_info['target_signals']
    metrics = model_info['metrics']
    avg_r2 = np.mean([m['R2'] for m in metrics.values()])

    info_lines = [
        "=" * 60,
        "模型信息",
        "=" * 60,
        f"模型类型: {model_info['type']}",
        f"边界信号数: {len(model_info['boundary_signals'])}",
        f"目标信号数: {len(model_info['target_signals'])}",
        f"平均R²: {avg_r2:.4f}",
        ""
    ]

    if model_info['type'] == 'HybridSensorTransformer':
        info_lines.append(f"时序模式: {'是' if model_info.get('use_temporal', False) else '否'}")
        if model_info.get('temporal_signals'):
            info_lines.append(f"时序信号数: {len(model_info['temporal_signals'])}")

    info_lines.append("")
    info_lines.append("目标信号列表:")
    for i, s in enumerate(target_signals[:10]):
        r2 = metrics[s]['R2']
        info_lines.append(f" {i+1:2d}. {s[:50]} (R²={r2:.3f})")
    if len(target_signals) > 10:
        info_lines.append(f"   ... 还有 {len(target_signals)-10} 个信号")
    info_lines.append("=" * 60)

    return '\n'.join(info_lines), gr.update(choices=target_signals, value=[])

def run_inference(model_name, start_idx, end_idx, selected_signals):
    """运行推理"""
    if not model_name or model_name not in global_state['trained_models']:
        return "❌ 请先加载模型", None, ""

    if global_state['df'] is None:
        return "❌ 数据未加载", None, ""

    if not selected_signals:
        return "❌ 请选择要可视化的信号", None, ""

    try:
        # 获取模型和配置
        model_info = global_state['trained_models'][model_name]
        model = model_info['model']
        model_type = model_info['type']
        boundary_signals = model_info['boundary_signals']
        target_signals = model_info['target_signals']
        config = model_info['config']
        scalers = global_state['scalers'][model_name]
        scaler_X = scalers['X']
        scaler_y = scalers['y']

        # 准备数据
        df = global_state['df']

        # 处理索引范围
        start_idx = int(start_idx)
        end_idx = int(end_idx)

        if start_idx < 0:
            start_idx = 0
        if end_idx > len(df):
            end_idx = len(df)
        if end_idx <= start_idx:
            end_idx = min(start_idx + 1000, len(df))

        df_slice = df.iloc[start_idx:end_idx]
        X = df_slice[boundary_signals].values
        y = df_slice[target_signals].values
        X_scaled = scaler_X.transform(X)

        # 推理
        model.eval()
        with torch.no_grad():
            if model_type == "HybridSensorTransformer" and model_info.get('use_temporal', False):
                context_window = config['context_window']
                if len(X_scaled) < 2 * context_window + 1:
                    return f"❌ 数据段太短，需要至少 {2*context_window+1} 个样本", None, ""

                X_ctx, _, valid_indices = create_temporal_context_data(
                    X_scaled,
                    np.zeros((len(X_scaled), len(target_signals))),
                    context_window
                )
                X_tensor = torch.FloatTensor(X_ctx).to(device)
                y_pred_scaled = model(X_tensor).cpu().numpy()

                # 调整y到有效索引
                y = y[valid_indices]
                actual_indices = np.array(valid_indices) + start_idx
            else:
                X_tensor = torch.FloatTensor(X_scaled).to(device)
                y_pred_scaled = model(X_tensor).cpu().numpy()
                actual_indices = np.arange(start_idx, end_idx)

        y_pred = scaler_y.inverse_transform(y_pred_scaled)

        # 计算评估指标
        metrics_lines = ["=" * 60, "评估指标", "=" * 60, ""]
        for signal in selected_signals:
            if signal not in target_signals:
                continue

            idx = target_signals.index(signal)
            y_true_signal = y[:, idx]
            y_pred_signal = y_pred[:, idx]
            residuals = y_true_signal - y_pred_signal

            r2 = r2_score(y_true_signal, y_pred_signal)
            rmse = np.sqrt(mean_squared_error(y_true_signal, y_pred_signal))
            mae = mean_absolute_error(y_true_signal, y_pred_signal)

            metrics_lines.append(f"{signal}:")
            metrics_lines.append(f" R² Score: {r2:.4f}")
            metrics_lines.append(f" RMSE: {rmse:.4f}")
            metrics_lines.append(f" MAE: {mae:.4f}")
            metrics_lines.append(f" 残差均值: {np.mean(residuals):.4f}")
            metrics_lines.append(f" 残差标准差: {np.std(residuals):.4f}")
            metrics_lines.append(f" 残差范围: [{np.min(residuals):.4f}, {np.max(residuals):.4f}]")
            metrics_lines.append("")

        metrics_lines.append("=" * 60)

        # 可视化
        n_signals = len(selected_signals)
        fig = plt.figure(figsize=(18, 5*n_signals))

        for i, signal in enumerate(selected_signals):
            if signal not in target_signals:
                continue

            idx = target_signals.index(signal)
            y_true_signal = y[:, idx]
            y_pred_signal = y_pred[:, idx]
            residuals = y_true_signal - y_pred_signal

            # 预测 vs 实际
            ax1 = plt.subplot(n_signals, 3, i*3 + 1)
            ax1.plot(actual_indices[:len(y_true_signal)], y_true_signal,
                    label='实际值', linewidth=2, alpha=0.8, color='#2c3e50')
            ax1.plot(actual_indices[:len(y_pred_signal)], y_pred_signal,
                    label='预测值', linewidth=2, alpha=0.8, color='#3498db')
            ax1.set_title(f'{signal}\n预测 vs 实际', fontsize=11, fontweight='bold')
            ax1.set_xlabel('数据索引', fontsize=10)
            ax1.set_ylabel('值', fontsize=10)
            ax1.legend(fontsize=9)
            ax1.grid(True, alpha=0.3)

            # 残差图
            ax2 = plt.subplot(n_signals, 3, i*3 + 2)
            ax2.plot(actual_indices[:len(residuals)], residuals,
                    color='#e74c3c', linewidth=1.5, alpha=0.7)
            ax2.axhline(y=0, color='black', linestyle='--', linewidth=2)
            ax2.fill_between(actual_indices[:len(residuals)], residuals,
                            alpha=0.3, color='#e74c3c')
            ax2.set_title(f'{signal}\n残差分析', fontsize=11, fontweight='bold')
            ax2.set_xlabel('数据索引', fontsize=10)
            ax2.set_ylabel('残差 (实际 - 预测)', fontsize=10)
            ax2.grid(True, alpha=0.3)

            # 散点图
            ax3 = plt.subplot(n_signals, 3, i*3 + 3)
            ax3.scatter(y_true_signal, y_pred_signal, alpha=0.6, s=20, color='#3498db')
            min_val = min(y_true_signal.min(), y_pred_signal.min())
            max_val = max(y_true_signal.max(), y_pred_signal.max())
            ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='完美预测')
            r2 = r2_score(y_true_signal, y_pred_signal)
            ax3.set_title(f'{signal}\n预测精度 (R²={r2:.3f})', fontsize=11, fontweight='bold')
            ax3.set_xlabel('实际值', fontsize=10)
            ax3.set_ylabel('预测值', fontsize=10)
            ax3.legend(fontsize=9)
            ax3.grid(True, alpha=0.3)

        plt.tight_layout()

        status_msg = f"✅ 推理完成!\n分析范围: index {start_idx} 到 {end_idx}\n实际分析样本数: {len(y_true_signal)}"

        return status_msg, fig, '\n'.join(metrics_lines)

    except Exception as e:
        error_msg = f"❌ 推理失败:\n{str(e)}\n\n详细错误:\n{traceback.format_exc()}"
        return error_msg, None, ""

# ============================================================================
# 创建Gradio界面 - Create Gradio Interface
# ============================================================================

with gr.Blocks(title="传感器预测模型训练和推理系统", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚀 传感器预测模型训练和推理系统")
    gr.Markdown("### 支持StaticTransformer (静态映射) 和 HybridSensorTransformer (混合时序+静态) 模型")

    with gr.Tabs():
        # ========== 数据加载Tab ==========
        with gr.Tab("📊 数据加载"):
            gr.Markdown("### 加载训练数据")
            with gr.Row():
                with gr.Column(scale=1):
                    data_file = gr.File(label="上传CSV文件 (可选，优先使用全局df)", file_types=[".csv"])
                    load_data_btn = gr.Button("📥 加载数据", variant="primary", size="lg")
                    data_status = gr.Textbox(label="数据状态", lines=10)

                with gr.Column(scale=1):
                    gr.Markdown("### 📋 可用信号列表")
                    gr.Markdown("数据加载后，以下所有信号可用于训练")
                    signals_list_display = gr.Textbox(
                        label="信号列表",
                        lines=25,
                        placeholder="加载数据后显示所有可用信号...",
                        interactive=False
                    )

        # ========== StaticTransformer训练Tab ==========
        with gr.Tab("🎯 StaticTransformer训练"):
            gr.Markdown("## StaticTransformer: 静态传感器映射 Transformer")

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📋 配置管理")
                    config_file_static = gr.File(label="导入配置文件", file_types=[".json"])
                    import_config_btn_static = gr.Button("📂 导入配置", variant="secondary")
                    export_config_btn_static = gr.Button("💾 导出配置", variant="secondary")
                    config_status_static = gr.Textbox(label="配置状态", lines=3)
                    config_download_static = gr.File(label="下载配置文件", visible=False)

                    gr.Markdown("### 🎛️ 信号选择")
                    gr.Markdown("⚠️ **注意**: 边界信号和目标信号不能重复，训练前会自动验证")

                    boundary_signals_static = gr.Dropdown(
                        choices=[],
                        multiselect=True,
                        label="边界条件信号 (输入)",
                        info="选择用于预测的输入信号"
                    )

                    target_signals_static = gr.Dropdown(
                        choices=[],
                        multiselect=True,
                        label="目标信号 (输出)",
                        info="选择要预测的信号"
                    )

                    gr.Markdown("### ⚙️ 数据分割")
                    with gr.Row():
                        test_size_static = gr.Slider(0.1, 0.4, value=0.2, step=0.05, label="测试集比例")
                        val_size_static = gr.Slider(0.1, 0.3, value=0.2, step=0.05, label="验证集比例")

                with gr.Column(scale=1):
                    gr.Markdown("### 🔧 训练参数")
                    with gr.Row():
                        epochs_static = gr.Slider(1, 200, value=100, step=1, label="训练轮数")
                        batch_size_static = gr.Slider(16, 256, value=64, step=16, label="批大小")

                    with gr.Row():
                        lr_static = gr.Number(value=0.001, label="学习率")
                        weight_decay_static = gr.Number(value=1e-5, label="权重衰减", precision=6)

                    gr.Markdown("### 🏗️ 模型架构")
                    with gr.Row():
                        d_model_static = gr.Slider(32, 256, value=128, step=32, label="模型维度")
                        nhead_static = gr.Slider(2, 16, value=8, step=2, label="注意力头数")

                    with gr.Row():
                        num_layers_static = gr.Slider(1, 6, value=3, step=1, label="Transformer层数")
                        dropout_static = gr.Slider(0.0, 0.5, value=0.1, step=0.05, label="Dropout")

                    gr.Markdown("### 📈 优化器参数")
                    with gr.Row():
                        scheduler_patience_static = gr.Slider(5, 20, value=10, step=1, label="学习率调度耐心")
                        scheduler_factor_static = gr.Slider(0.1, 0.9, value=0.5, step=0.1, label="学习率衰减因子")

                    with gr.Row():
                        grad_clip_static = gr.Slider(0.5, 5.0, value=1.0, step=0.1, label="梯度裁剪阈值")
                        early_stop_patience_static = gr.Slider(10, 50, value=25, step=5, label="早停耐心")

                    gr.Markdown("### 🚀 开始训练")
                    train_btn_static = gr.Button("▶️ 开始训练 StaticTransformer 模型", variant="primary", size="lg")
                    training_log_static = gr.Textbox(label="训练日志", lines=30, max_lines=50, autoscroll=True)

        # ========== HybridSensorTransformer训练Tab ==========
        with gr.Tab("🎯 HybridSensorTransformer训练"):
            gr.Markdown("## HybridSensorTransformer: 混合时序+静态 Transformer")

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📋 配置管理")
                    config_file_hybrid = gr.File(label="导入配置文件", file_types=[".json"])
                    import_config_btn_hybrid = gr.Button("📂 导入配置", variant="secondary")
                    export_config_btn_hybrid = gr.Button("💾 导出配置", variant="secondary")
                    config_status_hybrid = gr.Textbox(label="配置状态", lines=3)
                    config_download_hybrid = gr.File(label="下载配置文件", visible=False)

                    gr.Markdown("### 🎛️ 信号选择")
                    gr.Markdown("⚠️ **注意**: 边界信号和目标信号不能重复；时序信号必须是目标信号的子集，训练前会自动验证")

                    boundary_signals_hybrid = gr.Dropdown(
                        choices=[],
                        multiselect=True,
                        label="边界条件信号 (输入)",
                        info="选择用于预测的输入信号"
                    )

                    target_signals_hybrid = gr.Dropdown(
                        choices=[],
                        multiselect=True,
                        label="目标信号 (输出)",
                        info="选择要预测的信号"
                    )

                    gr.Markdown("### ⏱️ 时序选项")
                    gr.Markdown("💡 时序信号必须从目标信号中选择")
                    with gr.Row():
                        temporal_signals_hybrid = gr.Dropdown(
                            choices=[],
                            multiselect=True,
                            label="时序模式信号",
                            info="从目标信号中选择需要时序上下文的信号 (留空则全部使用静态模式)"
                        )
                        sync_temporal_btn_hybrid = gr.Button("🔄 同步目标信号", size="sm", scale=0)
                        context_window_hybrid = gr.Slider(1, 10, value=5, step=1, label="上下文窗口大小")
                        apply_smoothing_hybrid = gr.Checkbox(label="应用IFD平滑滤波", value=True)

                    gr.Markdown("### ⚙️ 数据分割")
                    with gr.Row():
                        test_size_hybrid = gr.Slider(0.1, 0.4, value=0.2, step=0.05, label="测试集比例")
                        val_size_hybrid = gr.Slider(0.1, 0.3, value=0.2, step=0.05, label="验证集比例")

                with gr.Column(scale=1):
                    gr.Markdown("### 🔧 训练参数")
                    with gr.Row():
                        epochs_hybrid = gr.Slider(1, 200, value=100, step=1, label="训练轮数")
                        batch_size_hybrid = gr.Slider(16, 256, value=64, step=16, label="批大小")

                    with gr.Row():
                        lr_hybrid = gr.Number(value=0.001, label="学习率")
                        weight_decay_hybrid = gr.Number(value=1e-5, label="权重衰减", precision=6)

                    gr.Markdown("### 🏗️ 模型架构")
                    with gr.Row():
                        d_model_hybrid = gr.Slider(32, 256, value=64, step=32, label="模型维度")
                        nhead_hybrid = gr.Slider(2, 16, value=4, step=2, label="注意力头数")

                    with gr.Row():
                        num_layers_hybrid = gr.Slider(1, 6, value=2, step=1, label="Transformer层数")
                        dropout_hybrid = gr.Slider(0.0, 0.5, value=0.1, step=0.05, label="Dropout")

                    with gr.Row():
                        gain_hybrid = gr.Slider(0.01, 1.0, value=0.1, step=0.01, label="权重初始化Gain")

                    gr.Markdown("### 📈 优化器参数")
                    with gr.Row():
                        scheduler_patience_hybrid = gr.Slider(5, 20, value=10, step=1, label="学习率调度耐心")
                        scheduler_factor_hybrid = gr.Slider(0.1, 0.9, value=0.5, step=0.1, label="学习率衰减因子")

                    with gr.Row():
                        grad_clip_hybrid = gr.Slider(0.5, 5.0, value=1.0, step=0.1, label="梯度裁剪阈值")
                        early_stop_patience_hybrid = gr.Slider(10, 50, value=25, step=5, label="早停耐心")

                    gr.Markdown("### 🚀 开始训练")
                    train_btn_hybrid = gr.Button("▶️ 开始训练 HybridSensorTransformer 模型", variant="primary", size="lg")
                    training_log_hybrid = gr.Textbox(label="训练日志", lines=30, max_lines=50, autoscroll=True)

        # ========== 推理Tab ==========
        with gr.Tab("🔮 模型推理"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 1️⃣ 选择模型")
                    model_selector = gr.Dropdown(
                        choices=[],
                        label="已训练模型",
                        info="选择要使用的模型"
                    )
                    refresh_models_btn = gr.Button("🔄 刷新模型列表")
                    load_model_btn = gr.Button("📂 加载模型", variant="primary")
                    model_info = gr.Textbox(label="模型信息", lines=15)

                    gr.Markdown("### 2️⃣ 推理设置")
                    with gr.Row():
                        start_idx = gr.Number(value=0, label="起始索引", precision=0)
                        end_idx = gr.Number(value=1000, label="结束索引", precision=0)

                    inference_signals = gr.Dropdown(
                        choices=[],
                        multiselect=True,
                        label="选择可视化信号",
                        info="选择要分析的目标信号（最多5个）"
                    )

                    inference_btn = gr.Button("▶️ 运行推理", variant="primary", size="lg")
                    inference_status = gr.Textbox(label="推理状态", lines=3)

                with gr.Column(scale=2):
                    gr.Markdown("### 3️⃣ 推理结果")
                    inference_plot = gr.Plot(label="可视化结果")
                    with gr.Row():
                        metrics_output = gr.Textbox(label="评估指标", lines=20)

            gr.Markdown("""
---
### 📖 使用说明

**数据加载:**
1. 在"数据加载"Tab上传CSV文件或使用全局df变量
2. 点击"加载数据"按钮
3. 在右侧可以看到所有可用信号的完整列表

**信号选择规则:**
- **StaticTransformer模型**: 边界条件信号和目标信号不能重复
- **HybridSensorTransformer模型**: 边界条件信号和目标信号不能重复；时序信号必须从目标信号中选择
- **验证时机**: 点击"开始训练"时自动验证，如有冲突会显示详细错误信息

**StaticTransformer模型训练流程:**
1. (可选) 导入之前保存的配置文件
2. 选择边界条件和目标信号
3. 调整训练参数和模型架构
4. 点击"开始训练 StaticTransformer 模型" - 系统会自动验证信号选择
5. **实时查看训练进度** - 每个epoch的训练和验证损失会实时显示（⭐表示最佳模型）
6. (可选) 导出当前配置以便后续使用

**HybridSensorTransformer模型训练流程:**
1. (可选) 导入之前保存的配置文件
2. 选择边界条件和目标信号
3. 从目标信号中选择需要时序模式的信号（可使用"同步目标信号"按钮快速更新选项）
4. 调整上下文窗口大小和其他参数
5. 调整训练参数和模型架构
6. 点击"开始训练 HybridSensorTransformer 模型" - 系统会自动验证信号选择
7. **实时查看训练进度** - 每个epoch的训练和验证损失会实时显示（⭐表示最佳模型）
8. (可选) 导出当前配置以便后续使用

**训练日志说明:**
- ⭐ 标记表示该epoch获得了最佳验证损失（模型会被保存）
- Train: 训练损失
- Val: 验证损失
- Best: 当前最佳验证损失
- LR: 当前学习率
- Patience: 早停计数器（达到设定值会自动停止训练）

**推理流程:**
1. 点击"刷新模型列表"
2. 选择已训练模型并点击"加载模型"
3. 设置推理的数据范围 (起始和结束索引)
4. 选择要可视化和分析的信号 (建议1-3个)
5. 点击"运行推理"查看结果

**可视化说明:**
- 左图: 预测值与实际值的时序对比
- 中图: 残差分析 (实际值 - 预测值)
- 右图: 预测精度散点图 (越接近对角线越好)

**提示:**
- HybridSensorTransformer模型的时序模式需要足够的上下文数据
- 配置文件可以保存你的信号选择和所有训练参数
- 建议先用小epoch数（如1-10）测试，确认无误后再进行大规模训练
- 如果信号选择有冲突，训练时会显示详细的错误信息，根据提示修改即可
- 训练过程中可以实时看到每个epoch的进度，无需等待训练结束
            """)

    # ========== 事件绑定 ==========

    # 数据加载 - 直接更新所有信号选择框
    load_data_btn.click(
        fn=on_load_data,
        inputs=[data_file],
        outputs=[
            data_status,
            signals_list_display,
            boundary_signals_static,
            target_signals_static,
            boundary_signals_hybrid,
            target_signals_hybrid
        ]
    )

    # StaticTransformer配置导入导出
    import_config_btn_static.click(
        fn=import_config,
        inputs=[config_file_static],
        outputs=[
            boundary_signals_static, target_signals_static, temporal_signals_hybrid,
            test_size_static, val_size_static, epochs_static, batch_size_static, lr_static,
            d_model_static, nhead_static, num_layers_static, dropout_static,
            context_window_hybrid, apply_smoothing_hybrid, gain_hybrid,
            weight_decay_static, scheduler_patience_static, scheduler_factor_static,
            grad_clip_static, early_stop_patience_static, config_status_static
        ]
    )

    export_config_btn_static.click(
        fn=export_config_static_transformer,
        inputs=[
            boundary_signals_static, target_signals_static,
            test_size_static, val_size_static, epochs_static, batch_size_static, lr_static,
            d_model_static, nhead_static, num_layers_static, dropout_static,
            weight_decay_static, scheduler_patience_static, scheduler_factor_static,
            grad_clip_static, early_stop_patience_static
        ],
        outputs=[config_download_static, config_status_static]
    ).then(
        fn=lambda x: gr.update(visible=True),
        inputs=[config_download_static],
        outputs=[config_download_static]
    )

    # HybridSensorTransformer配置导入导出
    sync_temporal_btn_hybrid.click(
        fn=lambda target_sigs: gr.update(choices=target_sigs if target_sigs else []),
        inputs=[target_signals_hybrid],
        outputs=[temporal_signals_hybrid]
    )

    import_config_btn_hybrid.click(
        fn=import_config,
        inputs=[config_file_hybrid],
        outputs=[
            boundary_signals_hybrid, target_signals_hybrid, temporal_signals_hybrid,
            test_size_hybrid, val_size_hybrid, epochs_hybrid, batch_size_hybrid, lr_hybrid,
            d_model_hybrid, nhead_hybrid, num_layers_hybrid, dropout_hybrid,
            context_window_hybrid, apply_smoothing_hybrid, gain_hybrid,
            weight_decay_hybrid, scheduler_patience_hybrid, scheduler_factor_hybrid,
            grad_clip_hybrid, early_stop_patience_hybrid, config_status_hybrid
        ]
    )

    export_config_btn_hybrid.click(
        fn=export_config_hybrid_sensor_transformer,
        inputs=[
            boundary_signals_hybrid, target_signals_hybrid, temporal_signals_hybrid,
            test_size_hybrid, val_size_hybrid, epochs_hybrid, batch_size_hybrid, lr_hybrid,
            d_model_hybrid, nhead_hybrid, num_layers_hybrid, dropout_hybrid,
            context_window_hybrid, apply_smoothing_hybrid, gain_hybrid,
            weight_decay_hybrid, scheduler_patience_hybrid, scheduler_factor_hybrid,
            grad_clip_hybrid, early_stop_patience_hybrid
        ],
        outputs=[config_download_hybrid, config_status_hybrid]
    ).then(
        fn=lambda x: gr.update(visible=True),
        inputs=[config_download_hybrid],
        outputs=[config_download_hybrid]
    )

    # StaticTransformer训练
    train_btn_static.click(
        fn=start_training_static_transformer,
        inputs=[
            boundary_signals_static, target_signals_static,
            test_size_static, val_size_static, epochs_static, batch_size_static, lr_static,
            d_model_static, nhead_static, num_layers_static, dropout_static,
            weight_decay_static, scheduler_patience_static, scheduler_factor_static,
            grad_clip_static, early_stop_patience_static
        ],
        outputs=[training_log_static]
    )

    # HybridSensorTransformer训练
    train_btn_hybrid.click(
        fn=start_training_hybrid_sensor_transformer,
        inputs=[
            boundary_signals_hybrid, target_signals_hybrid, temporal_signals_hybrid,
            test_size_hybrid, val_size_hybrid, epochs_hybrid, batch_size_hybrid, lr_hybrid,
            d_model_hybrid, nhead_hybrid, num_layers_hybrid, dropout_hybrid,
            context_window_hybrid, apply_smoothing_hybrid, gain_hybrid,
            weight_decay_hybrid, scheduler_patience_hybrid, scheduler_factor_hybrid,
            grad_clip_hybrid, early_stop_patience_hybrid
        ],
        outputs=[training_log_hybrid]
    )

    # 推理
    refresh_models_btn.click(
        fn=lambda: gr.update(choices=get_model_list()),
        outputs=[model_selector]
    )

    load_model_btn.click(
        fn=on_model_load,
        inputs=[model_selector],
        outputs=[model_info, inference_signals]
    )

    inference_btn.click(
        fn=run_inference,
        inputs=[model_selector, start_idx, end_idx, inference_signals],
        outputs=[inference_status, inference_plot, metrics_output]
    )

print("="*80)
print("完整Gradio界面已准备就绪！")
print("="*80)
print("\n🎯 模型名称已统一更新:")
print("  • V1 → StaticTransformer")
print("  • V4 → HybridSensorTransformer")
print("\n📝 包含完整的配置导入导出和回调函数")
print("\n🚀 执行 demo.launch() 启动界面")
print("\n💡 如果在本地，访问 http://127.0.0.1:7860")
print("="*80)

# 启动界面
if __name__ == "__main__":
    demo.launch(share=True, debug=True)