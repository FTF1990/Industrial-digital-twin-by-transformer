# Gradio界面使用指南

本项目提供**三个级别**的Gradio界面选项，您可以根据需求选择：

## 📊 文件说明

| 文件 | 说明 | 代码量 | 推荐场景 |
|------|------|--------|----------|
| `gradio_app.py` | 简化版界面 | ~400行 | 快速开始、学习结构 |
| `gradio_full_interface.py` | 完整功能模板 | ~600行 | 需要完整功能 |
| **您的Cell 3** | 原始完整代码 | ~2600行 | 100%原始功能 |

## 🚀 快速开始（3种方式）

### 方式1️⃣: 简化版（最快）

```bash
python gradio_app.py
```

**包含功能**:
- ✅ SST模型训练
- ✅ 基础推理
- ✅ 数据加载
- ⚠️ 不包含HST高级功能

---

### 方式2️⃣: 使用您的完整Cell 3代码（推荐）

#### 步骤1: 准备文件

创建新文件：`gradio_my_complete.py`

#### 步骤2: 复制以下内容到文件开头

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
\"\"\"
完整Gradio界面 - 基于原始Cell 3
包含所有SST和HST功能
\"\"\"

# ============ 标准库导入 ============
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
from scipy.signal import savgol_filter
from scipy.ndimage import maximum_filter1d

# ============ 🔥 关键：使用项目模块 ============
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

# ============ 设置 ============
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f\"V1模型 - 使用设备: {device}\")
print(f\"V4模型 - 使用设备: {device}\")
print(\"✓ V1模型定义完成\")
print(\"✓ V4模型定义完成\")

# ============ 下面粘贴您的Cell 3代码 ============
```

#### 步骤3: 粘贴Cell 3代码

从您的 `说明.txt` 文件中：

1. **找到Cell 3的开始**（大约在第362行，开始是 `# 全局变量存储`）
2. **复制到文件末尾**（大约到第3013行，结束是 `demo.launch(share=True, debug=True)`）
3. **粘贴到上面代码下方**

#### 步骤4: 运行

```bash
python gradio_my_complete.py
```

#### 完成！🎉

您的完整原始界面现在可以使用了，包含所有功能：
- ✅ 完整V1训练
- ✅ 完整V4训练（时序+静态）
- ✅ 配置导入/导出
- ✅ 实时训练进度
- ✅ 完整推理功能

---

### 方式3️⃣: 在Jupyter Notebook中使用

详见 `docs/CELL3_INTEGRATION_GUIDE.md`

## 📋 代码修改对照

### ❌ 不需要修改（原始Cell 3中）

```python
# 这些代码完全不需要改动：
global_state = {...}
def train_v1_model_complete(...):
def train_v4_model_complete(...):
def on_load_data(...):
# ... 所有其他函数和Gradio界面代码
```

### ✅ 唯一需要的修改

```python
# 原来（Cell 1 + Cell 2）:
class CompactSensorTransformer(nn.Module):
    def __init__(self, ...):
        # ... 100多行代码

class HybridTemporalTransformer(nn.Module):
    def __init__(self, ...):
        # ... 200多行代码

# 现在（导入即可）:
from models.v1_transformer import CompactSensorTransformer
from models.v4_hybrid_transformer import HybridTemporalTransformer
```

**就这么简单！** 只需要替换模型定义为导入语句。

## 🎯 推荐使用流程

### 第一次使用

1. **测试**: 先运行 `python gradio_app.py` 确保环境正常
2. **学习**: 查看 `gradio_app.py` 了解代码结构
3. **完整版**: 按方式2创建您的完整版本

### 日常使用

- **开发/调试**: 使用 `gradio_app.py`（代码简单）
- **生产/完整功能**: 使用您的完整Cell 3版本

## 📁 相关文档

- **详细集成指南**: `docs/CELL3_INTEGRATION_GUIDE.md`
- **Gradio说明**: `docs/GRADIO_FULL.md`
- **Gradio集成**: `docs/GRADIO_INTEGRATION.md`

## ❓ 常见问题

### Q: 我的Cell 3代码会改变吗？

**A**: 不会！除了顶部的导入语句，其他代码**100%保持不变**。

### Q: 为什么要这样做？

**A**:
- ✅ 模型定义只写一次，多处使用
- ✅ 更容易维护和更新
- ✅ 可以在不同地方（notebook、脚本、Gradio）使用同一个模型
- ✅ 符合软件工程最佳实践

### Q: 原始Cell 3的功能会丢失吗？

**A**: 完全不会！所有功能都保留：
- ✅ V1和V4完整训练流程
- ✅ 实时进度显示
- ✅ 配置管理
- ✅ 推理可视化
- ✅ 所有验证和错误处理

### Q: 如果遇到导入错误？

**A**: 确保在项目根目录运行：
```bash
cd Industrial-digital-twin-by-transformer
python gradio_my_complete.py
```

## 🎓 学习路径

1. **第1天**: 运行 `gradio_app.py`，了解基本流程
2. **第2天**: 查看代码结构，理解模块化
3. **第3天**: 创建完整版本，迁移您的Cell 3
4. **第4天**: 自定义和扩展功能

## 🔗 更多资源

- **主README**: `../README.md`
- **快速开始**: `docs/GETTING_STARTED.md`
- **项目结构**: `docs/PROJECT_STRUCTURE.md`

---

**总结**: 您可以用 **<5分钟** 将原始Cell 3代码集成到这个项目中，并立即获得所有好处！
