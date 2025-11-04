# Industrial Digital Twin by Transformer (基于 Transformer 的工业数字孪生)

**[English](README.md)** | **[中文](README_CN.md)**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

> **一个创新的基于 Transformer 的框架，专为复杂系统中的工业数字孪生建模设计，使用序列传感器输出和先进的残差提升训练方法。**

本项目引入了新颖的 Transformer 架构和残差提升训练方法，专门设计用于预测工业数字孪生应用中的传感器输出。与传统方法不同，我们的模型利用复杂工业环境中**多传感器系统的序列特性**，通过多阶段优化实现卓越的预测精度。

## 🌟 核心创新

**使用 Transformer 进行序列传感器预测**：这是首个将 Transformer 架构专门应用于工业数字孪生中序列传感器输出预测问题的框架。该模型将多个传感器视为一个序列，捕获传感器之间的空间关系及其测量值的时间依赖性。

### 为什么这很重要

在复杂的工业系统（制造工厂、化工过程、发电等）中，传感器不是孤立运行的。它们的输出具有以下特征：
- **空间相关性**：物理邻近性和工艺流程创建了依赖关系
- **时间依赖性**：历史测量值影响当前和未来的读数
- **层次结构**：一些传感器测量边界条件，而另一些测量内部状态

传统的机器学习方法独立对待传感器或使用简单的时间序列模型。我们基于 Transformer 的方法**捕获传感器相互关系的全部复杂性**。

## 🚀 功能特性

### 模型架构

#### **StaticSensorTransformer (SST)**
- **用途**：将边界条件传感器映射到目标传感器预测
- **架构**：具有学习位置编码的传感器序列 Transformer
- **创新点**：将固定传感器阵列视为序列（替代 NLP 中的词元序列）
- **应用场景**：具有复杂传感器相互依赖关系的工业系统
- **优势**：
  - 通过注意力机制捕获空间传感器关系
  - 快速训练和推理
  - 学习传感器之间的物理因果关系
  - 非常适合工业数字孪生应用

### 🆕 增强型残差提升训练系统 (v1.0)

#### **Stage2 提升训练** 🚀
- 在 SST 预测残差上训练第二阶段模型
- 进一步优化预测以提高准确性
- 可配置的架构和训练参数
- 自动模型保存和版本控制

#### **智能 Delta R² 阈值选择** 🎯
- 计算每个信号的 Delta R² (R²_ensemble - R²_stage1)
- 基于 Delta R² 阈值选择性地应用 Stage2 修正
- 生成结合 SST + Stage2 的集成模型
- 优化的性能/效率平衡
- 仅对有显著改进的信号使用 Stage2

#### **全面的推理对比** 📊
- 比较集成模型与纯 SST 模型
- 可视化所有输出信号的性能改进
- 详细的逐信号指标分析（MAE、RMSE、R²）
- CSV 导出包含预测值和 R² 分数
- 交互式索引范围选择

#### **全信号可视化** 📈
- 每个输出信号的独立预测 vs 实际值对比
- 动态布局适应信号数量
- 每个信号显示 R² 分数
- 轻松识别模型改进

### 附加功能

- ✅ **模块化设计**：易于扩展和定制
- ✅ **全面的训练流程**：内置数据预处理、训练和评估
- ✅ **交互式 Gradio 界面**：适用于所有训练阶段的用户友好型 Web 界面
- ✅ **Jupyter Notebooks**：完整的教程和示例
- ✅ **生产就绪**：可导出模型用于部署
- ✅ **详尽的文档**：清晰的 API 文档和使用示例
- ✅ **自动化模型管理**：智能模型保存和加载（含配置）

## 📊 使用场景

本框架非常适合：

- **制造业数字孪生**：从传感器阵列预测设备状态
- **化工过程监控**：建模反应器中的复杂传感器交互
- **发电厂优化**：预测涡轮机和发电机状况
- **HVAC 系统**：预测温度和压力分布
- **预测性维护**：从传感器模式中早期检测异常
- **质量控制**：从工艺传感器预测产品质量

## 🏗️ 架构概述

### 🔑 核心创新：传感器作为序列元素

**传统 NLP Transformer vs. SST（我们的创新）**

```
┌─────────────────────────────────────────────────────────────────┐
│                  NLP Transformer（传统）                        │
├─────────────────────────────────────────────────────────────────┤
│ 输入:  [The, cat, sits, on, the, mat]  ← 单词作为词元          │
│ 嵌入:  [E₁,  E₂,  E₃,   E₄,  E₅,  E₆]  ← 词嵌入                │
│ 位置:  [P₁,  P₂,  P₃,   P₄,  P₅,  P₆]  ← 时间顺序              │
│ 注意力: 单词之间的语义关系                                      │
└─────────────────────────────────────────────────────────────────┘

                              ⬇️  创新点  ⬇️

┌─────────────────────────────────────────────────────────────────┐
│              SST - 传感器序列 Transformer（我们的）             │
├─────────────────────────────────────────────────────────────────┤
│ 输入:  [S₁,  S₂,  S₃, ..., Sₙ]  ← 固定传感器阵列               │
│         (温度, 压力, 流量, ...)                                 │
│ 嵌入:  [E₁,  E₂,  E₃, ..., Eₙ]  ← 传感器值嵌入                 │
│ 位置:  [P₁,  P₂,  P₃, ..., Pₙ]  ← 空间位置                     │
│ 注意力: 物理因果关系和传感器相互依赖关系                        │
│                                                                  │
│ 关键差异：                                                       │
│ • 固定序列长度（N 个传感器预先确定）                            │
│ • 位置 = 传感器位置，而非时间顺序                               │
│ • 注意力学习跨传感器物理关系                                    │
│ • 针对工业系统的领域专用设计                                    │
└─────────────────────────────────────────────────────────────────┘
```

### 🎯 SST 架构深入解析

```
物理传感器阵列: [Sensor₁, Sensor₂, ..., Sensorₙ]
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    传感器嵌入层                                  │
│  • 将每个标量传感器读数投影到 d_model 维度                      │
│  • 每个传感器获得自己的嵌入变换                                  │
│  • 输入: (batch, N_sensors) → 输出: (batch, N_sensors, d_model) │
└──────────────────────┬──────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│               可学习位置编码                                     │
│  • 与 NLP 不同：编码空间传感器位置                              │
│  • 学习传感器位置重要性（例如，进口 vs 出口）                   │
│  • 形状: (N_sensors, d_model) - 每个传感器一个                 │
│  • 添加到嵌入中: Embed + PosEncode                             │
└──────────────────────┬──────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│              多头自注意力机制                                    │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ 头 1: 学习温度-压力关系                                  │  │
│  │ 头 2: 学习流量-速度相关性                                │  │
│  │ 头 3: 学习空间邻近效应                                   │  │
│  │ ...                                                      │  │
│  │ 头 N: 学习系统级依赖关系                                 │  │
│  └─────────────────────────────────────────────────────────┘  │
│  • 捕获复杂的非线性传感器交互                                   │
│  • 注意力权重揭示传感器重要性                                   │
└──────────────────────┬──────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│                   Transformer 编码器堆栈                         │
│  层 1: 注意力 + FFN + 残差                                      │
│  层 2: 注意力 + FFN + 残差                                      │
│  ...                                                             │
│  层 L: 注意力 + FFN + 残差                                      │
│  • 每一层优化传感器关系理解                                     │
└──────────────────────┬──────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│              全局池化（序列聚合）                                │
│  • 对传感器序列进行自适应平均池化                               │
│  • 聚合来自所有传感器的信息                                     │
│  • 输出: (batch, d_model) - 固定大小表示                       │
└──────────────────────┬──────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│                    输出投影层                                    │
│  • 将聚合表示投影到目标传感器值                                 │
│  • 线性变换: d_model → N_target_sensors                        │
│  • 最终预测: (batch, N_target_sensors)                         │
└──────────────────────┬──────────────────────────────────────────┘
                       ↓
              目标传感器预测
```

### 📊 Stage2 残差提升系统

建立在 SST 之上，Stage2 系统进一步优化预测：

```
步骤 1: 基础 SST 模型
   边界传感器 → [SST] → 预测 + 残差

步骤 2: Stage2 残差模型
   边界传感器 → [SST₂] → 残差修正

步骤 3: 智能 Delta R² 选择
   对于每个目标信号:
     Delta R² = R²_ensemble - R²_stage1
     if Delta R² > 阈值: 应用 Stage2 修正
     else: 使用基础 SST 预测

步骤 4: 最终集成模型
   预测 = Stage1 预测 + 选择性 Stage2 修正

```

## 🔧 安装

### 使用 Google Colab 快速开始

```bash
# 克隆仓库
!git clone https://github.com/YOUR_USERNAME/Industrial-digital-twin-by-transformer.git
%cd Industrial-digital-twin-by-transformer

# 安装依赖
!pip install -r requirements.txt
```

### 本地安装

```bash
# 克隆仓库
git clone https://github.com/YOUR_USERNAME/Industrial-digital-twin-by-transformer.git
cd Industrial-digital-twin-by-transformer

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Windows 系统: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

## 📚 快速入门

### 1. 准备数据

将您的 CSV 传感器数据文件放在 `data/raw/` 文件夹中。您的 CSV 应该具有：
- 每行代表一个时间步
- 每列代表一个传感器测量值
- （可选）第一列可以是时间戳

CSV 结构示例：
```csv
timestamp,sensor_1,sensor_2,sensor_3,...,sensor_n
2025-01-01 00:00:00,23.5,101.3,45.2,...,78.9
2025-01-01 00:00:01,23.6,101.4,45.1,...,79.0
...
```

### 2. 使用 Jupyter Notebook 训练 Stage1 模型（基础训练）

本节演示**基础 Stage1 (SST) 模型训练**，用于学习传感器预测建模的基础知识。

**注意**：Notebook 提供了理解 SST 架构和基础训练过程的基础。如需完整的 Stage2 提升训练和集成模型生成功能，请使用增强型 Gradio 界面（第3节）。

**可用的 Notebooks**：
- `notebooks/transformer_boost_Leap_final.ipynb` - 在 LEAP 数据集上的完整 Stage1 + Stage2 训练的高级示例

**基础训练示例**（用于您自己的数据）：

```python
from models.static_transformer import StaticSensorTransformer
from src.data_loader import SensorDataLoader
from src.trainer import ModelTrainer

# 加载数据
data_loader = SensorDataLoader(data_path='data/raw/your_data.csv')

# 配置信号
boundary_signals = ['sensor_1', 'sensor_2', 'sensor_3']  # 输入
target_signals = ['sensor_4', 'sensor_5']  # 要预测的输出

# 准备数据
data_splits = data_loader.prepare_data(boundary_signals, target_signals)

# 创建和训练 Stage1 SST 模型
model = StaticSensorTransformer(
    num_boundary_sensors=len(boundary_signals),
    num_target_sensors=len(target_signals)
)

trainer = ModelTrainer(model, device='cuda')
history = trainer.train(train_loader, val_loader)

# 保存训练好的模型
torch.save(model.state_dict(), 'saved_models/my_sst_model.pth')
```

**在 Stage1 中您将学到**：
- 加载和预处理传感器数据
- 配置边界传感器和目标传感器
- 训练静态传感器 Transformer (SST)
- 基础模型评估和预测

**如需完整功能**（Stage2 提升 + 集成模型），请继续第3节。

### 3. 使用增强型 Gradio 界面（完整 Stage1 + Stage2 训练）

启动增强型交互式 Web 界面，提供**完整的 Stage1 + Stage2 残差提升训练**：

```bash
python gradio_residual_tft_app.py
```

增强型界面提供**完整的端到端工作流程**：
- 📊 **数据加载**：上传 CSV 或创建示例数据
- 🎯 **Stage1 SST 训练**：配置和训练基础静态传感器 Transformer 模型
- 🔬 **残差提取**：从 Stage1 模型中提取和分析预测误差
- 🚀 **Stage2 提升训练**：在残差上训练第二阶段模型进行误差修正
- 🎯 **集成模型生成**：基于智能 Delta R² 阈值的模型组合
- 📊 **推理对比**：比较 Stage1 SST vs. 集成模型性能并可视化
- 💾 **导出**：自动模型保存（含完整配置）

**这是体验框架完整功能的推荐方式**，包括：
- 自动化多阶段训练流程
- 智能的逐信号 Stage2 选择
- 全面的性能指标和可视化
- 生产就绪的集成模型生成

**快速入门指南**：参见 `docs/QUICKSTART.md` 获取 5 分钟教程

## 📖 文档

### 项目结构

```
Industrial-digital-twin-by-transformer/
├── models/                      # 模型实现
│   ├── __init__.py
│   ├── static_transformer.py    # SST (StaticSensorTransformer)
│   ├── utils.py                # 工具函数
│   └── saved/                  # 保存的模型检查点
├── saved_models/               # 训练好的模型（含配置）
│   ├── StaticSensorTransformer_*.pth   # SST 模型
│   ├── stage2_boost/           # Stage2 残差模型
│   ├── ensemble/               # 集成模型配置
│   └── tft_models/            # TFT 模型（如果使用）
├── src/                        # 源代码
│   ├── __init__.py
│   ├── data_loader.py         # 数据加载和预处理
│   ├── trainer.py             # 训练流程
│   └── inference.py           # 推理引擎
├── docs/                       # 文档
│   ├── ENHANCED_VERSION_README.md  # 增强功能指南
│   ├── UPDATE_NOTES.md        # 详细更新说明
│   ├── QUICKSTART.md          # 5 分钟快速入门
│   └── FILE_MANIFEST.md       # 文件结构指南
├── notebooks/                  # Jupyter notebooks
│   └── transformer_boost_Leap_final.ipynb  # 使用 LEAP 数据集的高级 Stage1+Stage2 教程
├── data/                      # 数据文件夹
│   ├── raw/                   # 将您的 CSV 文件放在这里
│   └── residuals_*.csv       # 提取的残差
├── examples/                  # 示例脚本
│   └── quick_start.py        # 快速入门示例
├── configs/                   # 配置文件
├── archive/                   # 归档的旧文件
│   ├── gradio_app.py         # 旧的简单界面
│   ├── gradio_full_interface.py  # 旧的完整界面
│   └── hybrid_transformer.py  # 已弃用的 HST 模型
├── gradio_residual_tft_app.py # 🆕 增强型 Gradio 应用
├── requirements.txt          # Python 依赖
├── setup.py                  # 包设置
├── LICENSE                   # MIT 许可证
└── README.md                # 英文说明文件
```

### 模型 API

#### StaticSensorTransformer (SST)

```python
from models.static_transformer import StaticSensorTransformer

model = StaticSensorTransformer(
    num_boundary_sensors=10,    # 输入传感器数量
    num_target_sensors=5,       # 输出传感器数量
    d_model=128,                # 模型维度
    nhead=8,                    # 注意力头数量
    num_layers=3,               # Transformer 层数
    dropout=0.1                 # Dropout 率
)

# 前向传播
predictions = model(boundary_conditions)  # 形状: (batch_size, num_target_sensors)
```

#### Stage2 残差提升训练

```python
# 步骤 1: 训练基础 SST 模型
base_model = StaticSensorTransformer(...)
# ... 训练基础模型 ...

# 步骤 2: 提取残差
residuals = true_values - base_model_predictions

# 步骤 3: 在残差上训练 Stage2 模型
stage2_model = StaticSensorTransformer(...)
# ... 在残差上训练 stage2 ...

# 步骤 4: 使用智能 Delta R² 选择生成集成
for signal_idx in range(num_signals):
    r2_base = calculate_r2(true_values[:, signal_idx], base_predictions[:, signal_idx])
    r2_ensemble = calculate_r2(true_values[:, signal_idx], base_pred[:, signal_idx] + stage2_pred[:, signal_idx])
    delta_r2 = r2_ensemble - r2_base

    if delta_r2 > threshold:  # 例如, threshold=0.05 (5% 改进)
        # 使用 Stage2 修正（显著改进）
        ensemble_pred[:, signal_idx] = base_pred[:, signal_idx] + stage2_pred[:, signal_idx]
    else:
        # 保持基础预测（无显著改进）
        ensemble_pred[:, signal_idx] = base_pred[:, signal_idx]
```

**注意**：增强型 Gradio 界面（`gradio_residual_tft_app.py`）自动化了整个工作流程。

## 🎯 性能

### 基准测试结果（示例）

在典型的工业传感器数据集上，具有 50 个边界传感器和 20 个目标传感器：

| 模型 | 平均 R² | 平均 MAE | 平均 RMSE | 训练时间 | 推理时间 |
|-------|-----------|------------|--------------|---------------|----------------|
| **SST（基础）** | 0.92 | 2.34 | 3.45 | ~15 分钟 | 0.5 毫秒/样本 |
| **SST + Stage2（集成）** | 0.96 | 1.87 | 2.76 | ~30 分钟 | 0.8 毫秒/样本 |

**Stage2 提升带来的性能改进：**
- MAE：提高 15-25%
- RMSE：提高 12-20%
- R²：低 R² 信号显著改善

*注意：结果因数据集特性、R² 阈值和硬件而异。*

## 🤝 贡献

欢迎贡献！请随时提交 Pull Request。对于重大更改，请先开启 issue 讨论您想要更改的内容。

### 开发设置

```bash
# 克隆仓库
git clone https://github.com/YOUR_USERNAME/Industrial-digital-twin-by-transformer.git
cd Industrial-digital-twin-by-transformer

# 以开发模式安装
pip install -e .

# 运行测试（如果可用）
python -m pytest tests/
```

## 📄 许可证

本项目根据 MIT 许可证授权 - 详情请参阅 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- Transformer 架构基于 "Attention Is All You Need"（Vaswani et al., 2017）
- 灵感来自工业自动化中的数字孪生应用
- 使用 PyTorch、Gradio 和出色的开源社区构建

## 📞 联系方式

如有问题、议题或合作：
- **GitHub Issues**：[创建 issue](https://github.com/YOUR_USERNAME/Industrial-digital-twin-by-transformer/issues)
- **电子邮件**：your.email@example.com

## 🔗 引用

如果您在研究中使用此工作，请引用：

```bibtex
@software{industrial_digital_twin_transformer,
  author = {Your Name},
  title = {Industrial Digital Twin by Transformer},
  year = {2025},
  url = {https://github.com/YOUR_USERNAME/Industrial-digital-twin-by-transformer}
}
```

## 🗺️ 路线图

### v1.0（当前）✅
- [x] Stage2 提升训练系统
- [x] 智能 R² 阈值选择
- [x] 集成模型生成
- [x] 推理对比工具
- [x] 增强型 Gradio 界面

### v2.0（即将推出）🚀

#### **Stage3 时序提升系统** 🕐
多阶段架构的下一次演进，专注于纯时序特征提取：

- **Stage3 残差时序建模**：在 Stage1+Stage2 残差上训练时序模型
  - 捕获空间 Transformer 遗漏的时间序列模式
  - 使用 LSTM/时序 Transformer 进行纯时序特征提取
  - 最终残差的未来纯时序预测

- **完整的空间-时间分解架构**：
  - **Stage1 (SST)**：空间传感器关系和跨传感器依赖性
  - **Stage2 (Boost)**：空间残差修正和次级空间模式
  - **Stage3 (Temporal)**：纯时序特征和时间序列动态
  - **最终目标**：将空间和时间特征完全剥离并分层预测，除不可预测的噪音特征外，捕捉所有可预测模式，实现场景泛用化的数字孪生

- **分层特征提取哲学**：
  - 第一层：主要空间传感器相关性（SST）
  - 第二层：残差空间模式（Stage2 提升）
  - 第三层：时间动态和序列依赖性（Stage3 时序）
  - 残差：不可约随机噪声（不可预测成分）

此设计旨在通过系统性地分解和捕获不同领域的所有可预测特征，实现**通用数字孪生建模**。

#### **附加功能**
- [ ] 高级残差分析和可视化工具
- [ ] 注意力机制可视化以提高可解释性
- [ ] 实时流数据支持
- [ ] Docker 容器化便于部署
- [ ] 生产环境模型服务的 REST API
- [ ] 自动化超参数调优
- [ ] 额外的基准数据集和示例

---

**为工业 AI 社区精心打造 ❤️**
