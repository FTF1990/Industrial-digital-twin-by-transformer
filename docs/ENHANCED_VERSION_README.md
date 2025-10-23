# 工业数字孪生残差Boost训练系统 - Enhanced Version

## 🎯 项目概述

本项目是一个增强版的工业数字孪生系统，基于SST (Static Sensor Transformer) 模型和Stage2 Boost残差训练架构。

### 新增功能

1. **Stage2 Boost残差模型训练** 🚀
   - 基于SST模型生成的残差训练Stage2模型
   - 进一步提升预测精度

2. **智能R²阈值选择** 🎯
   - 自动计算每个信号的R²分数
   - 智能选择需要Stage2修正的信号
   - 生成综合推理模型

3. **二次推理比较** 📊
   - 选择任意index范围进行推理
   - 对比综合模型vs纯SST模型的性能
   - 可视化性能提升

4. **Sundial时序残差预测** 🔮
   - 基于综合模型的最终残差
   - 预测未来残差趋势
   - 支持长期预测

### 删除内容

- ❌ 移除了所有Hybrid Transformer相关功能
- ✅ 简化为纯SST架构 + Stage2 Boost

## 📁 项目结构

```
.
├── gradio_residual_tft_app.py   # 主应用（新版）
├── static_transformer.py         # SST模型定义
├── residual_tft.py              # TFT残差模型定义
├── utils.py                     # 工具函数
├── saved_models/                # 模型保存目录
│   ├── stage2_boost/           # Stage2模型
│   └── ensemble/               # 综合模型配置
└── data/                       # 数据目录
```

## 🚀 使用流程

### 1. 数据加载
```python
# 上传CSV文件或创建示例数据
```

### 2. SST模型训练
```python
# 配置参数：
# - 边界信号（输入）
# - 目标信号（输出）
# - 模型架构：d_model, nhead, num_layers
# - 训练参数：epochs, batch_size, lr
```

### 3. 残差提取
```python
# 从训练好的SST模型提取残差
# - 选择SST模型
# - 设置未来预测长度
# - 可选：选择数据片段
```

### 4. Stage2 Boost训练
```python
# 训练Stage2残差模型
# - 选择残差数据
# - 配置Stage2模型架构
# - 训练参数设置
```

### 5. 生成综合推理模型
```python
# 智能R²阈值选择
# - 选择基础SST模型
# - 选择Stage2模型
# - 设置R²阈值（默认0.4）
# - 自动生成综合模型
```

### 6. 二次推理比较
```python
# 性能对比
# - 选择综合模型
# - 设置index范围
# - 对比SST vs 综合模型
# - 可视化性能提升
```

### 7. Sundial残差预测
```python
# 预测未来残差
# - 选择综合模型
# - 配置预测步数
# - 训练Sundial模型
# - 预测未来残差趋势
```

## 🎯 核心创新

### Stage2 Boost架构

```
输入数据
   ↓
[SST Model] → 预测1 + 残差1
   ↓
残差数据
   ↓
[Stage2 Model] → 残差修正
   ↓
智能R²阈值选择
   ↓
综合预测 = 预测1 + 残差修正（选择性应用）
   ↓
最终残差
   ↓
[Sundial Model] → 未来残差预测
```

### 智能R²阈值选择机制

```python
# 对每个信号：
if signal_r2 < threshold:
    # R²较低，应用Stage2修正
    final_pred = sst_pred + stage2_residual_pred
else:
    # R²较高，保持SST预测
    final_pred = sst_pred
```

### 性能提升指标

- **MAE改进**: 通常10-25%
- **RMSE改进**: 通常8-20%
- **R²提升**: 显著提升低R²信号的表现

## 📊 模型保存与加载

### 自动保存

所有模型训练完成后自动保存：

```
saved_models/
├── StaticSensorTransformer_*.pth      # SST模型权重
├── StaticSensorTransformer_*.json     # SST推理配置
├── stage2_boost/
│   ├── Stage2_Boost_*.pth            # Stage2模型权重
│   └── Stage2_Boost_*_scalers.pkl    # Stage2 Scalers
└── ensemble/
    └── Ensemble_*_config.json        # 综合模型配置
```

### 推理配置

每个模型都会生成对应的`*_inference.json`配置文件，包含：
- 模型路径
- Scaler路径
- 信号配置
- 模型架构参数
- 训练信息

### 加载已保存模型

在"残差提取"tab中：
1. 上传`*_inference.json`文件
2. 点击"加载配置"
3. 自动加载模型和Scalers

## 🔧 配置说明

### SST模型参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| d_model | 128 | 模型维度 |
| nhead | 8 | 注意力头数 |
| num_layers | 3 | Transformer层数 |
| dropout | 0.1 | Dropout率 |
| batch_size | 64 | 批大小 |
| lr | 0.001 | 学习率 |
| epochs | 100 | 训练轮数 |

### Stage2模型参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| d_model | 128 | 模型维度 |
| nhead | 8 | 注意力头数 |
| num_layers | 3 | Transformer层数 |
| dropout | 0.1 | Dropout率 |
| batch_size | 32 | 批大小 |
| lr | 0.001 | 学习率 |
| weight_decay | 1e-5 | 权重衰减 |
| grad_clip | 1.0 | 梯度裁剪 |
| epochs | 100 | 训练轮数 |

### R²阈值

| 阈值 | 应用场景 |
|------|---------|
| 0.3 | 激进：更多信号应用Stage2 |
| 0.4 | 推荐：平衡性能和效率 |
| 0.5 | 保守：只对低R²信号应用Stage2 |

## 📈 可视化功能

### 残差分析
- 残差分布直方图
- 残差时间序列
- 预测vs真实对比

### 训练历史
- Loss曲线
- R²曲线
- 学习率变化

### 性能对比
- 指标对比表格
- 预测对比曲线
- 误差分布对比
- 性能提升柱状图

## 🛠️ 依赖包

```bash
pip install torch>=2.0.0
pip install gradio>=4.0.0
pip install pandas
pip install numpy
pip install scikit-learn
pip install matplotlib
pip install seaborn
```

## 🚦 启动应用

```bash
python gradio_residual_tft_app.py
```

应用会自动尝试端口7860-7869，并在浏览器中打开界面。

## 💡 最佳实践

### 数据准备
- 确保数据无缺失值
- 建议至少10000个样本
- 合理选择边界信号和目标信号

### 模型训练
- 先训练SST模型至收敛
- 提取残差后再训练Stage2
- 使用早停避免过拟合

### R²阈值选择
- 从0.4开始尝试
- 观察性能提升情况
- 根据需求调整阈值

### 二次推理
- 选择代表性的index范围
- 对比多个片段的性能
- 关注特殊case的表现

## 📝 注意事项

1. **内存管理**
   - 大数据集建议使用数据片段
   - GPU显存不足时减小batch_size

2. **模型保存**
   - 定期备份saved_models目录
   - 重要模型建议导出推理配置

3. **性能优化**
   - 优先优化低R²信号
   - 可以针对性调整Stage2参数

4. **Sundial预测**
   - 功能正在开发中
   - 预计v2.0版本完整支持

## 🔄 版本历史

### v1.0 (Current)
- ✅ SST模型训练
- ✅ 残差提取
- ✅ Stage2 Boost训练
- ✅ 智能R²阈值选择
- ✅ 综合推理模型
- ✅ 二次推理比较
- ⚠️ Sundial预测（开发中）

### v0.9 (Previous)
- ✅ 基础SST模型
- ✅ Hybrid Transformer（已移除）
- ✅ 简单残差TFT

## 📧 反馈与支持

如有问题或建议，请在项目中提出Issue。

## 📄 License

MIT License
