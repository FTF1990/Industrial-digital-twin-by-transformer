# 🚀 Hugging Face Spaces 部署指南

## 📋 部署前准备

### 1. 准备您的文件

确保您有以下文件准备好：

```
Industrial-digital-twin-by-transformer/
├── app.py                              # ✅ 已创建（入口文件）
├── gradio_sensor_transformer_app.py    # ✅ 已存在（主应用）
├── requirements_hf.txt                 # ✅ 已创建（依赖）
├── README_HF_SPACES.md                 # ✅ 已创建（配置）
├── .gitattributes                      # ✅ 已创建（Git LFS）
├── models/                             # ✅ 模型代码
│   ├── __init__.py
│   ├── static_transformer.py
│   ├── residual_tft.py
│   └── utils.py
├── src/                                # ✅ 源代码
│   ├── __init__.py
│   ├── data_loader.py
│   ├── trainer.py
│   └── inference.py
├── saved_models/                       # 📦 您的预训练模型（可选）
│   ├── your_model.pth
│   ├── your_model_config.json
│   └── your_model_scaler.pkl
└── data/                               # 📊 您的示例数据（可选）
    └── demo_data.csv
```

---

## 🛠️ 部署步骤

### 步骤 1: 创建 Hugging Face 账号

1. 访问 https://huggingface.co/
2. 注册/登录账号
3. 验证您的电子邮件地址

### 步骤 2: 创建新的 Space

1. 点击右上角头像 → "New Space"
2. 填写信息：
   - **Space name**: `industrial-digital-twin`（或您喜欢的名称）
   - **License**: MIT
   - **Select the Space SDK**: **Gradio**
   - **Space hardware**: CPU basic（免费）或选择 GPU
   - **Visibility**: Public 或 Private

3. 点击 "Create Space"

### 步骤 3: 安装 Git LFS（用于上传大文件）

```bash
# Ubuntu/Debian
sudo apt-get install git-lfs

# macOS
brew install git-lfs

# Windows
# 从 https://git-lfs.github.com/ 下载安装

# 初始化 Git LFS
git lfs install
```

### 步骤 4: 克隆您的 Space 仓库

```bash
# 在 Hugging Face Space 页面复制仓库 URL
git clone https://huggingface.co/spaces/YOUR_USERNAME/industrial-digital-twin
cd industrial-digital-twin
```

### 步骤 5: 复制项目文件到 Space 仓库

```bash
# 从您的项目目录复制必要文件

# 1. 核心文件
cp /path/to/Industrial-digital-twin-by-transformer/app.py ./
cp /path/to/Industrial-digital-twin-by-transformer/gradio_sensor_transformer_app.py ./
cp /path/to/Industrial-digital-twin-by-transformer/.gitattributes ./

# 2. 依赖文件
cp /path/to/Industrial-digital-twin-by-transformer/requirements_hf.txt ./requirements.txt

# 3. README（重要：HF Spaces 配置在这里）
cp /path/to/Industrial-digital-twin-by-transformer/README_HF_SPACES.md ./README.md

# 4. 复制代码目录
cp -r /path/to/Industrial-digital-twin-by-transformer/models ./
cp -r /path/to/Industrial-digital-twin-by-transformer/src ./

# 5. 【可选】复制您的预训练模型
cp -r /path/to/Industrial-digital-twin-by-transformer/saved_models ./

# 6. 【可选】复制示例数据
cp -r /path/to/Industrial-digital-twin-by-transformer/data ./
```

### 步骤 6: 上传文件到 Hugging Face

```bash
# 添加所有文件
git add .

# 如果有大文件（>10MB），使用 Git LFS
# 例如：模型文件和数据文件会自动通过 .gitattributes 配置使用 LFS
git lfs track "*.pth"
git lfs track "*.pkl"
git lfs track "*.csv"

# 提交
git commit -m "Initial deployment: Industrial Digital Twin app"

# 推送到 Hugging Face（第一次需要登录）
git push
```

**首次推送时的登录**：
- Username: 您的 HF 用户名
- Password: 使用 **Access Token**（不是密码）
  - 获取 Token: https://huggingface.co/settings/tokens
  - 创建一个 "Write" 权限的 Token

### 步骤 7: 等待构建完成

1. 推送后，访问您的 Space 页面：`https://huggingface.co/spaces/YOUR_USERNAME/industrial-digital-twin`
2. 查看 "Building" 状态
3. 通常 3-5 分钟完成构建
4. 构建完成后，应用会自动启动

---

## 📦 上传您的模型和数据

### 方法 1: 通过 Git（推荐用于小文件 <100MB）

```bash
# 将您的模型放在 saved_models/ 目录
cp your_trained_model.pth saved_models/
cp your_trained_model_config.json saved_models/
cp your_trained_model_scaler.pkl saved_models/

# 将数据放在 data/ 目录
cp your_data.csv data/

# 提交并推送
git add saved_models/ data/
git commit -m "Add pretrained models and demo data"
git push
```

### 方法 2: 通过 Hugging Face Hub（推荐用于大文件 >100MB）

```bash
# 安装 huggingface_hub
pip install huggingface_hub

# 上传单个文件
huggingface-cli upload YOUR_USERNAME/industrial-digital-twin \
    ./saved_models/large_model.pth \
    saved_models/large_model.pth

# 或使用 Python
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj="./saved_models/large_model.pth",
    path_in_repo="saved_models/large_model.pth",
    repo_id="YOUR_USERNAME/industrial-digital-twin",
    repo_type="space",
)
```

### 方法 3: 通过 Web 界面（最简单，适合单个文件）

1. 访问您的 Space 页面
2. 点击 "Files and versions" 标签
3. 点击 "Upload files" 按钮
4. 拖拽文件或选择文件上传

---

## 🎯 重要配置说明

### 1. README.md 顶部的 YAML 配置（必须）

```yaml
---
title: Industrial Digital Twin by Transformer
emoji: 🏭
colorFrom: blue
colorTo: green
sdk: gradio              # 必须是 gradio
sdk_version: 4.8.0       # Gradio 版本
app_file: app.py         # 入口文件（必须）
pinned: false
license: mit
---
```

### 2. 硬件选择

**免费选项**：
- **CPU basic**: 免费，2 vCPU，16GB RAM
- 适合小模型和演示

**付费选项**（如需 GPU）：
- **Tesla T4**: $0.60/小时
- **A10G**: $3.15/小时
- **A100**: $4.13/小时

在 Space Settings → Hardware 中选择

### 3. 持久化存储（Persistent Storage）

**重要**：默认情况下，Space 重启后用户上传的文件会丢失。

**启用持久化存储**：
1. 进入 Space Settings
2. 找到 "Persistent Storage"
3. 选择存储大小（例如 20GB）
4. 需要付费：约 $5/月/50GB

**在代码中使用**：
```python
import os
# 持久化目录（如果启用了 persistent storage）
PERSISTENT_DIR = os.environ.get("HF_HOME", "./saved_models")
os.makedirs(PERSISTENT_DIR, exist_ok=True)
```

---

## 🔧 常见问题解决

### 问题 1: 构建失败 - 依赖错误

**解决方案**：检查 `requirements.txt`
- 确保版本兼容
- 移除不必要的依赖（如 jupyter）

### 问题 2: 应用启动失败

**解决方案**：检查日志
1. 在 Space 页面点击 "Logs"
2. 查看错误信息
3. 常见问题：
   - 缺少文件或目录
   - 端口配置错误（确保使用 `server_port=7860`）

### 问题 3: 文件上传失败（文件太大）

**解决方案**：
- 单个文件 <5GB: 使用 Git LFS
- 单个文件 >5GB: 需要升级到 Pro 账户

### 问题 4: 模型加载失败

**解决方案**：
- 检查模型文件路径
- 确保 `.pth`, `.pkl`, `.json` 文件都已上传
- 检查 Git LFS 是否正确跟踪文件

---

## 📊 示例：完整的文件结构

```
industrial-digital-twin/  (HF Space 仓库)
├── .gitattributes                      # Git LFS 配置
├── README.md                           # ⚠️ 包含 HF Spaces YAML 配置
├── app.py                              # 入口文件
├── gradio_sensor_transformer_app.py    # 主应用
├── requirements.txt                    # Python 依赖
│
├── models/                             # 模型代码
│   ├── __init__.py
│   ├── static_transformer.py
│   ├── residual_tft.py
│   └── utils.py
│
├── src/                                # 源代码
│   ├── __init__.py
│   ├── data_loader.py
│   ├── trainer.py
│   └── inference.py
│
├── saved_models/                       # 预训练模型（可选）
│   ├── MyModel.pth                     # ← 您的模型文件
│   ├── MyModel_config.json
│   ├── MyModel_scaler.pkl
│   └── stage2_boost/
│       └── ...
│
└── data/                               # 示例数据（可选）
    └── demo_sensor_data.csv            # ← 您的演示数据
```

---

## 🎉 部署成功后

您的应用将在以下 URL 可访问：
```
https://huggingface.co/spaces/YOUR_USERNAME/industrial-digital-twin
```

**分享您的应用**：
- 直接分享 URL
- 嵌入到网页：使用 HF 提供的嵌入代码
- 设为 Private（仅限邀请用户访问）

---

## 📚 更多资源

- **Hugging Face Spaces 文档**: https://huggingface.co/docs/hub/spaces
- **Gradio 文档**: https://gradio.app/docs/
- **项目 GitHub**: https://github.com/FTF1990/Industrial-digital-twin-by-transformer

---

## 💡 提示

1. **首次部署**：先不上传模型和数据，确保应用能正常运行
2. **测试**：使用用户上传功能测试应用
3. **添加模型**：确认应用正常后，再添加预训练模型
4. **监控**：定期检查 Space 日志和使用情况

---

需要帮助？提交 Issue: https://github.com/FTF1990/Industrial-digital-twin-by-transformer/issues
