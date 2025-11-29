# 🚀 快速部署指南（中文）

## 三种部署方法

### 方法 1: 自动脚本部署（推荐）⭐

```bash
# 1. 确保已安装 Git LFS
git lfs install

# 2. 在 Hugging Face 网站创建 Space
# 访问: https://huggingface.co/new-space
# 选择 SDK: Gradio

# 3. 运行部署脚本
./deploy_to_hf.sh YOUR_HF_USERNAME YOUR_SPACE_NAME

# 例如:
./deploy_to_hf.sh john-doe industrial-twin
```

---

### 方法 2: 手动部署（完全控制）

```bash
# 1. 克隆您的 HF Space
git clone https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
cd YOUR_SPACE_NAME

# 2. 复制文件
cp /path/to/project/app.py ./
cp /path/to/project/gradio_sensor_transformer_app.py ./
cp /path/to/project/requirements_hf.txt ./requirements.txt
cp /path/to/project/README_HF_SPACES.md ./README.md
cp /path/to/project/.gitattributes ./

# 3. 复制代码目录
cp -r /path/to/project/models ./
cp -r /path/to/project/src ./

# 4. 【可选】复制模型和数据
cp -r /path/to/project/saved_models ./
cp -r /path/to/project/data ./

# 5. 提交并推送
git add .
git commit -m "Initial deployment"
git push
```

---

### 方法 3: HF Web 界面上传（最简单）

1. 创建 Space: https://huggingface.co/new-space
2. 选择 **Gradio** SDK
3. 进入 Space → **Files and versions**
4. 逐个上传文件：
   - `app.py`
   - `gradio_sensor_transformer_app.py`
   - `requirements.txt`（使用 `requirements_hf.txt` 内容）
   - `README.md`（使用 `README_HF_SPACES.md` 内容）
   - 上传 `models/` 和 `src/` 文件夹
5. 等待自动构建

---

## 📦 上传您的模型和数据

### 小文件（< 100MB）- 使用 Git

```bash
cd YOUR_SPACE_NAME

# 添加您的模型
cp /path/to/your_model.pth ./saved_models/
cp /path/to/your_model_config.json ./saved_models/
cp /path/to/your_model_scaler.pkl ./saved_models/

# 添加数据
cp /path/to/your_data.csv ./data/

# 提交
git add saved_models/ data/
git commit -m "Add pretrained models and data"
git push
```

### 大文件（> 100MB）- 使用 Git LFS

```bash
# 确保 Git LFS 已初始化
git lfs install

# 添加大文件
cp /path/to/large_model.pth ./saved_models/
git add saved_models/large_model.pth
git commit -m "Add large model"
git push
```

### 超大文件（> 5GB）- 使用 HF Hub CLI

```bash
# 安装工具
pip install huggingface_hub

# 上传文件
huggingface-cli upload YOUR_USERNAME/YOUR_SPACE_NAME \
    ./saved_models/huge_model.pth \
    saved_models/huge_model.pth
```

---

## ✅ 检查清单

部署前确保：

- [ ] 已创建 HF Space（选择 Gradio SDK）
- [ ] 已安装 Git LFS
- [ ] `README.md` 包含正确的 YAML 配置
- [ ] `app.py` 是入口文件
- [ ] `requirements.txt` 包含所有依赖
- [ ] 所有代码目录（`models/`, `src/`）已复制

部署后检查：

- [ ] Space 构建成功（无错误）
- [ ] 应用可以正常打开
- [ ] 可以上传 CSV 数据
- [ ] 可以进行训练（测试功能）

---

## 🔧 常见问题

### 问题：构建失败

**检查**：
1. Space 页面 → Logs → 查看错误信息
2. 确认 `requirements.txt` 中的依赖版本正确
3. 确认 `README.md` 顶部的 YAML 配置正确

### 问题：应用无法启动

**检查**：
1. 确认 `app.py` 存在
2. 确认 `app.py` 中端口设置为 `7860`
3. 确认 `server_name="0.0.0.0"`

### 问题：模型文件丢失

**检查**：
1. 使用 `git lfs ls-files` 查看 LFS 文件
2. 确认 `.gitattributes` 正确配置
3. 大文件需要使用 Git LFS

---

## 📞 需要帮助？

- **详细指南**：查看 `DEPLOYMENT_GUIDE.md`
- **HF 文档**：https://huggingface.co/docs/hub/spaces
- **项目 Issues**：https://github.com/FTF1990/Industrial-digital-twin-by-transformer/issues

---

祝您部署顺利！🎉
