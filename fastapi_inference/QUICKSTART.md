# FastAPI Inference Service - Quick Start Guide

## 🚀 3分钟快速开始

### 方式 1: 使用启动脚本（推荐）

```bash
cd /path/to/Industrial-digital-twin-by-transformer
bash fastapi_inference/start_server.sh
```

### 方式 2: 手动启动

```bash
# 1. 安装依赖
pip install -r fastapi_inference/requirements.txt

# 2. 启动服务
python -m fastapi_inference.main
```

### 方式 3: Colab 环境

```python
# 后台启动
!cd /content/Industrial-digital-twin-by-transformer && \
  nohup python -m fastapi_inference.main > fastapi.log 2>&1 &

# 等待启动
import time
time.sleep(5)

# 测试连接
import requests
response = requests.get("http://localhost:8000/api/v1/health")
print(response.json())
```

## 📖 访问文档

启动后访问：

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🧪 测试连接

```bash
# Bash
curl http://localhost:8000/api/v1/health

# Python
python fastapi_inference/tests/demo_api_client.py

# 或者
bash fastapi_inference/tests/test_api.sh
```

## 📝 完整使用示例

### 1. 准备数据和模型

确保你有：
- ✅ Stage1 模型的 `inference.json` 配置文件
- ✅ Residual Boost 模型的 `inference.json` 配置文件
- ✅ 评估数据 CSV（包含边界信号 + 目标信号真值）
- ✅ 推理数据 CSV（仅需边界信号）

### 2. 使用 Python 客户端

```python
import requests

BASE_URL = "http://localhost:8000"

# 1. 加载 Stage1 模型
response = requests.post(
    f"{BASE_URL}/api/v1/models/stage1/load",
    json={"inference_config_path": "saved_models/my_sst_model_inference.json"}
)
stage1_info = response.json()
print(f"Stage1 loaded: {stage1_info['model_name']}")

# 2. 加载 Residual Boost 模型
response = requests.post(
    f"{BASE_URL}/api/v1/models/residual-boost/load",
    json={"inference_config_path": "saved_models/tft_models/my_tft_inference.json"}
)
rb_info = response.json()
print(f"Residual Boost loaded: {rb_info['model_name']}")

# 3. 创建 Ensemble
response = requests.post(
    f"{BASE_URL}/api/v1/ensemble/create",
    json={
        "stage1_model_name": stage1_info['model_name'],
        "residual_boost_model_name": rb_info['model_name'],
        "evaluation_data_path": "data/evaluation_data.csv",
        "delta_r2_threshold": 0.05
    }
)
ensemble_info = response.json()
ensemble_name = ensemble_info['ensemble_name']
print(f"Ensemble created: {ensemble_name}")

# 4. 批量推理
response = requests.post(
    f"{BASE_URL}/api/v1/inference/batch",
    json={
        "ensemble_name": ensemble_name,
        "input_data_path": "data/new_data.csv",
        "output_dir": "fastapi_inference/results"
    }
)
result = response.json()
print(f"Predictions saved to: {result['output_path']}")
```

### 3. 使用 curl

```bash
# 1. 加载模型
curl -X POST "http://localhost:8000/api/v1/models/stage1/load" \
  -H "Content-Type: application/json" \
  -d '{"inference_config_path": "saved_models/my_sst_model_inference.json"}'

# 2. 创建 Ensemble
curl -X POST "http://localhost:8000/api/v1/ensemble/create" \
  -H "Content-Type: application/json" \
  -d '{
    "stage1_model_name": "my_sst_model",
    "residual_boost_model_name": "my_tft_model",
    "evaluation_data_path": "data/evaluation_data.csv",
    "delta_r2_threshold": 0.05
  }'

# 3. 推理
curl -X POST "http://localhost:8000/api/v1/inference/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "ensemble_name": "Ensemble_my_sst_model_20251215_103000",
    "input_data_path": "data/new_data.csv",
    "output_dir": "fastapi_inference/results"
  }'
```

## 📁 输出文件

推理结果保存在 `output_dir` 指定的目录：

- `predictions_{ensemble_name}_{timestamp}.csv` - 预测结果
- `predictions_{ensemble_name}_{timestamp}_metadata.txt` - 元数据信息

## 🔍 查看日志

```bash
# 如果使用启动脚本，日志在终端显示
# 如果后台运行，查看日志文件
tail -f fastapi.log
```

## ❓ 常见问题

### Q: 端口被占用怎么办？

修改 `fastapi_inference/config.py` 中的 `PORT` 配置。

### Q: GPU 没有被使用？

检查 PyTorch CUDA 是否安装正确：
```python
import torch
print(torch.cuda.is_available())
```

### Q: 模型文件找不到？

确保路径相对于项目根目录，或使用绝对路径。

## 📚 更多文档

详细文档请参考：`fastapi_inference/README.md`
