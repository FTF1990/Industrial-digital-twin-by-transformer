# Industrial Digital Twin - FastAPI Inference Service

FastAPI 推理服务，用于工业数字孪生的 Stage1 + Residual Boost 模型推理。

## 📋 功能特性

- ✅ **模型管理**: 加载和管理 Stage1 (SST) 和 Residual Boost (TFT) 模型
- ✅ **Ensemble 生成**: 基于新数据使用 Delta R² 策略生成 Ensemble 模型
- ✅ **批量推理**: 对新数据进行批量预测
- ✅ **动态阈值调整**: 无需重新推理即可更新 Delta R² 阈值
- ✅ **手动信号控制**: 推理时可手动覆盖哪些信号使用 Residual Boost

## 🚀 快速开始

### 1. 安装依赖

```bash
cd fastapi_inference
pip install -r requirements.txt
```

### 2. 启动服务

#### 方式 A: 使用 Python 模块

```bash
# 从项目根目录运行
cd /path/to/Industrial-digital-twin-by-transformer
python -m fastapi_inference.main
```

#### 方式 B: 使用 uvicorn 直接运行

```bash
cd /path/to/Industrial-digital-twin-by-transformer
uvicorn fastapi_inference.main:app --host 0.0.0.0 --port 8000
```

### 3. 访问 API 文档

服务启动后，访问以下地址：

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **API Info**: http://localhost:8000/api/v1/info

## 📖 使用流程

### 步骤 1: 加载模型

#### 1.1 加载 Stage1 模型

```bash
curl -X POST "http://localhost:8000/api/v1/models/stage1/load" \
  -H "Content-Type: application/json" \
  -d '{
    "inference_config_path": "saved_models/my_sst_model_inference.json"
  }'
```

#### 1.2 加载 Residual Boost 模型

```bash
curl -X POST "http://localhost:8000/api/v1/models/residual-boost/load" \
  -H "Content-Type: application/json" \
  -d '{
    "inference_config_path": "saved_models/tft_models/my_tft_inference.json"
  }'
```

### 步骤 2: 创建 Ensemble 模型

```bash
curl -X POST "http://localhost:8000/api/v1/ensemble/create" \
  -H "Content-Type: application/json" \
  -d '{
    "stage1_model_name": "my_sst_model",
    "residual_boost_model_name": "my_tft_model",
    "evaluation_data_path": "data/evaluation_data.csv",
    "delta_r2_threshold": 0.05,
    "save_config": true
  }'
```

**注意**: `evaluation_data.csv` 必须包含：
- 所有边界信号 (boundary signals)
- 所有目标信号的真值 (target signals ground truth)

### 步骤 3: 批量推理

```bash
curl -X POST "http://localhost:8000/api/v1/inference/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "ensemble_name": "Ensemble_my_sst_model_20251215_103000",
    "input_data_path": "data/new_data.csv",
    "output_dir": "fastapi_inference/results",
    "include_metadata": true
  }'
```

**注意**: `new_data.csv` 只需包含边界信号 (boundary signals)，不需要真值。

### 步骤 4 (可选): 更新 Delta R² 阈值

```bash
curl -X POST "http://localhost:8000/api/v1/ensemble/Ensemble_xxx/update-threshold" \
  -H "Content-Type: application/json" \
  -d '{
    "new_threshold": 0.08
  }'
```

### 步骤 5 (可选): 手动控制信号选择

```bash
curl -X POST "http://localhost:8000/api/v1/inference/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "ensemble_name": "Ensemble_xxx",
    "input_data_path": "data/new_data.csv",
    "output_dir": "fastapi_inference/results",
    "manual_boost_signals": {
      "Temperature_1": true,
      "Pressure_2": false
    }
  }'
```

## 🐍 Python 客户端示例

参见 `tests/demo_api_client.py` 获取完整的 Python 客户端示例。

基本用法：

```python
import requests

# 1. 加载 Stage1 模型
response = requests.post(
    "http://localhost:8000/api/v1/models/stage1/load",
    json={"inference_config_path": "saved_models/my_sst_model_inference.json"}
)
print(response.json())

# 2. 加载 Residual Boost 模型
response = requests.post(
    "http://localhost:8000/api/v1/models/residual-boost/load",
    json={"inference_config_path": "saved_models/tft_models/my_tft_inference.json"}
)
print(response.json())

# 3. 创建 Ensemble
response = requests.post(
    "http://localhost:8000/api/v1/ensemble/create",
    json={
        "stage1_model_name": "my_sst_model",
        "residual_boost_model_name": "my_tft_model",
        "evaluation_data_path": "data/evaluation_data.csv",
        "delta_r2_threshold": 0.05
    }
)
ensemble_name = response.json()['ensemble_name']
print(f"Ensemble created: {ensemble_name}")

# 4. 批量推理
response = requests.post(
    "http://localhost:8000/api/v1/inference/batch",
    json={
        "ensemble_name": ensemble_name,
        "input_data_path": "data/new_data.csv",
        "output_dir": "fastapi_inference/results"
    }
)
result = response.json()
print(f"Predictions saved to: {result['output_path']}")
```

## 📊 API 端点

### 模型管理

- `POST /api/v1/models/stage1/load` - 加载 Stage1 模型
- `POST /api/v1/models/residual-boost/load` - 加载 Residual Boost 模型
- `GET /api/v1/models/list` - 列出所有已加载模型
- `GET /api/v1/models/{model_type}/{name}` - 获取模型详情
- `DELETE /api/v1/models/{model_type}/{name}` - 卸载模型

### Ensemble 管理

- `POST /api/v1/ensemble/create` - 创建 Ensemble 模型
- `POST /api/v1/ensemble/{name}/update-threshold` - 更新 Delta R² 阈值
- `GET /api/v1/ensemble/list` - 列出所有 Ensemble
- `GET /api/v1/ensemble/{name}/info` - 获取 Ensemble 详情
- `DELETE /api/v1/ensemble/{name}` - 删除 Ensemble

### 推理

- `POST /api/v1/inference/batch` - 批量推理
- `GET /api/v1/health` - 健康检查

## 🔧 配置说明

### Delta R² 阈值

Delta R² 阈值决定哪些信号使用 Residual Boost：

- `Delta R² = R²_ensemble - R²_stage1`
- 如果 `Delta R² > threshold`，该信号使用 Stage1 + Residual Boost
- 否则仅使用 Stage1 预测

**推荐值**:
- `0.05` (5%) - 保守，仅对明显改善的信号使用 Boost
- `0.02` (2%) - 中等，对中等改善的信号也使用 Boost
- `0.01` (1%) - 激进，对轻微改善的信号也使用 Boost

### 输出目录

所有推理结果保存到 `output_dir` 指定的目录，文件命名格式：

```
predictions_{ensemble_name}_{timestamp}.csv
predictions_{ensemble_name}_{timestamp}_metadata.txt
```

## 🌐 Colab 环境使用

在 Google Colab 中使用 FastAPI 服务：

```python
# 1. 启动服务（后台运行）
!cd /content/Industrial-digital-twin-by-transformer && \
  nohup python -m fastapi_inference.main > fastapi.log 2>&1 &

# 2. 等待服务启动
import time
time.sleep(5)

# 3. 使用 requests 调用 API
import requests
response = requests.get("http://localhost:8000/api/v1/health")
print(response.json())
```

详见 `tests/colab_demo.ipynb`。

## 🛠️ 开发和测试

### 运行测试脚本

```bash
# 完整流程测试
python fastapi_inference/tests/demo_api_client.py

# Bash 脚本测试
bash fastapi_inference/tests/test_api.sh
```

### 查看日志

服务日志会输出到终端。如果后台运行，可以查看日志文件：

```bash
tail -f fastapi.log
```

## 📁 目录结构

```
fastapi_inference/
├── main.py                 # FastAPI 主应用
├── config.py               # 配置文件
├── requirements.txt        # 依赖
├── api/                    # API 端点
│   ├── models.py           # 模型管理
│   ├── ensemble.py         # Ensemble 管理
│   └── inference.py        # 推理端点
├── core/                   # 核心模块
│   ├── model_loader.py     # 模型加载
│   ├── ensemble_builder.py # Ensemble 生成
│   └── predictor.py        # 推理引擎
├── schemas/                # 数据模型
│   ├── requests.py         # 请求模型
│   └── responses.py        # 响应模型
├── utils/                  # 工具函数
│   ├── device.py           # GPU/CPU 管理
│   └── file_handler.py     # 文件处理
├── tests/                  # 测试脚本
│   ├── demo_api_client.py  # Python 客户端示例
│   ├── test_api.sh         # Bash 测试脚本
│   └── colab_demo.ipynb    # Colab 测试笔记本
└── results/                # 推理结果输出目录
```

## ❓ 常见问题

### Q: 如何指定使用 GPU？

A: 服务会自动检测 GPU。如果 CUDA 可用，会自动使用 GPU。

### Q: 模型文件路径如何指定？

A: 路径相对于项目根目录。例如：
- `saved_models/my_model_inference.json`
- `../saved_models/my_model.pth`

### Q: 评估数据必须包含哪些列？

A: 必须包含：
1. 所有边界信号（与模型训练时一致）
2. 所有目标信号的真值（用于计算 R²）

### Q: 推理数据必须包含哪些列？

A: 仅需包含所有边界信号。不需要目标信号的真值。

### Q: 如何查看已创建的 Ensemble 配置？

A: 配置文件保存在 `saved_models/ensemble/{ensemble_name}_config.json`

## 📞 支持

如有问题，请在 GitHub 仓库提交 Issue：
https://github.com/FTF1990/Industrial-digital-twin-by-transformer/issues

## 📝 License

MIT License

## 🌊 流式推理 (WebSocket)

### 新功能：实时流式推理

FastAPI 服务现在支持通过 WebSocket 进行实时流式推理！

#### 特性

- ✅ **实时推理**: 低延迟（10-20ms）
- ✅ **双向通信**: WebSocket 双向实时通信
- ✅ **两种模式**: 单条模式 + 批量模式
- ✅ **统计信息**: 实时监控连接和性能
- ✅ **历史保存**: 保存推理历史到文件

#### 快速开始

```python
import asyncio
import websockets
import json

async def stream_inference():
    uri = "ws://localhost:8000/api/v1/inference/stream"
    
    async with websockets.connect(uri) as ws:
        # 配置
        await ws.send(json.dumps({
            "type": "config",
            "data": {
                "ensemble_name": "Ensemble_your_model_20251215_103000",
                "mode": "single"
            }
        }))
        await ws.recv()  # 接收确认
        
        # 发送数据并获取预测
        await ws.send(json.dumps({
            "type": "predict",
            "data": {
                "boundary_signals": {
                    "Temperature_boundary_1": 23.5,
                    "Pressure_boundary_1": 101.3,
                    # ...
                }
            }
        }))
        
        result = json.loads(await ws.recv())
        print(result['data']['predictions'])

asyncio.run(stream_inference())
```

#### 完整文档

详见 **[流式推理文档](STREAMING.md)**

#### Demo 客户端

```bash
# 运行流式推理 Demo
python fastapi_inference/tests/demo_stream_client.py
```

#### API 端点

- **WebSocket**: `ws://localhost:8000/api/v1/inference/stream`
- **统计信息**: `GET /api/v1/inference/stream/stats`
- **保存历史**: `POST /api/v1/inference/stream/save`

