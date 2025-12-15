# API Endpoints Reference

FastAPI 推理服务的完整 API 端点参考文档。

## 基础信息

- **Base URL**: `http://localhost:8000`
- **API Version**: v1
- **Documentation**: http://localhost:8000/docs

---

## 📋 模型管理 API

### 1. 加载 Stage1 模型

**Endpoint**: `POST /api/v1/models/stage1/load`

**Request Body**:
```json
{
  "inference_config_path": "saved_models/my_sst_model_inference.json",
  "model_name": "custom_name"  // 可选
}
```

**Response**:
```json
{
  "model_name": "my_sst_model",
  "model_type": "stage1",
  "num_boundary_signals": 10,
  "num_target_signals": 20,
  "config_path": "saved_models/my_sst_model_inference.json",
  "loaded_time": "2025-12-15 10:30:00"
}
```

---

### 2. 加载 Residual Boost 模型

**Endpoint**: `POST /api/v1/models/residual-boost/load`

**Request Body**:
```json
{
  "inference_config_path": "saved_models/tft_models/my_tft_inference.json",
  "model_name": "custom_name"  // 可选
}
```

**Response**:
```json
{
  "model_name": "my_tft_model",
  "model_type": "residual_boost",
  "num_boundary_signals": 10,
  "num_target_signals": 20,
  "config_path": "saved_models/tft_models/my_tft_inference.json",
  "loaded_time": "2025-12-15 10:31:00"
}
```

---

### 3. 列出所有模型

**Endpoint**: `GET /api/v1/models/list`

**Response**:
```json
{
  "stage1_models": ["my_sst_model", "another_sst"],
  "residual_boost_models": ["my_tft_model"],
  "ensemble_models": ["Ensemble_xxx_20251215_103000"]
}
```

---

### 4. 获取模型详情

**Endpoint**: `GET /api/v1/models/{model_type}/{model_name}`

**Parameters**:
- `model_type`: `stage1` | `residual-boost`
- `model_name`: 模型名称

**Example**: `GET /api/v1/models/stage1/my_sst_model`

**Response**: 同加载模型的响应

---

### 5. 卸载模型

**Endpoint**: `DELETE /api/v1/models/{model_type}/{model_name}`

**Response**:
```json
{
  "message": "Stage1 model 'my_sst_model' unloaded successfully"
}
```

---

## 🎯 Ensemble 管理 API

### 1. 创建 Ensemble 模型

**Endpoint**: `POST /api/v1/ensemble/create`

**Request Body**:
```json
{
  "stage1_model_name": "my_sst_model",
  "residual_boost_model_name": "my_tft_model",
  "evaluation_data_path": "data/evaluation_data.csv",
  "ensemble_name": "custom_ensemble_name",  // 可选，自动生成
  "delta_r2_threshold": 0.05,  // Delta R² 阈值
  "save_config": true
}
```

**Response**:
```json
{
  "ensemble_name": "Ensemble_my_sst_model_20251215_103000",
  "stage1_model_name": "my_sst_model",
  "residual_boost_model_name": "my_tft_model",
  "delta_r2_threshold": 0.05,
  "signal_analysis": [
    {
      "signal": "Temperature_1",
      "r2_stage1": 0.85,
      "r2_ensemble": 0.92,
      "delta_r2": 0.07,
      "use_boost": true
    },
    {
      "signal": "Pressure_2",
      "r2_stage1": 0.90,
      "r2_ensemble": 0.91,
      "delta_r2": 0.01,
      "use_boost": false
    }
  ],
  "num_use_boost": 12,
  "num_use_stage1_only": 8,
  "metrics": {
    "stage1": {
      "mae": 0.123,
      "rmse": 0.234,
      "r2": 0.85
    },
    "ensemble": {
      "mae": 0.098,
      "rmse": 0.187,
      "r2": 0.92
    },
    "improvement": {
      "mae_pct": 20.33,
      "rmse_pct": 20.09,
      "r2_pct": 46.67
    }
  },
  "config_path": "saved_models/ensemble/Ensemble_xxx_config.json",
  "created_time": "2025-12-15 10:35:00"
}
```

---

### 2. 更新 Delta R² 阈值

**Endpoint**: `POST /api/v1/ensemble/{ensemble_name}/update-threshold`

**Request Body**:
```json
{
  "new_threshold": 0.08
}
```

**Response**: 同创建 Ensemble 的响应，包含更新后的 `signal_analysis`

---

### 3. 列出所有 Ensemble

**Endpoint**: `GET /api/v1/ensemble/list`

**Response**:
```json
[
  "Ensemble_my_sst_model_20251215_103000",
  "Ensemble_another_20251215_110000"
]
```

---

### 4. 获取 Ensemble 详情

**Endpoint**: `GET /api/v1/ensemble/{ensemble_name}/info`

**Response**: 同创建 Ensemble 的响应

---

### 5. 删除 Ensemble

**Endpoint**: `DELETE /api/v1/ensemble/{ensemble_name}`

**Response**:
```json
{
  "message": "Ensemble 'Ensemble_xxx' deleted successfully"
}
```

---

## 🚀 推理 API

### 1. 批量推理

**Endpoint**: `POST /api/v1/inference/batch`

**Request Body**:
```json
{
  "ensemble_name": "Ensemble_my_sst_model_20251215_103000",
  "input_data_path": "data/new_data.csv",
  "output_dir": "fastapi_inference/results",
  "manual_boost_signals": {  // 可选：手动覆盖信号选择
    "Temperature_1": true,
    "Pressure_2": false
  },
  "include_metadata": true
}
```

**Response**:
```json
{
  "ensemble_name": "Ensemble_my_sst_model_20251215_103000",
  "output_path": "fastapi_inference/results/predictions_xxx_20251215_104500.csv",
  "num_samples": 1000,
  "num_signals": 20,
  "signals_used_boost": ["Temperature_1", "Pressure_3", "Flow_5"],
  "num_signals_used_boost": 3,
  "timestamp": "2025-12-15 10:45:00",
  "predictions": null  // 仅当样本数 <= 100 时返回
}
```

---

### 2. 健康检查

**Endpoint**: `GET /api/v1/health`

**Response**:
```json
{
  "status": "healthy",
  "gpu_available": true,
  "num_stage1_models": 2,
  "num_residual_boost_models": 1,
  "num_ensemble_models": 1,
  "device": "cuda"
}
```

---

## 📊 其他端点

### API 信息

**Endpoint**: `GET /api/v1/info`

**Response**:
```json
{
  "title": "Industrial Digital Twin - Inference API",
  "version": "1.0.0",
  "description": "FastAPI service for Industrial Digital Twin inference",
  "endpoints": {
    "models": "/api/v1/models",
    "ensemble": "/api/v1/ensemble",
    "inference": "/api/v1/inference",
    "health": "/api/v1/health"
  },
  "documentation": {
    "swagger": "/docs",
    "redoc": "/redoc"
  }
}
```

---

## 🔍 状态码

- `200 OK`: 请求成功
- `400 Bad Request`: 请求参数错误
- `404 Not Found`: 资源不存在（模型或 Ensemble 未找到）
- `500 Internal Server Error`: 服务器内部错误

---

## 📝 使用流程

推荐的标准使用流程：

1. **加载模型**
   - `POST /api/v1/models/stage1/load`
   - `POST /api/v1/models/residual-boost/load`

2. **创建 Ensemble**
   - `POST /api/v1/ensemble/create`

3. **批量推理**
   - `POST /api/v1/inference/batch`

4. **（可选）调整阈值**
   - `POST /api/v1/ensemble/{name}/update-threshold`
   - 重新运行推理

---

## 💡 示例脚本

完整的 Python 客户端示例：`fastapi_inference/tests/demo_api_client.py`

Bash 测试脚本：`fastapi_inference/tests/test_api.sh`

Colab 笔记本：`fastapi_inference/tests/colab_demo.ipynb`
