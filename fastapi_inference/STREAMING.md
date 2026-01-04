# WebSocket 流式推理文档

FastAPI 推理服务的 WebSocket 流式推理功能完整文档。

## 🌊 概述

流式推理通过 WebSocket 提供实时、低延迟的预测服务，适用于：
- 实时传感器数据流
- 持续监控系统
- 高频率预测场景
- IoT 设备数据处理

---

## 🚀 快速开始

### Python 客户端

```python
import asyncio
import websockets
import json

async def stream_inference():
    uri = "ws://localhost:8000/api/v1/inference/stream"
    
    async with websockets.connect(uri) as websocket:
        # 1. 发送配置
        await websocket.send(json.dumps({
            "type": "config",
            "data": {
                "ensemble_name": "Ensemble_your_model_20251215_103000",
                "mode": "single"
            }
        }))
        
        # 2. 接收确认
        config_ack = json.loads(await websocket.recv())
        print(f"Connected: {config_ack['session_id']}")
        
        # 3. 发送数据并获取预测
        await websocket.send(json.dumps({
            "type": "predict",
            "data": {
                "boundary_signals": {
                    "Temperature_boundary_1": 23.5,
                    "Pressure_boundary_1": 101.3,
                    # ... 其他边界信号
                }
            }
        }))
        
        # 4. 接收预测结果
        result = json.loads(await websocket.recv())
        print(f"Prediction: {result['data']['predictions']}")

asyncio.run(stream_inference())
```

---

## 📡 WebSocket 协议

### 连接 URL

```
ws://localhost:8000/api/v1/inference/stream
```

### 消息格式

所有消息使用 JSON 格式，包含 `type` 和 `data` 字段。

---

## 📋 消息类型

### 1. 配置消息 (config)

**客户端发送**:
```json
{
  "type": "config",
  "data": {
    "ensemble_name": "Ensemble_my_sst_model_20251215_103000",
    "manual_boost_signals": {
      "Temperature_1": true,
      "Pressure_2": false
    },
    "mode": "single",
    "batch_size": 10,
    "include_metadata": true,
    "output_format": "json"
  }
}
```

**服务器响应**:
```json
{
  "type": "config_ack",
  "status": "success",
  "message": "Configuration applied",
  "session_id": "session_abc123",
  "ensemble_info": {
    "ensemble_name": "Ensemble_xxx",
    "num_signals": 20,
    "signals_using_boost": 12
  }
}
```

---

### 2. 单条预测 (predict)

**客户端发送**:
```json
{
  "type": "predict",
  "data": {
    "boundary_signals": {
      "Temperature_boundary_1": 23.5,
      "Pressure_boundary_1": 101.3,
      "Flow_boundary_1": 50.2
    },
    "timestamp": "2025-12-15T10:30:00"
  }
}
```

**服务器响应**:
```json
{
  "type": "prediction",
  "status": "success",
  "data": {
    "predictions": {
      "Temperature_1": 25.3,
      "Pressure_2": 102.1,
      "Flow_3": 55.6
    },
    "signals_used_boost": ["Temperature_1", "Flow_3"],
    "latency_ms": 12.5,
    "timestamp": "2025-12-15T10:30:00"
  }
}
```

---

### 3. 批量预测 (predict_batch)

**客户端发送**:
```json
{
  "type": "predict_batch",
  "data": {
    "batch": [
      {
        "Temperature_boundary_1": 23.5,
        "Pressure_boundary_1": 101.3
      },
      {
        "Temperature_boundary_1": 23.6,
        "Pressure_boundary_1": 101.4
      }
    ],
    "timestamps": ["2025-12-15T10:30:00", "2025-12-15T10:30:01"]
  }
}
```

**服务器响应**:
```json
{
  "type": "prediction_batch",
  "status": "success",
  "data": {
    "predictions": [
      {"Temperature_1": 25.3, "Pressure_2": 102.1},
      {"Temperature_1": 25.4, "Pressure_2": 102.2}
    ],
    "count": 2,
    "latency_ms": 25.8
  }
}
```

---

### 4. 心跳检测 (ping/pong)

**客户端发送**:
```json
{
  "type": "ping"
}
```

**服务器响应**:
```json
{
  "type": "pong",
  "timestamp": "2025-12-15T10:30:00"
}
```

---

### 5. 错误响应 (error)

**服务器响应**:
```json
{
  "type": "error",
  "error_code": "MISSING_SIGNALS",
  "message": "Missing boundary signals: ['Temperature_boundary_1']",
  "details": {
    "missing_signals": ["Temperature_boundary_1"]
  }
}
```

**错误代码**:
- `INVALID_MESSAGE` - 消息格式错误
- `MISSING_ENSEMBLE` - 缺少 ensemble 名称
- `ENSEMBLE_NOT_FOUND` - Ensemble 不存在
- `MISSING_SIGNALS` - 缺少必需的边界信号
- `EMPTY_BATCH` - 批量数据为空
- `BATCH_TOO_LARGE` - 批量大小超过限制
- `PREDICTION_ERROR` - 预测过程错误
- `UNKNOWN_MESSAGE_TYPE` - 未知消息类型
- `INVALID_JSON` - JSON 解析错误
- `INTERNAL_ERROR` - 服务器内部错误

---

## 🔧 HTTP 端点

### 获取统计信息

**端点**: `GET /api/v1/inference/stream/stats`

**响应**:
```json
{
  "active_connections": 3,
  "total_predictions": 12543,
  "average_latency_ms": 15.2,
  "connections": [
    {
      "session_id": "session_abc123",
      "ensemble_name": "Ensemble_xxx",
      "connected_at": "2025-12-15T10:00:00",
      "predictions_count": 523,
      "mode": "single"
    }
  ]
}
```

### 保存历史记录

**端点**: `POST /api/v1/inference/stream/save`

**请求**:
```json
{
  "session_id": "session_abc123",
  "output_dir": "fastapi_inference/results/stream",
  "format": "csv"
}
```

**响应**:
```json
{
  "status": "success",
  "message": "History saved successfully",
  "output_path": "fastapi_inference/results/stream/stream_history_session_abc123_20251215_103000.csv",
  "samples_saved": 523
}
```

---

## 💡 使用示例

### 示例 1: 实时数据流

```python
import asyncio
import websockets
import json
import time

async def realtime_stream():
    uri = "ws://localhost:8000/api/v1/inference/stream"
    
    async with websockets.connect(uri) as ws:
        # 配置
        await ws.send(json.dumps({
            "type": "config",
            "data": {
                "ensemble_name": "Ensemble_my_model_20251215_103000",
                "mode": "single"
            }
        }))
        await ws.recv()  # 接收 config_ack
        
        # 持续发送数据
        while True:
            # 模拟传感器读数
            data = {
                "type": "predict",
                "data": {
                    "boundary_signals": {
                        "Temperature_boundary_1": 20 + time.time() % 10,
                        "Pressure_boundary_1": 100 + time.time() % 5,
                        # ...
                    }
                }
            }
            
            await ws.send(json.dumps(data))
            result = json.loads(await ws.recv())
            
            if result['type'] == 'prediction':
                print(f"Latency: {result['data']['latency_ms']:.2f} ms")
            
            await asyncio.sleep(0.05)  # 20 Hz

asyncio.run(realtime_stream())
```

### 示例 2: 批量处理

```python
async def batch_processing():
    uri = "ws://localhost:8000/api/v1/inference/stream"
    
    async with websockets.connect(uri) as ws:
        # 配置批量模式
        await ws.send(json.dumps({
            "type": "config",
            "data": {
                "ensemble_name": "Ensemble_my_model_20251215_103000",
                "mode": "batch",
                "batch_size": 50
            }
        }))
        await ws.recv()
        
        # 准备批量数据
        batch = []
        for i in range(50):
            batch.append({
                "Temperature_boundary_1": 23.5 + i * 0.1,
                "Pressure_boundary_1": 101.3 + i * 0.05,
                # ...
            })
        
        # 发送批量请求
        await ws.send(json.dumps({
            "type": "predict_batch",
            "data": {"batch": batch}
        }))
        
        result = json.loads(await ws.recv())
        print(f"Processed {result['data']['count']} samples")
        print(f"Total latency: {result['data']['latency_ms']:.2f} ms")

asyncio.run(batch_processing())
```

### 示例 3: 使用客户端类

参见完整示例：`fastapi_inference/tests/demo_stream_client.py`

```bash
# 运行 Demo
python fastapi_inference/tests/demo_stream_client.py
```

---

## 📊 性能特性

### 延迟

- **单条模式**: 通常 10-20ms（取决于模型大小和硬件）
- **批量模式**: 更高吞吐量，但单样本延迟稍高
- **WebSocket 开销**: < 1ms

### 吞吐量

- **单条模式**: ~50-100 predictions/sec（取决于模型）
- **批量模式**: ~500-1000 predictions/sec（批量大小 50-100）
- **多连接**: 支持多个并发 WebSocket 连接

### 建议

- **高频低延迟**: 使用单条模式
- **高吞吐量**: 使用批量模式，批量大小 20-50
- **持续监控**: 定期发送 ping 检查连接

---

## 🔒 安全考虑

### 连接限制

- 默认无限制，生产环境建议配置
- 可通过中间件添加速率限制

### 数据验证

- 所有输入数据严格验证
- 信号名称和数量检查
- 批量大小限制

### 超时处理

- 长时间无活动连接会被自动断开
- 建议定期发送 ping 保持连接

---

## 🐛 故障排除

### 连接失败

```
WebSocketException: Connection refused
```

**解决**:
1. 确认服务器正在运行
2. 检查 URL 和端口是否正确
3. 检查防火墙设置

### 配置失败

```json
{
  "type": "error",
  "error_code": "ENSEMBLE_NOT_FOUND"
}
```

**解决**:
1. 确认 Ensemble 已创建
2. 使用 `GET /api/v1/ensemble/list` 查看可用 Ensemble
3. 检查拼写错误

### 预测失败

```json
{
  "type": "error",
  "error_code": "MISSING_SIGNALS"
}
```

**解决**:
1. 检查所有必需的边界信号是否提供
2. 使用 `GET /api/v1/ensemble/{name}/info` 查看所需信号
3. 检查信号名称拼写

---

## 📚 更多资源

- **完整 Demo**: `fastapi_inference/tests/demo_stream_client.py`
- **API 文档**: http://localhost:8000/docs
- **统计信息**: `GET /api/v1/inference/stream/stats`
- **主文档**: `fastapi_inference/README.md`

---

## 🎯 最佳实践

1. **连接管理**
   - 使用连接池管理多个连接
   - 实现自动重连机制
   - 定期发送 ping 保持连接

2. **错误处理**
   - 捕获所有异常
   - 实现重试逻辑
   - 记录错误日志

3. **性能优化**
   - 根据场景选择单条/批量模式
   - 调整批量大小平衡延迟和吞吐量
   - 监控延迟指标

4. **资源管理**
   - 及时关闭不用的连接
   - 定期清理历史数据
   - 监控服务器资源使用

---

**版本**: 1.0.0  
**更新时间**: 2025-12-15
