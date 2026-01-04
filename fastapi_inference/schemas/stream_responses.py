"""
Stream Response Data Models for WebSocket
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Literal, Any
from datetime import datetime


class StreamConfigAckResponse(BaseModel):
    """Configuration acknowledgment response"""
    type: Literal["config_ack"] = "config_ack"
    status: Literal["success", "error"] = "success"
    message: str = "Configuration applied"
    session_id: str
    ensemble_info: Optional[Dict[str, Any]] = None


class StreamPredictionResponse(BaseModel):
    """Single prediction response"""
    type: Literal["prediction"] = "prediction"
    status: Literal["success", "error"] = "success"
    data: Optional["StreamPredictionData"] = None
    error: Optional[str] = None


class StreamPredictionData(BaseModel):
    """Prediction data"""
    predictions: Dict[str, float] = Field(..., description="Predicted values")
    signals_used_boost: List[str] = Field(..., description="Signals that used Residual Boost")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")
    timestamp: Optional[str] = None


class StreamPredictionBatchResponse(BaseModel):
    """Batch prediction response"""
    type: Literal["prediction_batch"] = "prediction_batch"
    status: Literal["success", "error"] = "success"
    data: Optional["StreamPredictionBatchData"] = None
    error: Optional[str] = None


class StreamPredictionBatchData(BaseModel):
    """Batch prediction data"""
    predictions: List[Dict[str, float]]
    count: int
    latency_ms: float


class StreamPongResponse(BaseModel):
    """Pong response for heartbeat"""
    type: Literal["pong"] = "pong"
    timestamp: str


class StreamErrorResponse(BaseModel):
    """Error response"""
    type: Literal["error"] = "error"
    error_code: str
    message: str
    details: Optional[Dict[str, Any]] = None


class StreamStatsResponse(BaseModel):
    """Streaming statistics response"""
    active_connections: int
    total_predictions: int
    average_latency_ms: float
    connections: List["ConnectionStats"]


class ConnectionStats(BaseModel):
    """Individual connection statistics"""
    session_id: str
    ensemble_name: str
    connected_at: str
    predictions_count: int
    mode: str


class StreamSaveResponse(BaseModel):
    """Save history response"""
    status: Literal["success", "error"]
    message: str
    output_path: Optional[str] = None
    samples_saved: Optional[int] = None
