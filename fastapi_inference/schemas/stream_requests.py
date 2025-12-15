"""
Stream Request Data Models for WebSocket
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Literal
from datetime import datetime


class StreamConfigRequest(BaseModel):
    """WebSocket configuration request"""
    type: Literal["config"] = "config"
    data: "StreamConfigData"


class StreamConfigData(BaseModel):
    """Configuration data for streaming session"""
    ensemble_name: str = Field(..., description="Name of ensemble model to use")
    manual_boost_signals: Optional[Dict[str, bool]] = Field(
        None,
        description="Optional manual override for signal boost selection"
    )
    mode: Literal["single", "batch"] = Field(
        "single",
        description="Inference mode: 'single' for one sample at a time, 'batch' for multiple samples"
    )
    batch_size: Optional[int] = Field(
        None,
        description="Batch size (only used in batch mode)",
        ge=1,
        le=1000
    )
    include_metadata: bool = Field(
        True,
        description="Whether to include metadata in responses"
    )
    output_format: Literal["json", "csv"] = Field(
        "json",
        description="Output format for predictions"
    )


class StreamPredictRequest(BaseModel):
    """Single prediction request"""
    type: Literal["predict"] = "predict"
    data: "StreamPredictData"


class StreamPredictData(BaseModel):
    """Prediction data for single sample"""
    boundary_signals: Dict[str, float] = Field(
        ...,
        description="Boundary signal values as dict {signal_name: value}"
    )
    timestamp: Optional[str] = Field(
        None,
        description="Optional timestamp for this sample"
    )


class StreamPredictBatchRequest(BaseModel):
    """Batch prediction request"""
    type: Literal["predict_batch"] = "predict_batch"
    data: "StreamPredictBatchData"


class StreamPredictBatchData(BaseModel):
    """Prediction data for batch of samples"""
    batch: List[Dict[str, float]] = Field(
        ...,
        description="List of boundary signal dicts"
    )
    timestamps: Optional[List[str]] = Field(
        None,
        description="Optional timestamps for each sample"
    )


class StreamPingRequest(BaseModel):
    """Ping request for heartbeat"""
    type: Literal["ping"] = "ping"


class StreamSaveRequest(BaseModel):
    """Request to save streaming history"""
    session_id: str = Field(..., description="Session ID to save")
    output_dir: str = Field(
        "fastapi_inference/results/stream",
        description="Output directory for saved results"
    )
    format: Literal["csv", "json"] = Field(
        "csv",
        description="Output format"
    )
