"""
Response Data Models for FastAPI
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any


class SignalAnalysis(BaseModel):
    """Signal-level Delta R² analysis"""
    signal: str = Field(..., description="Signal name")
    r2_stage1: float = Field(..., description="Stage1 R² score")
    r2_ensemble: float = Field(..., description="Ensemble R² score")
    delta_r2: float = Field(..., description="Delta R² (ensemble - stage1)")
    use_boost: bool = Field(..., description="Whether this signal uses Residual Boost")


class MetricsInfo(BaseModel):
    """Performance metrics"""
    mae: float = Field(..., description="Mean Absolute Error")
    rmse: float = Field(..., description="Root Mean Squared Error")
    r2: float = Field(..., description="R² Score")


class ImprovementInfo(BaseModel):
    """Performance improvement metrics"""
    mae_pct: float = Field(..., description="MAE improvement percentage")
    rmse_pct: float = Field(..., description="RMSE improvement percentage")
    r2_pct: float = Field(..., description="R² improvement percentage")


class ModelInfo(BaseModel):
    """Model information response"""
    model_name: str
    model_type: str = Field(..., description="Model type: 'stage1' or 'residual_boost'")
    num_boundary_signals: int
    num_target_signals: int
    config_path: str
    loaded_time: str


class EnsembleInfo(BaseModel):
    """Ensemble model information"""
    ensemble_name: str
    stage1_model_name: str
    residual_boost_model_name: str
    delta_r2_threshold: float
    signal_analysis: List[SignalAnalysis]
    num_use_boost: int = Field(..., description="Number of signals using Residual Boost")
    num_use_stage1_only: int = Field(..., description="Number of signals using Stage1 only")
    metrics: Dict[str, Any] = Field(..., description="Performance metrics")
    config_path: Optional[str] = None
    created_time: str
    updated_time: Optional[str] = None


class InferenceResult(BaseModel):
    """Batch inference result"""
    ensemble_name: str
    output_path: str = Field(..., description="Path to saved predictions CSV")
    num_samples: int = Field(..., description="Number of samples processed")
    num_signals: int = Field(..., description="Number of target signals")
    signals_used_boost: List[str] = Field(..., description="List of signals that used Residual Boost")
    num_signals_used_boost: int
    timestamp: str
    predictions: Optional[List[List[float]]] = Field(
        None,
        description="Predictions array (only included if num_samples <= 100)"
    )


class ModelListResponse(BaseModel):
    """Response for list models endpoint"""
    stage1_models: List[str] = Field(..., description="List of loaded Stage1 models")
    residual_boost_models: List[str] = Field(..., description="List of loaded Residual Boost models")
    ensemble_models: List[str] = Field(..., description="List of created ensemble models")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    gpu_available: bool = Field(..., description="Whether GPU is available")
    num_stage1_models: int
    num_residual_boost_models: int
    num_ensemble_models: int
    device: str = Field(..., description="Current device (cuda or cpu)")


class ErrorResponse(BaseModel):
    """Error response"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")


class SuccessResponse(BaseModel):
    """Generic success response"""
    message: str
    details: Optional[Dict[str, Any]] = None
