"""
Request Data Models for FastAPI
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict


class LoadStage1Request(BaseModel):
    """Request model for loading Stage1 SST model"""
    inference_config_path: str = Field(
        ...,
        description="Path to inference config JSON file",
        example="saved_models/my_sst_model_inference.json"
    )
    model_name: Optional[str] = Field(
        None,
        description="Optional model name (auto-extracted from config if not provided)"
    )


class LoadResidualBoostRequest(BaseModel):
    """Request model for loading Residual Boost (TFT) model"""
    inference_config_path: str = Field(
        ...,
        description="Path to inference config JSON file",
        example="saved_models/tft_models/my_tft_inference.json"
    )
    model_name: Optional[str] = Field(
        None,
        description="Optional model name (auto-extracted from config if not provided)"
    )


class CreateEnsembleRequest(BaseModel):
    """Request model for creating ensemble model"""
    stage1_model_name: str = Field(
        ...,
        description="Name of loaded Stage1 model"
    )
    residual_boost_model_name: str = Field(
        ...,
        description="Name of loaded Residual Boost model"
    )
    evaluation_data_path: str = Field(
        ...,
        description="Path to CSV file with boundary and target signals (with ground truth)"
    )
    ensemble_name: Optional[str] = Field(
        None,
        description="Optional ensemble name (auto-generated if not provided)"
    )
    delta_r2_threshold: float = Field(
        0.05,
        description="Delta R² threshold for signal selection (default 0.05 = 5%)",
        ge=0.0,
        le=1.0
    )
    save_config: bool = Field(
        True,
        description="Whether to save ensemble config to file"
    )


class UpdateThresholdRequest(BaseModel):
    """Request model for updating ensemble Delta R² threshold"""
    new_threshold: float = Field(
        ...,
        description="New Delta R² threshold",
        ge=0.0,
        le=1.0
    )


class BatchInferenceRequest(BaseModel):
    """Request model for batch inference"""
    ensemble_name: str = Field(
        ...,
        description="Name of ensemble model to use"
    )
    input_data_path: str = Field(
        ...,
        description="Path to input CSV file (must contain boundary signals)",
        example="data/new_data.csv"
    )
    output_dir: str = Field(
        ...,
        description="Output directory for predictions",
        example="fastapi_inference/results"
    )
    manual_boost_signals: Optional[Dict[str, bool]] = Field(
        None,
        description="Optional manual override for signal boost selection {'signal_name': True/False}"
    )
    include_metadata: bool = Field(
        True,
        description="Whether to save metadata file alongside predictions"
    )


class Config:
    schema_extra = {
        "example": {
            "ensemble_name": "Ensemble_my_model_20251215_103000",
            "input_data_path": "data/test_data.csv",
            "output_dir": "fastapi_inference/results",
            "manual_boost_signals": {
                "Temperature_1": True,
                "Pressure_2": False
            },
            "include_metadata": True
        }
    }
