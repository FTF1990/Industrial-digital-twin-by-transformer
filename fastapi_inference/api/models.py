"""
Model Management API Endpoints
"""
from fastapi import APIRouter, HTTPException
from typing import Dict

from ..schemas.requests import LoadStage1Request, LoadResidualBoostRequest
from ..schemas.responses import ModelInfo, ModelListResponse, SuccessResponse
from ..core.model_loader import ModelLoader
from ..utils.device import setup_device

# Create router
router = APIRouter(prefix="/api/v1/models", tags=["Models"])

# Global state to store loaded models
model_state = {
    'stage1_models': {},
    'residual_boost_models': {},
    'device': None
}


def get_device():
    """Get or initialize device"""
    if model_state['device'] is None:
        model_state['device'] = setup_device()
    return model_state['device']


@router.post("/stage1/load", response_model=ModelInfo)
async def load_stage1(request: LoadStage1Request):
    """
    Load Stage1 SST model from inference config

    Args:
        request: Load request with inference_config_path

    Returns:
        ModelInfo: Loaded model information
    """
    try:
        device = get_device()
        model_name, model_info = ModelLoader.load_stage1_model(
            request.inference_config_path,
            device
        )

        # Override name if provided
        if request.model_name:
            model_name = request.model_name
            model_info['model_name'] = model_name

        # Store in global state
        model_state['stage1_models'][model_name] = model_info

        return ModelInfo(
            model_name=model_name,
            model_type='stage1',
            num_boundary_signals=len(model_info['boundary_signals']),
            num_target_signals=len(model_info['target_signals']),
            config_path=model_info['config_path'],
            loaded_time=model_info['loaded_time']
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load Stage1 model: {str(e)}")


@router.post("/residual-boost/load", response_model=ModelInfo)
async def load_residual_boost(request: LoadResidualBoostRequest):
    """
    Load Residual Boost (TFT) model from inference config

    Args:
        request: Load request with inference_config_path

    Returns:
        ModelInfo: Loaded model information
    """
    try:
        device = get_device()
        model_name, model_info = ModelLoader.load_residual_boost_model(
            request.inference_config_path,
            device
        )

        # Override name if provided
        if request.model_name:
            model_name = request.model_name
            model_info['model_name'] = model_name

        # Store in global state
        model_state['residual_boost_models'][model_name] = model_info

        return ModelInfo(
            model_name=model_name,
            model_type='residual_boost',
            num_boundary_signals=len(model_info['boundary_signals']),
            num_target_signals=len(model_info['target_signals']),
            config_path=model_info['config_path'],
            loaded_time=model_info['loaded_time']
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load Residual Boost model: {str(e)}")


@router.get("/list", response_model=ModelListResponse)
async def list_models():
    """
    List all loaded models

    Returns:
        ModelListResponse: Lists of all loaded models
    """
    # Get ensemble models from ensemble router state
    from .ensemble import ensemble_state

    return ModelListResponse(
        stage1_models=list(model_state['stage1_models'].keys()),
        residual_boost_models=list(model_state['residual_boost_models'].keys()),
        ensemble_models=list(ensemble_state['ensemble_models'].keys())
    )


@router.get("/{model_type}/{model_name}", response_model=ModelInfo)
async def get_model_info(model_type: str, model_name: str):
    """
    Get detailed information about a loaded model

    Args:
        model_type: 'stage1' or 'residual-boost'
        model_name: Name of the model

    Returns:
        ModelInfo: Model information
    """
    if model_type == 'stage1':
        if model_name not in model_state['stage1_models']:
            raise HTTPException(status_code=404, detail=f"Stage1 model '{model_name}' not found")

        model_info = model_state['stage1_models'][model_name]
        return ModelInfo(
            model_name=model_name,
            model_type='stage1',
            num_boundary_signals=len(model_info['boundary_signals']),
            num_target_signals=len(model_info['target_signals']),
            config_path=model_info['config_path'],
            loaded_time=model_info['loaded_time']
        )

    elif model_type == 'residual-boost':
        if model_name not in model_state['residual_boost_models']:
            raise HTTPException(status_code=404, detail=f"Residual Boost model '{model_name}' not found")

        model_info = model_state['residual_boost_models'][model_name]
        return ModelInfo(
            model_name=model_name,
            model_type='residual_boost',
            num_boundary_signals=len(model_info['boundary_signals']),
            num_target_signals=len(model_info['target_signals']),
            config_path=model_info['config_path'],
            loaded_time=model_info['loaded_time']
        )

    else:
        raise HTTPException(status_code=400, detail=f"Invalid model_type: {model_type}")


@router.delete("/{model_type}/{model_name}", response_model=SuccessResponse)
async def unload_model(model_type: str, model_name: str):
    """
    Unload a model from memory

    Args:
        model_type: 'stage1' or 'residual-boost'
        model_name: Name of the model

    Returns:
        SuccessResponse: Success message
    """
    if model_type == 'stage1':
        if model_name not in model_state['stage1_models']:
            raise HTTPException(status_code=404, detail=f"Stage1 model '{model_name}' not found")

        del model_state['stage1_models'][model_name]
        return SuccessResponse(message=f"Stage1 model '{model_name}' unloaded successfully")

    elif model_type == 'residual-boost':
        if model_name not in model_state['residual_boost_models']:
            raise HTTPException(status_code=404, detail=f"Residual Boost model '{model_name}' not found")

        del model_state['residual_boost_models'][model_name]
        return SuccessResponse(message=f"Residual Boost model '{model_name}' unloaded successfully")

    else:
        raise HTTPException(status_code=400, detail=f"Invalid model_type: {model_type}")
