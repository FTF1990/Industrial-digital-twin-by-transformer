"""
Inference API Endpoints
"""
from fastapi import APIRouter, HTTPException
import torch

from ..schemas.requests import BatchInferenceRequest
from ..schemas.responses import InferenceResult, HealthResponse
from ..core.predictor import Predictor
from ..utils.file_handler import read_csv_data, validate_data_format
from ..utils.device import print_gpu_memory
from .models import model_state
from .ensemble import ensemble_state

# Create router
router = APIRouter(prefix="/api/v1", tags=["Inference"])


@router.post("/inference/batch", response_model=InferenceResult)
async def batch_inference(request: BatchInferenceRequest):
    """
    Perform batch inference using ensemble model

    Process:
    1. Load input data (must contain boundary signals)
    2. Run Stage1 inference
    3. Run Residual Boost inference
    4. Apply ensemble configuration (or manual override)
    5. Save predictions to output directory

    Args:
        request: Batch inference request

    Returns:
        InferenceResult: Inference results and output path
    """
    try:
        # Verify ensemble exists
        if request.ensemble_name not in ensemble_state['ensemble_models']:
            raise HTTPException(
                status_code=404,
                detail=f"Ensemble '{request.ensemble_name}' not found. Create it first via /api/v1/ensemble/create"
            )

        ensemble_info = ensemble_state['ensemble_models'][request.ensemble_name]

        # Get model names from ensemble
        stage1_model_name = ensemble_info['stage1_model_name']
        residual_boost_model_name = ensemble_info['residual_boost_model_name']

        # Verify models are still loaded
        if stage1_model_name not in model_state['stage1_models']:
            raise HTTPException(
                status_code=404,
                detail=f"Stage1 model '{stage1_model_name}' not loaded. It may have been unloaded."
            )

        if residual_boost_model_name not in model_state['residual_boost_models']:
            raise HTTPException(
                status_code=404,
                detail=f"Residual Boost model '{residual_boost_model_name}' not loaded. It may have been unloaded."
            )

        stage1_model_info = ensemble_info['stage1_model_info']
        residual_boost_model_info = ensemble_info['residual_boost_model_info']

        # Get boundary signals
        boundary_signals = ensemble_info['signals']['boundary']

        # Load input data
        print(f"=Â Loading input data: {request.input_data_path}")
        df_input = read_csv_data(request.input_data_path)

        # Validate data format (only boundary signals required for inference)
        is_valid, error_msg = validate_data_format(df_input, boundary_signals)
        if not is_valid:
            raise HTTPException(status_code=400, detail=f"Invalid input data format: {error_msg}")

        # Extract boundary data
        input_data = df_input[boundary_signals].values

        # Get device
        device = model_state['device']
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Run batch prediction
        result_info = Predictor.batch_predict(
            ensemble_info=ensemble_info,
            stage1_model_info=stage1_model_info,
            residual_boost_model_info=residual_boost_model_info,
            input_data=input_data,
            output_dir=request.output_dir,
            manual_boost_signals=request.manual_boost_signals,
            device=device,
            include_metadata=request.include_metadata
        )

        # Print GPU memory if available
        if device.type == 'cuda':
            print_gpu_memory()

        return InferenceResult(**result_info)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch inference failed: {str(e)}")


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint

    Returns:
        HealthResponse: Service health status
    """
    device = model_state.get('device')
    if device is None:
        device_str = 'not_initialized'
        gpu_available = torch.cuda.is_available()
    else:
        device_str = str(device)
        gpu_available = device.type == 'cuda'

    return HealthResponse(
        status='healthy',
        gpu_available=gpu_available,
        num_stage1_models=len(model_state['stage1_models']),
        num_residual_boost_models=len(model_state['residual_boost_models']),
        num_ensemble_models=len(ensemble_state['ensemble_models']),
        device=device_str
    )
