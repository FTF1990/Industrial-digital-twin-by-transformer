"""
Ensemble Management API Endpoints
"""
from fastapi import APIRouter, HTTPException
from typing import List
from datetime import datetime

from ..schemas.requests import CreateEnsembleRequest, UpdateThresholdRequest
from ..schemas.responses import EnsembleInfo, SuccessResponse, SignalAnalysis
from ..core.ensemble_builder import EnsembleBuilder
from ..utils.file_handler import read_csv_data, validate_data_format
from .models import model_state, get_device

# Create router
router = APIRouter(prefix="/api/v1/ensemble", tags=["Ensemble"])

# Global state to store ensemble models
ensemble_state = {
    'ensemble_models': {}
}


@router.post("/create", response_model=EnsembleInfo)
async def create_ensemble(request: CreateEnsembleRequest):
    """
    Create ensemble model from evaluation data using Delta R² strategy

    Process:
    1. Load evaluation data (must contain boundary + target signals with ground truth)
    2. Run Stage1 and Residual Boost inference
    3. Calculate Delta R² for each signal
    4. Select signals based on threshold
    5. Save ensemble configuration

    Args:
        request: Ensemble creation request

    Returns:
        EnsembleInfo: Created ensemble information
    """
    try:
        # Verify models are loaded
        if request.stage1_model_name not in model_state['stage1_models']:
            raise HTTPException(
                status_code=404,
                detail=f"Stage1 model '{request.stage1_model_name}' not loaded. Load it first via /api/v1/models/stage1/load"
            )

        if request.residual_boost_model_name not in model_state['residual_boost_models']:
            raise HTTPException(
                status_code=404,
                detail=f"Residual Boost model '{request.residual_boost_model_name}' not loaded. Load it first via /api/v1/models/residual-boost/load"
            )

        stage1_model_info = model_state['stage1_models'][request.stage1_model_name]
        residual_boost_model_info = model_state['residual_boost_models'][request.residual_boost_model_name]

        # Get signal lists
        boundary_signals = stage1_model_info['boundary_signals']
        target_signals = stage1_model_info['target_signals']

        # Verify signal compatibility
        if set(boundary_signals) != set(residual_boost_model_info['boundary_signals']):
            raise HTTPException(
                status_code=400,
                detail="Boundary signals mismatch between Stage1 and Residual Boost models"
            )
        if set(target_signals) != set(residual_boost_model_info['target_signals']):
            raise HTTPException(
                status_code=400,
                detail="Target signals mismatch between Stage1 and Residual Boost models"
            )

        # Load evaluation data
        print(f"=Â Loading evaluation data: {request.evaluation_data_path}")
        df_eval = read_csv_data(request.evaluation_data_path)

        # Validate data format
        is_valid, error_msg = validate_data_format(df_eval, boundary_signals, target_signals)
        if not is_valid:
            raise HTTPException(status_code=400, detail=f"Invalid data format: {error_msg}")

        # Extract data as numpy array
        all_signals = boundary_signals + target_signals
        evaluation_data = df_eval[all_signals].values

        # Generate ensemble name if not provided
        ensemble_name = request.ensemble_name
        if not ensemble_name:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            ensemble_name = f"Ensemble_{request.stage1_model_name}_{timestamp}"

        # Create ensemble
        device = get_device()
        ensemble_info_dict = EnsembleBuilder.create_ensemble_from_data(
            ensemble_name=ensemble_name,
            stage1_model_info=stage1_model_info,
            residual_boost_model_info=residual_boost_model_info,
            evaluation_data=evaluation_data,
            boundary_signals=boundary_signals,
            target_signals=target_signals,
            delta_r2_threshold=request.delta_r2_threshold,
            device=device,
            save_config=request.save_config
        )

        # Add model names for reference
        ensemble_info_dict['stage1_model_info'] = stage1_model_info
        ensemble_info_dict['residual_boost_model_info'] = residual_boost_model_info

        # Store in global state
        ensemble_state['ensemble_models'][ensemble_name] = ensemble_info_dict

        # Build response
        return EnsembleInfo(
            ensemble_name=ensemble_info_dict['ensemble_name'],
            stage1_model_name=ensemble_info_dict['stage1_model_name'],
            residual_boost_model_name=ensemble_info_dict['residual_boost_model_name'],
            delta_r2_threshold=ensemble_info_dict['delta_r2_threshold'],
            signal_analysis=[SignalAnalysis(**item) for item in ensemble_info_dict['signal_analysis']],
            num_use_boost=ensemble_info_dict['num_use_boost'],
            num_use_stage1_only=ensemble_info_dict['num_use_stage1_only'],
            metrics=ensemble_info_dict['metrics'],
            config_path=ensemble_info_dict.get('config_path'),
            created_time=ensemble_info_dict['created_time']
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create ensemble: {str(e)}")


@router.post("/{ensemble_name}/update-threshold", response_model=EnsembleInfo)
async def update_threshold(ensemble_name: str, request: UpdateThresholdRequest):
    """
    Update Delta R² threshold and regenerate signal selection

    This updates the ensemble configuration without re-running inference.
    Uses the stored predictions from original evaluation.

    Args:
        ensemble_name: Name of the ensemble
        request: Update request with new threshold

    Returns:
        EnsembleInfo: Updated ensemble information
    """
    try:
        # Check if ensemble exists
        if ensemble_name not in ensemble_state['ensemble_models']:
            raise HTTPException(
                status_code=404,
                detail=f"Ensemble '{ensemble_name}' not found. Create it first via /api/v1/ensemble/create"
            )

        ensemble_info_dict = ensemble_state['ensemble_models'][ensemble_name]

        # Update threshold
        updated_info = EnsembleBuilder.update_ensemble_threshold(
            ensemble_name=ensemble_name,
            ensemble_info=ensemble_info_dict,
            new_threshold=request.new_threshold,
            save_config=True
        )

        # Update global state
        ensemble_state['ensemble_models'][ensemble_name] = updated_info

        # Build response
        return EnsembleInfo(
            ensemble_name=updated_info['ensemble_name'],
            stage1_model_name=updated_info['stage1_model_name'],
            residual_boost_model_name=updated_info['residual_boost_model_name'],
            delta_r2_threshold=updated_info['delta_r2_threshold'],
            signal_analysis=[SignalAnalysis(**item) for item in updated_info['signal_analysis']],
            num_use_boost=updated_info['num_use_boost'],
            num_use_stage1_only=updated_info['num_use_stage1_only'],
            metrics=updated_info['metrics'],
            config_path=updated_info.get('config_path'),
            created_time=updated_info['created_time'],
            updated_time=updated_info.get('updated_time')
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update threshold: {str(e)}")


@router.get("/list")
async def list_ensembles() -> List[str]:
    """
    List all created ensemble models

    Returns:
        List of ensemble names
    """
    return list(ensemble_state['ensemble_models'].keys())


@router.get("/{ensemble_name}/info", response_model=EnsembleInfo)
async def get_ensemble_info(ensemble_name: str):
    """
    Get detailed information about an ensemble model

    Args:
        ensemble_name: Name of the ensemble

    Returns:
        EnsembleInfo: Ensemble information
    """
    if ensemble_name not in ensemble_state['ensemble_models']:
        raise HTTPException(status_code=404, detail=f"Ensemble '{ensemble_name}' not found")

    ensemble_info_dict = ensemble_state['ensemble_models'][ensemble_name]

    return EnsembleInfo(
        ensemble_name=ensemble_info_dict['ensemble_name'],
        stage1_model_name=ensemble_info_dict['stage1_model_name'],
        residual_boost_model_name=ensemble_info_dict['residual_boost_model_name'],
        delta_r2_threshold=ensemble_info_dict['delta_r2_threshold'],
        signal_analysis=[SignalAnalysis(**item) for item in ensemble_info_dict['signal_analysis']],
        num_use_boost=ensemble_info_dict['num_use_boost'],
        num_use_stage1_only=ensemble_info_dict['num_use_stage1_only'],
        metrics=ensemble_info_dict['metrics'],
        config_path=ensemble_info_dict.get('config_path'),
        created_time=ensemble_info_dict['created_time'],
        updated_time=ensemble_info_dict.get('updated_time')
    )


@router.delete("/{ensemble_name}", response_model=SuccessResponse)
async def delete_ensemble(ensemble_name: str):
    """
    Delete an ensemble model from memory

    Args:
        ensemble_name: Name of the ensemble

    Returns:
        SuccessResponse: Success message
    """
    if ensemble_name not in ensemble_state['ensemble_models']:
        raise HTTPException(status_code=404, detail=f"Ensemble '{ensemble_name}' not found")

    del ensemble_state['ensemble_models'][ensemble_name]
    return SuccessResponse(message=f"Ensemble '{ensemble_name}' deleted successfully")
