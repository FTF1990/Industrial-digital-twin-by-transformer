"""
WebSocket Streaming Inference API Endpoints
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse
import json
from typing import Dict, Any
from datetime import datetime
import pandas as pd
import os

from ..core.stream_predictor import stream_manager, StreamSession
from ..schemas.stream_requests import StreamSaveRequest
from ..schemas.stream_responses import (
    StreamStatsResponse,
    StreamSaveResponse,
    ConnectionStats
)
from .models import model_state
from .ensemble import ensemble_state

# Create router
router = APIRouter(prefix="/api/v1/inference", tags=["Streaming Inference"])


@router.websocket("/stream")
async def websocket_stream_inference(websocket: WebSocket):
    """
    WebSocket endpoint for streaming inference
    
    Protocol:
    1. Client connects
    2. Client sends config message with ensemble name and options
    3. Server responds with config_ack
    4. Client sends predict/predict_batch messages
    5. Server responds with prediction/prediction_batch
    6. Client can send ping, server responds with pong
    7. Client disconnects
    """
    await websocket.accept()
    
    session: StreamSession = None
    session_id = None
    
    try:
        # Wait for configuration message
        config_data = await websocket.receive_text()
        config_msg = json.loads(config_data)
        
        if config_msg.get('type') != 'config':
            await websocket.send_json({
                'type': 'error',
                'error_code': 'INVALID_MESSAGE',
                'message': 'First message must be a config message'
            })
            await websocket.close()
            return
        
        # Extract configuration
        config = config_msg.get('data', {})
        ensemble_name = config.get('ensemble_name')
        
        if not ensemble_name:
            await websocket.send_json({
                'type': 'error',
                'error_code': 'MISSING_ENSEMBLE',
                'message': 'ensemble_name is required in config'
            })
            await websocket.close()
            return
        
        # Verify ensemble exists
        if ensemble_name not in ensemble_state['ensemble_models']:
            await websocket.send_json({
                'type': 'error',
                'error_code': 'ENSEMBLE_NOT_FOUND',
                'message': f"Ensemble '{ensemble_name}' not found"
            })
            await websocket.close()
            return
        
        ensemble_info = ensemble_state['ensemble_models'][ensemble_name]
        
        # Get model infos
        stage1_model_info = ensemble_info['stage1_model_info']
        residual_boost_model_info = ensemble_info['residual_boost_model_info']
        
        # Get device
        device = model_state.get('device')
        if device is None:
            import torch
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create session
        session = stream_manager.create_session(
            ensemble_info=ensemble_info,
            stage1_model_info=stage1_model_info,
            residual_boost_model_info=residual_boost_model_info,
            config=config,
            device=device
        )
        session_id = session.session_id
        
        # Send config acknowledgment
        await websocket.send_json({
            'type': 'config_ack',
            'status': 'success',
            'message': 'Configuration applied',
            'session_id': session_id,
            'ensemble_info': {
                'ensemble_name': ensemble_name,
                'num_signals': len(session.target_signals),
                'signals_using_boost': sum(session.boost_signals_mask)
            }
        })
        
        # Message loop
        while True:
            try:
                message_data = await websocket.receive_text()
                message = json.loads(message_data)
                msg_type = message.get('type')
                
                if msg_type == 'ping':
                    # Handle ping
                    await websocket.send_json({
                        'type': 'pong',
                        'timestamp': datetime.now().isoformat()
                    })
                
                elif msg_type == 'predict':
                    # Handle single prediction
                    data = message.get('data', {})
                    boundary_signals = data.get('boundary_signals', {})
                    timestamp = data.get('timestamp')
                    
                    # Validate boundary signals
                    missing_signals = set(session.boundary_signals) - set(boundary_signals.keys())
                    if missing_signals:
                        await websocket.send_json({
                            'type': 'error',
                            'error_code': 'MISSING_SIGNALS',
                            'message': f'Missing boundary signals: {missing_signals}',
                            'details': {'missing_signals': list(missing_signals)}
                        })
                        continue
                    
                    # Predict
                    try:
                        predictions, signals_used_boost, latency_ms = session.predict_single(
                            boundary_signals, timestamp
                        )
                        
                        await websocket.send_json({
                            'type': 'prediction',
                            'status': 'success',
                            'data': {
                                'predictions': predictions,
                                'signals_used_boost': signals_used_boost,
                                'latency_ms': latency_ms,
                                'timestamp': timestamp or datetime.now().isoformat()
                            }
                        })
                    except Exception as e:
                        await websocket.send_json({
                            'type': 'error',
                            'error_code': 'PREDICTION_ERROR',
                            'message': f'Prediction failed: {str(e)}'
                        })
                
                elif msg_type == 'predict_batch':
                    # Handle batch prediction
                    data = message.get('data', {})
                    batch = data.get('batch', [])
                    timestamps = data.get('timestamps')
                    
                    if not batch:
                        await websocket.send_json({
                            'type': 'error',
                            'error_code': 'EMPTY_BATCH',
                            'message': 'Batch cannot be empty'
                        })
                        continue
                    
                    # Validate batch size
                    max_batch_size = config.get('batch_size', 100)
                    if len(batch) > max_batch_size:
                        await websocket.send_json({
                            'type': 'error',
                            'error_code': 'BATCH_TOO_LARGE',
                            'message': f'Batch size {len(batch)} exceeds maximum {max_batch_size}'
                        })
                        continue
                    
                    # Validate signals in first sample
                    missing_signals = set(session.boundary_signals) - set(batch[0].keys())
                    if missing_signals:
                        await websocket.send_json({
                            'type': 'error',
                            'error_code': 'MISSING_SIGNALS',
                            'message': f'Missing boundary signals in batch: {missing_signals}'
                        })
                        continue
                    
                    # Predict
                    try:
                        predictions, count, latency_ms = session.predict_batch(batch, timestamps)
                        
                        await websocket.send_json({
                            'type': 'prediction_batch',
                            'status': 'success',
                            'data': {
                                'predictions': predictions,
                                'count': count,
                                'latency_ms': latency_ms
                            }
                        })
                    except Exception as e:
                        await websocket.send_json({
                            'type': 'error',
                            'error_code': 'PREDICTION_ERROR',
                            'message': f'Batch prediction failed: {str(e)}'
                        })
                
                else:
                    await websocket.send_json({
                        'type': 'error',
                        'error_code': 'UNKNOWN_MESSAGE_TYPE',
                        'message': f'Unknown message type: {msg_type}'
                    })
            
            except json.JSONDecodeError:
                await websocket.send_json({
                    'type': 'error',
                    'error_code': 'INVALID_JSON',
                    'message': 'Invalid JSON message'
                })
            except Exception as e:
                await websocket.send_json({
                    'type': 'error',
                    'error_code': 'INTERNAL_ERROR',
                    'message': f'Internal error: {str(e)}'
                })
    
    except WebSocketDisconnect:
        print(f"🔌 WebSocket disconnected: {session_id}")
    
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
    
    finally:
        # Clean up session
        if session_id:
            stream_manager.remove_session(session_id)


@router.get("/stream/stats", response_model=StreamStatsResponse)
async def get_stream_stats():
    """
    Get streaming statistics
    
    Returns statistics about active WebSocket connections and predictions
    """
    stats = stream_manager.get_stats()
    
    return StreamStatsResponse(
        active_connections=stats['active_connections'],
        total_predictions=stats['total_predictions'],
        average_latency_ms=stats['average_latency_ms'],
        connections=[ConnectionStats(**conn) for conn in stats['connections']]
    )


@router.post("/stream/save", response_model=StreamSaveResponse)
async def save_stream_history(request: StreamSaveRequest):
    """
    Save streaming history to file
    
    Saves the prediction history of a streaming session to a CSV or JSON file
    """
    try:
        session = stream_manager.get_session(request.session_id)
        
        if not session:
            raise HTTPException(
                status_code=404,
                detail=f"Session '{request.session_id}' not found or already closed"
            )
        
        # Get history
        history = session.get_history()
        
        if not history:
            return StreamSaveResponse(
                status='success',
                message='No history to save',
                samples_saved=0
            )
        
        # Create output directory
        os.makedirs(request.output_dir, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"stream_history_{request.session_id}_{timestamp}.{request.format}"
        output_path = os.path.join(request.output_dir, filename)
        
        # Save to file
        if request.format == 'csv':
            # Flatten history for CSV
            rows = []
            for item in history:
                row = {
                    'timestamp': item['timestamp'],
                    'latency_ms': item['latency_ms']
                }
                # Add boundary signals
                for sig, val in item['boundary_data'].items():
                    row[f'boundary_{sig}'] = val
                # Add predictions
                for sig, val in item['predictions'].items():
                    row[f'prediction_{sig}'] = val
                rows.append(row)
            
            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False)
        
        elif request.format == 'json':
            import json
            with open(output_path, 'w') as f:
                json.dump(history, f, indent=2)
        
        print(f"💾 Saved streaming history: {output_path}")
        
        return StreamSaveResponse(
            status='success',
            message=f'History saved successfully',
            output_path=output_path,
            samples_saved=len(history)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save history: {str(e)}")
