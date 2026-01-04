"""
Stream Prediction Engine for WebSocket Inference
"""
import time
import uuid
import numpy as np
import torch
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from collections import defaultdict

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from models.residual_tft import batch_inference
except ImportError:
    print("⚠️  Warning: Could not import from models.residual_tft")


class StreamSession:
    """Streaming session manager"""
    
    def __init__(
        self,
        session_id: str,
        ensemble_info: Dict[str, Any],
        stage1_model_info: Dict[str, Any],
        residual_boost_model_info: Dict[str, Any],
        config: Dict[str, Any],
        device: torch.device
    ):
        self.session_id = session_id
        self.ensemble_info = ensemble_info
        self.stage1_model_info = stage1_model_info
        self.residual_boost_model_info = residual_boost_model_info
        self.config = config
        self.device = device
        
        # Session state
        self.connected_at = datetime.now()
        self.predictions_count = 0
        self.total_latency_ms = 0.0
        self.history = []  # Store prediction history
        
        # Get signal configuration
        self.boundary_signals = ensemble_info['signals']['boundary']
        self.target_signals = ensemble_info['signals']['target']
        self.signal_analysis = ensemble_info['signal_analysis']
        
        # Determine which signals use boost
        self.boost_signals_mask = self._get_boost_mask()
        
    def _get_boost_mask(self) -> List[bool]:
        """Get boolean mask for which signals use Residual Boost"""
        manual_boost = self.config.get('manual_boost_signals')
        
        if manual_boost is not None:
            # Use manual override
            return [manual_boost.get(sig, False) for sig in self.target_signals]
        else:
            # Use ensemble configuration
            return [item['use_boost'] for item in self.signal_analysis]
    
    def predict_single(
        self,
        boundary_data: Dict[str, float],
        timestamp: Optional[str] = None
    ) -> Tuple[Dict[str, float], List[str], float]:
        """
        Predict single sample
        
        Args:
            boundary_data: Boundary signal values
            timestamp: Optional timestamp
            
        Returns:
            predictions: Predicted values
            signals_used_boost: Signals that used Residual Boost
            latency_ms: Inference latency
        """
        start_time = time.time()
        
        # Convert to array
        X = np.array([boundary_data[sig] for sig in self.boundary_signals]).reshape(1, -1)
        
        # Stage1 prediction
        y_pred_stage1 = batch_inference(
            self.stage1_model_info['model'],
            X,
            self.stage1_model_info['scalers']['X'],
            self.stage1_model_info['scalers']['y'],
            self.device,
            batch_size=1,
            model_name="Stage1"
        )
        
        # Residual Boost prediction
        y_residual_pred = batch_inference(
            self.residual_boost_model_info['model'],
            X,
            self.residual_boost_model_info['scalers']['X'],
            self.residual_boost_model_info['scalers']['y'],
            self.device,
            batch_size=1,
            model_name="Residual Boost"
        )
        
        # Apply boost mask
        y_final = y_pred_stage1.copy()
        signals_used_boost = []
        for i, (signal, use_boost) in enumerate(zip(self.target_signals, self.boost_signals_mask)):
            if use_boost:
                y_final[0, i] = y_pred_stage1[0, i] + y_residual_pred[0, i]
                signals_used_boost.append(signal)
        
        # Convert to dict
        predictions = {sig: float(y_final[0, i]) for i, sig in enumerate(self.target_signals)}
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Update stats
        self.predictions_count += 1
        self.total_latency_ms += latency_ms
        
        # Store in history
        if self.config.get('include_metadata', True):
            self.history.append({
                'timestamp': timestamp or datetime.now().isoformat(),
                'boundary_data': boundary_data,
                'predictions': predictions,
                'latency_ms': latency_ms
            })
        
        return predictions, signals_used_boost, latency_ms
    
    def predict_batch(
        self,
        batch_data: List[Dict[str, float]],
        timestamps: Optional[List[str]] = None
    ) -> Tuple[List[Dict[str, float]], int, float]:
        """
        Predict batch of samples
        
        Args:
            batch_data: List of boundary signal dicts
            timestamps: Optional timestamps
            
        Returns:
            predictions: List of predicted values
            count: Number of samples
            latency_ms: Total inference latency
        """
        start_time = time.time()
        
        # Convert to array
        X = np.array([[sample[sig] for sig in self.boundary_signals] for sample in batch_data])
        
        # Stage1 prediction
        y_pred_stage1 = batch_inference(
            self.stage1_model_info['model'],
            X,
            self.stage1_model_info['scalers']['X'],
            self.stage1_model_info['scalers']['y'],
            self.device,
            batch_size=len(batch_data),
            model_name="Stage1"
        )
        
        # Residual Boost prediction
        y_residual_pred = batch_inference(
            self.residual_boost_model_info['model'],
            X,
            self.residual_boost_model_info['scalers']['X'],
            self.residual_boost_model_info['scalers']['y'],
            self.device,
            batch_size=len(batch_data),
            model_name="Residual Boost"
        )
        
        # Apply boost mask
        y_final = y_pred_stage1.copy()
        for i, use_boost in enumerate(self.boost_signals_mask):
            if use_boost:
                y_final[:, i] = y_pred_stage1[:, i] + y_residual_pred[:, i]
        
        # Convert to list of dicts
        predictions = [
            {sig: float(y_final[j, i]) for i, sig in enumerate(self.target_signals)}
            for j in range(len(batch_data))
        ]
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Update stats
        self.predictions_count += len(batch_data)
        self.total_latency_ms += latency_ms
        
        # Store in history
        if self.config.get('include_metadata', True):
            for j, (boundary, pred) in enumerate(zip(batch_data, predictions)):
                ts = timestamps[j] if timestamps else datetime.now().isoformat()
                self.history.append({
                    'timestamp': ts,
                    'boundary_data': boundary,
                    'predictions': pred,
                    'latency_ms': latency_ms / len(batch_data)
                })
        
        return predictions, len(batch_data), latency_ms
    
    def get_average_latency(self) -> float:
        """Get average latency per prediction"""
        if self.predictions_count == 0:
            return 0.0
        return self.total_latency_ms / self.predictions_count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get session statistics"""
        return {
            'session_id': self.session_id,
            'ensemble_name': self.ensemble_info['ensemble_name'],
            'connected_at': self.connected_at.isoformat(),
            'predictions_count': self.predictions_count,
            'mode': self.config.get('mode', 'single'),
            'average_latency_ms': self.get_average_latency()
        }
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get prediction history"""
        return self.history


class StreamManager:
    """Global manager for all streaming sessions"""
    
    def __init__(self):
        self.sessions: Dict[str, StreamSession] = {}
        self.total_predictions = 0
        self.total_latency_ms = 0.0
        
    def create_session(
        self,
        ensemble_info: Dict[str, Any],
        stage1_model_info: Dict[str, Any],
        residual_boost_model_info: Dict[str, Any],
        config: Dict[str, Any],
        device: torch.device
    ) -> StreamSession:
        """Create a new streaming session"""
        session_id = f"session_{uuid.uuid4().hex[:12]}"
        
        session = StreamSession(
            session_id=session_id,
            ensemble_info=ensemble_info,
            stage1_model_info=stage1_model_info,
            residual_boost_model_info=residual_boost_model_info,
            config=config,
            device=device
        )
        
        self.sessions[session_id] = session
        print(f"✅ Created streaming session: {session_id}")
        
        return session
    
    def get_session(self, session_id: str) -> Optional[StreamSession]:
        """Get session by ID"""
        return self.sessions.get(session_id)
    
    def remove_session(self, session_id: str):
        """Remove a session"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            self.total_predictions += session.predictions_count
            self.total_latency_ms += session.total_latency_ms
            del self.sessions[session_id]
            print(f"🗑️  Removed streaming session: {session_id}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get global statistics"""
        active_count = len(self.sessions)
        total_preds = self.total_predictions + sum(s.predictions_count for s in self.sessions.values())
        total_lat = self.total_latency_ms + sum(s.total_latency_ms for s in self.sessions.values())
        
        avg_latency = total_lat / total_preds if total_preds > 0 else 0.0
        
        return {
            'active_connections': active_count,
            'total_predictions': total_preds,
            'average_latency_ms': avg_latency,
            'connections': [s.get_stats() for s in self.sessions.values()]
        }


# Global stream manager instance
stream_manager = StreamManager()
