#!/usr/bin/env python3
"""
FastAPI Inference Service - Python Client Demo

This script demonstrates how to use the FastAPI inference service.
"""
import requests
import json
import time
from typing import Dict, Any


class InferenceAPIClient:
    """Python client for FastAPI inference service"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()

    def health_check(self) -> Dict[str, Any]:
        """Check service health"""
        response = self.session.get(f"{self.base_url}/api/v1/health")
        response.raise_for_status()
        return response.json()

    def load_stage1_model(self, inference_config_path: str, model_name: str = None) -> Dict[str, Any]:
        """Load Stage1 SST model"""
        payload = {"inference_config_path": inference_config_path}
        if model_name:
            payload["model_name"] = model_name

        response = self.session.post(
            f"{self.base_url}/api/v1/models/stage1/load",
            json=payload
        )
        response.raise_for_status()
        return response.json()

    def load_residual_boost_model(self, inference_config_path: str, model_name: str = None) -> Dict[str, Any]:
        """Load Residual Boost (TFT) model"""
        payload = {"inference_config_path": inference_config_path}
        if model_name:
            payload["model_name"] = model_name

        response = self.session.post(
            f"{self.base_url}/api/v1/models/residual-boost/load",
            json=payload
        )
        response.raise_for_status()
        return response.json()

    def list_models(self) -> Dict[str, Any]:
        """List all loaded models"""
        response = self.session.get(f"{self.base_url}/api/v1/models/list")
        response.raise_for_status()
        return response.json()

    def create_ensemble(
        self,
        stage1_model_name: str,
        residual_boost_model_name: str,
        evaluation_data_path: str,
        delta_r2_threshold: float = 0.05,
        ensemble_name: str = None
    ) -> Dict[str, Any]:
        """Create ensemble model"""
        payload = {
            "stage1_model_name": stage1_model_name,
            "residual_boost_model_name": residual_boost_model_name,
            "evaluation_data_path": evaluation_data_path,
            "delta_r2_threshold": delta_r2_threshold,
            "save_config": True
        }
        if ensemble_name:
            payload["ensemble_name"] = ensemble_name

        response = self.session.post(
            f"{self.base_url}/api/v1/ensemble/create",
            json=payload
        )
        response.raise_for_status()
        return response.json()

    def update_ensemble_threshold(self, ensemble_name: str, new_threshold: float) -> Dict[str, Any]:
        """Update ensemble Delta R² threshold"""
        response = self.session.post(
            f"{self.base_url}/api/v1/ensemble/{ensemble_name}/update-threshold",
            json={"new_threshold": new_threshold}
        )
        response.raise_for_status()
        return response.json()

    def list_ensembles(self) -> list:
        """List all ensemble models"""
        response = self.session.get(f"{self.base_url}/api/v1/ensemble/list")
        response.raise_for_status()
        return response.json()

    def get_ensemble_info(self, ensemble_name: str) -> Dict[str, Any]:
        """Get ensemble model information"""
        response = self.session.get(f"{self.base_url}/api/v1/ensemble/{ensemble_name}/info")
        response.raise_for_status()
        return response.json()

    def batch_inference(
        self,
        ensemble_name: str,
        input_data_path: str,
        output_dir: str,
        manual_boost_signals: Dict[str, bool] = None,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """Perform batch inference"""
        payload = {
            "ensemble_name": ensemble_name,
            "input_data_path": input_data_path,
            "output_dir": output_dir,
            "include_metadata": include_metadata
        }
        if manual_boost_signals:
            payload["manual_boost_signals"] = manual_boost_signals

        response = self.session.post(
            f"{self.base_url}/api/v1/inference/batch",
            json=payload
        )
        response.raise_for_status()
        return response.json()


def main():
    """Demo usage of the API client"""
    print("=" * 80)
    print("FastAPI Inference Service - Python Client Demo")
    print("=" * 80)

    # Initialize client
    client = InferenceAPIClient("http://localhost:8000")

    # Step 1: Health check
    print("\n[Step 1] Health Check")
    print("-" * 80)
    try:
        health = client.health_check()
        print(f"✅ Service Status: {health['status']}")
        print(f"   GPU Available: {health['gpu_available']}")
        print(f"   Device: {health['device']}")
    except Exception as e:
        print(f"❌ Service not available: {e}")
        print("   Please start the service first:")
        print("   python -m fastapi_inference.main")
        return

    # Step 2: Load models (REPLACE WITH YOUR MODEL PATHS)
    print("\n[Step 2] Load Models")
    print("-" * 80)
    print("⚠️  Please update the model paths below with your actual model paths:")
    print()

    # Example paths - UPDATE THESE
    stage1_config = "saved_models/your_sst_model_inference.json"
    residual_boost_config = "saved_models/tft_models/your_tft_inference.json"
    evaluation_data = "data/your_evaluation_data.csv"
    inference_data = "data/your_inference_data.csv"

    print(f"Stage1 config: {stage1_config}")
    print(f"Residual Boost config: {residual_boost_config}")
    print(f"Evaluation data: {evaluation_data}")
    print(f"Inference data: {inference_data}")
    print()
    print("Example usage:")
    print()

    # Example API calls (commented out - uncomment and modify when ready)
    example_code = """
    # Load Stage1 model
    stage1_info = client.load_stage1_model(stage1_config)
    print(f"✅ Stage1 loaded: {stage1_info['model_name']}")
    
    # Load Residual Boost model
    rb_info = client.load_residual_boost_model(residual_boost_config)
    print(f"✅ Residual Boost loaded: {rb_info['model_name']}")
    
    # List loaded models
    models = client.list_models()
    print(f"📋 Loaded models:")
    print(f"   Stage1: {models['stage1_models']}")
    print(f"   Residual Boost: {models['residual_boost_models']}")
    
    # Create ensemble
    ensemble_info = client.create_ensemble(
        stage1_model_name=stage1_info['model_name'],
        residual_boost_model_name=rb_info['model_name'],
        evaluation_data_path=evaluation_data,
        delta_r2_threshold=0.05
    )
    ensemble_name = ensemble_info['ensemble_name']
    print(f"✅ Ensemble created: {ensemble_name}")
    print(f"   Signals using Residual Boost: {ensemble_info['num_use_boost']}/{ensemble_info['num_use_boost'] + ensemble_info['num_use_stage1_only']}")
    
    # Batch inference
    result = client.batch_inference(
        ensemble_name=ensemble_name,
        input_data_path=inference_data,
        output_dir="fastapi_inference/results"
    )
    print(f"✅ Inference completed!")
    print(f"   Output: {result['output_path']}")
    print(f"   Samples: {result['num_samples']}")
    print(f"   Signals used Residual Boost: {result['num_signals_used_boost']}/{result['num_signals']}")
    """
    
    print(example_code)
    
    print("\n" + "=" * 80)
    print("To run with your models, update the paths above and uncomment the code.")
    print("=" * 80)


if __name__ == "__main__":
    main()
