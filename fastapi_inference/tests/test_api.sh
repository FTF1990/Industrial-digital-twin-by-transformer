#!/bin/bash
# FastAPI Inference Service - Bash Test Script

BASE_URL="http://localhost:8000"

echo "========================================"
echo "FastAPI Inference Service - API Test"
echo "========================================"

# Check if service is running
echo -e "\n[1] Health Check"
echo "----------------------------------------"
curl -s "${BASE_URL}/api/v1/health" | python3 -m json.tool

# Get API info
echo -e "\n[2] API Info"
echo "----------------------------------------"
curl -s "${BASE_URL}/api/v1/info" | python3 -m json.tool

# List models
echo -e "\n[3] List Models"
echo "----------------------------------------"
curl -s "${BASE_URL}/api/v1/models/list" | python3 -m json.tool

# List ensembles
echo -e "\n[4] List Ensembles"
echo "----------------------------------------"
curl -s "${BASE_URL}/api/v1/ensemble/list" | python3 -m json.tool

echo -e "\n========================================"
echo "Basic connectivity test completed!"
echo "========================================"
echo ""
echo "To load models and create ensembles, use the Python client:"
echo "  python fastapi_inference/tests/demo_api_client.py"
echo ""
echo "Or use curl with your model paths:"
echo "  curl -X POST ${BASE_URL}/api/v1/models/stage1/load \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"inference_config_path\": \"saved_models/your_model_inference.json\"}'"
