#!/bin/bash

echo "🧪 Testing Docker Model Runner GPU Offload..."

COORDINATOR_URL="http://localhost:8090"  # Updated to port 8090

# Check GPU status
echo "📊 Checking GPU status..."
curl -s "$COORDINATOR_URL/gpu-status" | jq .

echo ""
echo "🔬 Testing GPU-accelerated inference..."

# Test text classification with GPU offload
echo "Testing text classification..."
curl -s -X POST "$COORDINATOR_URL/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "text_classifier",
    "inputs": {
      "INPUT": {
        "data": [[0.1, 0.2, 0.3, 0.4, 0.5]],
        "shape": [1, 5],
        "datatype": "FP32"
      }
    }
  }' | jq .

echo ""
echo "📈 Monitor GPU usage with: watch nvidia-smi"
echo "📊 View detailed status: curl $COORDINATOR_URL/gpu-status | jq ."
