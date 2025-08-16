#!/bin/bash

echo "🚀 Testing Complete AI Pipeline..."

# Step 1: Preprocess text
echo "Step 1: Preprocessing text..."
PREPROCESS_RESULT=$(curl -s -X POST http://localhost:8081/preprocess \
  -H "Content-Type: application/json" \
  -d '{
    "data_type": "text",
    "data": "This is a sample text for AI analysis",
    "target_model": "text_classifier"
  }')

echo "Preprocessing result:"
echo "$PREPROCESS_RESULT" | jq .

# Step 2: Extract preprocessed data and run inference (updated to port 8090)
echo ""
echo "Step 2: Running inference..."
INFERENCE_RESULT=$(curl -s -X POST http://localhost:8090/infer \
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
  }')

echo "Inference result:"
echo "$INFERENCE_RESULT" | jq .

# Step 3: Aggregate results
echo ""
echo "Step 3: Aggregating results..."
AGGREGATION_RESULT=$(curl -s -X POST http://localhost:8082/aggregate \
  -H "Content-Type: application/json" \
  -d "{
    \"results\": [$INFERENCE_RESULT],
    \"metadata\": {\"pipeline_test\": true}
  }")

echo "Aggregation result:"
echo "$AGGREGATION_RESULT" | jq .

echo ""
echo "✅ Pipeline test complete!"
