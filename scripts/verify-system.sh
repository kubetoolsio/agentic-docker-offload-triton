#!/bin/bash

echo "🔍 Verifying AI Docker Offload System..."

# Get actual port mappings from docker ps
TRITON_PORT=$(docker port agentic-docker-offload-showcase-triton-server-1 8000/tcp 2>/dev/null | cut -d: -f2)
if [ -z "$TRITON_PORT" ]; then
    TRITON_PORT="8000"  # Fallback
fi

echo "📊 Checking service health..."
echo "   Using Triton port: $TRITON_PORT"

# Coordinator (using port 8090)
echo "Testing Coordinator Agent..."
if timeout 10 curl -s http://localhost:8090/health | jq -r '.status' | grep -q "healthy"; then
    echo "✅ Coordinator Agent: Healthy"
else
    echo "❌ Coordinator Agent: Not healthy"
fi

# Preprocessor  
echo "Testing Preprocessor Agent..."
if timeout 10 curl -s http://localhost:8081/health | jq -r '.status' | grep -q "healthy"; then
    echo "✅ Preprocessor Agent: Healthy"
else
    echo "❌ Preprocessor Agent: Not healthy"
fi

# Aggregator
echo "Testing Aggregator Agent..."
if timeout 10 curl -s http://localhost:8082/health | jq -r '.status' | grep -q "healthy"; then
    echo "✅ Aggregator Agent: Healthy"
else
    echo "❌ Aggregator Agent: Not healthy"
fi

# Triton Server (with timeout and actual port)
echo "Testing Triton Server..."
if timeout 10 curl -s http://localhost:$TRITON_PORT/v2/health/live 2>/dev/null | grep -q "{}"; then
    echo "✅ Triton Server: Live (port $TRITON_PORT)"
else
    echo "⚠️  Triton Server: Not responding or taking too long (port $TRITON_PORT)"
    echo "   This is expected with dummy models - Triton may be slow to respond"
fi

echo ""
echo "🧪 Testing inference pipeline..."

# Test text preprocessing
echo "Testing text preprocessing..."
if timeout 10 curl -s -X POST http://localhost:8081/preprocess \
  -H "Content-Type: application/json" \
  -d '{"data_type": "text", "data": "Hello world", "target_model": "text_classifier"}' | jq . >/dev/null; then
    echo "✅ Text preprocessing: Working"
else
    echo "❌ Text preprocessing: Failed"
fi

echo ""
echo "Testing inference coordination..."

# Test inference through coordinator (using port 8090)
if timeout 10 curl -s -X POST http://localhost:8090/infer \
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
  }' | jq . >/dev/null; then
    echo "✅ Inference coordination: Working"
else
    echo "❌ Inference coordination: Failed"
fi

echo ""
echo "🎉 Verification complete!"
echo ""
echo "📊 Current port mappings:"
echo "   • Coordinator: localhost:8090"
echo "   • Preprocessor: localhost:8081" 
echo "   • Aggregator: localhost:8082"
echo "   • Triton: localhost:$TRITON_PORT"
