#!/bin/bash

echo "🧪 Simple GPU Test for Docker Model Runner..."

# Test Docker GPU access
echo "1. Testing Docker GPU access..."
if docker run --gpus all --rm ubuntu nvidia-smi; then
    echo "✅ Docker GPU access works"
else
    echo "❌ Docker GPU access failed"
    exit 1
fi

# Test Triton GPU access
echo ""
echo "2. Testing Triton GPU configuration..."
TRITON_PORT=$(docker port agentic-docker-offload-showcase-triton-server-1 8000/tcp 2>/dev/null | cut -d: -f2)
if [ ! -z "$TRITON_PORT" ]; then
    echo "Triton server running on port: $TRITON_PORT"
    timeout 10 curl -s http://localhost:$TRITON_PORT/v2/health/live || echo "Triton may still be starting"
else
    echo "⚠️  Triton server not found"
fi

# Test Coordinator GPU status
echo ""
echo "3. Testing Coordinator GPU status..."
timeout 10 curl -s http://localhost:8090/gpu-status | jq . || echo "Coordinator not ready yet"

echo ""
echo "✅ GPU test complete!"
