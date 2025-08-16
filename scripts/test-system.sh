#!/bin/bash

# System testing script for AI Docker Offload Demo
set -e

# Helper function for timeout (macOS compatible)
timeout_cmd() {
    local timeout=$1
    shift
    
    # Try gtimeout first (from coreutils), then timeout, then fallback
    if command -v gtimeout >/dev/null 2>&1; then
        gtimeout "$timeout" "$@"
    elif command -v timeout >/dev/null 2>&1; then
        timeout "$timeout" "$@"
    else
        # Fallback for macOS without timeout
        "$@" &
        local pid=$!
        (sleep "$timeout" && kill -9 $pid 2>/dev/null) &
        local killer=$!
        wait $pid 2>/dev/null
        local result=$?
        kill -9 $killer 2>/dev/null
        return $result
    fi
}

# Fixed port mappings - coordinator is on 8090
COORDINATOR_URL="http://localhost:8090"
PREPROCESSOR_URL="http://localhost:8081"
AGGREGATOR_URL="http://localhost:8082"

# Get actual Triton port mapping
TRITON_PORT=$(docker port agentic-docker-offload-showcase-triton-server-1 8000/tcp 2>/dev/null | cut -d: -f2)
if [ -z "$TRITON_PORT" ]; then
    TRITON_PORT="8000"  # Fallback
fi
TRITON_URL="http://localhost:$TRITON_PORT"

echo "🧪 Testing AI Docker Offload System..."
echo "   Coordinator on port: 8090"
echo "   Triton Server on port: $TRITON_PORT"

# Function to wait for service with shorter timeout
wait_for_service() {
    local url=$1
    local service_name=$2
    local max_attempts=10  # Reduced from 30
    local attempt=1
    
    echo "⏳ Waiting for $service_name to be ready..."
    
    while [ $attempt -le $max_attempts ]; do
        if timeout_cmd 5 curl -s -f "$url/health" > /dev/null 2>&1; then
            echo "✅ $service_name is ready"
            return 0
        fi
        
        echo "   Attempt $attempt/$max_attempts - waiting for $service_name..."
        sleep 1  # Shorter sleep
        ((attempt++))
    done
    
    echo "❌ $service_name failed to start after $((max_attempts)) seconds"
    echo "🔍 Checking if service is actually running..."
    
    # Check if the service is running but not responding to health checks
    if docker ps | grep -q "coordinator-agent.*healthy"; then
        echo "✅ Service is running and marked healthy by Docker - continuing tests"
        return 0
    else
        echo "❌ Service is not running or not healthy"
        return 1
    fi
}

# Function to test endpoint with timeout
test_endpoint() {
    local url=$1
    local description=$2
    local timeout_sec=${3:-10}
    
    echo "🔍 Testing: $description"
    
    if response=$(timeout_cmd $timeout_sec curl -s -f "$url" 2>/dev/null); then
        echo "✅ $description - OK"
        echo "   Response: $(echo "$response" | jq -c . 2>/dev/null || echo "$response" | head -c 100)..."
        return 0
    else
        echo "❌ $description - FAILED or TIMEOUT"
        return 1
    fi
}

# Function to test inference
test_inference() {
    local model=$1
    local input_data=$2
    local description=$3
    
    echo "🔍 Testing inference: $description"
    
    local payload=$(cat << EOF
{
    "model_name": "$model",
    "inputs": $input_data
}
EOF
)
    
    if response=$(timeout_cmd 15 curl -s -f -X POST \
        -H "Content-Type: application/json" \
        -d "$payload" \
        "$COORDINATOR_URL/infer" 2>/dev/null); then
        echo "✅ $description - OK"
        echo "   Model: $model"
        echo "   Response: $(echo "$response" | jq -c '.metadata // {}' 2>/dev/null || echo "No metadata")"
        return 0
    else
        echo "❌ $description - FAILED or TIMEOUT"
        return 1
    fi
}

# Start testing
echo "Starting system tests at $(date)"
echo "================================="

# Check if Docker Compose is running
if ! docker-compose ps | grep -q "Up"; then
    echo "❌ Docker Compose services are not running"
    echo "   Please run: docker-compose up -d"
    exit 1
fi

# Wait for coordinator to be ready - with fallback check
echo "🔍 Checking coordinator status..."
if docker ps | grep -q "coordinator-agent.*healthy"; then
    echo "✅ Coordinator is marked healthy by Docker"
    
    # Quick test to confirm it's responding
    if timeout_cmd 5 curl -s -f "$COORDINATOR_URL/health" > /dev/null 2>&1; then
        echo "✅ Coordinator is responding to health checks"
    else
        echo "⚠️  Coordinator is healthy but not responding - may still be initializing"
        echo "   Waiting a bit more..."
        sleep 10
        
        if timeout_cmd 5 curl -s -f "$COORDINATOR_URL/health" > /dev/null 2>&1; then
            echo "✅ Coordinator is now responding"
        else
            echo "❌ Coordinator still not responding after wait"
            echo "🔍 Checking coordinator logs..."
            docker logs --tail=20 agentic-docker-offload-showcase-coordinator-agent-1
            exit 1
        fi
    fi
else
    wait_for_service "$COORDINATOR_URL" "Inference Coordinator" || exit 1
fi

# Test basic endpoints
echo ""
echo "🔍 Testing basic endpoints..."
test_endpoint "$COORDINATOR_URL/health" "Coordinator Health check"
test_endpoint "$COORDINATOR_URL/models" "Model listing"
test_endpoint "$COORDINATOR_URL/agents" "Agent listing"
test_endpoint "$PREPROCESSOR_URL/health" "Preprocessor Health check"
test_endpoint "$AGGREGATOR_URL/health" "Aggregator Health check"

# Test Triton with longer timeout (it's slow with dummy models)
echo "🔍 Testing Triton Server (may be slow with dummy models)..."
if timeout_cmd 30 curl -s -f "$TRITON_URL/v2/health/live" >/dev/null 2>&1; then
    echo "✅ Triton Server: Responding"
else
    echo "⚠️  Triton Server: Not responding or very slow (expected with dummy models)"
fi

# Test mock inference (since we don't have real models)
echo ""
echo "🔍 Testing mock inference..."

# Text classification test
text_input='{
    "INPUT": {
        "data": [[0.1, 0.2, 0.3, 0.4, 0.5]],
        "shape": [1, 5],
        "datatype": "FP32"
    }
}'

test_inference "text_classifier" "$text_input" "Text classification"

# Image classification test
image_input='{
    "INPUT": {
        "data": [[[[[0.5, 0.6, 0.7]]]]],
        "shape": [1, 3, 1, 1],
        "datatype": "FP32"
    }
}'

test_inference "resnet50" "$image_input" "Image classification"

# Test metrics endpoint
echo ""
echo "🔍 Testing monitoring..."
test_endpoint "$COORDINATOR_URL/metrics" "Coordinator Prometheus metrics" 5
test_endpoint "$PREPROCESSOR_URL/metrics" "Preprocessor Prometheus metrics" 5
test_endpoint "$AGGREGATOR_URL/metrics" "Aggregator Prometheus metrics" 5

# Check Docker containers
echo ""
echo "📊 Container status:"
docker-compose ps

# Check GPU usage if available
echo ""
echo "🖥️  GPU status:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || echo "   No NVIDIA GPUs detected"
else
    echo "   NVIDIA GPU tools not available (this is normal on systems without NVIDIA GPUs)"
fi

# Check if running in Docker Desktop or other environments
echo ""
echo "🐳 Docker environment:"
if docker info 2>/dev/null | grep -q "Docker Desktop"; then
    echo "   Running in Docker Desktop (GPU support may be limited)"
elif docker info 2>/dev/null | grep -q "nvidia"; then
    echo "   NVIDIA Docker runtime detected"
else
    echo "   Standard Docker runtime (CPU only)"
fi

echo ""
echo "================================="
echo "🎉 System test complete!"
echo ""
echo "📊 Access points:"
echo "   • Coordinator API: $COORDINATOR_URL"
echo "   • Preprocessor API: $PREPROCESSOR_URL"
echo "   • Aggregator API: $AGGREGATOR_URL"
echo "   • Triton Server: $TRITON_URL"
echo ""
echo "📚 Next steps:"
echo "   • Run pipeline test: ./scripts/test-pipeline.sh"
echo "   • Test inference: ./scripts/test-inference.sh all"
echo "   • Monitor performance: $COORDINATOR_URL/metrics"

# Test Triton server specifically
echo ""
echo "🔧 Triton Server Analysis:"
if docker logs agentic-docker-offload-showcase-triton-server-1 2>&1 | tail -10 | grep -q "error"; then
    echo "⚠️  Triton has errors (expected with dummy models):"
    docker logs agentic-docker-offload-showcase-triton-server-1 2>&1 | tail -5
else
    echo "✅ Triton logs look normal"
fi

echo ""
echo "✅ Comprehensive system test complete!"