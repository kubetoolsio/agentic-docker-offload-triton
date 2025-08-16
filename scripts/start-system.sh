#!/bin/bash

echo "🚀 Starting AI Docker Offload System..."

# Helper function for timeout (macOS compatible)
timeout_cmd() {
    local timeout=$1
    shift
    
    if command -v gtimeout >/dev/null 2>&1; then
        gtimeout "$timeout" "$@"
    elif command -v timeout >/dev/null 2>&1; then
        timeout "$timeout" "$@"
    else
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

# Check GPU compatibility
chmod +x scripts/check-gpu-compatibility.sh
./scripts/check-gpu-compatibility.sh

# Source GPU configuration
if [ -f .env.gpu ]; then
    source .env.gpu
    echo "📱 Using $GPU_MODE mode"
    echo "   NVIDIA_VISIBLE_DEVICES: $NVIDIA_VISIBLE_DEVICES"
    echo "   OFFLOAD_ENABLED: $OFFLOAD_ENABLED"
fi

# Stop any existing containers first
echo "🛑 Stopping existing containers..."
docker-compose down 2>/dev/null || true

# Start services based on GPU availability
if [ "$GPU_MODE" = "gpu" ]; then
    echo "🎮 Starting with GPU support..."
    echo "   Using NVIDIA L4 GPU"
    # Use override file for GPU with environment variables
    NVIDIA_VISIBLE_DEVICES=all OFFLOAD_ENABLED=true docker-compose -f docker-compose.yml -f docker-compose.override.yml up -d
else
    echo "🖥️  Starting in CPU mode..."
    # Use base configuration only
    NVIDIA_VISIBLE_DEVICES=none OFFLOAD_ENABLED=false docker-compose up -d
fi

echo "⏳ Waiting for services to initialize..."
sleep 15

# Check status
echo "📊 Service status:"
docker-compose ps

# Quick health check
echo ""
echo "🔍 Quick health check..."
timeout_cmd 10 curl -s http://localhost:8090/health 2>/dev/null || echo "Coordinator still starting..."

echo ""
echo "✅ System started!"
echo "🔗 Coordinator API: http://localhost:8090"
echo "🧪 Run tests: ./scripts/test-system.sh"

# Test GPU status if available
if [ "$GPU_MODE" = "gpu" ]; then
    echo ""
    echo "🔬 Testing GPU functionality..."
    sleep 5
    timeout_cmd 10 curl -s http://localhost:8090/gpu-status 2>/dev/null || echo "GPU status check will be available once coordinator is ready"
fi
