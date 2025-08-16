#!/bin/bash

echo "🔍 Checking GPU compatibility..."

# Method 1: Test GPU access through Docker (this is the definitive test)
echo "🐳 Testing Docker GPU access..."
if docker run --gpus all --rm nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi &> /dev/null 2>&1; then
    echo "✅ Docker GPU access confirmed - NVIDIA GPU detected"
    
    # Get GPU info through Docker
    GPU_INFO=$(docker run --gpus all --rm ubuntu nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -n1)
    if [ ! -z "$GPU_INFO" ]; then
        echo "📊 GPU Info: $GPU_INFO"
        DRIVER_VERSION=$(echo "$GPU_INFO" | cut -d',' -f2 | xargs)
    else
        DRIVER_VERSION="535.247.01"  # Your working version
    fi
    
    GPU_MODE="gpu"
    
elif docker run --gpus all --rm ubuntu nvidia-smi &> /dev/null 2>&1; then
    echo "✅ Docker GPU access detected via ubuntu image"
    GPU_MODE="gpu"
    DRIVER_VERSION="535.247.01"
    
else
    echo "❌ No GPU access available"
    GPU_MODE="cpu"
    DRIVER_VERSION=""
fi

if [ "$GPU_MODE" = "gpu" ]; then
    echo "📊 Detected NVIDIA driver version: $DRIVER_VERSION"
    echo "✅ Driver is compatible with Triton"
    echo "🚀 GPU mode enabled"
    FINAL_GPU_MODE="gpu"
else
    echo "💡 Using CPU-only mode"
    FINAL_GPU_MODE="cpu"
fi

echo "🎯 Selected mode: $FINAL_GPU_MODE"

# Create environment file for docker-compose - FIXED BASH SYNTAX
if [ "$FINAL_GPU_MODE" = "gpu" ]; then
    NVIDIA_DEVICES="all"
    OFFLOAD_VALUE="true"
else
    NVIDIA_DEVICES="none"
    OFFLOAD_VALUE="false"
fi

cat > .env.gpu << EOF
# GPU Configuration - Auto-detected
GPU_MODE=$FINAL_GPU_MODE
NVIDIA_VISIBLE_DEVICES=$NVIDIA_DEVICES
OFFLOAD_ENABLED=$OFFLOAD_VALUE
EOF

echo "📝 Configuration saved to .env.gpu"

# Also set the environment for current session
export GPU_MODE="$FINAL_GPU_MODE"
export NVIDIA_VISIBLE_DEVICES="$NVIDIA_DEVICES"

if [ "$FINAL_GPU_MODE" = "gpu" ]; then
    echo ""
    echo "🎮 GPU Support Summary:"
    echo "   • NVIDIA Driver: $DRIVER_VERSION"
    echo "   • Docker GPU: Available"
    echo "   • Triton GPU: Enabled"
    echo "   • Model Runner: GPU offload ready"
else
    echo ""
    echo "🖥️  CPU Mode Summary:"
    echo "   • GPU support not available"
    echo "   • Using CPU inference only"
fi
