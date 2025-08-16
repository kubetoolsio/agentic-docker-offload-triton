#!/bin/bash

echo "🚀 Setting up Docker Model Runner GPU support..."

# Check if NVIDIA Docker is installed
if ! docker run --gpus all --rm nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi &> /dev/null; then
    echo "❌ NVIDIA Docker support not found"
    echo "📖 Please install NVIDIA Container Toolkit:"
    echo "   https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    echo ""
    echo "🔧 Quick installation (Ubuntu/Debian):"
    echo "   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
    echo "   curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list"
    echo "   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit"
    echo "   sudo systemctl restart docker"
    exit 1
fi

echo "✅ NVIDIA Docker support detected"

# Create directories for Docker Model Runner
echo "📁 Creating Docker Model Runner directories..."
mkdir -p docker-model-cache/{models,cache,logs}
mkdir -p model-images/{text-classifier,resnet50,identity}

# Set environment variables for Docker Model Runner
cat > .env.docker-model-runner << 'EOF'
# Docker Model Runner Configuration
DOCKER_MODEL_RUNNER_ENABLED=true
DOCKER_MODEL_RUNNER_GPU_MEMORY_FRACTION=0.8
DOCKER_MODEL_RUNNER_CACHE_DIR=./docker-model-cache
OFFLOAD_ENABLED=true

# GPU Configuration
NVIDIA_VISIBLE_DEVICES=all
NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
CUDA_VISIBLE_DEVICES=0

# Model Images
TEXT_CLASSIFIER_IMAGE=huggingface/transformers-pytorch-gpu:latest
RESNET50_IMAGE=pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
IDENTITY_MODEL_IMAGE=python:3.11-slim
EOF

echo "📋 Testing GPU availability..."
if nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
else
    echo "⚠️  nvidia-smi not available - GPU detection may not work"
fi

echo "🐳 Testing Docker GPU access..."
if docker run --gpus all --rm nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi --query-gpu=name --format=csv,noheader &> /dev/null; then
    echo "✅ Docker can access GPU"
else
    echo "❌ Docker cannot access GPU - check NVIDIA Container Toolkit installation"
fi

echo ""
echo "🎉 Docker Model Runner GPU setup complete!"
echo ""
echo "📝 Next steps:"
echo "1. Source environment: source .env.docker-model-runner"
echo "2. Start with GPU offload: OFFLOAD_ENABLED=true docker-compose up -d"
echo "3. Test GPU status: curl http://localhost:8080/gpu-status"
echo "4. Monitor GPU usage: watch nvidia-smi"
