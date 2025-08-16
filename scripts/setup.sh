#!/bin/bash

# Setup script for AI Docker Offload Demo
set -e

echo "🚀 Setting up AI Docker Offload Demo..."

# Check prerequisites
echo "📋 Checking prerequisites..."

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker Engine 20.10+"
    exit 1
fi

# Check Docker Compose
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose v2.0+"
    exit 1
fi

# Check NVIDIA Docker support
if ! docker run --gpus all --rm nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi &> /dev/null; then
    echo "⚠️  GPU support not available. Some features will run in CPU mode."
    echo "   To enable GPU support, install NVIDIA Container Toolkit:"
    echo "   https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
else
    echo "✅ GPU support detected"
fi

# Create directories
echo "📁 Creating project structure..."
mkdir -p triton-server/model-repository
mkdir -p test-data
mkdir -p monitoring/{prometheus,grafana/{dashboards,datasources}}
mkdir -p logs
mkdir -p agents/{coordinator,preprocessor,aggregator}

# Create empty model repository for now (no invalid models)
echo "📦 Setting up empty model repository..."
echo "# Triton Model Repository" > triton-server/model-repository/README.md
echo "This directory will contain AI models for inference." >> triton-server/model-repository/README.md
echo "Run ./scripts/download-models.sh to add sample models." >> triton-server/model-repository/README.md

# Create monitoring configuration
cat > monitoring/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'triton'
    static_configs:
      - targets: ['triton-server:8002']
  
  - job_name: 'coordinator'
    static_configs:
      - targets: ['inference-coordinator:8080']
    metrics_path: '/metrics'
EOF

# Create test data
echo "🧪 Creating test data..."
echo "This is a sample text for testing the AI inference pipeline" > test-data/sample.txt

# Download a sample image (create a simple test image)
python3 -c "
import numpy as np
from PIL import Image
import os

# Create a simple test image
img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
image = Image.fromarray(img)
image.save('test-data/sample.jpg')
print('Created sample test image')
" 2>/dev/null || echo "⚠️  Could not create sample image (PIL not available)"

# Create environment file
cat > .env << 'EOF'
# AI Docker Offload Configuration
TRITON_URL=triton-server:8000
PREPROCESSOR_URL=preprocessor:8000
COORDINATOR_URL=inference-coordinator:8090  # Updated to port 8090
AGGREGATOR_URL=aggregator:8000

# GPU Configuration
NVIDIA_VISIBLE_DEVICES=all
GPU_MEMORY_LIMIT=8Gi

# Offload Configuration
OFFLOAD_ENABLED=false
CLOUD_ENDPOINT=

# Logging
LOG_LEVEL=INFO
EOF

# Set permissions
chmod +x scripts/*.sh 2>/dev/null || true

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Download sample models: ./scripts/download-models.sh"
echo "2. Start the system: docker-compose up -d"
echo "3. Wait for services to initialize (2-3 minutes)"
echo "4. Test the system: ./scripts/test-system.sh"
echo "5. Access coordinator API: http://localhost:8090"
echo ""
echo "Note: The system will start without models initially."
echo "Add models using the download script or place your own models in triton-server/model-repository/"