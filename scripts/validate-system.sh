#!/bin/bash

# System validation script - checks prerequisites and configuration
set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check Docker installation
check_docker() {
    log_info "Checking Docker installation..."
    
    if command -v docker &> /dev/null; then
        docker_version=$(docker --version | cut -d' ' -f3 | cut -d',' -f1)
        log_success "Docker $docker_version installed"
        
        # Check if Docker daemon is running
        if docker info &> /dev/null; then
            log_success "Docker daemon is running"
        else
            log_error "Docker daemon is not running"
            return 1
        fi
    else
        log_error "Docker is not installed"
        return 1
    fi
}

# Check Docker Compose installation
check_docker_compose() {
    log_info "Checking Docker Compose installation..."
    
    if command -v docker-compose &> /dev/null; then
        compose_version=$(docker-compose --version | cut -d' ' -f3 | cut -d',' -f1)
        log_success "Docker Compose $compose_version installed"
    elif docker compose version &> /dev/null; then
        compose_version=$(docker compose version --short)
        log_success "Docker Compose (plugin) $compose_version installed"
    else
        log_error "Docker Compose is not installed"
        return 1
    fi
}

# Check NVIDIA Docker support
check_nvidia_docker() {
    log_info "Checking NVIDIA Docker support..."
    
    if command -v nvidia-smi &> /dev/null; then
        log_success "NVIDIA drivers detected"
        
        # Check NVIDIA Container Runtime
        if docker run --gpus all --rm nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi &> /dev/null; then
            log_success "NVIDIA Container Toolkit is working"
        else
            log_warning "NVIDIA Container Toolkit may not be properly configured"
            log_info "Install with: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
        fi
    else
        log_warning "NVIDIA drivers not detected"
    fi
}

# Check system resources
check_system_resources() {
    log_info "Checking system resources..."
    
    # Check available memory
    available_mem=$(free -g | awk '/^Mem:/{print $7}')
    if [ "$available_mem" -ge 4 ]; then
        log_success "Available memory: ${available_mem}GB (sufficient)"
    else
        log_warning "Available memory: ${available_mem}GB (may be insufficient for AI workloads)"
    fi
    
    # Check disk space
    available_disk=$(df -BG / | awk 'NR==2{print $4}' | sed 's/G//')
    if [ "$available_disk" -ge 20 ]; then
        log_success "Available disk space: ${available_disk}GB (sufficient)"
    else
        log_warning "Available disk space: ${available_disk}GB (may be insufficient)"
    fi
    
    # Check CPU cores
    cpu_cores=$(nproc)
    if [ "$cpu_cores" -ge 4 ]; then
        log_success "CPU cores: $cpu_cores (sufficient)"
    else
        log_warning "CPU cores: $cpu_cores (may be insufficient for optimal performance)"
    fi
}

# Check required files
check_required_files() {
    log_info "Checking required files..."
    
    local required_files=(
        "docker-compose.yml"
        "docker-offload.yml"
        "agents/coordinator/Dockerfile"
        "agents/preprocessor/Dockerfile"
        "agents/aggregator/Dockerfile"
        "scripts/setup.sh"
        "scripts/download-models.sh"
    )
    
    for file in "${required_files[@]}"; do
        if [ -f "$file" ]; then
            log_success "$file exists"
        else
            log_error "$file is missing"
            return 1
        fi
    done
}

# Check network ports
check_network_ports() {
    log_info "Checking network port availability..."
    
    local ports=(8000 8001 8002 8080 9090 3000)
    
    for port in "${ports[@]}"; do
        if netstat -tuln 2>/dev/null | grep -q ":$port "; then
            log_warning "Port $port is already in use"
        else
            log_success "Port $port is available"
        fi
    done
}

# Validate Docker Compose configuration
validate_docker_compose() {
    log_info "Validating Docker Compose configuration..."
    
    if docker-compose config &> /dev/null; then
        log_success "docker-compose.yml is valid"
    else
        log_error "docker-compose.yml has configuration errors"
        return 1
    fi
    
    if [ -f "docker-offload.yml" ]; then
        if docker-compose -f docker-offload.yml config &> /dev/null; then
            log_success "docker-offload.yml is valid"
        else
            log_error "docker-offload.yml has configuration errors"
            return 1
        fi
    fi
}

# Check Python dependencies for model download
check_python_deps() {
    log_info "Checking Python dependencies..."
    
    if command -v python3 &> /dev/null; then
        log_success "Python 3 is available"
        
        # Check for PIL/Pillow for image processing
        if python3 -c "from PIL import Image" &> /dev/null; then
            log_success "PIL/Pillow is available for image processing"
        else
            log_warning "PIL/Pillow not available (may affect image processing tests)"
        fi
        
        # Check for numpy
        if python3 -c "import numpy" &> /dev/null; then
            log_success "NumPy is available"
        else
            log_warning "NumPy not available (may affect model testing)"
        fi
    else
        log_warning "Python 3 not available (may affect some testing features)"
    fi
}

# Main validation function
main() {
    echo "🔍 AI Docker Offload System Validation"
    echo "======================================"
    echo "Timestamp: $(date)"
    echo ""
    
    local validation_failed=false
    
    # Run all checks
    check_docker || validation_failed=true
    echo ""
    
    check_docker_compose || validation_failed=true
    echo ""
    
    check_nvidia_docker
    echo ""
    
    check_system_resources
    echo ""
    
    check_required_files || validation_failed=true
    echo ""
    
    check_network_ports
    echo ""
    
    validate_docker_compose || validation_failed=true
    echo ""
    
    check_python_deps
    echo ""
    
    # Final summary
    echo "======================================"
    if [ "$validation_failed" = true ]; then
        log_error "System validation failed! Please address the errors above."
        echo ""
        echo "📚 Helpful resources:"
        echo "   • Docker installation: https://docs.docker.com/get-docker/"
        echo "   • Docker Compose: https://docs.docker.com/compose/install/"
        echo "   • NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/"
        exit 1
    else
        log_success "System validation passed! ✅"
        echo ""
        echo "🚀 Your system is ready to run the AI Docker Offload demo"
        echo ""
        echo "Next steps:"
        echo "1. Run setup: ./scripts/setup.sh"
        echo "2. Download models: ./scripts/download-models.sh"
        echo "3. Start system: docker-compose up -d"
        echo "4. Run tests: ./scripts/test-complete.sh"
        exit 0
    fi
}

# Handle command line arguments
case "${1:-full}" in
    "docker")
        check_docker && check_docker_compose
        ;;
    "gpu")
        check_nvidia_docker
        ;;
    "resources")
        check_system_resources
        ;;
    "files")
        check_required_files
        ;;
    *)
        main
        ;;
esac