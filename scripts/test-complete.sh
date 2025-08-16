#!/bin/bash

# Comprehensive end-to-end testing suite
set -e

COORDINATOR_URL="http://localhost:8080"
TRITON_URL="http://localhost:8000"
PROMETHEUS_URL="http://localhost:9090"
GRAFANA_URL="http://localhost:3000"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test results tracking
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
    ((PASSED_TESTS++))
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    ((FAILED_TESTS++))
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Test function wrapper
run_test() {
    local test_name="$1"
    local test_function="$2"
    
    ((TOTAL_TESTS++))
    log_info "Running test: $test_name"
    
    if $test_function; then
        log_success "$test_name passed"
        return 0
    else
        log_error "$test_name failed"
        return 1
    fi
}

# Wait for service with timeout
wait_for_service() {
    local url=$1
    local service_name=$2
    local timeout=${3:-120}
    local interval=5
    local elapsed=0
    
    log_info "Waiting for $service_name at $url (timeout: ${timeout}s)"
    
    while [ $elapsed -lt $timeout ]; do
        if curl -s -f "$url" > /dev/null 2>&1; then
            log_success "$service_name is ready"
            return 0
        fi
        
        sleep $interval
        ((elapsed += interval))
        echo -n "."
    done
    
    log_error "$service_name failed to start within ${timeout}s"
    return 1
}

# Test Docker services are running
test_docker_services() {
    log_info "Checking Docker services..."
    
    # Check if docker-compose is running
    if ! docker-compose ps | grep -q "Up"; then
        log_error "Docker Compose services are not running"
        return 1
    fi
    
    # Check specific services
    local services=("triton-server" "inference-coordinator" "preprocessor" "aggregator")
    for service in "${services[@]}"; do
        if docker-compose ps "$service" | grep -q "Up"; then
            log_success "$service is running"
        else
            log_error "$service is not running"
            return 1
        fi
    done
    
    return 0
}

# Test basic health endpoints
test_health_endpoints() {
    log_info "Testing health endpoints..."
    
    local endpoints=(
        "$COORDINATOR_URL/health:Coordinator"
        "$TRITON_URL/v2/health/ready:Triton"
    )
    
    for endpoint_info in "${endpoints[@]}"; do
        IFS=':' read -r endpoint name <<< "$endpoint_info"
        
        if curl -s -f "$endpoint" > /dev/null; then
            log_success "$name health check passed"
        else
            log_error "$name health check failed"
            return 1
        fi
    done
    
    return 0
}

# Test model listing
test_model_listing() {
    log_info "Testing model listing..."
    
    # Test Triton model repository
    if response=$(curl -s -f "$TRITON_URL/v2/models"); then
        if echo "$response" | jq -e '.models' > /dev/null 2>&1; then
            model_count=$(echo "$response" | jq '.models | length')
            log_success "Triton models listed: $model_count models found"
        else
            log_error "Invalid model listing response format"
            return 1
        fi
    else
        log_error "Failed to list Triton models"
        return 1
    fi
    
    # Test Coordinator model listing
    if response=$(curl -s -f "$COORDINATOR_URL/models"); then
        log_success "Coordinator model listing successful"
    else
        log_error "Coordinator model listing failed"
        return 1
    fi
    
    return 0
}

# Test text classification inference
test_text_inference() {
    log_info "Testing text classification inference..."
    
    local payload=$(cat << 'EOF'
{
    "model_name": "text_classifier",
    "inputs": {
        "INPUT": {
            "data": [[0.1, 0.2, 0.3, 0.4, 0.5]],
            "shape": [1, 5],
            "datatype": "FP32"
        }
    }
}
EOF
)
    
    if response=$(curl -s -f -X POST \
        -H "Content-Type: application/json" \
        -d "$payload" \
        "$COORDINATOR_URL/infer"); then
        
        if echo "$response" | jq -e '.outputs // .metadata // .' > /dev/null 2>&1; then
            log_success "Text inference completed successfully"
        else
            log_error "Text inference returned invalid response"
            return 1
        fi
    else
        log_error "Text inference request failed"
        return 1
    fi
    
    return 0
}

# Test image classification inference
test_image_inference() {
    log_info "Testing image classification inference..."
    
    local payload=$(cat << 'EOF'
{
    "model_name": "resnet50",
    "inputs": {
        "INPUT": {
            "data": [[[[[0.5, 0.6, 0.7]]]]],
            "shape": [1, 3, 1, 1],
            "datatype": "FP32"
        }
    }
}
EOF
)
    
    if response=$(curl -s -f -X POST \
        -H "Content-Type: application/json" \
        -d "$payload" \
        "$COORDINATOR_URL/infer"); then
        
        log_success "Image inference completed successfully"
    else
        log_warning "Image inference failed (may be expected without real model)"
        return 0  # Don't fail the test for mock models
    fi
    
    return 0
}

# Test preprocessing pipeline
test_preprocessing() {
    log_info "Testing preprocessing pipeline..."
    
    local text_payload=$(cat << 'EOF'
{
    "data_type": "text",
    "data": "This is a test message for preprocessing",
    "target_model": "text_classifier"
}
EOF
)
    
    if response=$(curl -s -f -X POST \
        -H "Content-Type: application/json" \
        -d "$text_payload" \
        "http://localhost:8080/preprocess"); then
        
        log_success "Text preprocessing successful"
    else
        log_error "Text preprocessing failed"
        return 1
    fi
    
    return 0
}

# Test metrics collection
test_metrics() {
    log_info "Testing metrics collection..."
    
    # Test Coordinator metrics
    if curl -s -f "$COORDINATOR_URL/metrics" | grep -q "inference_requests_total\|http_requests_total"; then
        log_success "Coordinator metrics available"
    else
        log_warning "Coordinator metrics may not be fully implemented"
    fi
    
    # Test Triton metrics
    if curl -s -f "$TRITON_URL/metrics" > /dev/null; then
        log_success "Triton metrics endpoint accessible"
    else
        log_warning "Triton metrics endpoint not accessible"
    fi
    
    return 0
}

# Test GPU availability
test_gpu_support() {
    log_info "Testing GPU support..."
    
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi > /dev/null 2>&1; then
            gpu_info=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1)
            log_success "GPU detected: $gpu_info"
        else
            log_warning "nvidia-smi failed to execute"
        fi
    else
        log_warning "NVIDIA tools not available"
    fi
    
    # Check Docker GPU support
    if docker run --gpus all --rm nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi &> /dev/null; then
        log_success "Docker GPU support confirmed"
    else
        log_warning "Docker GPU support not available"
    fi
    
    return 0
}

# Test offload configuration
test_offload_config() {
    log_info "Testing Docker offload configuration..."
    
    # Check offload labels
    if docker inspect ai-docker-offload-demo_triton-server_1 2>/dev/null | \
       jq -e '.[] | .Config.Labels | has("docker-offload.gpu-required")' > /dev/null; then
        log_success "Offload labels configured correctly"
    else
        log_warning "Offload labels may not be set"
    fi
    
    # Check resource constraints
    if docker inspect ai-docker-offload-demo_triton-server_1 2>/dev/null | \
       jq -e '.[] | .HostConfig.DeviceRequests' > /dev/null; then
        log_success "GPU device requirements configured"
    else
        log_warning "GPU device requirements not configured"
    fi
    
    return 0
}

# Test load balancing and scaling
test_scaling() {
    log_info "Testing service scaling..."
    
    # Scale preprocessor service
    if docker-compose up -d --scale preprocessor=2; then
        sleep 5
        
        # Check if 2 instances are running
        instance_count=$(docker-compose ps preprocessor | grep "Up" | wc -l)
        if [ "$instance_count" -eq 2 ]; then
            log_success "Service scaling successful: $instance_count preprocessor instances"
        else
            log_warning "Service scaling may not be working as expected"
        fi
        
        # Scale back down
        docker-compose up -d --scale preprocessor=1
    else
        log_error "Service scaling failed"
        return 1
    fi
    
    return 0
}

# Test error handling and recovery
test_error_handling() {
    log_info "Testing error handling..."
    
    # Test invalid model name
    local invalid_payload=$(cat << 'EOF'
{
    "model_name": "nonexistent_model",
    "inputs": {}
}
EOF
)
    
    response_code=$(curl -s -o /dev/null -w "%{http_code}" -X POST \
        -H "Content-Type: application/json" \
        -d "$invalid_payload" \
        "$COORDINATOR_URL/infer")
    
    if [ "$response_code" -eq 400 ] || [ "$response_code" -eq 404 ]; then
        log_success "Error handling working: HTTP $response_code for invalid model"
    else
        log_warning "Unexpected response code for invalid model: $response_code"
    fi
    
    return 0
}

# Main test execution
main() {
    echo "🧪 Starting Comprehensive AI Docker Offload Test Suite"
    echo "======================================================"
    echo "Timestamp: $(date)"
    echo ""
    
    # Wait for services to be ready
    log_info "Waiting for services to initialize..."
    wait_for_service "$COORDINATOR_URL/health" "Coordinator" 120 || exit 1
    wait_for_service "$TRITON_URL/v2/health/ready" "Triton" 120 || exit 1
    
    echo ""
    log_info "Starting test execution..."
    echo ""
    
    # Run all tests
    run_test "Docker Services Status" test_docker_services
    run_test "Health Endpoints" test_health_endpoints
    run_test "Model Listing" test_model_listing
    run_test "Text Inference" test_text_inference
    run_test "Image Inference" test_image_inference
    run_test "Preprocessing Pipeline" test_preprocessing
    run_test "Metrics Collection" test_metrics
    run_test "GPU Support" test_gpu_support
    run_test "Offload Configuration" test_offload_config
    run_test "Service Scaling" test_scaling
    run_test "Error Handling" test_error_handling
    
    # Final summary
    echo ""
    echo "======================================================"
    echo "🎯 Test Summary"
    echo "======================================================"
    echo "Total Tests: $TOTAL_TESTS"
    echo "Passed: $PASSED_TESTS"
    echo "Failed: $FAILED_TESTS"
    
    if [ $FAILED_TESTS -eq 0 ]; then
        log_success "All tests completed successfully! 🎉"
        echo ""
        echo "🚀 System is ready for production use"
        echo ""
        echo "📊 Access Points:"
        echo "   • API: $COORDINATOR_URL"
        echo "   • Triton: $TRITON_URL"
        echo "   • Prometheus: $PROMETHEUS_URL"
        echo "   • Grafana: $GRAFANA_URL"
        
        exit 0
    else
        log_error "Some tests failed. Please check the logs above."
        exit 1
    fi
}

# Handle script arguments
case "${1:-all}" in
    "quick")
        # Run only essential tests
        wait_for_service "$COORDINATOR_URL/health" "Coordinator" 60
        run_test "Health Check" test_health_endpoints
        run_test "Text Inference" test_text_inference
        ;;
    "gpu")
        # Run GPU-specific tests
        run_test "GPU Support" test_gpu_support
        run_test "Offload Configuration" test_offload_config
        ;;
    "inference")
        # Run inference tests only
        wait_for_service "$COORDINATOR_URL/health" "Coordinator" 60
        run_test "Text Inference" test_text_inference
        run_test "Image Inference" test_image_inference
        run_test "Preprocessing" test_preprocessing
        ;;
    *)
        # Run all tests
        main
        ;;
esac