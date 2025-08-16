#!/bin/bash

# Performance benchmarking script
set -e

COORDINATOR_URL="http://localhost:8080"
DEFAULT_CONCURRENCY=10
DEFAULT_REQUESTS=100

# Function to run benchmark
run_benchmark() {
    local test_name="$1"
    local payload="$2"
    local concurrency="${3:-$DEFAULT_CONCURRENCY}"
    local requests="${4:-$DEFAULT_REQUESTS}"
    
    echo "🚀 Running benchmark: $test_name"
    echo "   Concurrency: $concurrency, Requests: $requests"
    
    # Create temporary payload file
    local payload_file=$(mktemp)
    echo "$payload" > "$payload_file"
    
    # Run Apache Bench if available
    if command -v ab &> /dev/null; then
        echo "   Using Apache Bench (ab)..."
        ab -n "$requests" -c "$concurrency" -p "$payload_file" -T "application/json" "$COORDINATOR_URL/infer" | \
            grep -E "(Requests per second|Time per request|Transfer rate)" || true
    else
        # Fallback to curl-based testing
        echo "   Using curl-based testing..."
        local start_time=$(date +%s.%N)
        
        for ((i=1; i<=requests; i++)); do
            if ((i % 10 == 0)); then
                echo -n "."
            fi
            
            curl -s -X POST \
                -H "Content-Type: application/json" \
                -d "$payload" \
                "$COORDINATOR_URL/infer" > /dev/null &
            
            # Limit concurrent requests
            if ((i % concurrency == 0)); then
                wait
            fi
        done
        wait
        
        local end_time=$(date +%s.%N)
        local duration=$(echo "$end_time - $start_time" | bc -l)
        local rps=$(echo "scale=2; $requests / $duration" | bc -l)
        
        echo ""
        echo "   Completed $requests requests in ${duration}s"
        echo "   Requests per second: $rps"
    fi
    
    rm -f "$payload_file"
    echo ""
}

# Function to get system metrics
get_metrics() {
    echo "📊 Current system metrics:"
    
    # GPU metrics
    if command -v nvidia-smi &> /dev/null; then
        echo "GPU Status:"
        nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | \
            awk '{print "   GPU: " $0}'
    fi
    
    # Container stats
    echo "Container Resource Usage:"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}" | head -10
    
    # Service health
    echo "Service Health:"
    curl -s "$COORDINATOR_URL/health" | jq -r '"   Coordinator: " + .status' 2>/dev/null || echo "   Coordinator: Unknown"
    
    echo ""
}

# Benchmark payloads
TEXT_PAYLOAD='{
    "model_name": "text_classifier",
    "inputs": {
        "INPUT": {
            "data": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]],
            "shape": [1, 10],
            "datatype": "FP32"
        }
    }
}'

IMAGE_PAYLOAD='{
    "model_name": "resnet50",
    "inputs": {
        "INPUT": {
            "data": [[[[[0.5, 0.6, 0.7, 0.8, 0.9]]]]],
            "shape": [1, 3, 1, 1, 5],
            "datatype": "FP32"
        }
    }
}'

echo "🏁 AI Docker Offload Benchmark Suite"
echo "====================================="

# Check if services are ready
if ! curl -s -f "$COORDINATOR_URL/health" > /dev/null; then
    echo "❌ Services not available. Please start with: docker-compose up -d"
    exit 1
fi

# Get initial metrics
get_metrics

case "${1:-all}" in
    "local-gpu")
        echo "🖥️  Benchmarking local GPU performance..."
        run_benchmark "Text Classification (Local GPU)" "$TEXT_PAYLOAD" 5 50
        run_benchmark "Image Classification (Local GPU)" "$IMAGE_PAYLOAD" 3 30
        ;;
    
    "cloud-offload")
        echo "☁️  Benchmarking cloud offload performance..."
        echo "⚠️  Note: Cloud offload requires proper configuration in docker-offload.yml"
        # Set offload environment variable
        export OFFLOAD_ENABLED=true
        run_benchmark "Text Classification (Cloud)" "$TEXT_PAYLOAD" 10 100
        run_benchmark "Image Classification (Cloud)" "$IMAGE_PAYLOAD" 5 50
        ;;
    
    "mixed-workload")
        echo "🔄 Benchmarking mixed workload performance..."
        
        # Concurrent text and image processing
        echo "Starting mixed workload test..."
        
        # Background text processing
        for i in {1..20}; do
            curl -s -X POST \
                -H "Content-Type: application/json" \
                -d "$TEXT_PAYLOAD" \
                "$COORDINATOR_URL/infer" > /dev/null &
        done
        
        # Foreground image processing
        for i in {1..10}; do
            curl -s -X POST \
                -H "Content-Type: application/json" \
                -d "$IMAGE_PAYLOAD" \
                "$COORDINATOR_URL/infer" > /dev/null &
        done
        
        wait
        echo "✅ Mixed workload test completed"
        ;;
    
    "stress")
        echo "💪 Running stress test..."
        run_benchmark "Stress Test - Text" "$TEXT_PAYLOAD" 20 200
        run_benchmark "Stress Test - Image" "$IMAGE_PAYLOAD" 10 100
        ;;
    
    "latency")
        echo "⚡ Testing latency..."
        
        echo "Single request latency test:"
        for i in {1..5}; do
            start_time=$(date +%s.%N)
            curl -s -X POST \
                -H "Content-Type: application/json" \
                -d "$TEXT_PAYLOAD" \
                "$COORDINATOR_URL/infer" > /dev/null
            end_time=$(date +%s.%N)
            latency=$(echo "($end_time - $start_time) * 1000" | bc -l)
            printf "   Request %d: %.2f ms\n" "$i" "$latency"
        done
        ;;
    
    "all"|*)
        echo "Running comprehensive benchmark suite..."
        echo ""
        
        # Basic performance tests
        run_benchmark "Text Classification Baseline" "$TEXT_PAYLOAD" 5 50
        run_benchmark "Image Classification Baseline" "$IMAGE_PAYLOAD" 3 30
        
        # Concurrency tests
        run_benchmark "Text High Concurrency" "$TEXT_PAYLOAD" 15 100
        run_benchmark "Image Medium Concurrency" "$IMAGE_PAYLOAD" 8 50
        
        # Get final metrics
        get_metrics
        ;;
esac

echo "🎉 Benchmark complete!"
echo ""
echo "📊 Recommendations:"
echo "   • Monitor GPU utilization during peak loads"
echo "   • Adjust batch sizes for optimal throughput"
echo "   • Configure auto-scaling policies based on results"
echo "   • Consider offloading strategies for cost optimization"
echo ""
echo "📚 Next steps:"
echo "   • Review metrics: curl $COORDINATOR_URL/metrics"
echo "   • Check logs: docker-compose logs"
echo "   • Monitor dashboard: http://localhost:3000"