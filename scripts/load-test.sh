#!/bin/bash

# Load testing script for AI inference system
set -e

COORDINATOR_URL="http://localhost:8080"
CONCURRENT_REQUESTS=${1:-10}
TOTAL_REQUESTS=${2:-100}
TEST_DURATION=${3:-60}

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "[INFO] $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to send inference request
send_inference_request() {
    local model_name=$1
    local request_id=$2
    
    local payload=$(cat << EOF
{
    "model_name": "$model_name",
    "inputs": {
        "INPUT": {
            "data": [[$(printf "%.1f," $(seq 0.1 0.1 0.5) | sed 's/,$//')]],
            "shape": [1, 5],
            "datatype": "FP32"
        }
    },
    "request_id": "load_test_$request_id"
}
EOF
)
    
    local start_time=$(date +%s%3N)
    local response_code=$(curl -s -o /dev/null -w "%{http_code}" -X POST \
        -H "Content-Type: application/json" \
        -d "$payload" \
        "$COORDINATOR_URL/infer")
    local end_time=$(date +%s%3N)
    
    local latency=$((end_time - start_time))
    echo "$response_code,$latency,$model_name,$request_id"
}

# Function to run concurrent requests
run_load_test() {
    local model_name=$1
    local num_requests=$2
    local concurrency=$3
    
    log_info "Starting load test: $num_requests requests with $concurrency concurrent connections"
    log_info "Target model: $model_name"
    log_info "Target URL: $COORDINATOR_URL"
    
    # Create results directory
    mkdir -p load_test_results
    local results_file="load_test_results/results_$(date +%Y%m%d_%H%M%S).csv"
    
    # Write CSV header
    echo "response_code,latency_ms,model_name,request_id" > "$results_file"
    
    # Run requests in parallel
    local pids=()
    local request_count=0
    
    while [ $request_count -lt $num_requests ]; do
        # Limit concurrent processes
        while [ ${#pids[@]} -ge $concurrency ]; do
            for i in "${!pids[@]}"; do
                if ! kill -0 "${pids[i]}" 2>/dev/null; then
                    unset "pids[i]"
                fi
            done
            pids=("${pids[@]}")  # Re-index array
            sleep 0.1
        done
        
        # Start new request
        (send_inference_request "$model_name" "$request_count") >> "$results_file" &
        pids+=($!)
        
        ((request_count++))
        
        # Progress indicator
        if [ $((request_count % 10)) -eq 0 ]; then
            echo -n "."
        fi
    done
    
    # Wait for all requests to complete
    for pid in "${pids[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    
    echo ""
    log_success "Load test completed. Results saved to: $results_file"
    
    # Analyze results
    analyze_results "$results_file"
}

# Function to analyze test results
analyze_results() {
    local results_file=$1
    
    log_info "Analyzing results..."
    
    # Count total requests
    local total_requests=$(tail -n +2 "$results_file" | wc -l)
    
    # Count successful requests (2xx status codes)
    local successful_requests=$(tail -n +2 "$results_file" | awk -F',' '$1 >= 200 && $1 < 300' | wc -l)
    
    # Calculate success rate
    local success_rate=$(echo "scale=2; $successful_requests * 100 / $total_requests" | bc -l 2>/dev/null || echo "N/A")
    
    # Calculate latency statistics
    local avg_latency=$(tail -n +2 "$results_file" | awk -F',' 'BEGIN{sum=0; count=0} $1 >= 200 && $1 < 300 {sum+=$2; count++} END{if(count>0) print sum/count; else print 0}')
    local min_latency=$(tail -n +2 "$results_file" | awk -F',' '$1 >= 200 && $1 < 300 {print $2}' | sort -n | head -1)
    local max_latency=$(tail -n +2 "$results_file" | awk -F',' '$1 >= 200 && $1 < 300 {print $2}' | sort -n | tail -1)
    
    # Calculate percentiles
    local p50_latency=$(tail -n +2 "$results_file" | awk -F',' '$1 >= 200 && $1 < 300 {print $2}' | sort -n | awk '{a[NR]=$1} END{print (NR%2==1)?a[(NR+1)/2]:(a[NR/2]+a[NR/2+1])/2}')
    local p95_latency=$(tail -n +2 "$results_file" | awk -F',' '$1 >= 200 && $1 < 300 {print $2}' | sort -n | awk '{a[NR]=$1} END{print a[int(NR*0.95)]}')
    
    # Count error types
    local error_4xx=$(tail -n +2 "$results_file" | awk -F',' '$1 >= 400 && $1 < 500' | wc -l)
    local error_5xx=$(tail -n +2 "$results_file" | awk -F',' '$1 >= 500' | wc -l)
    local timeout_errors=$(tail -n +2 "$results_file" | awk -F',' '$1 == 0 || $1 == 000' | wc -l)
    
    # Display results
    echo ""
    echo "📊 Load Test Results"
    echo "===================="
    echo "Total Requests: $total_requests"
    echo "Successful Requests: $successful_requests"
    echo "Success Rate: $success_rate%"
    echo ""
    echo "🕐 Latency Statistics (successful requests only):"
    echo "Average: ${avg_latency}ms"
    echo "Minimum: ${min_latency}ms"
    echo "Maximum: ${max_latency}ms"
    echo "50th Percentile: ${p50_latency}ms"
    echo "95th Percentile: ${p95_latency}ms"
    echo ""
    echo "❌ Error Statistics:"
    echo "4xx Errors: $error_4xx"
    echo "5xx Errors: $error_5xx"
    echo "Timeout/Connection Errors: $timeout_errors"
    echo ""
    
    # Generate performance report
    cat > "load_test_results/report_$(date +%Y%m%d_%H%M%S).txt" << EOF
AI Docker Offload Load Test Report
Generated: $(date)

Test Configuration:
- Target URL: $COORDINATOR_URL
- Total Requests: $total_requests
- Concurrent Connections: $CONCURRENT_REQUESTS
- Model: text_classifier

Results Summary:
- Success Rate: $success_rate%
- Average Latency: ${avg_latency}ms
- 95th Percentile Latency: ${p95_latency}ms
- Error Rate: $(echo "scale=2; ($error_4xx + $error_5xx + $timeout_errors) * 100 / $total_requests" | bc -l 2>/dev/null || echo "N/A")%

Performance Assessment:
$(if [ "${avg_latency%.*}" -lt 1000 ]; then echo "✅ Good: Average latency under 1 second"; else echo "⚠️  Warning: Average latency over 1 second"; fi)
$(if [ "${p95_latency%.*}" -lt 2000 ]; then echo "✅ Good: 95th percentile latency under 2 seconds"; else echo "⚠️  Warning: 95th percentile latency over 2 seconds"; fi)
$(if [ "${success_rate%.*}" -gt 95 ]; then echo "✅ Excellent: Success rate over 95%"; elif [ "${success_rate%.*}" -gt 90 ]; then echo "✅ Good: Success rate over 90%"; else echo "⚠️  Warning: Success rate below 90%"; fi)

Recommendations:
$(if [ "${avg_latency%.*}" -gt 1000 ]; then echo "- Consider scaling up GPU resources or optimizing model inference"; fi)
$(if [ "$error_5xx" -gt 0 ]; then echo "- Investigate server errors and resource constraints"; fi)
$(if [ "$timeout_errors" -gt 0 ]; then echo "- Check network connectivity and increase timeout values"; fi)
EOF
    
    log_success "Performance report saved to: load_test_results/report_$(date +%Y%m%d_%H%M%S).txt"
}

# Function to test system under sustained load
sustained_load_test() {
    local duration=$1
    local rps=$2  # requests per second
    
    log_info "Starting sustained load test for ${duration}s at ${rps} RPS"
    
    local interval=$(echo "scale=3; 1 / $rps" | bc -l)
    local end_time=$(($(date +%s) + duration))
    local request_count=0
    
    while [ $(date +%s) -lt $end_time ]; do
        (send_inference_request "text_classifier" "$request_count") &
        ((request_count++))
        sleep "$interval"
    done
    
    wait
    log_success "Sustained load test completed: $request_count requests sent"
}

# Main execution
main() {
    echo "🚀 AI Docker Offload Load Testing Suite"
    echo "======================================="
    echo "Timestamp: $(date)"
    echo ""
    
    # Check if system is running
    if ! curl -s -f "$COORDINATOR_URL/health" > /dev/null; then
        log_error "Coordinator service not available at $COORDINATOR_URL"
        log_error "Please ensure the system is running: docker-compose up -d"
        exit 1
    fi
    
    log_success "System is running and accessible"
    echo ""
    
    # Install bc for calculations if not available
    if ! command -v bc &> /dev/null; then
        log_error "bc (calculator) is required for this script"
        log_error "Install with: sudo apt-get install bc (Ubuntu/Debian) or brew install bc (macOS)"
        exit 1
    fi
    
    case "${1:-standard}" in
        "quick")
            log_info "Running quick load test (10 requests, 2 concurrent)"
            run_load_test "text_classifier" 10 2
            ;;
        "standard")
            log_info "Running standard load test ($TOTAL_REQUESTS requests, $CONCURRENT_REQUESTS concurrent)"
            run_load_test "text_classifier" "$TOTAL_REQUESTS" "$CONCURRENT_REQUESTS"
            ;;
        "stress")
            log_info "Running stress test (500 requests, 50 concurrent)"
            run_load_test "text_classifier" 500 50
            ;;
        "sustained")
            sustained_load_test "${TEST_DURATION}" 5  # 5 RPS for specified duration
            ;;
        *)
            echo "Usage: $0 [quick|standard|stress|sustained] [concurrent_requests] [total_requests] [duration]"
            echo ""
            echo "Examples:"
            echo "  $0 quick                    # 10 requests, 2 concurrent"
            echo "  $0 standard                 # 100 requests, 10 concurrent (default)"
            echo "  $0 stress                   # 500 requests, 50 concurrent"
            echo "  $0 sustained                # Sustained load for 60 seconds"
            echo "  $0 standard 20 200          # Custom: 200 requests, 20 concurrent"
            exit 1
            ;;
    esac
}

# Execute main function
main "$@"