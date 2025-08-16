import os
import time
import logging
from typing import Dict, Any, List
import asyncio

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import structlog
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = structlog.get_logger()

# Metrics
AGGREGATION_COUNT = Counter('aggregation_requests_total', 'Total aggregation requests', ['status'])
AGGREGATION_DURATION = Histogram('aggregation_duration_seconds', 'Aggregation duration')

app = FastAPI(title="Results Aggregator Agent", version="1.0.0")

class AggregationRequest(BaseModel):
    results: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class AggregationResponse(BaseModel):
    aggregated_result: Dict[str, Any]
    metadata: Dict[str, Any]

class AggregatorAgent:
    def __init__(self, coordinator_url: str):
        self.coordinator_url = coordinator_url
        self.start_time = time.time()
        logger.info("Aggregator agent initialized", coordinator_url=coordinator_url)
    
    async def aggregate_results(self, request: AggregationRequest) -> Dict[str, Any]:
        """Aggregate multiple inference results"""
        start_time = time.time()
        
        try:
            results = request.results
            
            if not results:
                raise HTTPException(400, "No results provided for aggregation")
            
            # Different aggregation strategies based on result type
            if len(results) == 1:
                # Single result - just format
                aggregated = self._format_single_result(results[0])
            else:
                # Multiple results - aggregate
                aggregated = self._aggregate_multiple_results(results)
            
            processing_time = time.time() - start_time
            AGGREGATION_COUNT.labels(status='success').inc()
            AGGREGATION_DURATION.observe(processing_time)
            
            metadata = {
                "aggregation_method": "ensemble" if len(results) > 1 else "single",
                "num_results": len(results),
                "processing_time_ms": int(processing_time * 1000),
                "timestamp": time.time(),
                "agent_id": "aggregator-001"
            }
            
            logger.info("Results aggregated successfully", 
                       num_results=len(results),
                       processing_time=processing_time)
            
            return {
                "aggregated_result": aggregated,
                "metadata": metadata
            }
            
        except Exception as e:
            AGGREGATION_COUNT.labels(status='error').inc()
            logger.error("Aggregation failed", error=str(e))
            raise HTTPException(500, f"Aggregation failed: {str(e)}")
    
    def _format_single_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Format a single result"""
        return {
            "prediction": result.get("outputs", {}),
            "confidence": self._extract_confidence(result),
            "model": result.get("model", "unknown"),
            "source": "single_model"
        }
    
    def _aggregate_multiple_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate multiple results using ensemble methods"""
        # Extract predictions and confidence scores
        predictions = []
        confidences = []
        models = []
        
        for result in results:
            predictions.append(result.get("outputs", {}))
            confidences.append(self._extract_confidence(result))
            models.append(result.get("model", "unknown"))
        
        # Simple ensemble - average confidences, majority vote for classifications
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        # For now, use the result with highest confidence as primary
        best_idx = confidences.index(max(confidences)) if confidences else 0
        primary_prediction = predictions[best_idx]
        
        return {
            "prediction": primary_prediction,
            "confidence": avg_confidence,
            "ensemble_confidence": avg_confidence,
            "individual_confidences": confidences,
            "models": models,
            "source": "ensemble",
            "num_models": len(results)
        }
    
    def _extract_confidence(self, result: Dict[str, Any]) -> float:
        """Extract confidence score from result"""
        try:
            # Try to extract confidence from metadata
            metadata = result.get("metadata", {})
            if "confidence" in metadata:
                return float(metadata["confidence"])
            
            # Try to extract from outputs (for classification tasks)
            outputs = result.get("outputs", {})
            for key, value in outputs.items():
                if isinstance(value, list) and len(value) > 0:
                    if isinstance(value[0], (int, float)):
                        # Assume it's a probability distribution
                        return float(max(value))
            
            # Default confidence
            return 0.5
            
        except Exception:
            return 0.0

# Global agent instance
coordinator_url = f"http://{os.getenv('COORDINATOR_URL', 'coordinator-agent:8080')}"
aggregator = AggregatorAgent(coordinator_url)

@app.post("/aggregate")
async def aggregate_results(request: AggregationRequest):
    """Aggregate multiple inference results"""
    return await aggregator.aggregate_results(request)

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "agent": "aggregator",
        "uptime_seconds": time.time() - aggregator.start_time
    }

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/status")
async def status():
    """Detailed status information"""
    return {
        "agent_type": "aggregator",
        "coordinator_url": aggregator.coordinator_url,
        "uptime_seconds": time.time() - aggregator.start_time,
        "supported_aggregation_methods": ["single", "ensemble", "majority_vote", "confidence_weighted"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)