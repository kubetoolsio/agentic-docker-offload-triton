import os
import time
import logging
from typing import Dict, Any, Optional
import base64
import io

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
import structlog
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = structlog.get_logger()

# Metrics
PREPROCESS_COUNT = Counter('preprocessing_requests_total', 'Total preprocessing requests', ['data_type', 'status'])
PREPROCESS_DURATION = Histogram('preprocessing_duration_seconds', 'Preprocessing duration', ['data_type'])

app = FastAPI(title="Preprocessing Agent", version="1.0.0")

class PreprocessRequest(BaseModel):
    data_type: str  # text, image, audio
    data: Any
    target_model: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None

class PreprocessResponse(BaseModel):
    preprocessed_data: Dict[str, Any]
    metadata: Dict[str, Any]
    target_model: str

class PreprocessorAgent:
    def __init__(self):
        self.start_time = time.time()
        self.supported_models = {
            'text_classifier': {'data_types': ['text'], 'input_format': 'string'},
            'resnet50': {'data_types': ['image'], 'input_format': 'tensor'},
            'identity_model': {'data_types': ['text', 'image'], 'input_format': 'tensor'}
        }
        logger.info("Preprocessor agent initialized")
    
    async def preprocess(self, request: PreprocessRequest) -> Dict[str, Any]:
        """Main preprocessing entry point"""
        if request.data_type == "text":
            return self.preprocess_text(request.data, request.target_model or "text_classifier")
        elif request.data_type == "image":
            return self.preprocess_image(request.data, request.target_model or "resnet50")
        elif request.data_type == "audio":
            return self.preprocess_audio(request.data, request.target_model or "speech_to_text")
        else:
            raise HTTPException(400, f"Unsupported data type: {request.data_type}")
    
    def preprocess_text(self, text: str, target_model: str = "text_classifier") -> Dict[str, Any]:
        """Preprocess text data for inference"""
        start_time = time.time()
        
        try:
            # Mock text preprocessing
            processed_text = text.strip().lower()
            
            # Convert to format expected by model
            if target_model == "text_classifier":
                preprocessed_data = {
                    "INPUT_TEXT": {
                        "data": [processed_text],
                        "shape": [1],
                        "datatype": "BYTES"
                    }
                }
            else:
                # Generic text preprocessing - convert to embeddings-like format
                mock_embedding = np.random.rand(1, 512).astype(np.float32)
                preprocessed_data = {
                    "INPUT": {
                        "data": mock_embedding.tolist(),
                        "shape": [1, 512],
                        "datatype": "FP32"
                    }
                }
            
            processing_time = time.time() - start_time
            PREPROCESS_COUNT.labels(data_type='text', status='success').inc()
            PREPROCESS_DURATION.labels(data_type='text').observe(processing_time)
            
            metadata = {
                "original_length": len(text),
                "processed_length": len(processed_text),
                "processing_time_ms": int(processing_time * 1000),
                "timestamp": time.time()
            }
            
            logger.info("Text preprocessing completed", 
                       length=len(text), 
                       processing_time=processing_time)
            
            return {
                "preprocessed_data": preprocessed_data,
                "metadata": metadata,
                "target_model": target_model
            }
            
        except Exception as e:
            PREPROCESS_COUNT.labels(data_type='text', status='error').inc()
            logger.error("Text preprocessing failed", error=str(e))
            raise HTTPException(500, f"Text preprocessing failed: {str(e)}")
    
    def preprocess_image(self, image_data: str, target_model: str = "resnet50") -> Dict[str, Any]:
        """Preprocess image data for inference"""
        start_time = time.time()
        
        try:
            # Decode base64 image
            try:
                image_bytes = base64.b64decode(image_data)
            except Exception:
                # If not base64, assume it's already binary
                image_bytes = image_data.encode() if isinstance(image_data, str) else image_data
            
            # Mock image preprocessing
            if target_model == "resnet50" or target_model == "image_classifier":
                # Standard ImageNet preprocessing
                mock_image = np.random.rand(1, 3, 224, 224).astype(np.float32)
                preprocessed_data = {
                    "INPUT_IMAGE": {
                        "data": mock_image.tolist(),
                        "shape": [1, 3, 224, 224],
                        "datatype": "FP32"
                    }
                }
            else:
                # Generic image preprocessing
                mock_image = np.random.rand(1, 224, 224, 3).astype(np.float32)
                preprocessed_data = {
                    "INPUT": {
                        "data": mock_image.tolist(),
                        "shape": [1, 224, 224, 3],
                        "datatype": "FP32"
                    }
                }
            
            processing_time = time.time() - start_time
            PREPROCESS_COUNT.labels(data_type='image', status='success').inc()
            PREPROCESS_DURATION.labels(data_type='image').observe(processing_time)
            
            metadata = {
                "image_size_bytes": len(image_bytes),
                "target_shape": [224, 224, 3],
                "processing_time_ms": int(processing_time * 1000),
                "timestamp": time.time()
            }
            
            logger.info("Image preprocessing completed", 
                       size_bytes=len(image_bytes),
                       processing_time=processing_time)
            
            return {
                "preprocessed_data": preprocessed_data,
                "metadata": metadata,
                "target_model": target_model
            }
            
        except Exception as e:
            PREPROCESS_COUNT.labels(data_type='image', status='error').inc()
            logger.error("Image preprocessing failed", error=str(e))
            raise HTTPException(500, f"Image preprocessing failed: {str(e)}")
    
    def preprocess_audio(self, audio_data: str, target_model: str = "speech_to_text") -> Dict[str, Any]:
        """Preprocess audio data for inference"""
        start_time = time.time()
        
        try:
            # Mock audio preprocessing
            mock_audio_features = np.random.rand(1, 16000).astype(np.float32)  # 1 second at 16kHz
            
            preprocessed_data = {
                "INPUT_AUDIO": {
                    "data": mock_audio_features.tolist(),
                    "shape": [1, 16000],
                    "datatype": "FP32"
                }
            }
            
            processing_time = time.time() - start_time
            PREPROCESS_COUNT.labels(data_type='audio', status='success').inc()
            PREPROCESS_DURATION.labels(data_type='audio').observe(processing_time)
            
            metadata = {
                "sample_rate": 16000,
                "duration_seconds": 1.0,
                "processing_time_ms": int(processing_time * 1000),
                "timestamp": time.time()
            }
            
            logger.info("Audio preprocessing completed", processing_time=processing_time)
            
            return {
                "preprocessed_data": preprocessed_data,
                "metadata": metadata,
                "target_model": target_model
            }
            
        except Exception as e:
            PREPROCESS_COUNT.labels(data_type='audio', status='error').inc()
            logger.error("Audio preprocessing failed", error=str(e))
            raise HTTPException(500, f"Audio preprocessing failed: {str(e)}")

# Global agent instance
preprocessor = PreprocessorAgent()

@app.post("/preprocess")
async def preprocess_data(request: PreprocessRequest):
    """Preprocess data for inference"""
    return await preprocessor.preprocess(request)

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "agent": "preprocessor",
        "supported_models": list(preprocessor.supported_models.keys())
    }

@app.get("/models")
async def supported_models():
    """List supported models for preprocessing"""
    return {
        "supported_models": list(preprocessor.supported_models.keys())
    }

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)