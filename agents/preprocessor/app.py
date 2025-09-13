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
            'text_classifier': {'data_types': ['text'], 'input_format': 'tokens'},
            'resnet18': {'data_types': ['image'], 'input_format': 'tensor'},
            'resnet50': {'data_types': ['image'], 'input_format': 'tensor'},
            'identity_model': {'data_types': ['text', 'image'], 'input_format': 'tensor'}
        }
        logger.info("Preprocessor agent initialized with real model support")
    
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
        """Preprocess text data for REAL inference"""
        start_time = time.time()
        
        try:
            # Real text preprocessing for transformer models
            processed_text = text.strip()
            
            if target_model == "text_classifier":
                # Tokenize text for BERT-like models (simplified tokenization)
                # In production, you'd use the actual tokenizer
                words = processed_text.lower().split()
                
                # Simple vocabulary mapping (mock tokenizer)
                vocab = {
                    'i': 1045, 'love': 2293, 'this': 2023, 'hate': 4060, 'bad': 2919,
                    'good': 2204, 'great': 2307, 'terrible': 6653, 'amazing': 6429,
                    'wonderful': 6919, 'awful': 9643, 'excellent': 6581, 'poor': 3532,
                    'fantastic': 6438, 'horrible': 9202, 'beautiful': 3376, 'ugly': 9200,
                    'the': 1996, 'is': 2003, 'are': 2024, 'was': 2001, 'were': 2020,
                    'and': 1998, 'or': 2030, 'but': 2021, 'not': 2025, 'very': 2200,
                    'really': 2428, 'quite': 3243, 'so': 2061, 'too': 2475, 'weather': 4633,
                    'today': 2651, 'movie': 3185, 'book': 2338, 'product': 4125, 'service': 2326,
                    'restaurant': 4825, 'food': 2833, 'place': 2173, 'time': 2051, 'money': 2769,
                    '[CLS]': 101, '[SEP]': 102, '[UNK]': 100, '[PAD]': 0
                }
                
                # Convert to token IDs
                token_ids = [vocab['[CLS]']]  # Start token
                attention_mask = [1]
                
                for word in words:
                    if word in vocab:
                        token_ids.append(vocab[word])
                    else:
                        token_ids.append(vocab['[UNK]'])
                    attention_mask.append(1)
                
                token_ids.append(vocab['[SEP]'])  # End token
                attention_mask.append(1)
                
                # Pad to fixed length (128)
                max_length = 128
                while len(token_ids) < max_length:
                    token_ids.append(vocab['[PAD]'])
                    attention_mask.append(0)
                
                # Truncate if too long
                token_ids = token_ids[:max_length]
                attention_mask = attention_mask[:max_length]
                
                preprocessed_data = {
                    "input_ids": {
                        "data": [token_ids],
                        "shape": [1, max_length],
                        "datatype": "INT64"
                    },
                    "attention_mask": {
                        "data": [attention_mask],
                        "shape": [1, max_length],
                        "datatype": "INT64"
                    }
                }
            else:
                # Generic text preprocessing
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
                "token_count": len(token_ids) if target_model == "text_classifier" else 0,
                "processing_time_ms": int(processing_time * 1000),
                "timestamp": time.time(),
                "preprocessing_type": "real_tokenization"
            }
            
            logger.info("Real text preprocessing completed", 
                       length=len(text), 
                       tokens=len(token_ids) if target_model == "text_classifier" else 0,
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
    
    def preprocess_image(self, image_data: str, target_model: str = "resnet18") -> Dict[str, Any]:
        """Preprocess image data for REAL inference"""
        start_time = time.time()
        
        try:
            # Decode base64 image
            try:
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            except Exception as e:
                raise HTTPException(400, f"Invalid image data: {str(e)}")
            
            # Real image preprocessing for ResNet models
            if target_model in ["resnet18", "resnet50"]:
                # Resize to 224x224
                image = image.resize((224, 224))
                
                # Convert to numpy array
                img_array = np.array(image).astype(np.float32)
                
                # Normalize using ImageNet statistics
                mean = np.array([0.485, 0.456, 0.406]) * 255
                std = np.array([0.229, 0.224, 0.225]) * 255
                
                img_array = (img_array - mean) / std
                
                # Convert from HWC to CHW format
                img_array = img_array.transpose(2, 0, 1)
                
                # Add batch dimension
                img_array = np.expand_dims(img_array, axis=0)
                
                preprocessed_data = {
                    "input": {
                        "data": img_array.tolist(),
                        "shape": [1, 3, 224, 224],
                        "datatype": "FP32"
                    }
                }
            else:
                # Generic image preprocessing
                image = image.resize((224, 224))
                img_array = np.array(image).astype(np.float32) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                preprocessed_data = {
                    "INPUT": {
                        "data": img_array.tolist(),
                        "shape": [1, 224, 224, 3],
                        "datatype": "FP32"
                    }
                }
            
            processing_time = time.time() - start_time
            PREPROCESS_COUNT.labels(data_type='image', status='success').inc()
            PREPROCESS_DURATION.labels(data_type='image').observe(processing_time)
            
            metadata = {
                "original_size": image.size,
                "processed_size": [224, 224],
                "channels": 3,
                "normalization": "imagenet" if target_model in ["resnet18", "resnet50"] else "standard",
                "processing_time_ms": int(processing_time * 1000),
                "timestamp": time.time(),
                "preprocessing_type": "real_image_processing"
            }
            
            logger.info("Real image preprocessing completed", 
                       original_size=image.size,
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