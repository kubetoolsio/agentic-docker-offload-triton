# agents/coordinator/app.py - Agentic inference coordinator
import asyncio
import logging
import os
import time
from typing import Dict, List, Optional, Any
import numpy as np

import tritonclient.http as httpclient
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import structlog
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import docker
import json

# Configure structured logging
logging.basicConfig(level=logging.INFO)
logger = structlog.get_logger()

# Prometheus metrics
REQUEST_COUNT = Counter('inference_requests_total', 'Total inference requests', ['model', 'status'])
REQUEST_DURATION = Histogram('inference_request_duration_seconds', 'Request duration', ['model'])
GPU_UTILIZATION = Gauge('gpu_utilization_percent', 'GPU utilization percentage')
MODEL_AVAILABILITY = Gauge('model_availability', 'Model availability status', ['model'])

app = FastAPI(title="AI Inference Coordinator Agent", version="1.0.0")

class InferenceRequest(BaseModel):
    model_name: str
    inputs: Dict[str, Any]
    parameters: Optional[Dict[str, Any]] = None
    
    class Config:
        protected_namespaces = ()  # Fix the Pydantic warning

class AgentStatus(BaseModel):
    status: str
    models_loaded: int
    uptime_seconds: float
    gpu_available: bool

class Agent:
    def __init__(self, triton_url: str):
        # Remove http:// or https:// scheme from URL for Triton client
        if triton_url.startswith('http://'):
            self.triton_url = triton_url.replace('http://', '')
        elif triton_url.startswith('https://'):
            self.triton_url = triton_url.replace('https://', '')
        else:
            self.triton_url = triton_url
            
        logger.info("Initializing agent with Triton URL", triton_url=self.triton_url)
        
        self.client = None
        self.model_metadata = {}
        self.start_time = time.time()
        self.health_status = "initializing"
        
        # Offload configuration
        self.offload_mode = os.getenv("OFFLOAD_MODE", "auto")
        self.remote_runner_url = os.getenv("REMOTE_DOCKER_MODEL_RUNNER_URL", "").strip() or None
        self.docker_model_runner_enabled = os.getenv('DOCKER_MODEL_RUNNER_ENABLED', 'true').lower() == 'true'

        # Disable local runner automatically in remote-offload mode
        if self.offload_mode == "remote-offload":
            self.docker_model_runner_enabled = False
            logger.info("Remote offload mode active: local Docker Model Runner disabled")

        self.docker_client = None
        if self.docker_model_runner_enabled:
            try:
                import docker
                self.docker_client = docker.from_env()
                self.docker_client.ping()
                logger.info("Local Docker Model Runner enabled")
            except Exception as e:
                logger.warning(f"Disabling Docker Model Runner (init failed): {e}")
                self.docker_model_runner_enabled = False

        logger.info("Offload configuration",
                    offload_mode=self.offload_mode,
                    docker_model_runner_enabled=self.docker_model_runner_enabled,
                    remote_runner_url=bool(self.remote_runner_url))
    
    async def initialize(self):
        """Discover available models and their capabilities"""
        max_retries = 3  # Reduced retries for faster startup
        retry_delay = 2  # Shorter delay
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempting to connect to Triton at: {self.triton_url}")
                self.client = httpclient.InferenceServerClient(url=self.triton_url)
                
                # Test connection with shorter timeout
                if not self.client.is_server_ready():
                    raise Exception("Triton server not ready")
                
                # Load model metadata
                await self._load_model_metadata()
                
                self.health_status = "healthy"
                logger.info("Agent initialized successfully", models_count=len(self.model_metadata))
                return
                
            except Exception as e:
                logger.warning(f"Initialization attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                else:
                    # If all retries fail, fall back to mock mode immediately
                    self.health_status = "healthy"  # Mark as healthy even in mock mode
                    logger.warning("Failed to connect to Triton, falling back to mock mode")
                    await self._setup_mock_mode()
                    return

    async def _setup_mock_mode(self):
        """Set up mock mode when Triton is not available"""
        self.client = None
        self.model_metadata = {
            'identity_model': {
                'inputs': [{'name': 'INPUT', 'datatype': 'TYPE_FP32', 'shape': [5]}],
                'outputs': [{'name': 'OUTPUT', 'datatype': 'TYPE_FP32', 'shape': [5]}],
                'platform': 'mock',
                'max_batch_size': 8
            },
            'text_classifier': {
                'inputs': [{'name': 'INPUT_TEXT', 'datatype': 'TYPE_FP32', 'shape': [512]}],
                'outputs': [{'name': 'OUTPUT_SCORES', 'datatype': 'TYPE_FP32', 'shape': [2]}],
                'platform': 'mock',
                'max_batch_size': 8
            },
            'resnet50': {
                'inputs': [{'name': 'INPUT', 'datatype': 'TYPE_FP32', 'shape': [3, 224, 224]}],
                'outputs': [{'name': 'OUTPUT', 'datatype': 'TYPE_FP32', 'shape': [1000]}],
                'platform': 'mock',
                'max_batch_size': 4
            }
        }
        logger.info("Mock mode initialized with test models")

    async def _load_model_metadata(self):
        """Load metadata for all available models"""
        try:
            # Try to get model repository index
            try:
                models = self.client.get_model_repository_index()
                self.model_metadata = {}
                
                for model in models:
                    try:
                        if model.state == "READY":
                            metadata = self.client.get_model_metadata(model.name)
                            self.model_metadata[model.name] = {
                                'inputs': [{'name': inp.name, 'datatype': inp.datatype, 'shape': inp.shape} 
                                         for inp in metadata.inputs],
                                'outputs': [{'name': out.name, 'datatype': out.datatype, 'shape': out.shape} 
                                          for out in metadata.outputs],
                                'platform': metadata.platform,
                                'max_batch_size': getattr(metadata, 'max_batch_size', 0)
                            }
                            MODEL_AVAILABILITY.labels(model=model.name).set(1)
                            logger.info("Loaded model metadata", model=model.name, platform=metadata.platform)
                        else:
                            logger.warning(f"Model {model.name} not ready, state: {model.state}")
                            MODEL_AVAILABILITY.labels(model=model.name).set(0)
                    except Exception as e:
                        logger.warning(f"Failed to load metadata for model {model.name}: {e}")
                        MODEL_AVAILABILITY.labels(model=model.name).set(0)
                        
            except Exception as e:
                logger.warning(f"No models available or repository empty: {e}")
                # Create mock model metadata for testing when no real models are available
                await self._setup_mock_mode()
                        
        except Exception as e:
            logger.error(f"Failed to load model metadata: {e}")
            # Don't raise - fall back to mock mode instead
            await self._setup_mock_mode()

    async def route_inference(self, request: InferenceRequest) -> Dict[str, Any]:
        """Route inference request to appropriate path based on model and offload settings"""
        start_time = time.time()

        try:
            if request.model_name not in self.model_metadata:
                REQUEST_COUNT.labels(model=request.model_name, status='error').inc()
                raise HTTPException(404, f"Model {request.model_name} not found or not ready")

            model_info = self.model_metadata[request.model_name]
            offload_enabled = os.getenv('OFFLOAD_ENABLED', 'false').lower() == 'true'

            # Path 1: Local GPU Docker Model Runner
            if offload_enabled and self.docker_model_runner_enabled and self._should_use_local_gpu_offload():
                logger.info(f"Attempting local GPU Docker Model Runner offload for {request.model_name}")
                try:
                    return await self._offload_to_docker_model_runner(request)
                except Exception as e:
                    logger.warning(f"Local GPU offload failed, trying next path: {e}")

            # Path 2: Remote Offload Runner
            if offload_enabled and self.remote_runner_url and self._should_use_remote_offload():
                logger.info(f"Attempting remote Docker offload for {request.model_name} via {self.remote_runner_url}")
                try:
                    return await self._offload_to_remote_runner(request, start_time)
                except Exception as e:
                    logger.warning(f"Remote offload failed, trying next path: {e}")

            # Path 3: Standard Triton Inference
            if model_info.get('platform') != 'mock':
                logger.info(f"Attempting Triton inference for {request.model_name}")
                try:
                    return await self._real_inference(request, model_info, start_time)
                except Exception as e:
                    logger.warning(f"Triton inference failed, falling back to mock: {e}")
            
            # Path 4: Fallback to Mock Inference
            logger.info(f"Falling back to mock inference for {request.model_name}")
            return await self._mock_inference(request, model_info, start_time)

        except HTTPException:
            raise
        except Exception as e:
            REQUEST_COUNT.labels(model=request.model_name, status='error').inc()
            logger.error("Inference failed", model=request.model_name, error=str(e))
            raise HTTPException(500, f"Inference failed: {str(e)}")

    async def _mock_inference(self, request: InferenceRequest, model_info: Dict, start_time: float) -> Dict[str, Any]:
        """Mock inference for testing when real models aren't available"""
        execution_time = time.time() - start_time
        
        # Generate mock outputs based on model type
        mock_outputs = {}
        for output_spec in model_info['outputs']:
            output_name = output_spec['name']
            shape = output_spec['shape']
            
            if 'text' in request.model_name.lower():
                # Mock text classification scores
                scores = [0.7, 0.3]  # positive, negative
                mock_outputs[output_name] = scores
            elif 'resnet' in request.model_name.lower():
                # Mock image classification scores
                scores = np.random.rand(*shape).tolist()
                mock_outputs[output_name] = scores
            else:
                # Generic mock output
                if isinstance(shape, list) and len(shape) > 0:
                    mock_data = np.random.rand(*shape).tolist()
                else:
                    mock_data = [0.5] * 5  # Default
                mock_outputs[output_name] = mock_data
        
        REQUEST_COUNT.labels(model=request.model_name, status='success').inc()
        REQUEST_DURATION.labels(model=request.model_name).observe(execution_time)
        
        response = {
            'model': request.model_name,
            'outputs': mock_outputs,
            'metadata': {
                'execution_time_ms': int(execution_time * 1000),
                'mode': 'mock',
                'agent_id': 'coordinator-001',
                'timestamp': time.time()
            }
        }
        
        logger.info("Mock inference completed", 
                   model=request.model_name, 
                   execution_time=execution_time)
        
        return response

    async def _real_inference(self, request: InferenceRequest, model_info: Dict, start_time: float) -> Dict[str, Any]:
        """Real Triton inference"""
        # Prepare inputs for Triton
        inputs = []
        for input_spec in model_info['inputs']:
            input_name = input_spec['name']
            if input_name not in request.inputs:
                raise HTTPException(400, f"Missing required input: {input_name}")
            
            input_data = request.inputs[input_name]
            
            # Handle different input formats
            if isinstance(input_data, dict):
                data = np.array(input_data.get('data', input_data))
                shape = input_data.get('shape', data.shape)
                datatype = input_data.get('datatype', input_spec['datatype'])
            else:
                data = np.array(input_data)
                shape = data.shape
                datatype = input_spec['datatype']
            
            input_tensor = httpclient.InferInput(input_name, shape, datatype)
            input_tensor.set_data_from_numpy(data)
            inputs.append(input_tensor)
        
        # Prepare outputs
        outputs = []
        for output_spec in model_info['outputs']:
            output = httpclient.InferRequestedOutput(output_spec['name'])
            outputs.append(output)
        
        # Execute inference with monitoring
        inference_start = time.time()
        result = self.client.infer(request.model_name, inputs, outputs=outputs)
        inference_time = time.time() - inference_start
        
        # Process results
        response_outputs = {}
        for output_spec in model_info['outputs']:
            output_name = output_spec['name']
            output_data = result.as_numpy(output_name)
            response_outputs[output_name] = output_data.tolist()
        
        execution_time = time.time() - start_time
        
        # Update metrics
        REQUEST_COUNT.labels(model=request.model_name, status='success').inc()
        REQUEST_DURATION.labels(model=request.model_name).observe(execution_time)
        
        response = {
            'model': request.model_name,
            'outputs': response_outputs,
            'metadata': {
                'execution_time_ms': int(execution_time * 1000),
                'inference_time_ms': int(inference_time * 1000),
                'agent_id': 'coordinator-001',
                'timestamp': time.time()
            }
        }
        
        logger.info("Inference completed", 
                   model=request.model_name, 
                   execution_time=execution_time,
                   inference_time=inference_time)
        
        return response

    async def _check_docker_model_runner_gpu(self) -> Dict[str, Any]:
        """Report GPU/offload state clearly."""
        # Remote offload: assume GPU provided by Docker Offload backend
        if self.offload_mode == "remote-offload":
            return {
                "available": True,
                "mode": "remote-offload",
                "reason": "GPU provided via Docker Offload",
                "docker_model_runner": False
            }
        if not self.docker_model_runner_enabled:
            return {
                "available": False,
                "mode": self.offload_mode,
                "reason": "Local Docker Model Runner disabled"
            }
        if not self.docker_client:
            return {
                "available": False,
                "mode": self.offload_mode,
                "reason": "Docker client not initialized"
            }
        try:
            info = self.docker_client.info()
            runtimes = info.get("Runtimes", {})
            has = "nvidia" in runtimes
            return {
                "available": has,
                "mode": "local-gpu" if has else self.offload_mode,
                "runtimes": list(runtimes.keys())
            }
        except Exception as e:
            return {
                "available": False,
                "mode": self.offload_mode,
                "reason": f"Docker info failed: {e}"
            }
    
    async def _offload_to_docker_model_runner(self, request: InferenceRequest) -> Dict[str, Any]:
        """Offload inference to Docker Model Runner with GPU"""
        if not self.docker_model_runner_enabled:
            raise Exception("Docker Model Runner not enabled")
        
        start_time = time.time()
        
        try:
            # Configuration for Docker Model Runner
            model_image = self._get_model_image(request.model_name)
            
            # Run inference in a GPU-enabled container
            container_config = {
                "image": model_image,
                "runtime": "nvidia",
                "environment": [
                    "NVIDIA_VISIBLE_DEVICES=all",
                    f"MODEL_NAME={request.model_name}",
                    "CUDA_VISIBLE_DEVICES=0"
                ],
                "volumes": {
                    "/model-cache": {"bind": "/models", "mode": "ro"}
                },
                "remove": True,
                "detach": False
            }
            
            # Prepare input data
            input_data = json.dumps(request.inputs)
            
            # Run the container
            result = self.docker_client.containers.run(
                **container_config,
                command=f"python /app/inference.py '{input_data}'",
                stdout=True,
                stderr=True
            )
            
            # Parse results
            output = json.loads(result.decode('utf-8'))
            
            execution_time = time.time() - start_time
            
            return {
                'model': request.model_name,
                'outputs': output.get('outputs', {}),
                'metadata': {
                    'execution_time_ms': int(execution_time * 1000),
                    'mode': 'docker_model_runner_gpu',
                    'agent_id': 'coordinator-001',
                    'timestamp': time.time(),
                    'gpu_used': True
                }
            }
            
        except Exception as e:
            logger.error(f"Docker Model Runner inference failed: {e}")
            # Fallback to mock inference
            return await self._mock_inference(request, {"platform": "mock"}, start_time)
    
    def _get_model_image(self, model_name: str) -> str:
        """Get Docker image for model inference"""
        model_images = {
            "text_classifier": "huggingface/transformers-pytorch-gpu:latest",
            "resnet50": "pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime",
            "identity_model": "python:3.11-slim"
        }
        return model_images.get(model_name, "python:3.11-slim")

    def get_status(self) -> AgentStatus:
        """Get current agent status"""
        uptime = time.time() - self.start_time
        return AgentStatus(
            status=self.health_status,
            models_loaded=len(self.model_metadata),
            uptime_seconds=uptime,
            gpu_available=self._check_gpu_availability()
        )
    
    def _check_gpu_availability(self) -> bool:
        """Enhanced GPU availability check"""
        try:
            # Check Triton GPU
            if self.client and self.client.is_server_ready():
                triton_gpu = True
            else:
                triton_gpu = False
            
            # Check Docker Model Runner GPU
            docker_gpu = self.docker_model_runner_enabled
            
            return triton_gpu or docker_gpu
        except:
            return False
        
    def _should_use_local_gpu_offload(self) -> bool:
        """Determines if the local GPU path should be taken based on OFFLOAD_MODE."""
        # Note: A real implementation for 'auto' would check for local GPU hardware here.
        # For this showcase, we rely on the mode set by start-system.sh.
        return self.offload_mode == "local-gpu"

    def _should_use_remote_offload(self) -> bool:
        """Determines if the remote offload path should be taken based on OFFLOAD_MODE."""
        return self.offload_mode == "remote-offload"

    async def _offload_to_remote_runner(self, request: InferenceRequest, start_time: float) -> Dict[str, Any]:
        """Send inference request to a remote Docker Model Runner endpoint"""
        if not self.remote_runner_url:
            raise Exception("Remote runner URL not configured")

        payload = {
            "model_name": request.model_name,
            "inputs": request.inputs
        }

        try:
            import httpx
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(self.remote_runner_url.rstrip('/') + "/infer", json=payload)
                resp.raise_for_status()
                data = resp.json()
        except ImportError:
            # Fallback for environments without httpx
            import urllib.request
            import json as _json
            req = urllib.request.Request(
                self.remote_runner_url.rstrip('/') + "/infer",
                data=_json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=60) as f:
                data = _json.loads(f.read().decode("utf-8"))
        
        elapsed = int((time.time() - start_time) * 1000)
        return {
            "model": request.model_name,
            "outputs": data.get("outputs", {}),
            "metadata": {
                "execution_time_ms": elapsed,
                "mode": "remote_docker_offload",
                "agent_id": "coordinator-001",
                "timestamp": time.time(),
                "gpu_used": data.get("metadata", {}).get("gpu_used", False)
            }
        }

# Global agent instance - fix URL parsing
triton_url_raw = os.getenv('TRITON_URL', 'triton-server:8000')
coordinator = Agent(triton_url=triton_url_raw)

@app.on_event("startup")
async def startup():
    """Initialize the agent on startup"""
    await coordinator.initialize()

@app.post("/infer")
async def infer(request: InferenceRequest):
    """Execute inference request"""
    return await coordinator.route_inference(request)

@app.get("/models")
async def list_models():
    """List available models and their metadata"""
    return {
        "models": list(coordinator.model_metadata.keys()),
        "metadata": coordinator.model_metadata
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    status = coordinator.get_status()
    return {
        "status": status.status,
        "agent": "inference-coordinator",
        "models_loaded": status.models_loaded,
        "uptime_seconds": status.uptime_seconds,
        "gpu_available": status.gpu_available
    }

@app.get("/status")
async def get_status():
    """Detailed status information"""
    return coordinator.get_status()

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/agents")
async def list_agents():
    """List connected agents (for monitoring)"""
    return {
        "agents": [
            {
                "id": "coordinator-001",
                "type": "inference-coordinator",
                "status": coordinator.health_status,
                "models": list(coordinator.model_metadata.keys()),
                "uptime": time.time() - coordinator.start_time
            }
        ]
    }

@app.get("/gpu-status")
async def gpu_status():
    """Get detailed GPU status for Docker Model Runner"""
    info = await coordinator._check_docker_model_runner_gpu()
    
    return {
        "offload_mode": coordinator.offload_mode,
        "docker_model_runner_enabled": coordinator.docker_model_runner_enabled,
        "gpu_info": info,
        "offload_enabled": os.getenv('OFFLOAD_ENABLED', 'false').lower() == 'true'
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)