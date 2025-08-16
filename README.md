# AI Inference Docker Offload

## Repository Structure

```
agentic-docker-offload-triton/
├── agents/
│   ├── coordinator/
│   │   ├── app.py
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── preprocessor/
│   │   ├── app.py
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   └── aggregator/
│       ├── app.py
│       ├── Dockerfile
│       └── requirements.txt
├── triton-server/
│   └── model-repository/
│       ├── identity_model/
│       ├── text_classifier/
│       ├── resnet50/
│       └── resnet18/
├── test-data/
│   ├── create-real-samples.sh
│   ├── text_samples.txt
│   ├── *.jpg
│   ├── *.json
│   └── sample_audio.wav
├── monitoring/
│   ├── prometheus.yml
│   └── grafana/
│       ├── dashboards/
│       └── datasources/
├── scripts/
│   ├── setup.sh
│   ├── download-models.sh
│   ├── download-real-models.sh
│   ├── check-gpu-compatibility.sh
│   ├── start-system.sh
│   ├── test*.sh
│   ├── verify-system.sh
│   ├── setup-docker-model-runner.sh
│   └── test-gpu-offload.sh
├── docker-compose.yml
├── docker-compose.override.yml
├── .env
├── .env.gpu
├── README.md
└── ...
```

---

## Overview

This repository demonstrates a scalable, agentic AI inference pipeline using Docker containers, NVIDIA Triton Inference Server, and GPU offload logic. The system is designed for both mock and real model serving ( Real Model Serving not tested Yet)

---

## File & Codebase Explanation

- **agents/**  
  Contains the source code for each microservice:
  - **coordinator/**: Handles inference routing, GPU offload, and Triton integration.
  - **preprocessor/**: Preprocesses text, image, and audio data for model input.
  - **aggregator/**: Aggregates results from multiple models or requests.

- **triton-server/model-repository/**  
  Stores ONNX models and configuration files for Triton Inference Server. Includes both dummy (for testing) and real models ( Not Verified Yet).

- **test-data/**  
  Contains scripts and files for generating and storing sample data used in testing (text, images, payloads).

- **monitoring/**  
  Configuration for Prometheus (metrics scraping) and Grafana (dashboard visualization).

- **scripts/**  
  Utility scripts for setup, model download, GPU checks, starting/stopping services, and running tests.

- **docker-compose.yml / docker-compose.override.yml**  
  Main Docker Compose files for orchestrating all services and enabling GPU support.

- **.env / .env.gpu**  
  Environment configuration files for service URLs, GPU settings, and offload flags.

---

## Installation

1. **Clone the repository**
    ```bash
    git clone https://github.com/kubetoolsio/agentic-docker-offload-triton.git
    cd agentic-docker-offload-triton
    ```

2. **Run setup and download models**
    ```bash
    ./scripts/setup.sh
    ./scripts/download-models.sh          # For mock models
    ./scripts/download-real-models.sh     # For real models (Not Verified Yet)
    ```

3. **Start the system**
    ```bash
    ./scripts/start-system.sh
    ```

4. **Verify services**
    ```bash
    ./scripts/verify-system.sh
    ```

---

## Quick Start

```bash
# Setup
./scripts/setup.sh
./scripts/download-models.sh

# Start services
docker-compose up -d

# Test system
./scripts/test-system.sh
```

## Service Endpoints

- **Coordinator API**: http://localhost:8090
- **Preprocessor API**: http://localhost:8081  
- **Aggregator API**: http://localhost:8082
- **Triton Server**: http://localhost:8000

## Docker Model Runner GPU Support

The coordinator now supports **Docker Model Runner** for GPU-accelerated inference:

```bash
# Setup GPU support
./scripts/setup-docker-model-runner.sh

# Start with GPU offload
OFFLOAD_ENABLED=true docker-compose up -d

# Test GPU functionality
./scripts/test-gpu-offload.sh

# Monitor GPU usage
curl http://localhost:8090/gpu-status
```

## Testing

- **System health and endpoints**
    ```bash
    ./scripts/test-system.sh
    ```

- **Pipeline test**
    ```bash
    ./scripts/test-pipeline.sh
    ```

- **Inference tests**
    ```bash
    ./scripts/test-inference.sh text "Your text here"
    ./scripts/test-inference.sh image ./test-data/sample.jpg
    ./scripts/test-inference.sh all
    ```

- **Real model tests**
    ```bash
    ./scripts/test-real-inference.sh
    ```

- **GPU offload tests**
    ```bash
    ./scripts/test-gpu-offload.sh
    ```

All tests now use port **8090** for the coordinator service.

---

## Troubleshooting

- **Service not starting**
    - Check Docker logs: `docker-compose logs <service-name>`
    - Ensure ports are not in use.
    - Verify `.env` and `.env.gpu` are correctly set.

- **GPU not detected**
    - Run: `./scripts/check-gpu-compatibility.sh`
    - Ensure NVIDIA drivers and container toolkit are installed.

- **Triton not loading models**
    - Check model files in `triton-server/model-repository/`
    - Use real models via `./scripts/download-real-models.sh`
    - Inspect Triton logs for errors.

- **Metrics not available** ( Still not Verified Yet)
    - Ensure Prometheus and Grafana are running.
    - Check `monitoring/prometheus.yml` configuration.

---

## To-Dos

- Add support for more model types (audio, multi-modal).
- Improve Docker Model Runner integration.
- Expand aggregation strategies.
- Add more real-world sample data.
- Enhance documentation and examples.

---

## Contributing Guidelines

- Fork the repository and create a feature branch.
- Follow PEP8 and best practices for Python code.
- Use clear commit messages and document changes.
- Submit pull requests with a description of your changes.
- For major features or refactoring, open an issue first for discussion.
- All contributions are welcome!

---

## License

This project is released under the MIT License.

---

## Acknowledgements

- NVIDIA Triton Inference Server
- Docker & Docker Compose
- FastAPI, Prometheus, Grafana
- HuggingFace Transformers, PyTorch, ONNX

