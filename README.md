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
├── model_repository   # mock testing
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
│   ├── test-system.sh
│   ├── test-real-inference.sh
├── docker-compose.yml
├── docker-compose.gpu.override.yml
├── .env
├── README.md
└── ...
```

---

## Overview

This repository demonstrates a scalable, agentic AI inference pipeline using Docker containers, NVIDIA Triton Inference Server, and GPU offload logic. The system is designed for both mock and real model serving 

---

## File & Codebase Explanation

- **agents/**  
  Contains the source code for each microservice:
  - **coordinator/**: Handles inference routing, GPU offload, and Triton integration.
  - **preprocessor/**: Preprocesses text, image, and audio data for model input.
  - **aggregator/**: Aggregates results from multiple models or requests.

- **triton-server/model-repository/**  
  Stores ONNX models and configuration files for Triton Inference Server. Includes both dummy (for testing) and real models (Verified).

- **test-data/**  
  Contains scripts and files for generating and storing sample data used in testing (text, images, payloads).

- **monitoring/**  
  Configuration for Prometheus (metrics scraping) and Grafana (dashboard visualization).

- **scripts/**  
  Utility scripts for setup, model download, GPU checks, starting/stopping services, and running tests.

- **docker-compose.yml / docker-compose.gpu.override.yml**  
  Main Docker Compose files for orchestrating all services and enabling GPU support.

- **.env / .env.gpu**  
  Environment configuration files for service URLs, GPU settings, and offload flags.

---

## Execution Modes

| OFFLOAD_MODE     | Behavior                                      | Requirements                               |
|------------------|-----------------------------------------------|---------------------------------------------|
| cpu              | Everything runs CPU                           | None                                        |
| local-gpu        | Use local NVIDIA GPU                          | Host GPU + NVIDIA runtime                   |
| remote-offload   | Runs stack on Docker Offload cloud GPU        | `docker offload start --gpu` (eligible acct)|
| auto (default)   | local-gpu if GPU → else remote-offload → cpu  | Optional GPU / Offload                      |

---

## Quick Start

### 1. Clone & basic setup
```bash
git clone https://github.com/kubetoolsio/agentic-docker-offload-triton.git
cd agentic-docker-offload-triton
./scripts/setup.sh
```

### 2. Add mock models (already in repo)
```bash
./scripts/download-models.sh
```

### 3. (Optional) Add real small model (bert‑tiny ONNX)
```bash
./scripts/download-real-models.sh
```

### 4A. CPU only
```bash
OFFLOAD_MODE=cpu ./scripts/start-system.sh
```

### 4B. Remote GPU (Docker Offload)
```bash
docker offload start --gpu
OFFLOAD_MODE=remote-offload ./scripts/start-system.sh
```

### 4C. Local GPU
```bash
OFFLOAD_MODE=local-gpu ./scripts/start-system.sh
```

---

## Verify

```bash
./scripts/test-system.sh
curl -s http://localhost:8090/health | jq .
curl -s http://localhost:8090/gpu-status | jq .
```

Remote or local GPU check:
```bash
./scripts/verify-remote-gpu.sh
```

---

## Real Model Inference (bert‑tiny)

After adding the model:
```bash
./scripts/test-real-inference.sh "The movie was surprisingly good"
```

Example output includes logits + probabilities.

---

## Coordinator Endpoints

| Endpoint                 | Purpose                          |
|--------------------------|----------------------------------|
| /health                  | Health & basic status            |
| /models                  | Model list (mock + discovered)   |
| /gpu-status              | Offload + GPU mode info          |
| /metrics                 | Prometheus metrics               |
| /infer                   | Mock routing (legacy path)       |

Real model inference via Triton direct HTTP:
```
POST /v2/models/text_classifier/infer
```

---

## Scripts Summary

| Script                        | Purpose                                      |
|-------------------------------|----------------------------------------------|
| start-system.sh              | Mode detection & stack launch                |
| download-models.sh           | Mock/dummy models                            |
| download-real-models.sh      | Exports bert‑tiny ONNX                       |
| test-system.sh               | End‑to‑end health tests                      |
| test-inference.sh            | Mock inference helpers                       |
| test-real-inference.sh       | Real ONNX model inference (bert‑tiny)        |          |

---

## Offload Mode Logic (Simplified)

1. If OFFLOAD_MODE explicitly set → use it.
2. If auto:
   - Detect local GPU → local-gpu
   - Else if Docker Offload active → remote-offload
   - Else → cpu

Remote mode disables local Docker Model Runner logic (no nested docker).

---

## Testing Examples

```bash
# Text (mock)
./scripts/test-inference.sh text "hello world"

# Image (mock)
./scripts/test-inference.sh image ./test-data/sample.jpg

# Real model (bert‑tiny)
./scripts/test-real-inference.sh "A concise test sentence"
```
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
- Batch & multi‑model fan‑out
- GPU utilization metrics surface
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

