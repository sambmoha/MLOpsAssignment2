M2: Model Packaging & Containerization
====================================
Objective: Package the trained model into a reproducible, containerized service.

Status:
- Inference service: DONE (FastAPI) in `src/api.py` ✅
- Requirements: DONE (pinned versions) in `requirements.txt` ✅
- Dockerfile: DONE in `Dockerfile` ✅
- Docker build & verification: DONE ✅

Features Implemented:
- FastAPI with /health and /predict endpoints
- Structured logging with timestamps and error tracking
- Pydantic response models for type validation
- File validation (size, format, image integrity)
- Docker health checks (automatic container monitoring)
- Security: Non-root user (appuser), no system vulnerabilities
- Proxy support for corporate environments (fastweb.bell.ca:80)

Quick Setup (Local, no Docker)
------------------------------
1) Install deps:
   pip install -r requirements.txt

2) Ensure model exists:
   dvc pull artifacts.dvc

3) Run API:
   uvicorn src.api:app --host 0.0.0.0 --port 8000

4) Test health:
   curl http://localhost:8000/health

5) Test prediction:
   curl -X POST "http://localhost:8000/predict" -F "file=@path/to/image.jpg"

Docker Build/Run
----------------
1) Build image:
   docker build -t dogs-cats-api .

2) Run container (background):
   docker run -d -p 8000:8000 dogs-cats-api

3) Test health endpoint:
   curl http://localhost:8000/health

4) Test prediction:
   curl.exe -X POST "http://localhost:8000/predict" -F "file=@C:\path\to\image.jpg"

5) Stop container:
   docker stop <container-id>

Note: Base image is python:3.10-slim-bullseye for security & size optimization


M3: CI Pipeline for Build, Test & Image Creation
================================================
Objective: Implement CI to test, build, and publish container images.

Remaining Steps (NOT done yet):
1) Automated Testing
   - Add pytest tests:
     - one data pre-processing function test
     - one model utility/inference function test

2) CI Setup (GitHub Actions / GitLab CI / Jenkins / Tekton)
   - On push/PR: checkout -> install deps -> run pytest -> build Docker image

3) Artifact Publishing
   - Push Docker image to registry (Docker Hub / GHCR / local registry)

Suggested Files to Add:
- `tests/test_preprocess.py`
- `tests/test_inference.py`
- `.github/workflows/ci.yml` (if using GitHub Actions)


M4: CD Pipeline & Deployment
============================
Objective: Deploy the containerized model and automate updates.

Remaining Steps (NOT done yet):
1) Choose deployment target:
   - Docker Compose OR Kubernetes (kind/minikube)

2) Provide manifests:
   - If Docker Compose: `docker-compose.yml`
   - If Kubernetes: `k8s/deployment.yaml` and `k8s/service.yaml`

3) CD / GitOps flow:
   - On main branch merge, pull new image and redeploy

4) Smoke Tests:
   - Post-deploy health check + one prediction call
   - Fail pipeline if tests fail


Notes
-----
- API endpoints implemented:
  - GET /health
  - POST /predict
- Model used: `artifacts/models/baseline_cnn.pt`
- If you change the model path, set environment variable `MODEL_PATH`.


High-Accuracy Training (Target >85%)
====================================
Run the optimized trainer with transfer learning:

python3 src/train.py \
  --data-dir ../data/processed_224 \
  --output-dir artifacts \
  --arch resnet18 \
  --epochs 8 \
  --batch-size 64 \
  --lr 3e-4 \
  --device auto

This configuration enables:
- ImageNet normalization
- weighted loss for class imbalance
- best-checkpoint selection by validation accuracy
- pretrained transfer learning with staged unfreezing
