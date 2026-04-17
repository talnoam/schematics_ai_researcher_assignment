# Adaptive Questionnaire Agent (Interview Project)

This repository contains an adaptive mortgage-profile questionnaire:

- **Backend:** FastAPI (`backend/`)
- **Frontend:** Streamlit chat UI (`frontend/`)
- **LLM:** Ollama (`qwen2.5:7b`) for conversational question generation and free-text extraction

The system asks fewer questions by combining deterministic rules, probabilistic inference, and information-gain scoring.

## What You Need Pre-Installed

Install these tools before running the project:

- Python 3.12+
- `uv` (package/environment manager)
- Ollama
- (Optional) Docker + Docker Compose

## Quick Start (Local, Recommended for Review)

### 1) Clone and install dependencies

```bash
git clone https://github.com/talnoam/schematics_ai_researcher_assignment.git
cd schematics_ai_researcher_assignment
uv sync --all-extras
```

### 2) Configure environment

```bash
cp .env.example .env
```

Edit `.env` for local run:

```dotenv
DEBUG=true
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen2.5:7b
DATA_DIR=data
RANDOM_SEED=42
BACKEND_URL=http://127.0.0.1:8000
```

Important:

- Do **not** use inline comments on env value lines (for example `... # comment`), because they can be parsed as part of the value.

### 3) Start Ollama

```bash
ollama serve
```

In another terminal (one-time model download):

```bash
ollama pull qwen2.5:7b
```

Optional warmup call:

```bash
curl http://127.0.0.1:11434/api/generate \
  -d '{"model":"qwen2.5:7b","prompt":"hello","stream":false}'
```

### 4) Start backend

```bash
bash scripts/run_backend.sh
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

### 5) Start frontend

```bash
bash scripts/run_frontend.sh
```

Open:

- Streamlit UI: `http://127.0.0.1:8501`

## Quick Start (Docker)

Use this if you prefer containers:

```bash
cp .env.example .env
```

For Docker mode keep:

```dotenv
OLLAMA_BASE_URL=http://host.docker.internal:11434
BACKEND_URL=http://backend:8000
```

Start Ollama on host:

```bash
ollama serve
ollama pull qwen2.5:7b
```

Start stack:

```bash
docker compose build --no-cache
docker compose up -d
```

Open:

- Frontend: `http://localhost:8501`
- Backend (host-mapped): `http://localhost:8003/health`

## Useful Commands

Run tests:

```bash
uv run python -m pytest tests/ -v --tb=short
```

Generate mock dataset:

```bash
uv run python scripts/seed_mock_data.py
```

Validate generated statistics:

```bash
uv run python scripts/validate_statistics.py
```

## Data Validation & Analytics Dashboard

Run the dashboard generation script:

```bash
uv run python scripts/validate_statistics.py
```

Interactive HTML output:

- `data/visualizations/cohort_feature_distributions_dashboard.html`

This dashboard is designed to visually validate the Latent Factor + Softmax data-generation pipeline. It shows distinct, cohort-normalized probability distributions (0-100%) across the Facebook campaign cohorts, making it easy to verify that each cohort exhibits logical, differentiated behavior.

## API Endpoints (Backend)

- `GET /health`
- `POST /api/v1/sessions/start`
- `POST /api/v1/sessions/{session_id}/answer`
- `POST /api/v1/sessions/{session_id}/answer_text`

## Common Local Issues

- `Backend request failed: nodename nor servname provided`
  - Usually wrong `BACKEND_URL` or `OLLAMA_BASE_URL`.
  - For local debug use `127.0.0.1` values shown above.
- `404` with URL containing `#...`
  - Caused by inline comments in `.env` values. Remove inline comments.
- `500` on `/answer_text`
  - Ollama is not reachable or model not loaded.
  - Ensure `ollama serve` is running and `qwen2.5:7b` is pulled.

