# Adaptive Questionnaire Agent

An intelligent questionnaire system that uses **information theory** and **deterministic inference** to minimize the number of questions asked while maximizing the information gathered about a user. Instead of asking all 10 questions in a fixed order, the agent dynamically selects the most valuable next question, skips questions it can infer deterministically, and stops early when further questioning is no longer worth the user friction.

The system also supports **free-text answers** powered by a local LLM (Ollama), allowing users to type natural language responses that are automatically parsed into structured field values.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Target Fields](#target-fields)
- [Prerequisites](#prerequisites)
- [Quick Start (Docker)](#quick-start-docker)
- [Local Development Setup](#local-development-setup)
- [Running the Application Locally](#running-the-application-locally)
- [Ollama LLM Setup](#ollama-llm-setup)
- [Mock Data Generation](#mock-data-generation)
- [Running Tests](#running-tests)
- [Configuration Reference](#configuration-reference)
- [API Endpoints](#api-endpoints)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)

## Architecture Overview

```text
┌──────────────────┐       HTTP        ┌──────────────────┐      HTTP       ┌─────────────┐
│  Streamlit UI    │  ──────────────>  │  FastAPI Backend  │  ───────────>  │   Ollama     │
│  (port 8501)     │  <──────────────  │  (port 8000)      │  <───────────  │ (port 11434) │
│                  │                   │                    │                │              │
│  - Questionnaire │                   │  - Session store   │                │  - qwen2.5   │
│  - Debug sidebar │                   │  - Adaptive agent  │                │  - Free-text │
│  - Free-text     │                   │  - Rules engine    │                │    parsing   │
└──────────────────┘                   │  - Info-theory     │                └─────────────┘
                                       │  - LLM extractor   │
                                       └──────────────────┘
```

- **Frontend** (Streamlit) — Interactive questionnaire UI with a debug sidebar showing the agent's internal state.
- **Backend** (FastAPI) — Core logic: adaptive question selection, deterministic inference, session management, and LLM-based text extraction.
- **Ollama** — Local LLM server for parsing free-text user answers into structured fields. Runs on the host machine (not inside Docker).

## Target Fields

The agent aims to collect 10 fields about a mortgage applicant:

| # | Field | Type | Example Values |
|---|-------|------|----------------|
| 1 | Credit Score Rate | Enum | EXCELLENT, GOOD, FAIR, POOR |
| 2 | Loan Primary Purpose | Enum | PURCHASE, REFINANCE, CASH_OUT_REFINANCE |
| 3 | Property Type | Enum | SINGLE_FAMILY, CONDO, TOWNHOUSE, MULTI_FAMILY |
| 4 | Property Use | Enum | PRIMARY_RESIDENCE, SECOND_HOME, INVESTMENT |
| 5 | Annual Income Band | Enum | BELOW_50K, 50K_100K, 100K_150K, 150K_250K, ABOVE_250K |
| 6 | Property Value Band | Enum | BELOW_200K, 200K_400K, 400K_700K, 700K_1M, ABOVE_1M |
| 7 | Credit Line Band | Enum | BELOW_10K, 10K_30K, 30K_60K, 60K_100K, ABOVE_100K |
| 8 | Age Band | Enum | 18_25, 26_35, 36_50, 51_65, ABOVE_65 |
| 9 | Currently Have Mortgage | Boolean | true, false |
| 10 | Military Veteran | Boolean | true, false |

## Prerequisites

Before you begin, make sure you have the following installed:

| Tool | Version | Purpose | Install |
|------|---------|---------|---------|
| **Python** | >= 3.12 | Runtime | [python.org](https://www.python.org/downloads/) |
| **uv** | latest | Python package/project manager | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| **Docker** & **Docker Compose** | latest | Containerized deployment | [docs.docker.com](https://docs.docker.com/get-docker/) |
| **Ollama** | latest | Local LLM for free-text parsing | [ollama.com/download](https://ollama.com/download) |

> **Note:** Ollama is only required if you want to use the free-text answer feature. The explicit-answer (selectbox/radio) flow works without it.

## Quick Start (Docker)

This is the fastest way to get everything running.

### 1. Clone the repository

```bash
git clone git@github.com:talnoam/schematics_ai_researcher_assignment.git
cd schematics_ai_researcher_assignment
```

### 2. Set up environment variables

```bash
cp .env.example .env
```

Edit `.env` if you need to change defaults (e.g., the Ollama model). The defaults work out of the box:

```dotenv
DEBUG=false
OLLAMA_BASE_URL=http://host.docker.internal:11434
OLLAMA_MODEL=qwen2.5:7b
DATA_DIR=data
RANDOM_SEED=42
BACKEND_URL=http://backend:8000
```

### 3. Start Ollama on your host machine

Ollama runs **on your host** (not inside Docker). The containers reach it via `host.docker.internal`.

```bash
# Terminal 1: Start the Ollama server
ollama serve
```

```bash
# Terminal 2: Pull the model (one-time download, ~4.7 GB)
ollama pull qwen2.5:7b
```

Verify Ollama is running:

```bash
curl http://localhost:11434/api/tags
```

You should see a JSON response listing `qwen2.5:7b` in the models array.

**Warm up the model** (recommended to avoid cold-start timeouts on first request):

```bash
curl http://localhost:11434/api/generate \
  -d '{"model":"qwen2.5:7b","prompt":"hello","stream":false}'
```

### 4. Build and start containers

```bash
docker compose build --no-cache
docker compose up -d
```

This starts two containers:
- **backend** — FastAPI on host port `8003` (maps to container port `8000`)
- **frontend** — Streamlit on host port `8501`

The frontend waits for the backend health check to pass before starting.

### 5. Open the application

Open your browser to **http://localhost:8501** and start a questionnaire session.

### Useful Docker commands

```bash
# View logs
docker compose logs -f backend
docker compose logs -f frontend

# Restart after code changes
docker compose build --no-cache && docker compose up -d

# Stop everything
docker compose down

# Check container health
docker compose ps
```

## Local Development Setup

For development with hot-reload and direct access to the Python environment.

### 1. Clone and set up dependencies

```bash
git clone git@github.com:talnoam/schematics_ai_researcher_assignment.git
cd schematics_ai_researcher_assignment
uv sync --all-extras
```

This installs all dependencies (including dev tools: pytest, mypy, ruff) into a uv-managed virtual environment.

### 2. Set up environment variables

```bash
cp .env.example .env
```

For **local** development (not Docker), update `OLLAMA_BASE_URL` and `BACKEND_URL` in `.env`:

```dotenv
DEBUG=true
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b
DATA_DIR=data
RANDOM_SEED=42
BACKEND_URL=http://localhost:8000
```

> **Important:** When running locally (not in Docker), use `http://localhost:11434` for Ollama and `http://localhost:8000` for the backend. The `host.docker.internal` addresses only work from inside Docker containers.

## Running the Application Locally

You need **three terminal windows**: one for Ollama, one for the backend, and one for the frontend.

### Terminal 1 — Ollama

```bash
ollama serve
```

(If you haven't already, pull the model: `ollama pull qwen2.5:7b`)

### Terminal 2 — Backend (FastAPI)

```bash
bash scripts/run_backend.sh
```

Or equivalently:

```bash
uv run uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

The backend starts on **http://localhost:8000**. Verify with:

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

### Terminal 3 — Frontend (Streamlit)

```bash
bash scripts/run_frontend.sh
```

Or equivalently:

```bash
uv run streamlit run frontend/app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true
```

The frontend opens on **http://localhost:8501**.

## Ollama LLM Setup

Ollama provides local LLM inference for the free-text answer feature. The agent sends the user's free-text response to Ollama, which extracts structured field values from it.

### Supported Models

The default model is `qwen2.5:7b` (~4.7 GB). You can change it in `.env`:

| Model | Size | Speed | Quality |
|-------|------|-------|---------|
| `qwen2.5:7b` | ~4.7 GB | Moderate | Good (recommended) |
| `qwen2.5:3b` | ~2 GB | Fast | Acceptable |
| `qwen2.5:14b` | ~9 GB | Slow | Better |

To switch models:

1. Pull the new model: `ollama pull <model-name>`
2. Update `OLLAMA_MODEL` in `.env`
3. If using Docker, rebuild: `docker compose build --no-cache && docker compose up -d`

### Timeout Configuration

The LLM request timeout is set in `backend/llm/config.py`:

```python
LLM_REQUEST_TIMEOUT_SECONDS: float = 90.0
```

If you experience timeouts (especially on first request after model load), increase this value or warm up the model before using the UI.

### Troubleshooting Ollama

| Symptom | Cause | Fix |
|---------|-------|-----|
| `httpx.ConnectError` | Ollama not running | Run `ollama serve` |
| `httpx.ReadTimeout` | Model cold start or too slow | Warm up model with `curl` (see above) or increase timeout |
| `Model not found` | Model not pulled | Run `ollama pull qwen2.5:7b` |
| Timeout in Docker | Wrong `OLLAMA_BASE_URL` | Must be `http://host.docker.internal:11434` in `.env` |
| Timeout locally | Wrong `OLLAMA_BASE_URL` | Must be `http://localhost:11434` in `.env` |

## Mock Data Generation

The project includes a synthetic data generator for testing and evaluation. It produces realistic user profiles using cohort-based sampling with latent factor correlations.

### Generate mock data

```bash
uv run python scripts/seed_mock_data.py
```

This generates `data/generated/mock_users.csv` with 5,000 synthetic user records.

### Validate statistics

```bash
uv run python scripts/validate_statistics.py
```

This loads the generated CSV, computes correlations and conditional probabilities, and generates HTML plots in `data/generated/plots/`.

### Cohort definitions

Three cohorts are defined in `data/cohorts/cohort_definitions.yaml`:

- **Tech Veterans** — Higher income, likely military veterans, single-family homes
- **Middle-aged Suburban Families** — Moderate income, primary residence owners, likely have mortgages
- **Young Urban Renters** — Lower income, younger age, condos/townhouses, less likely to have mortgages

## Running Tests

All commands use `uv run` to execute within the project environment.

### Run all tests

```bash
uv run python -m pytest tests/ -v --tb=short
```

### Run only unit tests

```bash
uv run python -m pytest tests/unit/ -v --tb=short
```

### Run only integration tests

```bash
uv run python -m pytest tests/integration/ -v --tb=short
```

### Run with coverage

```bash
uv run python -m pytest tests/ -v --tb=short --cov=backend --cov=frontend
```

> **Note:** Unit tests do not require Ollama, the backend, or any external services. All external calls are mocked. Integration tests that hit the live API or Ollama are marked with `@pytest.mark.integration`.

## Configuration Reference

### Environment Variables (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `DEBUG` | `false` | Enable debug logging |
| `OLLAMA_BASE_URL` | `http://host.docker.internal:11434` | Ollama API URL (use `localhost` for local dev) |
| `OLLAMA_MODEL` | `qwen2.5:7b` | Ollama model for free-text extraction |
| `DATA_DIR` | `data` | Root directory for data files |
| `RANDOM_SEED` | `42` | Global seed for reproducibility |
| `BACKEND_URL` | `http://backend:8000` | Backend URL (used by frontend; use `http://localhost:8000` for local dev) |

### Core Logic Constants (`backend/core_logic/config.py`)

| Constant | Description |
|----------|-------------|
| `FRICTION_WEIGHT` | Weight penalty for question friction in utility calculation |
| `MIN_UTILITY_THRESHOLD` | Below this utility, the agent stops asking and infers remaining fields |
| `DETERMINISTIC_CONFIDENCE` | Confidence level assigned to deterministically inferred fields |
| `ENTROPY_EPSILON` | Small value to prevent log(0) in entropy calculations |

### LLM Constants (`backend/llm/config.py`)

| Constant | Default | Description |
|----------|---------|-------------|
| `LLM_REQUEST_TIMEOUT_SECONDS` | `90.0` | HTTP timeout for Ollama requests |
| `LLM_TEMPERATURE` | `0.0` | Sampling temperature (0 = deterministic) |
| `LLM_MAX_TOKENS` | `400` | Max tokens in LLM response |

### Docker Port Mapping

| Service | Container Port | Host Port |
|---------|---------------|-----------|
| Backend | 8000 | 8003 |
| Frontend | 8501 | 8501 |

## API Endpoints

All endpoints are prefixed with `/api/v1/sessions`.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check (returns `{"status": "ok"}`) |
| `POST` | `/api/v1/sessions/start` | Start a new questionnaire session |
| `POST` | `/api/v1/sessions/{session_id}/answer` | Submit an explicit answer for the current question |
| `POST` | `/api/v1/sessions/{session_id}/answer_text` | Submit a free-text answer (parsed by LLM) |

### Example: Start a session

```bash
curl -X POST http://localhost:8000/api/v1/sessions/start \
  -H "Content-Type: application/json" \
  -d '{"cohort_name": "Tech Veterans"}'
```

### Example: Submit an explicit answer

```bash
curl -X POST http://localhost:8000/api/v1/sessions/<session-id>/answer \
  -H "Content-Type: application/json" \
  -d '{"target_field": "loan_primary_purpose", "answer_value": "REFINANCE"}'
```

### Example: Submit a free-text answer

```bash
curl -X POST http://localhost:8000/api/v1/sessions/<session-id>/answer_text \
  -H "Content-Type: application/json" \
  -d '{"user_text": "I am 30 years old, buying my first home, a single family house as primary residence"}'
```

## Project Structure

```text
schematics_ai_researcher_assignment/
├── .env.example                 # Template for environment variables
├── config.py                    # Global settings (Pydantic BaseSettings)
├── pyproject.toml               # Project metadata and dependencies
├── uv.lock                      # Locked dependency versions
├── docker-compose.yml           # Container orchestration
├── Dockerfile.backend           # Multi-stage backend image
├── Dockerfile.frontend          # Multi-stage frontend image
│
├── backend/
│   ├── main.py                  # FastAPI app factory
│   ├── api/
│   │   ├── config.py            # API prefix/CORS constants
│   │   ├── routes.py            # Session endpoints
│   │   ├── schemas.py           # Request/response Pydantic models
│   │   └── session_store.py     # Thread-safe in-memory session store
│   ├── core_logic/
│   │   ├── agent.py             # AdaptiveQuestionnaireAgent
│   │   ├── config.py            # Thresholds and weights
│   │   ├── deterministic_rules.py  # Rules engine (e.g., REFINANCE → has mortgage)
│   │   ├── field_mappings.py    # Shared field name/type mappings
│   │   ├── information_math.py  # Shannon entropy & expected information gain
│   │   ├── question_bank.py     # Question metadata with friction costs
│   │   └── scoring.py           # Utility = EIG − friction_weight × friction_cost
│   ├── data_generation/
│   │   ├── cohort_loader.py     # YAML cohort parser
│   │   ├── config.py            # Data generation constants
│   │   ├── enums.py             # StrEnum types for all fields
│   │   ├── generator.py         # MockDataGenerator
│   │   ├── latent_factors.py    # Latent factor model (affluence, risk)
│   │   └── schemas.py           # UserProfile, CohortDefinition models
│   └── llm/
│       ├── client.py            # Async Ollama HTTP client
│       ├── config.py            # Timeout/temperature settings
│       └── extractor.py         # Free-text → structured field extraction
│
├── frontend/
│   ├── api_client.py            # Sync HTTP client for backend
│   └── app.py                   # Streamlit application
│
├── scripts/
│   ├── run_backend.sh           # Start backend with hot-reload
│   ├── run_frontend.sh          # Start Streamlit frontend
│   ├── seed_mock_data.py        # Generate synthetic user data
│   ├── seed_mock_data.sh        # Shell wrapper for data generation
│   └── validate_statistics.py   # Statistical validation and plots
│
├── data/
│   ├── cohorts/
│   │   └── cohort_definitions.yaml  # 3 cohort probability distributions
│   └── generated/               # Output directory for mock data & plots
│
└── tests/
    ├── conftest.py              # Shared fixtures
    ├── unit/                    # Isolated tests (all mocked)
    │   ├── test_agent.py
    │   ├── test_cohort_loader.py
    │   ├── test_data_generation_schemas.py
    │   ├── test_deterministic_rules.py
    │   ├── test_frontend_api_client.py
    │   ├── test_generator.py
    │   ├── test_information_math.py
    │   ├── test_latent_factors.py
    │   ├── test_llm_extractor.py
    │   ├── test_scoring.py
    │   └── test_session_store.py
    └── integration/             # Cross-module and API tests
        ├── test_agent_phase3_flow.py
        ├── test_api_endpoints.py
        ├── test_core_logic_phase2_flow.py
        ├── test_data_generation_phase1_flow.py
        ├── test_frontend_api_client_contract.py
        └── test_generator_statistics.py
```

## How It Works

### Adaptive Question Selection

The agent follows this decision loop for each interaction:

1. **Apply deterministic rules** — If a known field logically implies another (e.g., `loan_purpose = REFINANCE` implies `currently_have_mortgage = true`), fill it automatically without asking.

2. **Calculate entropy** — For each unknown field, compute the Shannon entropy of its probability distribution. High entropy = high uncertainty.

3. **Compute Expected Information Gain (EIG)** — For each candidate question, estimate how much entropy would be reduced on average by asking it.

4. **Score utility** — `Utility = EIG − (friction_weight × friction_cost)`. Sensitive questions (like income) have high friction costs, making them less attractive unless their information gain is very high.

5. **Decide** — If the best question's utility exceeds `MIN_UTILITY_THRESHOLD`, ask it. Otherwise, stop and infer remaining fields from the probability distributions.

### Free-Text Processing

When a user types a natural language answer (e.g., "I'm 30, buying my first condo as an investment"), the system:

1. Sends the text to Ollama with a structured prompt listing the missing fields and their valid values.
2. Ollama returns a JSON object mapping field names to extracted values.
3. The backend coerces and validates each value, applies deterministic rules, and re-evaluates the next question.

### Mock Data Generation

The synthetic data pipeline creates realistic user profiles through:

1. **Cohort sampling** — Each user is assigned to one of three cohorts with different base probability distributions.
2. **Latent factors** — Hidden continuous variables (`affluence_score`, `risk_profile`) shift probabilities to create cross-field correlations (e.g., high affluence → higher income AND higher property value).
3. **Categorical sampling** — Final field values are sampled from the adjusted probability distributions using `numpy.random.choice`.
