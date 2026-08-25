# 🤖 Enterprise-Grade Agentic RAG Application

An end-to-end, production-grade **Retrieval-Augmented Generation (RAG)** application powered by **LangGraph**, **FastAPI**, **Qdrant**, **Jina AI**, **Portkey AI**, and **Streamlit**. 

Designed for enterprise technical documentation (Kubernetes, Intel Hardware, Enterprise Networking) with multi-format document ingestion, semantic reranking, structured guardrails, distributed observability, serverless persistence, and continuous integration.

---

## 🌟 Key Features

* **🧠 Agentic Workflow (LangGraph)**: Stateful graph routing with Planner, Retriever, and Generator nodes supporting multi-turn conversation memory.
* **⚡ Dual-Engine Embeddings & Reranking (Jina AI)**: Uses `jina-embeddings-v3` (1024-dim) for high-density vector representations and `jina-reranker-v2-base-multilingual` for top-N precision filtering.
* **🗄️ Vector Database (Qdrant)**: Scalable cloud vector search with payload filtering and Cosine similarity metric.
* **🛡️ Structured Safety Guardrails**: Guardrails gate powered by Pydantic structured output validation for greetings, capabilities, off-topic rejection, and jailbreak defense.
* **🌐 Production LLM Gateway (Portkey AI)**: Enterprise LLM routing with fallback strategies, rate-limit retries, and semantic response caching, backed by direct Groq fallbacks (`llama-3.3-70b-versatile` & `llama-3.1-8b-instant`).
* **🗃️ Persistent Database & Caching**: **Neon Serverless PostgreSQL** for session history & logging, plus **Upstash Redis REST API** for ultra-fast query caching.
* **🔍 Distributed Observability**: Deep tracing and telemetry integrated with **Pydantic Logfire** and **LangSmith**.
* **📄 Universal Document Ingestion Processor**: Batch ingestion supporting PDF (`pypdf` + `pdfplumber`), HTML (`BeautifulSoup`), DOCX, PPTX, and TXT with smart token-boundary chunking.
* **🎨 Modern Streamlit User Interface**: Interactive chat UI featuring live thought-process execution, source context inspection, session management, and automatic Render backend cold-start wake-up.
* **🚀 Automated CI/CD Pipeline**: GitHub Actions workflow covering syntax linting (`ruff`), automated unit testing (`pytest`), a static dataset-integrity eval gate, a live **RAGAS quality gate** (Faithfulness + Answer Relevancy ≥ 0.6), and Docker container build verification.
* **📊 Quality Evaluation Suite**: A golden-dataset-driven eval harness (`evals/`) scoring RAGAS Faithfulness, Answer Relevancy, Context Precision/Recall, Answer Correctness, Tool Correctness, and guardrail precision/recall — see [`evals/EVAL_RESULTS.md`](evals/EVAL_RESULTS.md).

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    User([👤 User / Streamlit UI]) -->|HTTP POST /query| API[⚡ FastAPI Backend]
    
    subgraph Guardrails Gate
        API --> Guard[🛡️ Guardrails Validator]
        Guard -->|Off-Topic / Jailbreak / Greeting| DirectResponse[💬 Instant Direct Response]
        Guard -->|Clean Technical Query| Graph[🧠 LangGraph Orchestrator]
    end

    subgraph LangGraph Agentic Pipeline
        Graph --> Planner[1. 🧠 Planner Node]
        Planner -->|Conversational| Responder[3. ✍️ Responder Node]
        Planner -->|Technical Search Query| Retriever[2. 🔍 Retriever Node]
        
        Retriever -->|Query Vector| Qdrant[(🗄️ Qdrant Vector DB)]
        Qdrant -->|Top-15 Chunks| JinaRerank[⚖️ Jina Reranker API]
        JinaRerank -->|Top-5 Relevant Chunks| Responder
        
        Responder --> Portkey[🌐 Portkey LLM Gateway / Groq]
    end

    Portkey -->|Synthesizer Response| API
    DirectResponse --> API
    API -->|JSON Answer + Sources + Thoughts| User

    subgraph Persistence & Observability
        Graph -.-> Neon[(🐘 Neon Postgres DB)]
        Graph -.-> Upstash[(⚡ Upstash Redis Cache)]
        API -.-> Logfire[🔍 Pydantic Logfire Tracing]
    end
```

---

## 📁 Repository Directory Structure

```text
Enterprise-Grade-RAG-Application/
├── .github/
│   └── workflows/
│       └── ci.yml                 # GitHub Actions CI pipeline (lint, test, docker build)
├── app/
│   ├── agents/
│   │   ├── nodes/                 # LangGraph nodes (planner, retriever, responder)
│   │   ├── graph.py               # StateGraph compilation with MemorySaver
│   │   └── state.py               # TypedDict AgentState definition
│   ├── config.py                  # Dynamic environment-driven Settings loader
│   ├── gateway/                   # Portkey LLM Gateway client & fallbacks
│   ├── guardrails/
│   │   ├── colang_rules.py        # Intent rules & system prompts
│   │   └── rails.py               # Pydantic structured output guardrail validator
│   ├── injection/
│   │   ├── chunking/              # Token & character text splitters
│   │   └── processor.py           # Universal ingestion CLI processor
│   ├── services/
│   │   ├── cache/                 # Upstash Redis REST service
│   │   ├── database/              # Neon PostgreSQL DB service
│   │   └── retrieval/             # Qdrant search & Jina embedding/reranker services
│   └── main.py                    # FastAPI application & REST endpoints
├── DATA/                          # Raw enterprise documents (PDF, HTML, TXT)
├── evals/
│   ├── golden_dataset.json        # 15 RAG + 6 guardrail golden samples
│   ├── offline_gate.py            # Static dataset-integrity CI gate
│   ├── ci_ragas_gate.py           # Live RAGAS CI gate (in-process agent)
│   ├── pipeline.py                # Phase 1: live /query response capture
│   ├── metrics.py                 # Phase 2: RAGAS + tool-correctness metrics
│   ├── guardrails_eval.py         # Guardrail precision / recall
│   ├── app.py                     # Streamlit evaluation UI
│   └── EVAL_RESULTS.md            # Evaluation docs & latest results
├── tests/
│   └── test_app.py                # Automated pytest suite
├── ui/
│   ├── .streamlit/                # Production Streamlit configuration
│   ├── app.py                     # Streamlit web interface with auto backend wake-up
│   ├── errors.py                  # Custom UI exception classes
│   └── Dockerfile                 # UI Docker container definition
├── Dockerfile                     # Backend FastAPI Docker container definition
├── docker-compose.yml             # Local multi-container orchestration
├── requirements-prod.txt          # Production dependency manifest
└── README.md
```

---

## 🛠️ Technology Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Framework** | FastAPI / Python 3.12 | Asynchronous high-performance REST API |
| **Orchestration** | LangGraph / LangChain | Stateful graph agent with memory checkpointer |
| **Vector DB** | Qdrant Cloud | Managed vector database with Cosine distance |
| **Embeddings** | Jina AI (`jina-embeddings-v3`) | Hosted 1024-dimensional dense embeddings |
| **Reranker** | Jina AI (`jina-reranker-v2`) | Cross-encoder semantic reranking model |
| **LLM Engine** | Groq / Portkey AI | Llama-3.3-70b & Llama-3.1-8b via Portkey proxy |
| **Database** | Neon Postgres | Serverless PostgreSQL for session logging |
| **Caching** | Upstash Redis | REST-based key-value response caching |
| **Observability** | Logfire & LangSmith | Distributed tracing and performance monitoring |
| **UI** | Streamlit | Responsive web UI with streaming response simulation |
| **CI/CD** | GitHub Actions & Render | Automated linting, testing, Docker builds, and deployment |

---

## ⚙️ Environment Variables Setup

Create a `.env` file in the root directory:

```env
# Vector Database
QDRANT_API_KEY="your_qdrant_api_key"
QDRANT_URL="https://your-cluster.qdrant.tech"
QDRANT_COLLECTION_NAME="Rag-Application"

# Jina AI (Embedding & Reranker)
JINA_API_KEY="your_jina_api_key"
JINA_EMBEDDING_MODEL="jina-embeddings-v3"
JINA_EMBEDDING_DIM="1024"
JINA_RERANK_MODEL="jina-reranker-v2-base-multilingual"

# Groq LLMs
GROQ_API_KEY="your_groq_api_key"
GROQ_MODEL="llama-3.3-70b-versatile"
GROQ_SLUG="your_portkey_groq_slug"

# Portkey Gateway
PORTKEY_API_KEY="your_portkey_api_key"
PORTKEY_CONFIG_ID="your_portkey_config_id"

# Persistence & Cache
NEON_DB_URL="postgresql://user:password@ep-host.aws.neon.tech/neondb?sslmode=require"
UPSTASH_REDIS_REST_URL="https://your-redis.upstash.io"
UPSTASH_REDIS_REST_TOKEN="your_upstash_token"

# Observability
LOGFIRE_TOKEN="your_logfire_token"
LANGSMITH_TRACING="true"
LANGSMITH_API_KEY="your_langsmith_api_key"

# Frontend / UI
BACKEND_URL="http://localhost:8080"
```

---

## 🚀 Quickstart Guide

### 1. Local Environment Setup

```bash
# Clone repository
git clone https://github.com/kaushal320/Enterprise-Grade-RAG-Application.git
cd Enterprise-Grade-RAG-Application

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install production dependencies
pip install -r requirements-prod.txt
```

### 2. Ingest Documents into Vector DB

Run the universal ingestion pipeline to wipe, parse, chunk, embed, and index all documents in `DATA/` to Qdrant:

```bash
python -m app.injection.processor DATA --wipe
```

### 3. Run FastAPI Backend

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

Interactive API documentation available at: `http://localhost:8080/docs`

### 4. Run Streamlit UI

```bash
streamlit run ui/app.py
```

---

## 🐳 Docker Deployment

### Run Locally with Docker Compose

```bash
docker compose up --build
```

- **Backend API**: `http://localhost:8080`
- **Streamlit UI**: `http://localhost:8501`

### Single Container Build

```bash
# Build & Run Backend
docker build -t enterprise-rag-api .
docker run -p 8080:8080 --env-file .env enterprise-rag-api

# Build & Run UI
docker build -t enterprise-rag-ui -f ui/Dockerfile .
docker run -p 8501:8501 --env-file .env enterprise-rag-ui
```

---

## 🧪 Testing & CI/CD Pipeline

### Run Unit Tests Locally

```bash
pytest tests/ -v
```

### Run the Quality Gates Locally

```bash
# Static dataset-integrity gate (offline, no secrets)
python evals/offline_gate.py --golden evals/golden_dataset.json

# Live RAGAS gate — in-process agent, requires live secrets (GROQ / QDRANT / JINA)
python evals/ci_ragas_gate.py --golden evals/golden_dataset.json
```

### Automated GitHub Actions Workflow

The CI workflow (`.github/workflows/ci.yml`) executes on every push/PR to `main`:
1. **`lint-and-syntax`**: Validates code quality using `ruff`.
2. **`unit-tests`**: Runs the test suite against mock environment settings.
3. **`eval-gate`**: Statically validates `evals/golden_dataset.json` (offline, no secrets).
4. **`ragas-eval-gate`**: Runs the compiled LangGraph agent in-process on the first 3 golden samples and scores **RAGAS Faithfulness + Answer Relevancy** — fails the build if either average is below **0.6**. Requires live secrets (`GROQ_API_KEY`, `JUDGE_GROQ_KEY`, `QDRANT_API_KEY`, `QDRANT_URL`, `QDRANT_COLLECTION_NAME`, `JINA_API_KEY`, `PORTKEY_API_KEY`).
5. **`docker-build-check`**: Validates backend and UI Docker builds (depends on all four jobs above).

Full metric definitions, methodology, and the latest results are documented in
[`evals/EVAL_RESULTS.md`](evals/EVAL_RESULTS.md).

---

## 📡 API Endpoints Summary

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Health check endpoint |
| `GET` | `/graph` | Returns the visual Mermaid diagram of the LangGraph workflow |
| `POST` | `/query` | Executes the RAG flow with session memory (`q`, `thread_id`) |

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.