# Hexagonal RAG Agent Demo (Microsoft Agent Framework & FastAPI)

Retrieval-Augmented Generation (RAG) system built with **FastAPI** and the **Microsoft Agent Framework (MAF)**. This project implements **Hexagonal Architecture (Ports and Adapters)**, providing a clean separation of concerns and a robust foundation for production-ready AI agents.

---

## Key Features

- **Microsoft Agent Framework (MAF)**: Utilizes MAF's native `Agent` and `FoundryChatClient` architectures for clean orchestration.
- **Hexagonal Architecture**: Isolates core business domain logic from external dependencies (FastAPI, Azure AI Search, LLM providers).
- **Asynchronous Execution**: Fully utilizes Python's `asyncio` for non-blocking I/O across search queries, agent execution, and HTTP requests.
- **Dynamic Search Overrides**: Supports overriding connection parameters (index name, endpoint, api key, search mode, top_k) dynamically on a per-request basis.
- **Conversation State Persistence**: Manages session history threads asynchronously through a serialized in-memory adapter (designed to be easily swapped for Azure Cosmos DB or Redis).
- **Offline Mock / Developer Mode**: Includes a local mock execution strategy allowing you to test the API endpoints locally without active Azure credentials.

---

## Architecture Blueprint

```
                                 +---------------------------------+
                                 |         DRIVING ADAPTER         |
                                 |     (FastAPI HTTP Endpoints)    |
                                 +----------------+----------------+
                                                  |
                                                  | (Uses)
                                                  v
                                 +----------------+----------------+
                                 |           INPUT PORT            |
                                 |        (RAGUseCasePort)         |
                                 +----------------+----------------+
                                                  |
                                                  | (Implemented by)
                                                  v
                                 +----------------+----------------+
                                 |          CORE DOMAIN            |
                                 |       (RAGDomainService)        |
                                 +---+------------+------------+---+
                                     |            |            |
                           (Uses)    v   (Uses)   v   (Uses)   v
                     +---------------+--+  +------+------+  +--+---------------+
                     |   OUTPUT PORT    |  | OUTPUT PORT |  |   OUTPUT PORT    |
                     |   (AgentPort)    |  | (SearchPort)|  | (SessionStorePort|
                     +-------+----------+  +------+------+  +--+---------------+
                             |                    |            |
             (Implemented by)|    (Implemented by)|            | (Implemented by)
                             v                    v            v
                     +-------+----------+  +------+------+  +--+---------------+
                     |  DRIVEN ADAPTER  |  |DRIVEN ADAPTR|  |  DRIVEN ADAPTERS |
                     |  (AgentAdapter)  |  |(AISearchAdpt|  | (Memory / Redis /|
                     |  [MAF Workflow]  |  |[AzureSearch]|  |    Cosmos DB)    |
                     +-------+----------+  +-------------+  +------------------+
                             |
                             | (Orchestrates Multi-Agent System)
                             v
       +---------------------+---------------------+---------------------+
       |                     |                     |                     |
       v                     v                     v                     v
+------+------+       +------+------+       +------+------+       +------+------+
| TriageAgent |       |RAGSearchAgent       |CryptoPricing|       |OpenZeppelin |
|             | ----> |             | ----> |    Agent    | ----> |    Agent    |
+-------------+       +-------------+       +-------------+       +-------------+
```

---

## Directory Structure

```
app/
├── domain/
│   ├── models.py       # Core Domain entities & Request/Response schemas (Pydantic v2)
│   └── services.py     # Domain Service orchestrating session, search config, and agent runs
├── ports/
│   ├── inputs.py       # Input Ports (Use cases interfaces)
│   └── outputs.py      # Output Ports (Driven adapters interfaces)
├── adapters/
│   ├── driving/
│   │   └── fastapi_api.py # Driving Adapter (FastAPI controllers/endpoints)
│   └── driven/
│       ├── agent/
│       │   └── agent_adapter.py # Driven Adapter implementing AgentPort with MAF
│       ├── search/
│       │   └── ai_search_adapter.py # Driven Adapter implementing SearchPort for Azure AI Search
│       └── storage/
│           ├── cosmos_store_adapter.py # Driven Adapter for Cosmos DB
│           ├── redis_store_adapter.py  # Driven Adapter for Redis
│           └── in_memory_store_adapter.py # Driven Adapter for in-memory session persistence
├── config.py           # Application settings (Pydantic BaseSettings)
└── main.py             # Composition Root (Dependency Injection setup & FastAPI initialization)
```

---

## Setup & Running

### 1. Prerequisites
Ensure you have **Python >=3.10** installed. In this workspace, a virtual environment `.venv` has already been configured.

### 2. Configure Environment Variables
Copy the `.env.example` to `.env` (already prepared) and fill in your Azure settings:
```ini
# Toggle mock mode (set to False to connect to real Azure resources)
MOCK_MODE=True

# Azure AI Foundry Configuration
AZURE_AI_FOUNDRY_ENDPOINT=https://your-foundry-project.services.ai.azure.com
AZURE_AI_MODEL_DEPLOYMENT_NAME=gpt-4.1-mini-demo

# Azure AI Search Configuration
AZURE_SEARCH_ENDPOINT=https://your-search-service.search.windows.net
AZURE_SEARCH_INDEX_NAME=your-index-name
AZURE_SEARCH_API_KEY=your-search-api-key
AZURE_SEARCH_MODE=semantic
AZURE_SEARCH_TOP_K=5
```

### 3. Run the FastAPI Application
Start the FastAPI server using Uvicorn:
```bash
.venv/bin/uvicorn app.main:app --reload
```
The application will start on `http://127.0.0.1:8000`. You can access the Swagger UI documentation at `http://127.0.0.1:8000/docs`.

---

## Dynamic Per-Request Configurations

You can interact with the RAG agent by making a `POST` request to `/api/v1/chat`. The index connection, retrieval mode, and search limits can be configured dynamically inside `search_overrides`:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is Bitcoin?",
    "thread_id": "session-123-abc"
  }'
```

---

## Running Verification Tests

Run the test suite using pytest to verify config loading, routing, session persistence, and adapter wiring:
```bash
.venv/bin/pytest tests/
```
