# Hexagonal RAG Agent Demo (Microsoft Agent Framework & FastAPI)

Retrieval-Augmented Generation (RAG) system built with **FastAPI** and the **Microsoft Agent Framework (MAF)**. This project implements **Hexagonal Architecture (Ports and Adapters)**, providing a clean separation of concerns and a robust foundation for production-ready AI agents.

---

## Key Features

- **Microsoft Agent Framework (MAF)**: Utilizes MAF's native `Agent` and `FoundryChatClient` architectures for clean orchestration.
- **Workflow-Driven Architecture**: Leverages MAF's `WorkflowBuilder` to build a stateful agent execution graph with conditional switch-case branching and looping.
- **Dynamic Multi-Agent Routing**: Integrates a custom dynamic `Router` executor that automatically redirects tasks to one or more specialized agents (`RAGSearchAgent`, `CryptoPricingAgent`, `OpenZeppelinAgent`) based on the conversation context.
- **Multi-Intent Response Synthesis**: Employs a `SummarizerAgent` to synthesize and format findings into a clean, bulleted list when multiple specialized agents are triggered in a single turn.
- **Human-in-the-Loop (HITL) Approvals**: Implements tool invocation approvals (e.g., Solidity contract compilation in `OpenZeppelinAgent`), allowing workflows to pause execution and resume seamlessly upon receiving approval/denial callbacks.
- **Conversation State & Checkpoint Persistence**: Integrates a custom `CheckpointStorage` adapter that maps MAF session checkpoints directly into the user session state (compatible with InMemory, Redis, or Cosmos DB).
- **Resource Usage & Metric Auditing**: Intercepts workflow runs to calculate and log input/output token counts and execution duration per agent.
- **Enhanced Specialist Tooling**:
  - `RAGSearchAgent` connects to Azure AI Search via custom Context Providers.
  - `CryptoPricingAgent` integrates with CoinGecko API via MCP and calculates purchase mathematics using a native `calculate_crypto_purchase` tool.
- **Mock-Driven Test Verification**: Includes a comprehensive test suite (16 tests) with test-level mocks using `pytest` and monkeypatches, allowing execution of all flows (including single-turn, multi-intent, multi-turn, and HITL) locally without active Azure credentials.

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
                             | (Orchestrates Dynamic Orchestration Workflow)
                             v
                 +-----------------------+
                 |      TriageAgent      | <-----------------------+
                 +-----------+-----------+                         |
                             |                                     |
                             v                                     |
                 +-----------------------+                         |
          +----> |        Router         |                         |
          |      +-----------+-----------+                         |
          |                  |                                     |
          |                  | (Evaluates needed / pending)        |
          |                  v                                     |
          |         [Select Target Agent]                          |
          |          /   |     |    \    \                         |
          |         /    |     |     \    \                        |
          |        v     v     v      v    \                       |
          |    +------+ +----+ +----+ +---+ \                      |
          |    | RAG  | |Cryp| | OZ | |Sum|  \                     |
          |    |Search| |Pric| |Agent| |agt|   \                   |
          |    +--+---+ +-+--+ +-+--+ +-+--+    \ (None remaining) |
          |       |       |      |      |        v                 |
          +-------+-------+------+------+  +-----+----+            |
                                           |Finalizer |            |
                                           +-----+----+            |
                                                 | (Yields & Pauses)
                                                 v                 |
                                           [User Input] -----------+
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
│   │   └── fastapi_api.py # Driving Adapter (FastAPI controllers/endpoints for chat and approvals)
│   └── driven/
│       ├── agent/
│       │   ├── agents/          # Modular specialist agent classes
│       │   │   ├── __init__.py
│       │   │   ├── triage_agent.py          # Classifying and routing agent
│       │   │   ├── rag_search_agent.py      # RAG-based context retrieval agent
│       │   │   ├── crypto_pricing_agent.py  # CoinGecko MCP & purchasing math agent
│       │   │   ├── openzeppelin_agent.py    # Solidity smart contract coding agent (HITL-required)
│       │   │   └── summarizer_agent.py      # Compiles/synthesizes multi-agent responses
│       │   ├── agent_adapter.py    # Driven Adapter implementing AgentPort with MAF
│       │   └── workflow_support.py # CheckpointStorage, Router, and Finalizer executors
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

## Interacting with the Chat API

### 1. Standard Chat Session
You can interact with the dynamic multi-agent system by making a `POST` request to `/api/v1/chat`:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is the price of Bitcoin, and can you also explain what Bitcoin is?",
    "thread_id": "session-123-abc"
  }'
```

Because this query contains multiple intents (pricing and conceptual definition), the workflow will:
1. Route to `CryptoPricingAgent` to fetch the live price.
2. Route to `RAGSearchAgent` to fetch document definitions.
3. Route to `SummarizerAgent` to combine and format the results.
4. Output the synthesized response to the client.

### 2. Human-in-the-Loop (HITL) Approvals
If a query triggers a tool with an `always_require` approval mode (such as generating code with `OpenZeppelinAgent`), the API response will return an `approval_request` block, and the execution will pause:

**Response from `/chat`:**
```json
{
  "response_text": "I am about to invoke the openzeppelin_develop_contract tool. Do you want to proceed?",
  "thread_id": "session-123-abc",
  "metadata": {},
  "approval_request": {
    "request_id": "req-xyz-789",
    "tool_name": "openzeppelin_develop_contract",
    "arguments": {
      "contract_type": "ERC20"
    }
  }
}
```

To resume the execution, send a `POST` request to `/api/v1/chat/approve`:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/chat/approve \
  -H "Content-Type: application/json" \
  -d '{
    "thread_id": "session-123-abc",
    "request_id": "req-xyz-789",
    "approved": true
  }'
```

The server will resume the suspended workflow and return the final compiled response.

---

## Token and Performance Auditing

Each completed agent execution logs a summary of resources consumed to the application console:

```
======= Token Usage & Processing Time Summary =======
  Agent: TriageAgent -> Input: 420 | Output: 50 | Total: 470 | Duration: 0.85s
  Agent: CryptoPricingAgent -> Input: 1250 | Output: 310 | Total: 1560 | Duration: 1.45s
  Agent: RAGSearchAgent -> Input: 1980 | Output: 450 | Total: 2430 | Duration: 1.95s
  Agent: SummarizerAgent -> Input: 1540 | Output: 220 | Total: 1760 | Duration: 0.90s
  TOTAL -> Input: 5190 | Output: 1030 | Total: 6220 | Duration: 5.15s
=====================================================
```

---

## Running Verification Tests

Run the full suite using pytest to verify agent routing, multi-intent synthesis, HITL state transitions, and session persistence:
```bash
.venv/bin/pytest tests/
```
