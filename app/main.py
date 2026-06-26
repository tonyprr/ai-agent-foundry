import logging
import sys
from fastapi import FastAPI
from contextlib import asynccontextmanager

from app.config import Settings
from app.adapters.driven.search.ai_search_adapter import AISearchAdapter
from app.adapters.driven.agent.agent_adapter import AgentAdapter
from app.adapters.driven.storage.in_memory_store_adapter import InMemorySessionStoreAdapter
from app.domain.services import RAGDomainService
from app.adapters.driving.fastapi_api import router as chat_router

# Configure logging to output to console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Asynchronous lifespan management for FastAPI.
    Sets up resources and dependency injection.
    """
    logger.info("Initializing RAG Agent application...")
    
    # 1. Load configurations
    settings = Settings()
    logger.info(f"Configuration loaded. Mock mode active: {settings.mock_mode}")
    logger.info(f"Azure AI Foundry endpoint: {settings.azure_ai_foundry_endpoint}")
    logger.info(f"Azure AI Model Deployment Name: {settings.azure_ai_model_deployment_name}")
    logger.info(f"Azure Search endpoint: {settings.azure_search_endpoint}")
    logger.info(f"Azure OpenAI resource url: {settings.azure_openai_resource_url}")

    # 2. Instantiate Driven Adapters
    search_adapter = AISearchAdapter(
        settings=settings
    )
    
    # Instantiate configured session store adapter
    if settings.session_store_type == "cosmos":
        from app.adapters.driven.storage.cosmos_store_adapter import CosmosDBSessionStoreAdapter
        session_store_adapter = CosmosDBSessionStoreAdapter(settings=settings)
    elif settings.session_store_type == "redis":
        from app.adapters.driven.storage.redis_store_adapter import RedisSessionStoreAdapter
        session_store_adapter = RedisSessionStoreAdapter(settings=settings)
    else:
        session_store_adapter = InMemorySessionStoreAdapter()
        
    agent_adapter = AgentAdapter(
        settings=settings,
        search_adapter=search_adapter,
        session_store=session_store_adapter
    )
    
    # 3. Instantiate Domain Service (Dependency Injection container)
    domain_service = RAGDomainService(
        settings=settings,
        agent_port=agent_adapter,
        session_store_port=session_store_adapter
    )

    # 4. Attach domain service to the app state to make it available to route handlers
    app.state.settings = settings
    app.state.rag_service = domain_service
    
    logger.info("Application successfully initialized and wired.")
    yield
    logger.info("Shutting down RAG Agent application...")
    if hasattr(session_store_adapter, "close"):
        await session_store_adapter.close()

# Initialize the FastAPI App with lifespan context manager
app = FastAPI(
    title="Hexagonal RAG Agent Demo (Microsoft Agent Framework)",
    version="1.0.0",
    description="A senior-level Python demo showcasing a decoupled asynchronous RAG system using MAF, FastAPI, and Hexagonal Architecture.",
    lifespan=lifespan
)

# Register driving adapter routers
app.include_router(chat_router)

@app.get("/", tags=["Health"])
async def root():
    """
    Simple health check and index page.
    """
    return {
        "status": "healthy",
        "service": "Hexagonal RAG Agent API",
        "framework": "Microsoft Agent Framework (MAF)",
        "docs_url": "/docs"
    }
