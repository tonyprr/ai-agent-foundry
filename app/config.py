from typing import Optional, Literal
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import model_validator

class Settings(BaseSettings):
    """
    Application configuration loaded from environment variables or a .env file.
    Supports auto-loading with specific prefixes.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Azure AI Foundry Config
    azure_ai_foundry_endpoint: Optional[str] = None
    azure_ai_model_deployment_name: str = "gpt-4.1-mini-demo"

    # Agent Orchestration Config
    router_type: Literal["deterministic", "agentic"] = "deterministic"

    # Azure AI Search Config
    azure_search_endpoint: Optional[str] = None
    azure_search_index_name: Optional[str] = None
    azure_search_api_key: Optional[str] = None
    azure_search_mode: Literal["semantic", "agentic"] = "semantic"
    azure_search_top_k: int = 5
    azure_search_vector_field: Optional[str] = None
    azure_search_semantic_config: Optional[str] = None

    # Azure OpenAI Resource Config (needed for agentic mode search context provider)
    azure_openai_resource_url: Optional[str] = None
    azure_openai_api_key: Optional[str] = None

    # Session Persistence Settings
    session_store_type: Literal["memory", "cosmos", "redis"] = "memory"

    # Cosmos DB Settings
    cosmos_endpoint: Optional[str] = None
    cosmos_key: Optional[str] = None
    cosmos_database_name: str = "rag-agent-db"
    cosmos_container_name: str = "sessions"

    # Redis Settings
    redis_url: str = "redis://localhost:6379/0"
    redis_ttl_seconds: int = 86400  # Default 24 hours (1 day)

    # Memory Compaction / Optimization Settings
    memory_compaction_enabled: bool = False
    memory_compaction_strategy: Literal["summarization", "sliding_window", "truncation"] = "summarization"
    memory_compaction_target_count: int = 4
    memory_compaction_threshold: int = 2
