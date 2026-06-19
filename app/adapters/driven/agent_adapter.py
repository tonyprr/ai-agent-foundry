import logging
from typing import List
from agent_framework import Agent, AgentSession, ContextProvider, SessionContext
from agent_framework.foundry import FoundryChatClient
from agent_framework.openai import OpenAIChatClient

from app.ports.outputs import AgentPort
from app.domain.models import SearchConfigOverride
from app.config import Settings
from app.adapters.driven.search_adapter import SearchAdapter

logger = logging.getLogger(__name__)

class SimpleMockContextProvider(ContextProvider):
    """
    A custom mock ContextProvider that simulates retrieval results for local/offline testing.
    """
    def __init__(self, index_name: str, top_k: int):
        super().__init__(source_id="mock_search")
        self.index_name = index_name
        self.top_k = top_k

    async def get_context(self, context: SessionContext) -> str:
        # Simulate retrieved text based on index settings
        return (
            f"[Source: Document_1.txt (Mocked from index '{self.index_name}')]\n"
            "This is simulated context. Microsoft Agent Framework (MAF) is a pro-code, "
            "enterprise-grade SDK that unifies concepts from AutoGen and Semantic Kernel.\n\n"
            f"[Source: Document_2.txt (Mocked retrieval, top_k={self.top_k})]\n"
            "FastAPI is an asynchronous web framework for building APIs in Python. "
            "Hexagonal Architecture isolates business logic from external frameworks and technologies."
        )

class AgentAdapter(AgentPort):
    """
    Driven Adapter implementing the AgentPort.
    Interacts with the Microsoft Agent Framework to coordinate RAG execution.
    """
    def __init__(self, settings: Settings, search_adapter: SearchAdapter):
        self._settings = settings
        self._search_adapter = search_adapter

    async def run_agent(
        self,
        message: str,
        session: AgentSession,
        search_config: SearchConfigOverride
    ) -> str:
        """
        Coordinates the agent run, supporting both real Azure integration and a mock mode for developer testing.
        """
        if self._settings.mock_mode:
            logger.info("Executing Agent in MOCK mode.")
            return await self._run_mock_agent(message, session, search_config)
        
        logger.info("Executing Agent in PRODUCTION mode with Azure AI Foundry and Search.")
        return await self._run_production_agent(message, session, search_config)

    async def _run_production_agent(
        self,
        message: str,
        session: AgentSession,
        search_config: SearchConfigOverride
    ) -> str:
        # 1. Build the dynamic Azure AI Search Context Provider
        context_provider = self._search_adapter.build_context_provider(search_config)

        # 2. Set up Azure Credentials
        from azure.identity.aio import DefaultAzureCredential
        credential = DefaultAzureCredential()

        # 3. Initialize the Microsoft Agent Framework Foundry client
        client = FoundryChatClient(
            project_endpoint=self._settings.azure_ai_foundry_endpoint,
            model=self._settings.azure_ai_model_deployment_name,
            credential=credential
        )

        # 4. Setup context providers, including compaction if enabled
        context_providers = [context_provider]
        if self._settings.memory_compaction_enabled:
            logger.info(f"Memory compaction enabled with strategy: {self._settings.memory_compaction_strategy}")
            from agent_framework._compaction import (
                CompactionProvider,
                SummarizationStrategy,
                SlidingWindowStrategy,
                TruncationStrategy,
            )
            
            strategy = None
            if self._settings.memory_compaction_strategy == "summarization":
                strategy = SummarizationStrategy(
                    client=client,
                    target_count=self._settings.memory_compaction_target_count,
                    threshold=self._settings.memory_compaction_threshold
                )
            elif self._settings.memory_compaction_strategy == "sliding_window":
                strategy = SlidingWindowStrategy(
                    keep_last_groups=self._settings.memory_compaction_target_count
                )
            elif self._settings.memory_compaction_strategy == "truncation":
                strategy = TruncationStrategy(
                    max_n=self._settings.memory_compaction_target_count + self._settings.memory_compaction_threshold,
                    compact_to=self._settings.memory_compaction_target_count
                )
            
            if strategy:
                compaction_provider = CompactionProvider(
                    before_strategy=None,
                    after_strategy=strategy,
                    history_source_id="in_memory"
                )
                context_providers.append(compaction_provider)

        # 5. Create the Agent with context providers integrated
        agent = Agent(
            name="RAGAgent",
            client=client,
            context_providers=context_providers,
            instructions=(
                "You are an expert RAG agent. You must answer questions using only "
                "the retrieved context from Azure AI Search. Always cite your sources "
                "using the [Source: filename] format."
            )
        )

        try:
            # 5. Run the agent asynchronously on the current session thread
            response = await agent.run(message, session=session)
            return response.text
        finally:
            await credential.close()

    async def _run_mock_agent(
        self,
        message: str,
        session: AgentSession,
        search_config: SearchConfigOverride
    ) -> str:
        # Use dynamic mock provider to simulate retrieval
        index_name = search_config.index_name or self._settings.azure_search_index_name or "default_mock_index"
        top_k = search_config.top_k or self._settings.azure_search_top_k
        mock_search_provider = SimpleMockContextProvider(index_name=index_name, top_k=top_k)

        # If an OpenAI Key is configured, use OpenAIChatClient for real LLM answers over mock search context
        if self._settings.openai_api_key and self._settings.openai_api_key != "mock-key":
            try:
                client = OpenAIChatClient(
                    api_key=self._settings.openai_api_key,
                    model="gpt-4o-mini"
                )
                
                context_providers = [mock_search_provider]
                if self._settings.memory_compaction_enabled:
                    logger.info(f"Memory compaction enabled for mock agent: {self._settings.memory_compaction_strategy}")
                    from agent_framework._compaction import (
                        CompactionProvider,
                        SummarizationStrategy,
                        SlidingWindowStrategy,
                        TruncationStrategy,
                    )
                    
                    strategy = None
                    if self._settings.memory_compaction_strategy == "summarization":
                        strategy = SummarizationStrategy(
                            client=client,
                            target_count=self._settings.memory_compaction_target_count,
                            threshold=self._settings.memory_compaction_threshold
                        )
                    elif self._settings.memory_compaction_strategy == "sliding_window":
                        strategy = SlidingWindowStrategy(
                            keep_last_groups=self._settings.memory_compaction_target_count
                        )
                    elif self._settings.memory_compaction_strategy == "truncation":
                        strategy = TruncationStrategy(
                            max_n=self._settings.memory_compaction_target_count + self._settings.memory_compaction_threshold,
                            compact_to=self._settings.memory_compaction_target_count
                        )
                    
                    if strategy:
                        compaction_provider = CompactionProvider(
                            before_strategy=None,
                            after_strategy=strategy,
                            history_source_id="in_memory"
                        )
                        context_providers.append(compaction_provider)

                agent = Agent(
                    name="MockRAGAgent",
                    client=client,
                    context_providers=context_providers,
                    instructions="Answer the user's questions based on the retrieved context. Cite mock sources."
                )
                response = await agent.run(message, session=session)
                return response.text
            except Exception as e:
                logger.warning(f"Failed to use OpenAI Chat Client for mock execution: {e}. Falling back to simulation.")

        # Static text simulation fallback (fully offline, no credentials required)
        simulated_response = (
            f"[MOCK RESPONSE]\n"
            f"You asked: '{message}'\n"
            f"Retrieved context simulated from index '{index_name}' (top_k={top_k}):\n"
            f"  - MAF unifies AutoGen & Semantic Kernel.\n"
            f"  - Hexagonal Architecture decouples core logic from FastAPI.\n"
            f"Session ID in use: {session.session_id}"
        )
        return simulated_response
