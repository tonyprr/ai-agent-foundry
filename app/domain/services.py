import uuid
import logging
from typing import Optional
from app.ports.inputs import RAGUseCasePort
from app.ports.outputs import AgentPort, SessionStorePort
from app.domain.models import RAGQueryRequest, RAGQueryResponse, SearchConfigOverride
from app.config import Settings

logger = logging.getLogger(__name__)

class RAGDomainService(RAGUseCasePort):
    """
    Core Domain Service implementing the RAGUseCasePort.
    Orchestrates RAG workflows by coordinating between ports.
    """
    def __init__(
        self,
        settings: Settings,
        agent_port: AgentPort,
        session_store_port: SessionStorePort
    ):
        self._settings = settings
        self._agent_port = agent_port
        self._session_store_port = session_store_port

    async def process_query(self, request: RAGQueryRequest) -> RAGQueryResponse:
        """
        Main domain business logic flow:
        1. Resolves/generates conversation thread ID.
        2. Retrieves session context.
        3. Merges default search configurations with request-level overrides.
        4. Triggers the MAF Agent query.
        5. Saves updated session state.
        6. Returns RAG response models.
        """
        # Resolve or generate a new unique conversation thread ID
        thread_id = request.thread_id or f"thread_{uuid.uuid4().hex[:12]}"
        logger.info(f"Processing query for thread '{thread_id}'")

        # 1. Fetch or initialize the conversation session
        session = await self._session_store_port.get_or_create_session(thread_id)

        # 2. Build configuration with dynamic per-query overrides
        search_config = self._build_search_config(request.search_overrides)

        # 3. Call the agent output port asynchronously
        response_text = await self._agent_port.run_agent(
            message=request.message,
            session=session,
            search_config=search_config
        )

        # 4. Save back the updated session context (persisting memory state)
        await self._session_store_port.save_session(session)

        # 5. Build response metadata showing the configuration details applied
        metadata = {
            "search_endpoint": search_config.endpoint,
            "search_index": search_config.index_name,
            "search_mode": search_config.mode,
            "search_top_k": search_config.top_k,
            "mock_mode": self._settings.mock_mode
        }

        return RAGQueryResponse(
            response_text=response_text,
            thread_id=thread_id,
            metadata=metadata
        )

    def _build_search_config(self, overrides: Optional[SearchConfigOverride]) -> SearchConfigOverride:
        """
        Merges global defaults from config with dynamic request-level overrides.
        """
        # Base config from settings
        default_config = SearchConfigOverride(
            endpoint=self._settings.azure_search_endpoint,
            index_name=self._settings.azure_search_index_name,
            api_key=self._settings.azure_search_api_key,
            mode=self._settings.azure_search_mode,
            top_k=self._settings.azure_search_top_k,
            vector_field_name=self._settings.azure_search_vector_field,
            semantic_configuration_name=self._settings.azure_search_semantic_config,
            model=self._settings.azure_ai_model_deployment_name,
            azure_openai_resource_url=self._settings.azure_openai_resource_url,
            azure_openai_api_key=self._settings.azure_openai_api_key
        )

        if not overrides:
            return default_config

        # Merge defaults and overrides
        merged_dict = default_config.model_dump()
        override_dict = overrides.model_dump(exclude_unset=True)
        merged_dict.update(override_dict)

        return SearchConfigOverride(**merged_dict)
