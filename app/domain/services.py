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
        1. Validates that search_overrides are not supplied.
        2. Resolves/generates conversation thread ID.
        3. Retrieves session context.
        4. Triggers the multi-agent Handoff Orchestration.
        5. Saves updated session state.
        6. Returns RAG response models (including any pending approval request).
        """
        if request.search_overrides is not None:
            logger.warning("Rejected request containing search_overrides.")
            raise ValueError("search_overrides are not permitted in the public API.")

        # Resolve or generate a new unique conversation thread ID
        thread_id = request.thread_id or f"thread_{uuid.uuid4().hex[:12]}"
        logger.info(f"Processing query for thread '{thread_id}'")

        # 1. Fetch or initialize the conversation session
        session = await self._session_store_port.get_or_create_session(thread_id)

        # 2. Call the agent output port asynchronously
        run_result = await self._agent_port.run_agent(
            message=request.message,
            session=session
        )

        # 3. Save back the updated session context (persisting memory state)
        await self._session_store_port.save_session(session)

        # 4. Build response metadata showing the configuration details applied
        metadata = {
            "mock_mode": self._settings.mock_mode,
            "memory_compaction_strategy": self._settings.memory_compaction_strategy
        }

        return RAGQueryResponse(
            response_text=run_result.response_text,
            thread_id=thread_id,
            metadata=metadata,
            approval_request=run_result.approval_request
        )

    async def approve_request(
        self,
        thread_id: str,
        request_id: str,
        approved: bool
    ) -> RAGQueryResponse:
        """
        Resumes a paused workflow run after receiving user approval/denial for a pending tool call.
        """
        logger.info(f"Processing approval for thread '{thread_id}', request '{request_id}': approved={approved}")

        # 1. Fetch the conversation session
        session = await self._session_store_port.get_or_create_session(thread_id)

        # 2. Resume agent execution
        run_result = await self._agent_port.resume_run(
            session=session,
            request_id=request_id,
            approved=approved
        )

        # 3. Save updated session state
        await self._session_store_port.save_session(session)

        metadata = {
            "mock_mode": self._settings.mock_mode,
            "memory_compaction_strategy": self._settings.memory_compaction_strategy
        }

        return RAGQueryResponse(
            response_text=run_result.response_text,
            thread_id=thread_id,
            metadata=metadata,
            approval_request=run_result.approval_request
        )

