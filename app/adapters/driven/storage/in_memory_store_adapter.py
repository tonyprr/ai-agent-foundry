import logging
from typing import Dict, Any
from agent_framework import AgentSession
from app.ports.outputs import SessionStorePort

logger = logging.getLogger(__name__)

class InMemorySessionStoreAdapter(SessionStorePort):
    """
    Driven Adapter for persisting Agent Sessions in memory.
    Implements SessionStorePort with asynchronous serialization.
    """
    def __init__(self):
        # Maps session_id (thread_id) to the serialized dictionary representation
        self._store: Dict[str, Dict[str, Any]] = {}

    async def get_or_create_session(self, thread_id: str) -> AgentSession:
        """
        Retrieves an existing session from memory or creates a new one if not found.
        """
        if thread_id in self._store:
            logger.info(f"Retrieving active conversation session: {thread_id}")
            serialized_session = self._store[thread_id]
            # Deserialize using MAF's built-in from_dict
            return AgentSession.from_dict(serialized_session)
        
        logger.info(f"Creating new conversation session: {thread_id}")
        # Create a new session using MAF AgentSession
        new_session = AgentSession(session_id=thread_id)
        # Store initial state
        self._store[thread_id] = new_session.to_dict()
        return new_session

    async def save_session(self, session: AgentSession) -> None:
        """
        Persists the session's state in memory by serializing it to a dict.
        """
        if not session.session_id:
            logger.warning("Attempted to save a session without a session_id.")
            return

        logger.info(f"Saving conversation session state: {session.session_id}")
        # Serialize using MAF's built-in to_dict
        self._store[session.session_id] = session.to_dict()
