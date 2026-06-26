import logging
from typing import Dict, Any, Optional
from azure.cosmos.aio import CosmosClient
from azure.cosmos import PartitionKey
from azure.cosmos.exceptions import CosmosHttpResponseError

from agent_framework import AgentSession
from app.ports.outputs import SessionStorePort
from app.config import Settings

logger = logging.getLogger(__name__)

class CosmosDBSessionStoreAdapter(SessionStorePort):
    """
    Driven Adapter for Cosmos DB to store conversation session states.
    Uses point reads and upserts for low-latency operations.
    Falls back to In-Memory mode if not configured or if connection fails.
    """
    def __init__(self, settings: Settings):
        self._settings = settings
        self._client: Optional[CosmosClient] = None
        self._container = None
        self._fallback_store: Dict[str, Dict[str, Any]] = {}
        
        # Verify credentials configuration
        if not settings.cosmos_endpoint or not settings.cosmos_key:
            logger.warning("Cosmos DB endpoint/key not fully configured. Using In-Memory fallback store.")
            self._use_fallback = True
        else:
            self._use_fallback = False

    async def _init_cosmos(self) -> None:
        if self._use_fallback or self._container is not None:
            return

        try:
            logger.info("Initializing Azure Cosmos DB client...")
            self._client = CosmosClient(
                url=self._settings.cosmos_endpoint,
                credential=self._settings.cosmos_key
            )
            # Create Database and Container if they do not exist
            db = await self._client.create_database_if_not_exists(id=self._settings.cosmos_database_name)
            self._container = await db.create_container_if_not_exists(
                id=self._settings.cosmos_container_name,
                partition_key=PartitionKey(path="/id")
            )
            logger.info("Azure Cosmos DB initialization complete.")
        except Exception as e:
            logger.error(f"Failed to connect/initialize Cosmos DB: {e}. Falling back to in-memory session persistence.")
            self._use_fallback = True

    async def get_or_create_session(self, thread_id: str) -> AgentSession:
        await self._init_cosmos()

        if self._use_fallback:
            if thread_id in self._fallback_store:
                logger.info(f"[Fallback] Retrieving session '{thread_id}' from memory.")
                return AgentSession.from_dict(self._fallback_store[thread_id])
            
            logger.info(f"[Fallback] Creating new session '{thread_id}' in memory.")
            new_session = AgentSession(session_id=thread_id)
            self._fallback_store[thread_id] = new_session.to_dict()
            return new_session

        try:
            # Read item directly (point read, 1 RU cost)
            doc = await self._container.read_item(item=thread_id, partition_key=thread_id)
            logger.info(f"Retrieved session '{thread_id}' from Cosmos DB.")
            return AgentSession.from_dict(doc)
        except CosmosHttpResponseError as e:
            if e.status_code == 404:
                logger.info(f"Session '{thread_id}' not found in Cosmos DB. Creating new one.")
                new_session = AgentSession(session_id=thread_id)
                # Save it immediately to Cosmos
                await self.save_session(new_session)
                return new_session
            else:
                logger.error(f"Cosmos DB read error for '{thread_id}': {e}. Falling back to new in-memory session.")
                return AgentSession(session_id=thread_id)
        except Exception as e:
            logger.error(f"Unexpected error retrieving session '{thread_id}' from Cosmos DB: {e}. Falling back to new session.")
            return AgentSession(session_id=thread_id)

    async def save_session(self, session: AgentSession) -> None:
        if not session.session_id:
            logger.warning("Attempted to save a session without a session_id.")
            return

        await self._init_cosmos()
        doc_data = session.to_dict()
        doc_data["id"] = session.session_id  # Cosmos DB expects 'id' in container root

        if self._use_fallback:
            logger.info(f"[Fallback] Saving session '{session.session_id}' in memory.")
            self._fallback_store[session.session_id] = doc_data
            return

        try:
            # Upsert will create the document or overwrite it if it exists
            await self._container.upsert_item(body=doc_data)
            logger.info(f"Successfully saved session '{session.session_id}' to Cosmos DB.")
        except Exception as e:
            logger.error(f"Failed to upsert session '{session.session_id}' to Cosmos DB: {e}.")

    async def close(self) -> None:
        if self._client:
            logger.info("Closing Azure Cosmos DB client connection...")
            await self._client.close()
