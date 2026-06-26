import logging
from typing import Optional
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
    """
    def __init__(self, settings: Settings):
        self._settings = settings
        self._client: Optional[CosmosClient] = None
        self._container = None
        
        # Verify credentials configuration
        if not settings.cosmos_endpoint or not settings.cosmos_key:
            raise ValueError("Cosmos DB endpoint and key must be fully configured.")

    async def _init_cosmos(self) -> None:
        if self._container is not None:
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
            logger.error(f"Failed to connect/initialize Cosmos DB: {e}")
            raise RuntimeError(f"Cosmos DB connection failed: {e}") from e

    async def get_or_create_session(self, thread_id: str) -> AgentSession:
        await self._init_cosmos()

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
                logger.error(f"Cosmos DB read error for '{thread_id}': {e}")
                raise RuntimeError(f"Cosmos DB read error for '{thread_id}': {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error retrieving session '{thread_id}' from Cosmos DB: {e}")
            raise RuntimeError(f"Cosmos DB session retrieval failed: {e}") from e

    async def save_session(self, session: AgentSession) -> None:
        if not session.session_id:
            logger.warning("Attempted to save a session without a session_id.")
            return

        await self._init_cosmos()
        doc_data = session.to_dict()
        doc_data["id"] = session.session_id  # Cosmos DB expects 'id' in container root

        try:
            # Upsert will create the document or overwrite it if it exists
            await self._container.upsert_item(body=doc_data)
            logger.info(f"Successfully saved session '{session.session_id}' to Cosmos DB.")
        except Exception as e:
            logger.error(f"Failed to upsert session '{session.session_id}' to Cosmos DB: {e}.")
            raise RuntimeError(f"Cosmos DB save operation failed: {e}") from e

    async def close(self) -> None:
        if self._client:
            logger.info("Closing Azure Cosmos DB client connection...")
            await self._client.close()
