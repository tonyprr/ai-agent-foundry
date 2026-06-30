import logging
import json
from typing import Optional
import redis.asyncio as aioredis

from agent_framework import AgentSession
from app.ports.outputs import SessionStorePort
from app.config import Settings

logger = logging.getLogger(__name__)

class RedisSessionStoreAdapter(SessionStorePort):
    """
    Driven Adapter for Redis to store conversation session states.
    Uses JSON serialization and configurable TTLs.
    """
    def __init__(self, settings: Settings):
        self._settings = settings
        self._client: Optional[aioredis.Redis] = None
        
        if not settings.redis_url:
            raise ValueError("Redis URL must be configured.")

    async def _init_redis(self) -> None:
        if self._client is not None:
            return

        try:
            logger.info(f"Connecting to Redis at {self._settings.redis_url}...")
            self._client = aioredis.from_url(
                self._settings.redis_url,
                decode_responses=True
            )
            # Ping to verify connectivity
            await self._client.ping()
            logger.info("Redis connection established successfully.")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise RuntimeError(f"Redis connection failed: {e}") from e

    async def get_or_create_session(self, thread_id: str) -> AgentSession:
        await self._init_redis()

        try:
            key = f"session:{thread_id}"
            raw_data = await self._client.get(key)
            if raw_data:
                doc = json.loads(raw_data)
                logger.info(f"Retrieved session '{thread_id}' from Redis.")
                return AgentSession.from_dict(doc)
            
            logger.info(f"Session '{thread_id}' not found in Redis. Creating new one.")
            new_session = AgentSession(session_id=thread_id)
            await self.save_session(new_session)
            return new_session
        except Exception as e:
            logger.error(f"Redis get error for '{thread_id}': {e}")
            raise RuntimeError(f"Redis retrieve operation failed: {e}") from e

    async def save_session(self, session: AgentSession) -> None:
        if not session.session_id:
            logger.warning("Attempted to save a session without a session_id.")
            return

        await self._init_redis()
        doc_data = session.to_dict()

        try:
            key = f"session:{session.session_id}"
            raw_data = json.dumps(doc_data)
            # Save string payload with key-specific Time To Live (TTL)
            await self._client.set(key, raw_data, ex=self._settings.redis_ttl_seconds)
            logger.info(f"Successfully saved session '{session.session_id}' to Redis (TTL {self._settings.redis_ttl_seconds}s).")
        except Exception as e:
            logger.error(f"Failed to save session '{session.session_id}' to Redis: {e}.")
            raise RuntimeError(f"Redis save operation failed: {e}") from e

    async def close(self) -> None:
        if self._client:
            logger.info("Closing Redis connection...")
            await self._client.aclose()
