import logging
import json
from typing import Dict, Any, Optional
import redis.asyncio as aioredis

from agent_framework import AgentSession
from app.ports.outputs import SessionStorePort
from app.config import Settings

logger = logging.getLogger(__name__)

class RedisSessionStoreAdapter(SessionStorePort):
    """
    Driven Adapter for Redis to store conversation session states.
    Uses JSON serialization and configurable TTLs.
    Falls back to In-Memory mode if connection fails or url is not specified.
    """
    def __init__(self, settings: Settings):
        self._settings = settings
        self._client: Optional[aioredis.Redis] = None
        self._fallback_store: Dict[str, Dict[str, Any]] = {}
        
        if not settings.redis_url:
            logger.warning("Redis URL not configured. Using In-Memory fallback store.")
            self._use_fallback = True
        else:
            self._use_fallback = False

    async def _init_redis(self) -> None:
        if self._use_fallback or self._client is not None:
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
            logger.error(f"Failed to connect to Redis: {e}. Falling back to in-memory session store.")
            self._use_fallback = True

    async def get_or_create_session(self, thread_id: str) -> AgentSession:
        await self._init_redis()

        if self._use_fallback:
            if thread_id in self._fallback_store:
                logger.info(f"[Fallback] Retrieving session '{thread_id}' from memory.")
                return AgentSession.from_dict(self._fallback_store[thread_id])
            
            logger.info(f"[Fallback] Creating new session '{thread_id}' in memory.")
            new_session = AgentSession(session_id=thread_id)
            self._fallback_store[thread_id] = new_session.to_dict()
            return new_session

        try:
            key = f"session:{thread_id}"
            raw_data = await self._client.get(key)
            if raw_data:
                doc = json.loads(raw_data)
                logger.info(f"Retrieved session '{thread_id}' and data '{doc}' from Redis.")
                return AgentSession.from_dict(doc)
            
            logger.info(f"Session '{thread_id}' not found in Redis. Creating new one.")
            new_session = AgentSession(session_id=thread_id)
            await self.save_session(new_session)
            return new_session
        except Exception as e:
            logger.error(f"Redis get error for '{thread_id}': {e}. Falling back to new in-memory session.")
            return AgentSession(session_id=thread_id)

    async def save_session(self, session: AgentSession) -> None:
        if not session.session_id:
            logger.warning("Attempted to save a session without a session_id.")
            return

        await self._init_redis()
        doc_data = session.to_dict()

        if self._use_fallback:
            logger.info(f"[Fallback] Saving session '{session.session_id}' in memory.")
            self._fallback_store[session.session_id] = doc_data
            return

        try:
            key = f"session:{session.session_id}"
            raw_data = json.dumps(doc_data)
            # Save string payload with key-specific Time To Live (TTL)
            await self._client.set(key, raw_data, ex=self._settings.redis_ttl_seconds)
            logger.info(f"Successfully saved session '{session.session_id}' to Redis (TTL {self._settings.redis_ttl_seconds}s).")
        except Exception as e:
            logger.error(f"Failed to save session '{session.session_id}' to Redis: {e}.")

    async def close(self) -> None:
        if self._client:
            logger.info("Closing Redis connection...")
            await self._client.aclose()
