import pytest
import json
from unittest.mock import AsyncMock, patch, MagicMock
from agent_framework import AgentSession, Message
from agent_framework._compaction import (
    SlidingWindowStrategy,
    TruncationStrategy,
    annotate_message_groups,
    project_included_messages,
    EXCLUDED_KEY
)

from app.config import Settings
from app.adapters.driven.storage.cosmos_store_adapter import CosmosDBSessionStoreAdapter
from app.adapters.driven.storage.redis_store_adapter import RedisSessionStoreAdapter

@pytest.mark.asyncio
async def test_cosmos_adapter_validation():
    """
    Verifies that CosmosDBSessionStoreAdapter raises ValueError when credentials are missing.
    """
    settings = Settings(
        cosmos_endpoint=None,
        cosmos_key=None,
        session_store_type="cosmos"
    )
    with pytest.raises(ValueError, match="Cosmos DB endpoint and key must be fully configured."):
        CosmosDBSessionStoreAdapter(settings=settings)

@pytest.mark.asyncio
async def test_cosmos_adapter_operations():
    """
    Verifies that CosmosDBSessionStoreAdapter correctly retrieves and saves sessions via Cosmos Client mocks.
    """
    settings = Settings(
        cosmos_endpoint="https://mock-endpoint.documents.azure.com:443/",
        cosmos_key="mock-key",
        session_store_type="cosmos"
    )
    
    adapter = CosmosDBSessionStoreAdapter(settings=settings)
    
    # Set up mock container and client
    mock_container = AsyncMock()
    mock_db = AsyncMock()
    mock_client = AsyncMock()
    
    mock_db.create_container_if_not_exists.return_value = mock_container
    mock_client.create_database_if_not_exists.return_value = mock_db
    
    adapter._container = mock_container
    adapter._client = mock_client
    
    thread_id = "cosmos_test_thread"
    session_data = {"session_id": thread_id, "state": {"test_key": "test_value"}}
    
    # 1. Mock get_or_create_session (Found case)
    mock_container.read_item.return_value = session_data
    session = await adapter.get_or_create_session(thread_id)
    assert session.session_id == thread_id
    assert session.state["test_key"] == "test_value"
    mock_container.read_item.assert_called_once_with(item=thread_id, partition_key=thread_id)
    
    # 2. Mock save_session
    await adapter.save_session(session)
    mock_container.upsert_item.assert_called_once()
    upserted_body = mock_container.upsert_item.call_args[1]["body"]
    assert upserted_body["id"] == thread_id
    assert upserted_body["state"]["test_key"] == "test_value"

@pytest.mark.asyncio
async def test_redis_adapter_validation():
    """
    Verifies that RedisSessionStoreAdapter raises ValueError when URL is missing.
    """
    settings = Settings(
        redis_url="",
        session_store_type="redis"
    )
    with pytest.raises(ValueError, match="Redis URL must be configured."):
        RedisSessionStoreAdapter(settings=settings)

@pytest.mark.asyncio
async def test_redis_adapter_operations():
    """
    Verifies that RedisSessionStoreAdapter correctly retrieves and saves sessions via Redis Client mocks.
    """
    settings = Settings(
        redis_url="redis://localhost:6379/0",
        session_store_type="redis"
    )
    
    adapter = RedisSessionStoreAdapter(settings=settings)
    
    # Set up mock Redis client
    mock_client = AsyncMock()
    adapter._client = mock_client
    
    thread_id = "redis_test_thread"
    session_data = {"session_id": thread_id, "state": {"test_key": "test_value"}}
    
    # 1. Mock get_or_create_session (Found case)
    mock_client.get.return_value = json.dumps(session_data)
    session = await adapter.get_or_create_session(thread_id)
    assert session.session_id == thread_id
    assert session.state["test_key"] == "test_value"
    mock_client.get.assert_called_once_with(f"session:{thread_id}")
    
    # 2. Mock save_session
    await adapter.save_session(session)
    mock_client.set.assert_called_once()
    set_key = mock_client.set.call_args[0][0]
    set_value = mock_client.set.call_args[0][1]
    assert set_key == f"session:{thread_id}"
    assert "test_value" in set_value


@pytest.mark.asyncio
async def test_sliding_window_compaction_strategy():
    """
    Verifies that SlidingWindowStrategy correctly flags older message groups for exclusion
    while preserving recent messages and system prompt instructions.
    """
    messages = [
        Message(role="system", contents=["System directive"]),
        Message(role="user", contents=["Message 1"]),
        Message(role="assistant", contents=["Response 1"]),
        Message(role="user", contents=["Message 2"]),
        Message(role="assistant", contents=["Response 2"]),
    ]
    
    # 1. Annotate message groups (required for compaction strategy)
    annotate_message_groups(messages)
    
    # 2. Configure sliding window to keep only the last 2 non-system groups (Message 2 & Response 2)
    strategy = SlidingWindowStrategy(keep_last_groups=2, preserve_system=True)
    changed = await strategy(messages)
    
    assert changed is True
    
    # 3. Project included messages
    included = project_included_messages(messages)
    
    # We should keep system prompt, plus the last 2 messages (Message 2 & Response 2)
    assert len(included) == 3
    assert included[0].contents[0].text == "System directive"
    assert included[1].contents[0].text == "Message 2"
    assert included[2].contents[0].text == "Response 2"
    
    # Verify the excluded flags
    assert messages[1].additional_properties.get("_excluded") is True
    assert messages[2].additional_properties.get("_excluded") is True
    assert messages[3].additional_properties.get("_excluded") is False
    assert messages[4].additional_properties.get("_excluded") is False

@pytest.mark.asyncio
async def test_truncation_compaction_strategy():
    """
    Verifies that TruncationStrategy excludes messages oldest-first when exceeding maximum limit.
    """
    messages = [
        Message(role="system", contents=["System directive"]),
        Message(role="user", contents=["Message 1"]),
        Message(role="assistant", contents=["Response 1"]),
        Message(role="user", contents=["Message 2"]),
        Message(role="assistant", contents=["Response 2"]),
    ]
    
    # Annotate message groups
    annotate_message_groups(messages)
    
    # Set limit to 4 messages maximum (system + 3 messages). Truncate to 3 messages.
    strategy = TruncationStrategy(max_n=4, compact_to=3, preserve_system=True)
    changed = await strategy(messages)
    
    assert changed is True
    
    # Verify that the oldest non-system messages were truncated first
    included = project_included_messages(messages)
    assert len(included) == 3
    assert included[0].contents[0].text == "System directive"
    assert included[1].contents[0].text == "Message 2"
    assert included[2].contents[0].text == "Response 2"
