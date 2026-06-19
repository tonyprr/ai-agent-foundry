import os
# Force mock mode for all tests
os.environ["MOCK_MODE"] = "True"
os.environ["AZURE_SEARCH_VECTOR_FIELD"] = ""
os.environ["AZURE_SEARCH_SEMANTIC_CONFIG"] = ""

import pytest
from agent_framework import AgentSession, Message
from agent_framework._compaction import (
    SlidingWindowStrategy,
    TruncationStrategy,
    annotate_message_groups,
    project_included_messages,
    EXCLUDED_KEY
)

from app.config import Settings
from app.adapters.driven.cosmos_store_adapter import CosmosDBSessionStoreAdapter
from app.adapters.driven.redis_store_adapter import RedisSessionStoreAdapter

@pytest.mark.asyncio
async def test_cosmos_adapter_fallback_mode():
    """
    Verifies that CosmosDBSessionStoreAdapter falls back gracefully to in-memory store
    when credentials are not provided, preserving session state correctly across calls.
    """
    settings = Settings(
        cosmos_endpoint=None,
        cosmos_key=None,
        session_store_type="cosmos"
    )
    
    adapter = CosmosDBSessionStoreAdapter(settings=settings)
    thread_id = "cosmos_test_thread"
    
    # 1. Retrieve or create session
    session = await adapter.get_or_create_session(thread_id)
    assert isinstance(session, AgentSession)
    assert session.session_id == thread_id
    assert "messages" not in session.state
    
    # 2. Add message to session state
    session.state["messages"] = [{"role": "user", "contents": [{"type": "text", "text": "Hello Cosmos"}]}]
    await adapter.save_session(session)
    
    # 3. Retrieve session again and verify persistence
    retrieved_session = await adapter.get_or_create_session(thread_id)
    assert retrieved_session.session_id == thread_id
    assert retrieved_session.state["messages"][0]["contents"][0]["text"] == "Hello Cosmos"
    
    await adapter.close()

@pytest.mark.asyncio
async def test_redis_adapter_fallback_mode():
    """
    Verifies that RedisSessionStoreAdapter falls back gracefully to in-memory store
    when the connection is not active or URL is omitted, preserving state.
    """
    settings = Settings(
        redis_url="",
        session_store_type="redis"
    )
    
    adapter = RedisSessionStoreAdapter(settings=settings)
    thread_id = "redis_test_thread"
    
    # 1. Retrieve or create session
    session = await adapter.get_or_create_session(thread_id)
    assert isinstance(session, AgentSession)
    assert session.session_id == thread_id
    
    # 2. Add message to session state
    session.state["messages"] = [{"role": "user", "contents": [{"type": "text", "text": "Hello Redis"}]}]
    await adapter.save_session(session)
    
    # 3. Retrieve session again and verify persistence
    retrieved_session = await adapter.get_or_create_session(thread_id)
    assert retrieved_session.session_id == thread_id
    assert retrieved_session.state["messages"][0]["contents"][0]["text"] == "Hello Redis"
    
    await adapter.close()

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
