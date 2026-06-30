import asyncio
import os
import logging
import uuid
from typing import Any

# Ensure mock settings are set
os.environ["SESSION_STORE_TYPE"] = "memory"
os.environ["AZURE_SEARCH_VECTOR_FIELD"] = ""
os.environ["AZURE_SEARCH_SEMANTIC_CONFIG"] = ""

from app.main import app, lifespan
from app.adapters.driven.agent.agent_adapter import AgentAdapter
from app.adapters.driven.search.ai_search_adapter import AISearchAdapter
from app.adapters.driven.agent.agents.crypto_pricing_agent import CryptoPricingAgent
from app.adapters.driven.agent.agents.openzeppelin_agent import OpenZeppelinAgent
from agent_framework import FunctionTool, ContextProvider, SessionContext, Message, Content
from agent_framework._middleware import ChatMiddlewareLayer
from agent_framework._tools import FunctionInvocationLayer
from agent_framework.observability import ChatTelemetryLayer
from agent_framework import BaseChatClient
from agent_framework._types import ChatResponse, UsageDetails

class SimpleMockContextProvider(ContextProvider):
    def __init__(self, index_name: str, top_k: int):
        super().__init__(source_id="mock_search")
        self.index_name = index_name
        self.top_k = top_k

    async def get_context(self, context: SessionContext) -> str:
        return "context"

class TestMockChatClient(FunctionInvocationLayer, ChatMiddlewareLayer, ChatTelemetryLayer, BaseChatClient):
    STORES_BY_DEFAULT = False
    def __init__(self, agent_name: str, **kwargs: Any):
        self.agent_name = agent_name
        self.model = "mock-model"
        super().__init__(**kwargs)
        
    async def _inner_get_response(self, *, messages, stream, options, **kwargs):
        return ChatResponse(messages=[Message(role="assistant", contents=["result"])], finish_reason="stop", model="mock")

AISearchAdapter.build_context_provider = lambda self: SimpleMockContextProvider("mock_index", 5)
AgentAdapter._get_chat_client = lambda self, agent_name: TestMockChatClient(agent_name)

def mock_crypto_init(self, client):
    from agent_framework import Agent
    Agent.__init__(self, id="CryptoPricingAgent", name="CryptoPricingAgent", client=client, tools=[], instructions="inst", require_per_service_call_history_persistence=True)
def mock_oz_init(self, client):
    from agent_framework import Agent
    Agent.__init__(self, id="OpenZeppelinAgent", name="OpenZeppelinAgent", client=client, tools=[], instructions="inst", require_per_service_call_history_persistence=True)
CryptoPricingAgent.__init__ = mock_crypto_init
OpenZeppelinAgent.__init__ = mock_oz_init

async def main():
    async with lifespan(app):
        rag_service = app.state.rag_service
        agent_port = rag_service._agent_port
        session = await rag_service._session_store_port.get_or_create_session("inspect_thread_1")
        workflow = agent_port._get_or_create_workflow()
        run_result = await workflow.run(message="test", checkpoint_storage=agent_port._checkpoint_storage)
        
        event = run_result[0]
        print(f"Attributes of WorkflowEvent: {dir(event)}")
        for attr in dir(event):
            if not attr.startswith("_"):
                try:
                    val = getattr(event, attr)
                    print(f"  {attr}: {type(val)} = {val}")
                except Exception as e:
                    print(f"  {attr}: ERROR: {e}")

if __name__ == "__main__":
    asyncio.run(main())
