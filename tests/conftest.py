import os
import uuid
import json
import logging
import pytest
from typing import Any

# Ensure test settings are correct
os.environ["SESSION_STORE_TYPE"] = "memory"
os.environ["AZURE_SEARCH_VECTOR_FIELD"] = ""
os.environ["AZURE_SEARCH_SEMANTIC_CONFIG"] = ""

from agent_framework._middleware import ChatMiddlewareLayer
from agent_framework._tools import FunctionInvocationLayer
from agent_framework.observability import ChatTelemetryLayer
from agent_framework import BaseChatClient, ContextProvider, SessionContext
from agent_framework._types import ChatResponse, UsageDetails
from agent_framework import Message, Content

logger = logging.getLogger(__name__)

class SimpleMockContextProvider(ContextProvider):
    """
    A custom mock ContextProvider that simulates retrieval results for testing.
    """
    def __init__(self, index_name: str, top_k: int):
        super().__init__(source_id="mock_search")
        self.index_name = index_name
        self.top_k = top_k

    async def get_context(self, context: SessionContext) -> str:
        return (
            "[Source: Document_1.txt]\n"
            "Microsoft Agent Framework (MAF) is a professional, multi-agent orchestration framework. "
            "Bitcoin is a decentralized digital currency."
        )

class TestMockChatClient(FunctionInvocationLayer, ChatMiddlewareLayer, ChatTelemetryLayer, BaseChatClient):
    STORES_BY_DEFAULT = False
    
    def __init__(self, agent_name: str, **kwargs: Any):
        self.agent_name = agent_name
        self.model = "mock-model"
        super().__init__(**kwargs)
        
    async def _inner_get_response(self, *, messages, stream, options, **kwargs):
        # Find the last user message
        last_msg = ""
        for m in reversed(messages):
            if m.role == "user":
                last_msg = m.text
                break
        last_msg_lower = last_msg.lower() if last_msg else ""
        
        # Extract if there is any tool result in the messages
        tool_result = None
        for m in reversed(messages):
            if m.role == "tool" or m.role == "assistant":
                for c in m.contents:
                    if c.type == "function_result":
                        tool_result = c
                        break
                if tool_result:
                    break

        reply_contents = []
        
        if self.agent_name == "TriageAgent":
            if "solidity" in last_msg_lower or "contract" in last_msg_lower or "openzeppelin" in last_msg_lower:
                reply_contents.append("Routing to OpenZeppelinAgent...")
                reply_contents.append(Content.from_function_call(
                    name="handoff_to_OpenZeppelinAgent",
                    arguments={},
                    call_id=f"triage_{uuid.uuid4().hex[:8]}"
                ))
            elif "price" in last_msg_lower or "pricing" in last_msg_lower or "coingecko" in last_msg_lower:
                reply_contents.append("Routing to CryptoPricingAgent...")
                reply_contents.append(Content.from_function_call(
                    name="handoff_to_CryptoPricingAgent",
                    arguments={},
                    call_id=f"triage_{uuid.uuid4().hex[:8]}"
                ))
            elif "bitcoin" in last_msg_lower or "blockchain" in last_msg_lower:
                reply_contents.append("Routing to RAGSearchAgent...")
                reply_contents.append(Content.from_function_call(
                    name="handoff_to_RAGSearchAgent",
                    arguments={},
                    call_id=f"triage_{uuid.uuid4().hex[:8]}"
                ))
            else:
                reply_contents.append(
                    "Hello! I am the Triage Agent. I can route you to: RAG Search, Crypto Pricing, or OpenZeppelin Solidity agents. What do you need?"
                )
                
        elif self.agent_name == "RAGSearchAgent":
            if any(k in last_msg_lower for k in ["price", "pricing", "coingecko", "solidity", "contract", "openzeppelin"]):
                reply_contents.append("Handing off to TriageAgent...")
                reply_contents.append(Content.from_function_call(
                    name="handoff_to_TriageAgent",
                    arguments={},
                    call_id=f"rag_handoff_{uuid.uuid4().hex[:8]}"
                ))
            else:
                reply_contents.append(
                    "[Source: Document_1.txt]\n"
                    "Microsoft Agent Framework (MAF) is a professional, multi-agent orchestration framework. "
                    "Bitcoin is a decentralized digital currency."
                )
            
        elif self.agent_name == "CryptoPricingAgent":
            if not any(k in last_msg_lower for k in ["price", "pricing", "coingecko"]):
                reply_contents.append("Handing off to TriageAgent...")
                reply_contents.append(Content.from_function_call(
                    name="handoff_to_TriageAgent",
                    arguments={},
                    call_id=f"crypto_handoff_{uuid.uuid4().hex[:8]}"
                ))
            elif tool_result:
                res_val = tool_result.result if hasattr(tool_result, "result") else ""
                reply_contents.append(f"[MOCK RESPONSE] Based on the pricing service: {res_val}")
            else:
                reply_contents.append(Content.from_function_call(
                    name="get_crypto_price",
                    arguments={"coin_id": "bitcoin"},
                    call_id=f"crypto_{uuid.uuid4().hex[:8]}"
                ))
                
        elif self.agent_name == "OpenZeppelinAgent":
            if not any(k in last_msg_lower for k in ["solidity", "contract", "openzeppelin"]):
                reply_contents.append("Handing off to TriageAgent...")
                reply_contents.append(Content.from_function_call(
                    name="handoff_to_TriageAgent",
                    arguments={},
                    call_id=f"openzeppelin_handoff_{uuid.uuid4().hex[:8]}"
                ))
            elif tool_result:
                contract_code = tool_result.result if hasattr(tool_result, "result") else ""
                reply_contents.append(
                    f"Here is your Solidity contract:\n{contract_code}"
                )
            else:
                reply_contents.append(Content.from_function_call(
                    name="openzeppelin_develop_contract",
                    arguments={"contract_type": "ERC20"},
                    call_id=f"openzeppelin_{uuid.uuid4().hex[:8]}"
                ))
                
        msg_contents = []
        has_func_call = False
        for c in reply_contents:
            if isinstance(c, str):
                msg_contents.append(Content.from_text(text=c))
            else:
                msg_contents.append(c)
                if c.type == "function_call":
                    has_func_call = True
                
        response_msg = Message(role="assistant", contents=msg_contents, author_name=self.agent_name)
        
        return ChatResponse(
            messages=[response_msg],
            finish_reason="tool_calls" if has_func_call else "stop",
            model=self.model,
            usage_details=UsageDetails(input_token_count=100, output_token_count=50, total_token_count=150)
        )

@pytest.fixture(autouse=True)
def mock_agent_adapters_and_clients(monkeypatch):
    """
    Autouse fixture that monkeypatches AgentAdapter and its dependencies
    to use testing mocks and stub context providers.
    """
    from app.adapters.driven.search.ai_search_adapter import AISearchAdapter
    from app.adapters.driven.agent.agent_adapter import AgentAdapter
    from app.adapters.driven.agent.agents.crypto_pricing_agent import CryptoPricingAgent
    from app.adapters.driven.agent.agents.openzeppelin_agent import OpenZeppelinAgent
    from agent_framework import FunctionTool

    # 1. Patch search adapter to return SimpleMockContextProvider
    def mock_build(self):
        return SimpleMockContextProvider("mock_index", 5)
    monkeypatch.setattr(AISearchAdapter, "build_context_provider", mock_build)

    # 2. Patch agent adapter to return TestMockChatClient
    def mock_get_chat_client(self, agent_name: str):
        return TestMockChatClient(agent_name)
    monkeypatch.setattr(AgentAdapter, "_get_chat_client", mock_get_chat_client)

    # 3. Patch specialist agent tools with mock FunctionTools
    def crypto_get_price(coin_id: str) -> str:
        return f"The live price of {coin_id} is $65,000 USD (mocked)."

    def openzeppelin_develop_contract(contract_type: str, features: list[str] | None = None) -> str:
        feat_str = f" with features {', '.join(features)}" if features else ""
        return (
            f"// SPDX-License-Identifier: MIT\n"
            f"pragma solidity ^0.8.20;\n\n"
            f"import \"@openzeppelin/contracts/token/{contract_type}/{contract_type}.sol\";\n\n"
            f"contract Mock{contract_type} is {contract_type} {{\n"
            f"    constructor() {contract_type}(\"MockToken\", \"MTK\") {{\n"
            f"        // Generated {contract_type}{feat_str}\n"
            f"    }}\n"
            f"}}"
        )

    def mock_crypto_init(self, client):
        crypto_tool = FunctionTool(
            name="get_crypto_price",
            description="Get the live price of a cryptocurrency in USD.",
            func=crypto_get_price,
            approval_mode="never_require"
        )
        from agent_framework import Agent
        Agent.__init__(
            self,
            id="CryptoPricingAgent",
            name="CryptoPricingAgent",
            client=client,
            tools=[crypto_tool],
            instructions=(
                "You are a Crypto Pricing Agent. Use the coingecko/crypto tool to fetch live "
                "prices for requested cryptocurrencies, then report them back to the user.\n"
                "If the user's query is outside your scope, you MUST delegate/route the conversation "
                "back to the TriageAgent by calling the handoff_to_TriageAgent tool."
            ),
            require_per_service_call_history_persistence=True
        )

    def mock_oz_init(self, client):
        oz_tool = FunctionTool(
            name="openzeppelin_develop_contract",
            description="Develops Solidity smart contracts using OpenZeppelin",
            func=openzeppelin_develop_contract,
            approval_mode="always_require"
        )
        from agent_framework import Agent
        Agent.__init__(
            self,
            id="OpenZeppelinAgent",
            name="OpenZeppelinAgent",
            client=client,
            tools=[oz_tool],
            instructions=(
                "You are an OpenZeppelin Agent. Use the openzeppelin tools to develop, write, "
                "or customize Solidity contracts. Every time you invoke these tools, human-in-the-loop "
                "approval is strictly required.\n"
                "If the user's query is outside your scope, you MUST delegate/route the conversation "
                "back to the TriageAgent by calling the handoff_to_TriageAgent tool."
            ),
            require_per_service_call_history_persistence=True
        )

    monkeypatch.setattr(CryptoPricingAgent, "__init__", mock_crypto_init)
    monkeypatch.setattr(OpenZeppelinAgent, "__init__", mock_oz_init)
