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
        
        # Find all assistant function calls and their results
        calls = {}  # call_id -> name
        results = {}  # call_id -> result_text
        for m in messages:
            if m.role == "assistant":
                for c in m.contents:
                    if c.type == "function_call":
                        calls[c.call_id] = c.name
            elif m.role == "tool":
                for c in m.contents:
                    if c.type == "function_result":
                        results[c.call_id] = c.result if hasattr(c, "result") else str(c)
            elif m.role == "user":
                for c in m.contents:
                    if c.type == "function_approval_response":
                        if c.approved:
                            results[c.call_id] = "pragma solidity ^0.8.0;" # Dummy result for approval

        has_price_result = any(calls.get(cid) == "get_crypto_price" for cid in results)
        has_calc_result = any(calls.get(cid) == "calculate_crypto_purchase" for cid in results)

        reply_contents = []
        
        if self.agent_name == "TriageAgent":
            needed = []
            is_purchase_calc = any(k in last_msg_lower for k in ("buy", "calculate", "purchase", "how many"))
            if "solidity" in last_msg_lower or "contract" in last_msg_lower or "openzeppelin" in last_msg_lower:
                needed.append("OpenZeppelinAgent")
            if "price" in last_msg_lower or "pricing" in last_msg_lower or "coingecko" in last_msg_lower or is_purchase_calc:
                needed.append("CryptoPricingAgent")
            if "bitcoin" in last_msg_lower or "blockchain" in last_msg_lower or "search" in last_msg_lower:
                if not is_purchase_calc and ("search" in last_msg_lower or "blockchain" in last_msg_lower or "document" in last_msg_lower or "explain" in last_msg_lower or "concept" in last_msg_lower or "what is" in last_msg_lower or "price" not in last_msg_lower):
                    needed.append("RAGSearchAgent")
            
            if needed:
                reply_contents.append(f"Routing to: {', '.join(needed)}")
            else:
                reply_contents.append(
                    "Hello! I am the Triage Agent. I can route you to: RAG Search, Crypto Pricing, or OpenZeppelin Solidity agents. What do you need?"
                )
                
        elif self.agent_name == "RAGSearchAgent":
            reply_contents.append(
                "[Source: Document_1.txt]\n"
                "Microsoft Agent Framework (MAF) is a professional, multi-agent orchestration framework. "
                "Bitcoin is a decentralized digital currency."
            )
            
        elif self.agent_name == "CryptoPricingAgent":
            is_calculation = "buy" in last_msg_lower or "calculate" in last_msg_lower or "purchase" in last_msg_lower or "how many" in last_msg_lower
            if is_calculation:
                if has_calc_result:
                    calc_val = next(results[cid] for cid in results if calls.get(cid) == "calculate_crypto_purchase")
                    reply_contents.append(f"[MOCK RESPONSE] Purchase calculation complete: {calc_val}")
                elif has_price_result:
                    usd_val = 500.0
                    import re
                    match = re.search(r'\$?(\d+(?:\.\d+)?)', last_msg_lower)
                    if match:
                        usd_val = float(match.group(1))
                    reply_contents.append(Content.from_function_call(
                        name="calculate_crypto_purchase",
                        arguments={"usd_amount": usd_val, "crypto_price": 65000.0},
                        call_id=f"calc_{uuid.uuid4().hex[:8]}"
                    ))
                else:
                    reply_contents.append(Content.from_function_call(
                        name="get_crypto_price",
                        arguments={"coin_id": "bitcoin"},
                        call_id=f"crypto_{uuid.uuid4().hex[:8]}"
                    ))
            else:
                if has_price_result:
                    price_val = next(results[cid] for cid in results if calls.get(cid) == "get_crypto_price")
                    reply_contents.append(f"[MOCK RESPONSE] Based on the pricing service: {price_val}")
                else:
                    reply_contents.append(Content.from_function_call(
                        name="get_crypto_price",
                        arguments={"coin_id": "bitcoin"},
                        call_id=f"crypto_{uuid.uuid4().hex[:8]}"
                    ))
                
        elif self.agent_name == "OpenZeppelinAgent":
            has_oz_result = any(calls.get(cid) == "openzeppelin_develop_contract" for cid in results)
            
            # Also check if there's an approval in the messages
            has_approval = any(
                c.approved 
                for m in messages if m.role == "user" 
                for c in m.contents if c.type == "function_approval_response"
            )
            
            if has_oz_result or has_approval:
                reply_contents.append(
                    "Here is your Solidity contract:\npragma solidity ^0.8.0;"
                )
            else:
                reply_contents.append(Content.from_function_call(
                    name="openzeppelin_develop_contract",
                    arguments={"contract_type": "ERC20"},
                    call_id=f"openzeppelin_{uuid.uuid4().hex[:8]}"
                ))
                
        elif self.agent_name == "RouterAgent":
            # Mock the router agent logic based on the user's initial query in history
            last_user_idx = -1
            user_msg = ""
            for idx, msg in enumerate(messages):
                if msg.role == "user" and any(c.type == "text" for c in msg.contents):
                    last_user_idx = idx
                    user_msg = next((c.text for c in msg.contents if c.type == "text"), "")
                    
            user_msg_lower = user_msg.lower()
            
            is_purchase_calc = any(k in user_msg_lower for k in ("buy", "calculate", "purchase", "how many"))
            is_market_analysis = any(k in user_msg_lower for k in ("report", "analysis", "deep status report", "financial report", "market analysis"))
            is_crypto = "price" in user_msg_lower or "pricing" in user_msg_lower or "coingecko" in user_msg_lower or is_purchase_calc
            is_rag = "search" in user_msg_lower or "blockchain" in user_msg_lower or "document" in user_msg_lower or ("bitcoin" in user_msg_lower and not is_purchase_calc and not is_market_analysis and "price" not in user_msg_lower)
            is_oz = "solidity" in user_msg_lower or "contract" in user_msg_lower or "openzeppelin" in user_msg_lower
            
            # Check what already responded after the last user message
            has_rag = False
            has_crypto = False
            has_oz = False
            has_ma = False
            if last_user_idx != -1:
                for m in messages[last_user_idx+1:]:
                    author = getattr(m, "author_name", None)
                    if author == "RAGSearchAgent": has_rag = True
                    elif author == "CryptoPricingAgent": has_crypto = True
                    elif author == "OpenZeppelinAgent": has_oz = True
                    elif author == "MarketAnalysisAgent": has_ma = True
            
            if is_oz and not has_oz:
                reply_contents.append("OpenZeppelinAgent")
            elif is_market_analysis and not has_ma:
                reply_contents.append("MarketAnalysisAgent")
            elif is_crypto and not has_crypto:
                reply_contents.append("CryptoPricingAgent")
            elif is_rag and not has_rag:
                reply_contents.append("RAGSearchAgent")
            elif sum([is_rag, is_crypto, is_oz, is_market_analysis]) > 1 and not any(getattr(m, "author_name", None) == "SummarizerAgent" for m in messages[last_user_idx+1:] if last_user_idx != -1):
                reply_contents.append("SummarizerAgent")
            else:
                reply_contents.append("Finalizer")

        elif self.agent_name == "SummarizerAgent":
            rag_res = ""
            crypto_res = ""
            oz_res = ""
            for m in messages:
                author = getattr(m, "author_name", None)
                if author == "RAGSearchAgent":
                    rag_res = m.text
                elif author == "CryptoPricingAgent":
                    crypto_res = m.text
                elif author == "OpenZeppelinAgent":
                    oz_res = m.text

            bullets = []
            if rag_res:
                bullets.append(f"- Search results: {rag_res.replace(chr(10), ' ')}")
            if crypto_res:
                bullets.append(f"- Crypto price: {crypto_res.replace(chr(10), ' ')}")
            if oz_res:
                bullets.append(f"- Contract code: {oz_res.replace(chr(10), ' ')}")

            reply_contents.append(
                "Here is a bullet summary of the answers:\n" + "\n".join(bullets)
            )

        elif self.agent_name == "RouterAgent":
            # Very basic mock routing logic mirroring the deterministic router
            reply_contents.append("Finalizer")
                
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
    from app.adapters.driven.agent.agents.summarizer_agent import SummarizerAgent
    from app.adapters.driven.agent.agents.router_agent import RouterAgent
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

    def calculate_crypto_purchase_func(usd_amount: float, crypto_price: float) -> str:
        if crypto_price <= 0:
            return "Error: Crypto price must be greater than zero."
        amount = usd_amount / crypto_price
        return f"With {usd_amount} USD, you can buy approximately {amount:.8f} units of the cryptocurrency at the price of {crypto_price} USD."

    def mock_crypto_init(self, client):
        crypto_tool = FunctionTool(
            name="get_crypto_price",
            description="Get the live price of a cryptocurrency in USD.",
            func=crypto_get_price,
            approval_mode="never_require"
        )
        calculation_tool = FunctionTool(
            name="calculate_crypto_purchase",
            description="Calculate the amount of cryptocurrency that can be purchased with a given amount of USD based on the current price.",
            func=calculate_crypto_purchase_func,
            approval_mode="never_require"
        )
        from agent_framework import Agent
        Agent.__init__(
            self,
            id="CryptoPricingAgent",
            name="CryptoPricingAgent",
            client=client,
            tools=[crypto_tool, calculation_tool],
            instructions=(
                "You are a Crypto Pricing Agent. Use the coingecko/crypto tool to fetch live "
                "prices for requested cryptocurrencies, then report them back to the user.\n\n"
                "If the user explicitly requests to calculate how much cryptocurrency they can purchase with a given amount of USD, you must:\n"
                "1. Fetch the live price of the requested cryptocurrency in USD first using the get_crypto_price tool.\n"
                "2. Call the `calculate_crypto_purchase` tool with the USD amount and the fetched cryptocurrency price to compute the purchase amount."
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
                "approval is strictly required."
            ),
            require_per_service_call_history_persistence=True
        )

    def mock_summarizer_init(self, client):
        from agent_framework import Agent
        Agent.__init__(
            self,
            id="SummarizerAgent",
            name="SummarizerAgent",
            client=client,
            instructions=(
                "You are a Summarizer Agent."
            ),
            require_per_service_call_history_persistence=True
        )

    def mock_router_init(self, client):
        from agent_framework import Agent
        Agent.__init__(
            self,
            id="RouterAgent",
            name="RouterAgent",
            client=client,
            instructions=(
                "You are the dynamic Router Agent."
            ),
            require_per_service_call_history_persistence=True
        )

    monkeypatch.setattr(CryptoPricingAgent, "__init__", mock_crypto_init)
    monkeypatch.setattr(OpenZeppelinAgent, "__init__", mock_oz_init)
    monkeypatch.setattr(SummarizerAgent, "__init__", mock_summarizer_init)
    monkeypatch.setattr(RouterAgent, "__init__", mock_router_init)
