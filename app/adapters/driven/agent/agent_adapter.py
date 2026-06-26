import logging
import uuid
import json
from typing import List, Dict, Any, Sequence, Callable, Literal, Collection, Optional, Mapping, AsyncIterable

from agent_framework import (
    Agent,
    AgentSession,
    Message,
    Content,
    WorkflowRunState,
    ContextProvider,
    SessionContext
)
from agent_framework._workflows._checkpoint import CheckpointStorage as BaseCheckpointStorage, WorkflowCheckpoint, CheckpointID
from agent_framework._workflows._checkpoint_encoding import encode_checkpoint_value, decode_checkpoint_value
from agent_framework.exceptions import WorkflowCheckpointException
from agent_framework_orchestrations import HandoffBuilder
from agent_framework_orchestrations._handoff import HandoffAgentUserRequest
from agent_framework.foundry import FoundryChatClient
from agent_framework.openai import OpenAIChatClient
from agent_framework._types import ChatResponse
from agent_framework._tools import FunctionTool

from app.ports.outputs import AgentPort, SessionStorePort, SearchPort
from app.domain.models import ApprovalRequestInfo, AgentRunResult
from app.config import Settings

logger = logging.getLogger(__name__)

class SimpleMockContextProvider(ContextProvider):
    """
    A custom mock ContextProvider that simulates retrieval results for local/offline testing.
    """
    def __init__(self, index_name: str, top_k: int):
        super().__init__(source_id="mock_search")
        self.index_name = index_name
        self.top_k = top_k

    async def get_context(self, context: SessionContext) -> str:
        # Simulate retrieved text based on index settings
        return (
            f"[Source: Document_1.txt (Mocked from index '{self.index_name}')]\n"
            "This is simulated context. Microsoft Agent Framework (MAF) is a pro-code, "
            "enterprise-grade SDK that unifies concepts from AutoGen and Semantic Kernel.\n\n"
            f"[Source: Document_2.txt (Mocked retrieval, top_k={self.top_k})]\n"
            "FastAPI is an asynchronous web framework for building APIs in Python. "
            "Hexagonal Architecture isolates business logic from external frameworks and technologies."
        )

class CheckpointStorage(BaseCheckpointStorage):
    """
    CheckpointStorage implementation that persists workflow checkpoints inside
    the AgentSession managed by SessionStorePort.
    """
    def __init__(self, session_store: SessionStorePort):
        self._session_store = session_store
        self.active_session = None

    async def save(self, checkpoint: WorkflowCheckpoint) -> CheckpointID:
        session_id = self.active_session.session_id if self.active_session else checkpoint.checkpoint_id
        try:
            object.__setattr__(checkpoint, "checkpoint_id", session_id)
        except Exception as e:
            logger.warning(f"Could not override checkpoint_id: {e}")
            
        session = self.active_session
        if not session:
            session = await self._session_store.get_or_create_session(session_id)
            
        checkpoint_dict = checkpoint.to_dict()
        encoded = encode_checkpoint_value(checkpoint_dict)
        session.state["_workflow_checkpoint"] = encoded
        await self._session_store.save_session(session)
        return session_id

    async def load(self, checkpoint_id: CheckpointID) -> WorkflowCheckpoint:
        session = self.active_session
        if not session or session.session_id != checkpoint_id:
            session = await self._session_store.get_or_create_session(checkpoint_id)
        encoded = session.state.get("_workflow_checkpoint")
        if not encoded:
            raise WorkflowCheckpointException(f"No checkpoint found with ID {checkpoint_id}")
            
        decoded = decode_checkpoint_value(encoded)
        return WorkflowCheckpoint.from_dict(decoded)

    async def list_checkpoints(self, *, workflow_name: str) -> list[WorkflowCheckpoint]:
        if self.active_session:
            encoded = self.active_session.state.get("_workflow_checkpoint")
            if encoded:
                try:
                    decoded = decode_checkpoint_value(encoded)
                    return [WorkflowCheckpoint.from_dict(decoded)]
                except Exception:
                    pass
        return []

    async def delete(self, checkpoint_id: CheckpointID) -> bool:
        session = self.active_session
        if not session or session.session_id != checkpoint_id:
            session = await self._session_store.get_or_create_session(checkpoint_id)
        if "_workflow_checkpoint" in session.state:
            del session.state["_workflow_checkpoint"]
            await self._session_store.save_session(session)
            return True
        return False

    async def get_latest(self, *, workflow_name: str) -> WorkflowCheckpoint | None:
        if self.active_session:
            encoded = self.active_session.state.get("_workflow_checkpoint")
            if encoded:
                try:
                    decoded = decode_checkpoint_value(encoded)
                    return WorkflowCheckpoint.from_dict(decoded)
                except Exception:
                    pass
        return None

    async def list_checkpoint_ids(self, *, workflow_name: str) -> list[CheckpointID]:
        if self.active_session and "_workflow_checkpoint" in self.active_session.state:
            return [self.active_session.session_id]
        return []

from agent_framework._middleware import ChatMiddlewareLayer
from agent_framework._tools import FunctionInvocationLayer
from agent_framework.observability import ChatTelemetryLayer
from agent_framework import BaseChatClient

class MockChatClient(FunctionInvocationLayer, ChatMiddlewareLayer, ChatTelemetryLayer, BaseChatClient):
    STORES_BY_DEFAULT = False
    
    def __init__(self, agent_name: str, **kwargs: Any):
        self.agent_name = agent_name
        self.model = "mock-model"
        super().__init__(**kwargs)
        
    async def _inner_get_response(self, *, messages, stream, options, **kwargs):
        logger.info(f"DEBUG MockChatClient [{self.agent_name}]: message count={len(messages)}")
        for idx, msg in enumerate(messages):
            logger.info(f"  Msg {idx}: role={msg.role}, author={getattr(msg, 'author_name', None)}")
            for c_idx, c in enumerate(msg.contents):
                logger.info(f"    Content {c_idx}: type={c.type}, call_id={getattr(c, 'call_id', None) or getattr(c, 'id', None)}, approved={getattr(c, 'approved', None)}")
        
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
            reply_contents.append(
                "[Source: Document_1.txt]\n"
                "Microsoft Agent Framework (MAF) is a professional, multi-agent orchestration framework. "
                "Bitcoin is a decentralized digital currency."
            )
            
        elif self.agent_name == "CryptoPricingAgent":
            if tool_result:
                res_val = tool_result.result if hasattr(tool_result, "result") else ""
                reply_contents.append(f"[MOCK RESPONSE] Based on the pricing service: {res_val}")
            else:
                reply_contents.append(Content.from_function_call(
                    name="get_crypto_price",
                    arguments={"coin_id": "bitcoin"},
                    call_id=f"crypto_{uuid.uuid4().hex[:8]}"
                ))
                
        elif self.agent_name == "OpenZeppelinAgent":
            if tool_result:
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
        logger.info(f"DEBUG MockChatClient [{self.agent_name}]: returned msg with call_ids {[c.call_id for c in msg_contents if hasattr(c, 'call_id')]}")
        
        return ChatResponse(
            messages=[response_msg],
            finish_reason="tool_calls" if has_func_call else "stop",
            model=self.model
        )

def crypto_get_price(coin_id: str) -> str:
    """
    Get the live price of a cryptocurrency in USD.
    
    Args:
        coin_id: The ID of the coin, e.g., 'bitcoin', 'ethereum', 'solana'.
    """
    return f"The live price of {coin_id} is $65,000 USD (mocked)."

def openzeppelin_develop_contract(contract_type: str, features: list[str] | None = None) -> str:
    """
    Develop a Solidity smart contract using OpenZeppelin.
    
    Args:
        contract_type: The type of contract, e.g., 'ERC20', 'ERC721', 'ERC1155'.
        features: Optional list of features, e.g., 'mintable', 'burnable', 'pausable'.
    """
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

class AgentAdapter(AgentPort):
    """
    Driven Adapter implementing the AgentPort.
    Interacts with the Microsoft Agent Framework to coordinate RAG execution.
    """
    _triage_agent = None
    _rag_search_agent = None
    _crypto_pricing_agent = None
    _openzeppelin_agent = None

    def __init__(self, settings: Settings, search_adapter: SearchPort, session_store: SessionStorePort):
        self._settings = settings
        self._search_adapter = search_adapter
        self._session_store = session_store
        self._checkpoint_storage = CheckpointStorage(session_store)

    def _get_or_create_workflow(self):
        # Create agents as singletons
        if AgentAdapter._triage_agent is None:
            triage_client = self._get_chat_client("TriageAgent")
            AgentAdapter._triage_agent = Agent(
                id="TriageAgent",
                name="TriageAgent",
                client=triage_client,
                instructions=(
                    "You are a Triage Agent. Analyze the user's input and delegate it to the appropriate specialist agent:\n"
                    "- Route to RAGSearchAgent for questions about Bitcoin, Blockchain, or general RAG search.\n"
                    "- Route to CryptoPricingAgent to get live cryptocurrency prices.\n"
                    "- Route to OpenZeppelinAgent to develop or write smart contracts.\n"
                    "If the query is general or does not fit these categories, respond politely yourself."
                ),
                require_per_service_call_history_persistence=True
            )

        if AgentAdapter._rag_search_agent is None:
            rag_client = self._get_chat_client("RAGSearchAgent")
            if self._settings.mock_mode:
                search_provider = SimpleMockContextProvider(
                    index_name=self._settings.azure_search_index_name or "default_mock_index",
                    top_k=self._settings.azure_search_top_k
                )
            else:
                search_provider = self._search_adapter.build_context_provider()
                
            AgentAdapter._rag_search_agent = Agent(
                id="RAGSearchAgent",
                name="RAGSearchAgent",
                client=rag_client,
                context_providers=[search_provider],
                instructions=(
                    "You are an expert RAG agent. You must answer questions using only "
                    "the retrieved context from Azure AI Search. Always cite your sources "
                    "using the [Source: filename] format."
                ),
                require_per_service_call_history_persistence=True
            )

        if AgentAdapter._crypto_pricing_agent is None:
            crypto_client = self._get_chat_client("CryptoPricingAgent")
            if self._settings.mock_mode:
                crypto_tool = FunctionTool(
                    name="get_crypto_price",
                    description="Get the live price of a cryptocurrency in USD.",
                    func=crypto_get_price,
                    approval_mode="never_require"
                )
            else:
                from agent_framework import MCPStreamableHTTPTool
                crypto_tool = MCPStreamableHTTPTool(
                    name="coingecko",
                    url="https://mcp.api.coingecko.com/mcp",
                    description="Crypto pricing tool using CoinGecko",
                    approval_mode="never_require"
                )
                
            AgentAdapter._crypto_pricing_agent = Agent(
                id="CryptoPricingAgent",
                name="CryptoPricingAgent",
                client=crypto_client,
                tools=[crypto_tool],
                instructions=(
                    "You are a Crypto Pricing Agent. Use the coingecko/crypto tool to fetch live "
                    "prices for requested cryptocurrencies, then report them back to the user."
                ),
                require_per_service_call_history_persistence=True
            )

        if AgentAdapter._openzeppelin_agent is None:
            openzeppelin_client = self._get_chat_client("OpenZeppelinAgent")
            if self._settings.mock_mode:
                oz_tool = FunctionTool(
                    name="openzeppelin_develop_contract",
                    description="Develops Solidity smart contracts using OpenZeppelin",
                    func=openzeppelin_develop_contract,
                    approval_mode="always_require"
                )
            else:
                from agent_framework import MCPStdioTool
                oz_tool = MCPStdioTool(
                    name="openzeppelin",
                    command="npx",
                    args=["-y", "@openzeppelin/contracts-mcp"],
                    description="Develops Solidity smart contracts using OpenZeppelin",
                    approval_mode="always_require"
                )
                
            AgentAdapter._openzeppelin_agent = Agent(
                id="OpenZeppelinAgent",
                name="OpenZeppelinAgent",
                client=openzeppelin_client,
                tools=[oz_tool],
                instructions=(
                    "You are an OpenZeppelin Agent. Use the openzeppelin tools to develop, write, "
                    "or customize Solidity contracts. Every time you invoke these tools, human-in-the-loop "
                    "approval is strictly required."
                ),
                require_per_service_call_history_persistence=True
            )

        # Always build and return a fresh Handoff Workflow instance to bind to the current asyncio event loop!
        builder = (
            HandoffBuilder(
                name="MultiAgentOrchestrationWorkflow",
                participants=[
                    AgentAdapter._triage_agent,
                    AgentAdapter._rag_search_agent,
                    AgentAdapter._crypto_pricing_agent,
                    AgentAdapter._openzeppelin_agent
                ],
                checkpoint_storage=self._checkpoint_storage
            )
            .with_start_agent(AgentAdapter._triage_agent)
            .add_handoff(AgentAdapter._triage_agent, [
                AgentAdapter._rag_search_agent,
                AgentAdapter._crypto_pricing_agent,
                AgentAdapter._openzeppelin_agent
            ])
            .add_handoff(AgentAdapter._rag_search_agent, [AgentAdapter._triage_agent])
            .add_handoff(AgentAdapter._crypto_pricing_agent, [AgentAdapter._triage_agent])
            .add_handoff(AgentAdapter._openzeppelin_agent, [AgentAdapter._triage_agent])
        )
        
        return builder.build()

    def _get_chat_client(self, agent_name: str):
        if self._settings.mock_mode:
            if self._settings.openai_api_key and self._settings.openai_api_key != "mock-key":
                return OpenAIChatClient(
                    api_key=self._settings.openai_api_key,
                    model="gpt-4o-mini"
                )
            else:
                return MockChatClient(agent_name)
        else:
            credential = None
            if not self._settings.mock_mode:
                from azure.identity.aio import DefaultAzureCredential
                credential = DefaultAzureCredential()
            return FoundryChatClient(
                project_endpoint=self._settings.azure_ai_foundry_endpoint,
                model=self._settings.azure_ai_model_deployment_name,
                credential=credential
            )

    async def run_agent(self, message: str, session: AgentSession) -> AgentRunResult:
        workflow = self._get_or_create_workflow()
        self._checkpoint_storage.active_session = session
        
        has_checkpoint = "_workflow_checkpoint" in session.state
        active_req_id = session.state.get("active_user_prompt_request_id")
        
        if has_checkpoint and active_req_id:
            responses = {
                active_req_id: [Message(role="user", contents=[message])]
            }
            # Clean up the active request ID as we are fulfilling it
            del session.state["active_user_prompt_request_id"]
            await self._session_store.save_session(session)
            
            run_result = await workflow.run(
                responses=responses,
                checkpoint_id=session.session_id,
                checkpoint_storage=self._checkpoint_storage
            )
        else:
            run_result = await workflow.run(
                message=message,
                checkpoint_storage=self._checkpoint_storage
            )
            
        return self._build_run_result(run_result, session)

    async def resume_run(self, session: AgentSession, request_id: str, approved: bool) -> AgentRunResult:
        req_key = f"pending_req_{request_id}"
        req_dict = session.state.get(req_key)
        if not req_dict:
            raise ValueError(f"No pending approval request found for ID '{request_id}'")
            
        original_request = Content.from_dict(req_dict)
        
        response_content = Content.from_function_approval_response(
            approved=approved,
            id=request_id,
            function_call=original_request.function_call
        )
        
        del session.state[req_key]
        await self._session_store.save_session(session)
        
        workflow = self._get_or_create_workflow()
        responses = {request_id: response_content}
        self._checkpoint_storage.active_session = session
        
        run_result = await workflow.run(
            responses=responses,
            checkpoint_id=session.session_id,
            checkpoint_storage=self._checkpoint_storage
        )
        
        return self._build_run_result(run_result, session)

    def _build_run_result(self, run_result: Any, session: AgentSession) -> AgentRunResult:
        outputs = run_result.get_outputs()
        texts = []
        for out in outputs:
            if isinstance(out, str):
                texts.append(out)
            elif hasattr(out, "text"):
                texts.append(out.text)
            elif hasattr(out, "messages"):
                for msg in out.messages:
                    if msg.text:
                        texts.append(msg.text)
                        
        response_text = "\n".join(texts)
        
        # Check for pending request info events
        approval_request = None
        final_state = run_result.get_final_state()
        if final_state == WorkflowRunState.IDLE_WITH_PENDING_REQUESTS:
            request_events = run_result.get_request_info_events()
            for event in request_events:
                if hasattr(event.data, "type") and event.data.type == "function_approval_request":
                    content = event.data
                    fn_call = content.function_call
                    args = fn_call.arguments if isinstance(fn_call.arguments, dict) else {}
                    if isinstance(fn_call.arguments, str):
                        try:
                            args = json.loads(fn_call.arguments)
                        except Exception:
                            args = {"raw": fn_call.arguments}
                            
                    approval_request = ApprovalRequestInfo(
                        request_id=event.request_id,
                        tool_name=fn_call.name,
                        arguments=args
                    )
                    
                    # Store original request content in session state
                    session.state[f"pending_req_{event.request_id}"] = content.to_dict()
                    break
                elif isinstance(event.data, HandoffAgentUserRequest):
                    session.state["active_user_prompt_request_id"] = event.request_id
                    
        return AgentRunResult(
            response_text=response_text,
            approval_request=approval_request
        )
