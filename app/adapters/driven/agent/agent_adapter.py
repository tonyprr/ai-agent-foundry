import json
import logging
import time
from typing import Any

from typing_extensions import Never

# Microsoft Agent Framework Imports
from agent_framework import (
    AgentSession,
    Case,
    Content,
    Default,
    Message,
    WorkflowBuilder,
    WorkflowRunState,
)
from agent_framework._workflows._agent_executor import AgentExecutor, AgentExecutorResponse
from agent_framework._workflows._events import WorkflowEvent
from agent_framework.foundry import FoundryChatClient

# Patch WorkflowEvent to record creation timestamp for agent metrics tracking
_original_workflow_event_init = WorkflowEvent.__init__
def _patched_workflow_event_init(self, *args, **kwargs):
    _original_workflow_event_init(self, *args, **kwargs)
    self.created_at = time.perf_counter()
WorkflowEvent.__init__ = _patched_workflow_event_init

# Local Application Imports
from app.config import Settings
from app.domain.models import AgentRunResult, ApprovalRequestInfo
from app.ports.outputs import AgentPort, SearchPort, SessionStorePort
from app.adapters.driven.agent.agents import (
    CryptoPricingAgent,
    OpenZeppelinAgent,
    RAGSearchAgent,
    SummarizerAgent,
    TriageAgent,
)
from app.adapters.driven.agent.workflow_support import CheckpointStorage, Finalizer, Router

logger = logging.getLogger(__name__)


class AgentAdapter(AgentPort):
    """
    Driven Adapter implementing the AgentPort.
    Interacts with the Microsoft Agent Framework to coordinate RAG execution.
    """
    _triage_agent = None
    _rag_search_agent = None
    _crypto_pricing_agent = None
    _openzeppelin_agent = None
    _summarizer_agent = None

    def __init__(self, settings: Settings, search_adapter: SearchPort, session_store: SessionStorePort):
        self._settings = settings
        self._search_adapter = search_adapter
        self._session_store = session_store
        self._checkpoint_storage = CheckpointStorage(session_store)
        self._workflow = None

    def _get_or_create_workflow(self):
        if self._workflow is not None:
            return self._workflow

        # Create agents as singletons using custom agent classes
        if AgentAdapter._triage_agent is None:
            triage_client = self._get_chat_client("TriageAgent")
            AgentAdapter._triage_agent = TriageAgent(client=triage_client)

        if AgentAdapter._rag_search_agent is None:
            rag_client = self._get_chat_client("RAGSearchAgent")
            search_provider = self._search_adapter.build_context_provider()
            AgentAdapter._rag_search_agent = RAGSearchAgent(
                client=rag_client,
                search_provider=search_provider
            )

        if AgentAdapter._crypto_pricing_agent is None:
            crypto_client = self._get_chat_client("CryptoPricingAgent")
            AgentAdapter._crypto_pricing_agent = CryptoPricingAgent(client=crypto_client)

        if AgentAdapter._openzeppelin_agent is None:
            openzeppelin_client = self._get_chat_client("OpenZeppelinAgent")
            AgentAdapter._openzeppelin_agent = OpenZeppelinAgent(client=openzeppelin_client)

        if AgentAdapter._summarizer_agent is None:
            summarizer_client = self._get_chat_client("SummarizerAgent")
            AgentAdapter._summarizer_agent = SummarizerAgent(client=summarizer_client)

        def clean_history(messages: list[Message]) -> list[Message]:
            cleaned = []
            for msg in messages:
                if msg.role == "tool":
                    continue
                clean_contents = [
                    c for c in msg.contents
                    if c.type not in ("function_call", "function_result", "function_approval_request", "function_approval_response")
                ]
                if clean_contents:
                    cleaned.append(Message(role=msg.role, contents=clean_contents, author_name=getattr(msg, "author_name", None)))
            return cleaned

        # Build and cache a Workflow instance to bind to the current asyncio event loop
        triage_exec = AgentExecutor(AgentAdapter._triage_agent, id="TriageAgent", context_mode="custom", context_filter=clean_history)
        rag_exec = AgentExecutor(AgentAdapter._rag_search_agent, id="RAGSearchAgent", context_mode="custom", context_filter=clean_history)
        crypto_exec = AgentExecutor(AgentAdapter._crypto_pricing_agent, id="CryptoPricingAgent", context_mode="custom", context_filter=clean_history)
        oz_exec = AgentExecutor(AgentAdapter._openzeppelin_agent, id="OpenZeppelinAgent", context_mode="custom", context_filter=clean_history)
        summarizer_exec = AgentExecutor(AgentAdapter._summarizer_agent, id="SummarizerAgent", context_mode="custom", context_filter=clean_history)

        
        router = Router()
        finalizer = Finalizer()

        builder = (
            WorkflowBuilder(
                name="MultiAgentOrchestrationWorkflow",
                start_executor=triage_exec,
                checkpoint_storage=self._checkpoint_storage,
                output_from=[finalizer]
            )
            .add_edge(triage_exec, router)
            .add_switch_case_edge_group(
                router,
                [
                    Case(condition=lambda r: getattr(r, "route_to", "") == "RAGSearchAgent", target=rag_exec),
                    Case(condition=lambda r: getattr(r, "route_to", "") == "CryptoPricingAgent", target=crypto_exec),
                    Case(condition=lambda r: getattr(r, "route_to", "") == "OpenZeppelinAgent", target=oz_exec),
                    Case(condition=lambda r: getattr(r, "route_to", "") == "SummarizerAgent", target=summarizer_exec),
                    Default(target=finalizer)
                ]
            )
            .add_edge(rag_exec, router)
            .add_edge(crypto_exec, router)
            .add_edge(oz_exec, router)
            .add_edge(summarizer_exec, router)
            .add_edge(finalizer, triage_exec)
        )
        
        self._workflow = builder.build()
        return self._workflow

    def _get_chat_client(self, agent_name: str):
        from azure.identity.aio import DefaultAzureCredential
        credential = DefaultAzureCredential()
        return FoundryChatClient(
            project_endpoint=self._settings.azure_ai_foundry_endpoint,
            model=self._settings.azure_ai_model_deployment_name,
            credential=credential
        )

    async def _ensure_mcp_tools_connected(self):
        import asyncio
        for agent in [AgentAdapter._crypto_pricing_agent, AgentAdapter._openzeppelin_agent]:
            if agent and hasattr(agent, "tools"):
                for tool in agent.tools:
                    if hasattr(tool, "connect") and asyncio.iscoroutinefunction(tool.connect):
                        try:
                            await tool.connect()
                        except Exception as e:
                            logger.warning(f"Could not connect MCP tool '{tool.name}': {e}")

    async def run_agent(self, message: str, session: AgentSession) -> AgentRunResult:
        if self._workflow is None:
            self._get_or_create_workflow()
        workflow = self._workflow
        self._checkpoint_storage.active_session = session
        await self._ensure_mcp_tools_connected()
        
        has_checkpoint = "_workflow_checkpoint" in session.state
        active_req_id = session.state.get("active_user_prompt_request_id")
        
        if has_checkpoint:
            if active_req_id:
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
                # Stale checkpoint found but no active user request. Start a fresh run.
                run_result = await workflow.run(
                    message=message,
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
        
        if self._workflow is None:
            self._get_or_create_workflow()
        workflow = self._workflow
        responses = {request_id: response_content}
        self._checkpoint_storage.active_session = session
        await self._ensure_mcp_tools_connected()
        
        run_result = await workflow.run(
            responses=responses,
            checkpoint_id=session.session_id,
            checkpoint_storage=self._checkpoint_storage
        )
        
        return self._build_run_result(run_result, session)

    def _build_run_result(self, run_result: Any, session: AgentSession) -> AgentRunResult:
        self._log_token_and_time_summary(run_result)
        
        response_text = self._extract_response_text(run_result)
        approval_request = self._extract_approval_request(run_result, session)
        
        return AgentRunResult(
            response_text=response_text,
            approval_request=approval_request
        )

    def _log_token_and_time_summary(self, run_result: Any) -> None:
        """
        Calculates and logs the token consumption and processing duration per agent.
        """
        AGENT_EXECUTOR_IDS = {
            "TriageAgent",
            "RAGSearchAgent",
            "CryptoPricingAgent",
            "OpenZeppelinAgent",
            "SummarizerAgent",
        }
        
        metrics = {}
        agent_start_times = {}
        
        def extract_usages_from_completed_event_data(data) -> list[Any]:
            if not data:
                return []
            
            if isinstance(data, (list, tuple)):
                agent_responses = []
                executor_responses = []
                for item in data:
                    class_name = item.__class__.__name__
                    if class_name == "AgentResponse" or hasattr(item, "usage_details"):
                        agent_responses.append(item)
                    elif class_name == "AgentExecutorResponse" or hasattr(item, "agent_response"):
                        executor_responses.append(item)
                
                usages = []
                if agent_responses:
                    for resp in agent_responses:
                        usage = getattr(resp, "usage_details", None)
                        if usage:
                            usages.append(usage)
                else:
                    for exec_resp in executor_responses:
                        agent_resp = getattr(exec_resp, "agent_response", None)
                        if agent_resp:
                            usage = getattr(agent_resp, "usage_details", None)
                            if usage:
                                usages.append(usage)
                return usages
            else:
                class_name = data.__class__.__name__
                if class_name == "AgentResponse" or hasattr(data, "usage_details"):
                    usage = getattr(data, "usage_details", None)
                    return [usage] if usage else []
                elif class_name == "AgentExecutorResponse" or hasattr(data, "agent_response"):
                    agent_resp = getattr(data, "agent_response", None)
                    if agent_resp:
                        usage = getattr(agent_resp, "usage_details", None)
                        return [usage] if usage else []
                return []

        try:
            for event in run_result:
                agent_name = getattr(event, "executor_id", None)
                if not agent_name or agent_name not in AGENT_EXECUTOR_IDS:
                    continue
                    
                if agent_name not in metrics:
                    metrics[agent_name] = {"input": 0, "output": 0, "duration": 0.0}
                    
                timestamp = getattr(event, "created_at", None)
                
                # Track duration
                if timestamp is not None:
                    if event.type == "executor_invoked":
                        agent_start_times[agent_name] = timestamp
                    elif event.type in ("executor_completed", "executor_failed", "executor_bypassed"):
                        start_time = agent_start_times.pop(agent_name, None)
                        if start_time is not None:
                            metrics[agent_name]["duration"] += timestamp - start_time
                            
                # Track token usage
                if event.type == "executor_completed" and event.data:
                    usages = extract_usages_from_completed_event_data(event.data)
                    for usage in usages:
                        input_tokens = 0
                        output_tokens = 0
                        if isinstance(usage, dict):
                            input_tokens = usage.get("input_token_count") or 0
                            output_tokens = usage.get("output_token_count") or 0
                        else:
                            input_tokens = getattr(usage, "input_token_count", 0) or 0
                            output_tokens = getattr(usage, "output_token_count", 0) or 0
                        
                        metrics[agent_name]["input"] += input_tokens
                        metrics[agent_name]["output"] += output_tokens
        except Exception as e:
            logger.warning(f"Error calculating token usage or duration: {e}")
            return

        if metrics:
            total_input = sum(item["input"] for item in metrics.values())
            total_output = sum(item["output"] for item in metrics.values())
            total_duration = sum(item["duration"] for item in metrics.values())
            
            log_lines = ["======= Token Usage & Processing Time Summary ======="]
            for agent, data in metrics.items():
                log_lines.append(
                    f"  Agent: {agent} -> Input: {data['input']} | Output: {data['output']} | Total: {data['input'] + data['output']} | Duration: {data['duration']:.2f}s"
                )
            log_lines.append(
                f"  TOTAL -> Input: {total_input} | Output: {total_output} | Total: {total_input + total_output} | Duration: {total_duration:.2f}s"
            )
            log_lines.append("=====================================================")
            
            logger.info("\n" + "\n".join(log_lines))

    def _extract_response_text(self, run_result: Any) -> str:
        """
        Extracts and concatenates the output texts from the workflow run result.
        """
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
        return "\n".join(texts)

    def _extract_approval_request(self, run_result: Any, session: AgentSession) -> ApprovalRequestInfo | None:
        """
        Checks for any pending requests or approvals, formats them, and stores necessary session state.
        """
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
                    return approval_request
        return None
