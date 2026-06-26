import logging
import json
import time
from typing import Any

from agent_framework._workflows._events import WorkflowEvent

# Patch WorkflowEvent to record creation timestamp for agent metrics tracking
_original_workflow_event_init = WorkflowEvent.__init__
def _patched_workflow_event_init(self, *args, **kwargs):
    _original_workflow_event_init(self, *args, **kwargs)
    self.created_at = time.perf_counter()
WorkflowEvent.__init__ = _patched_workflow_event_init

from agent_framework import (
    AgentSession,
    Message,
    Content,
    WorkflowRunState,
)
from agent_framework._workflows._checkpoint import CheckpointStorage as BaseCheckpointStorage, WorkflowCheckpoint, CheckpointID
from agent_framework._workflows._checkpoint_encoding import encode_checkpoint_value, decode_checkpoint_value
from agent_framework.exceptions import WorkflowCheckpointException
from agent_framework_orchestrations import HandoffBuilder
from agent_framework_orchestrations._handoff import HandoffAgentUserRequest
from agent_framework.foundry import FoundryChatClient

from app.ports.outputs import AgentPort, SessionStorePort, SearchPort
from app.domain.models import ApprovalRequestInfo, AgentRunResult
from app.config import Settings
from app.adapters.driven.agent.agents import (
    TriageAgent,
    RAGSearchAgent,
    CryptoPricingAgent,
    OpenZeppelinAgent,
)

logger = logging.getLogger(__name__)

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
        workflow = self._get_or_create_workflow()
        self._checkpoint_storage.active_session = session
        await self._ensure_mcp_tools_connected()
        
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
        metrics = {}
        agent_start_times = {}
        
        try:
            for event in run_result:
                timestamp = getattr(event, "created_at", None)
                agent_name = getattr(event, "executor_id", None)
                if not agent_name:
                    continue
                    
                if agent_name not in metrics:
                    metrics[agent_name] = {"input": 0, "output": 0, "duration": 0.0}
                    
                # Track duration
                if timestamp is not None:
                    if event.type == "executor_invoked":
                        agent_start_times[agent_name] = timestamp
                    elif event.type in ("executor_completed", "executor_failed", "executor_bypassed"):
                        start_time = agent_start_times.pop(agent_name, None)
                        if start_time is not None:
                            metrics[agent_name]["duration"] += timestamp - start_time
                            
                # Track token usage
                if event.data:
                    usage = getattr(event.data, "usage_details", None)
                    if usage:
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
                elif isinstance(event.data, HandoffAgentUserRequest):
                    session.state["active_user_prompt_request_id"] = event.request_id
        return None
