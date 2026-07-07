import logging
from typing import Any

from typing_extensions import Never

# Microsoft Agent Framework Imports
from agent_framework import (
    Executor,
    Message,
    handler,
    WorkflowContext,
)
from agent_framework._workflows._agent_executor import AgentExecutorResponse
from agent_framework._workflows._checkpoint import CheckpointStorage as BaseCheckpointStorage, CheckpointID, WorkflowCheckpoint
from agent_framework._workflows._checkpoint_encoding import decode_checkpoint_value, encode_checkpoint_value
from agent_framework.exceptions import WorkflowCheckpointException

# Local Application Imports
from app.ports.outputs import SessionStorePort

logger = logging.getLogger(__name__)


class Router(Executor):
    def __init__(self, id: str = "Router"):
        super().__init__(id)

    @handler
    async def route(self, response: AgentExecutorResponse, ctx: WorkflowContext[AgentExecutorResponse, Never]) -> None:
        # Extract user message from history
        user_msg = next((m.text for m in reversed(response.full_conversation) if m.role == "user"), "")
        user_msg_lower = user_msg.lower() if user_msg else ""

        # Extract triage agent's decision from history
        triage_msg = next((m.text for m in reversed(response.full_conversation) if getattr(m, "author_name", None) == "TriageAgent"), "")
        triage_msg_lower = triage_msg.lower() if triage_msg else ""

        is_purchase_calc = any(k in user_msg_lower for k in ("buy", "calculate", "purchase", "how many"))
        is_rag_needed = "ragsearchagent" in triage_msg_lower or ("search" in user_msg_lower or "blockchain" in user_msg_lower or "document" in user_msg_lower or ("bitcoin" in user_msg_lower and not is_purchase_calc and ("concept" in user_msg_lower or "what is" in user_msg_lower or "explain" in user_msg_lower or "price" not in user_msg_lower)))
        is_crypto_needed = "cryptopricingagent" in triage_msg_lower or ("price" in user_msg_lower or "pricing" in user_msg_lower or "coingecko" in user_msg_lower or is_purchase_calc)
        is_oz_needed = "openzeppelinagent" in triage_msg_lower or ("solidity" in user_msg_lower or "contract" in user_msg_lower or "openzeppelin" in user_msg_lower)

        # Find the index of the last user message to isolate responses of the current turn
        last_user_idx = -1
        for idx, m in enumerate(response.full_conversation):
            if m.role == "user":
                last_user_idx = idx

        has_rag_responded = False
        has_crypto_responded = False
        has_oz_responded = False
        has_summarizer_responded = False

        if last_user_idx != -1:
            for m in response.full_conversation[last_user_idx + 1:]:
                author = getattr(m, "author_name", None)
                if author == "RAGSearchAgent":
                    has_rag_responded = True
                elif author == "CryptoPricingAgent":
                    has_crypto_responded = True
                elif author == "OpenZeppelinAgent":
                    has_oz_responded = True
                elif author == "SummarizerAgent":
                    has_summarizer_responded = True

        route_to = "Finalizer"
        if is_oz_needed and not has_oz_responded:
            route_to = "OpenZeppelinAgent"
        elif is_crypto_needed and not has_crypto_responded:
            route_to = "CryptoPricingAgent"
        elif is_rag_needed and not has_rag_responded:
            route_to = "RAGSearchAgent"
        elif (sum([is_rag_needed, is_crypto_needed, is_oz_needed]) > 1) and not has_summarizer_responded:
            route_to = "SummarizerAgent"

        # Dynamically set routing attribute
        object.__setattr__(response, "route_to", route_to)
        await ctx.send_message(response)


class Finalizer(Executor):
    def __init__(self, id: str = "Finalizer"):
        super().__init__(id)

    @handler
    async def finalize(self, response: AgentExecutorResponse, ctx: WorkflowContext[list[Message], str]) -> None:
        last_user_idx = -1
        for idx, m in enumerate(response.full_conversation):
            if m.role == "user":
                last_user_idx = idx

        rag_response = ""
        crypto_response = ""
        oz_response = ""
        summarizer_response = ""

        if last_user_idx != -1:
            for m in response.full_conversation[last_user_idx + 1:]:
                author = getattr(m, "author_name", None)
                if author == "RAGSearchAgent":
                    rag_response = m.text
                elif author == "CryptoPricingAgent":
                    crypto_response = m.text
                elif author == "OpenZeppelinAgent":
                    oz_response = m.text
                elif author == "SummarizerAgent":
                    summarizer_response = m.text

        parts = []
        if rag_response:
            parts.append(rag_response)
        if crypto_response:
            parts.append(crypto_response)
        if oz_response:
            parts.append(oz_response)

        if summarizer_response:
            final_text = summarizer_response
        elif parts:
            final_text = "\n\n".join(parts)
        else:
            final_text = response.agent_response.text

        await ctx.yield_output(final_text)
        
        # Suspend workflow and wait for user's next message
        user_input = await ctx.request_info(final_text, list[Message])
        
        # Route new message back to TriageAgent for the next turn
        await ctx.send_message(user_input, target_id="TriageAgent")


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
