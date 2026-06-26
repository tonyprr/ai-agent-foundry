import abc
from agent_framework import AgentSession, ContextProvider
from app.domain.models import AgentRunResult

class AgentPort(abc.ABC):
    """
    Output Port (secondary port) for communicating with the Microsoft Agent Framework.
    """
    @abc.abstractmethod
    async def run_agent(
        self,
        message: str,
        session: AgentSession
    ) -> AgentRunResult:
        """
        Asynchronously runs the agent workflow.
        """
        pass

    @abc.abstractmethod
    async def resume_run(
        self,
        session: AgentSession,
        request_id: str,
        approved: bool
    ) -> AgentRunResult:
        """
        Resumes a paused workflow run after receiving user approval/denial.
        """
        pass


class SessionStorePort(abc.ABC):
    """
    Output Port (secondary port) for loading and saving agent conversation sessions.
    """
    @abc.abstractmethod
    async def get_or_create_session(self, thread_id: str) -> AgentSession:
        """
        Retrieves a session by thread ID, creating a new session if it does not exist.
        """
        pass

    @abc.abstractmethod
    async def save_session(self, session: AgentSession) -> None:
        """
        Persists the state of an agent session.
        """
        pass


class SearchPort(abc.ABC):
    """
    Output Port (secondary port) for building search context providers and validating connection.
    """
    @abc.abstractmethod
    def build_context_provider(self) -> ContextProvider:
        """
        Builds the Microsoft Agent Framework ContextProvider.
        """
        pass

    @abc.abstractmethod
    async def validate_connection(self) -> bool:
        """
        Asynchronously validates the connection to the search index.
        """
        pass
