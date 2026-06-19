import abc
from agent_framework import AgentSession
from app.domain.models import SearchConfigOverride

class AgentPort(abc.ABC):
    """
    Output Port (secondary port) for communicating with the Microsoft Agent Framework.
    """
    @abc.abstractmethod
    async def run_agent(
        self,
        message: str,
        session: AgentSession,
        search_config: SearchConfigOverride
    ) -> str:
        """
        Asynchronously runs the agent using a specific session and search index configuration.
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
