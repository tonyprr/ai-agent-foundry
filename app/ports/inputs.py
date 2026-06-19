import abc
from app.domain.models import RAGQueryRequest, RAGQueryResponse

class RAGUseCasePort(abc.ABC):
    """
    Input Port (primary port) for executing RAG queries.
    """
    @abc.abstractmethod
    async def process_query(self, request: RAGQueryRequest) -> RAGQueryResponse:
        """
        Asynchronously processes a user query, retrieving context and returning the RAG response.
        """
        pass
