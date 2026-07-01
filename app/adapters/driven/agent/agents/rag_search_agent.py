from agent_framework import Agent

class RAGSearchAgent(Agent):
    """
    Specialist RAG Search Agent configured with the Azure AI Search Context Provider.
    """
    def __init__(self, client, search_provider):
        super().__init__(
            id="RAGSearchAgent",
            name="RAGSearchAgent",
            client=client,
            context_providers=[search_provider],
            instructions=(
                "You are an expert RAG agent. You must answer questions using only "
                "the retrieved context from Azure AI Search. Always cite your sources "
                "using the [Source: filename] format.\n"
                "If the user's query is outside your scope (e.g., asking for live crypto prices, "
                "developing Solidity smart contracts, or general greeting/triage queries), "
                "you MUST delegate/route the conversation back to the TriageAgent by calling the "
                "handoff_to_TriageAgent tool."
            ),
            require_per_service_call_history_persistence=True
        )
