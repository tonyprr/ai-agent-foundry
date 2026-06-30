from agent_framework import Agent

class TriageAgent(Agent):
    """
    Triage Agent responsible for analyzing user query and routing it
    to the correct specialist agent (RAG, Crypto Pricing, or OpenZeppelin).
    """
    def __init__(self, client):
        super().__init__(
            id="TriageAgent",
            name="TriageAgent",
            client=client,
            instructions=(
                "You are a Triage Agent. Analyze the user's input and delegate it to the appropriate specialist agent:\n"
                "- Route to RAGSearchAgent for conceptual, historical, educational, or general explanation questions about Bitcoin, Blockchain, or general search queries.\n"
                "- Route to CryptoPricingAgent to get live cryptocurrency prices.\n"
                "- Route to OpenZeppelinAgent to develop or write smart contracts.\n"
                "If the query is general or does not fit these categories, respond politely yourself."
            ),
            require_per_service_call_history_persistence=True
        )
