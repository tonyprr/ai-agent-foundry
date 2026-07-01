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
                "You are a Triage Agent. Analyze the user's input and identify which specialist agents are needed to answer the request.\n"
                "Specialist agents available:\n"
                "- RAGSearchAgent: for conceptual, historical, educational, or general explanation questions about Bitcoin, Blockchain, or general search queries.\n"
                "- CryptoPricingAgent: to get live cryptocurrency prices.\n"
                "- OpenZeppelinAgent: to develop or write Solidity smart contracts.\n"
                "Respond with a list of the needed specialist agents (e.g., 'Routing to: RAGSearchAgent, CryptoPricingAgent').\n"
                "If the query is general or does not fit these categories, respond politely yourself."
            ),
            require_per_service_call_history_persistence=True
        )
