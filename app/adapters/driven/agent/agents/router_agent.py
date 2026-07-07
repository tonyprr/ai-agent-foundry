from agent_framework import Agent

class RouterAgent(Agent):
    """
    Agentic Router responsible for dynamically evaluating the conversation history
    and selecting the next specialist agent to route to.
    """
    def __init__(self, client):
        super().__init__(
            id="RouterAgent",
            name="RouterAgent",
            client=client,
            instructions=(
                "You are the dynamic Router Agent for a multi-agent orchestration workflow. "
                "Your task is to review the conversation history and determine which agent should act next.\n"
                "Agents available to route to: 'RAGSearchAgent', 'CryptoPricingAgent', 'OpenZeppelinAgent', 'SummarizerAgent', or 'Finalizer'.\n\n"
                "Rules:\n"
                "1. Look at the TriageAgent's decision in the history to see which specialist agents are needed.\n"
                "2. Check if the needed agents have already responded in the history.\n"
                "3. If a needed specialist agent has NOT responded yet, route to it.\n"
                "4. If all needed specialist agents have responded, AND there was more than one specialist agent involved, route to 'SummarizerAgent' (unless it has already responded).\n"
                "5. If all needed specialist agents have responded and no summarization is needed (or SummarizerAgent has already responded), route to 'Finalizer'.\n\n"
                "Output ONLY the exact name of the agent to route to. Do not include any other text."
            ),
            require_per_service_call_history_persistence=True
        )
