from agent_framework import Agent

class SummarizerAgent(Agent):
    """
    Summarizer Agent responsible for summarizing and synthesizing multi-intent response data
    into a clean, bulleted list format.
    """
    def __init__(self, client):
        super().__init__(
            id="SummarizerAgent",
            name="SummarizerAgent",
            client=client,
            instructions=(
                "You are a Summarizer Agent. Analyze the user's initial multi-purpose query and "
                "the answers provided by the specialist agents in the conversation history.\n"
                "Your task is to compile a synthesized response with a clear, concise bulleted list "
                "format where each bullet summarizes the answer to one of the user's questions.\n"
                "Ensure you keep the citations (e.g. [Source: filename]) and price information intact. "
                "Do not include developer/technical execution details, just present the answers beautifully."
            ),
            require_per_service_call_history_persistence=True
        )
