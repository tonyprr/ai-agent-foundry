from agent_framework import Agent

class FinancialAnalystAgent(Agent):
    """
    Agent responsible for synthesizing raw quantitative data and qualitative sentiment
    into a comprehensive financial market report.
    """
    def __init__(self, client):
        super().__init__(
            id="FinancialAnalystAgent",
            name="FinancialAnalystAgent",
            client=client,
            instructions=(
                "You are the Lead Financial Analyst for a crypto market analysis workflow.\n"
                "Review the conversation history to find the raw quantitative data provided by the DataFetcherAgent, "
                "and the qualitative news sentiment provided by the SentimentSearchAgent.\n\n"
                "Synthesize this information into a cohesive, professional Market Analysis Report in Markdown format.\n"
                "The report should include:\n"
                "- Executive Summary\n"
                "- Price & Volume Action\n"
                "- Market Sentiment & News\n"
                "- Concluding Outlook\n\n"
                "Do NOT use any tools. Just write the final report."
            ),
            require_per_service_call_history_persistence=True
        )
