from agent_framework import Agent, FunctionTool

def mock_sentiment_search(asset_name: str) -> str:
    """
    Mock tool that simulates searching recent news headlines and determining the market sentiment for an asset.
    """
    return f"Recent news for {asset_name} indicate a highly BULLISH sentiment. Major institutional adoption news just broke out, and regulatory outlook is positive."

class SentimentSearchAgent(Agent):
    """
    Agent responsible for gathering qualitative sentiment and news data.
    """
    def __init__(self, client):
        sentiment_tool = FunctionTool(
            name="mock_sentiment_search",
            description="Search recent news headlines and determine the market sentiment for an asset.",
            func=mock_sentiment_search,
            approval_mode="never_require"
        )
        super().__init__(
            id="SentimentSearchAgent",
            name="SentimentSearchAgent",
            client=client,
            tools=[sentiment_tool],
            instructions=(
                "You are the Sentiment Search Agent for a crypto market analysis workflow.\n"
                "Your ONLY task is to use the `mock_sentiment_search` tool to fetch the current market sentiment and news summary for the requested cryptocurrency.\n"
                "Output this qualitative data clearly so the final analyst agent can process it.\n"
                "Do NOT write the final report."
            ),
            require_per_service_call_history_persistence=True
        )
