from agent_framework import Agent, MCPStreamableHTTPTool

class DataFetcherAgent(Agent):
    """
    Agent responsible for fetching live market data (price, volume, market cap)
    using the CoinGecko MCP tool.
    """
    def __init__(self, client):
        crypto_tool = MCPStreamableHTTPTool(
            name="coingecko",
            url="https://mcp.api.coingecko.com/mcp",
            description="Crypto pricing tool using CoinGecko",
            approval_mode="never_require"
        )
        super().__init__(
            id="DataFetcherAgent",
            name="DataFetcherAgent",
            client=client,
            tools=[crypto_tool],
            instructions=(
                "You are the Data Fetcher Agent for a crypto market analysis workflow.\n"
                "Your ONLY task is to use the `coingecko` MCP tool to fetch the current price, 24h trading volume, and market cap of the cryptocurrency the user is asking about.\n"
                "Output this raw numerical data clearly so the next agent can process it.\n"
                "Do NOT perform any analysis or write any summaries. Just output the extracted metrics."
            ),
            require_per_service_call_history_persistence=True
        )
