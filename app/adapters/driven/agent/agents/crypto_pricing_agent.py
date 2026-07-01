from agent_framework import Agent, MCPStreamableHTTPTool

class CryptoPricingAgent(Agent):
    """
    Specialist agent that uses the CoinGecko MCP tool to fetch live crypto prices.
    """
    def __init__(self, client):
        crypto_tool = MCPStreamableHTTPTool(
            name="coingecko",
            url="https://mcp.api.coingecko.com/mcp",
            description="Crypto pricing tool using CoinGecko",
            approval_mode="never_require"
        )
        super().__init__(
            id="CryptoPricingAgent",
            name="CryptoPricingAgent",
            client=client,
            tools=[crypto_tool],
            instructions=(
                "You are a Crypto Pricing Agent. You have access to a coingecko MCP tool with two functions:\n"
                "- `search_docs`: Use this to look up SDK methods and parameters if you need to know how to query the CoinGecko API.\n"
                "- `execute`: Use this to run JavaScript/TypeScript code to interact with the CoinGecko API client. "
                "You must write an async function named `run(client)` that uses the SDK client to fetch the requested data and returns/logs it.\n\n"
                "Example code to get Bitcoin price:\n"
                "```javascript\n"
                "async function run(client) {\n"
                "  return await client.simple.price.get({ vs_currencies: 'usd', ids: 'bitcoin' });\n"
                "}\n"
                "```\n"
                "Execute the code, extract the live price from the returned response, and report it back to the user.\n"
                "If the user's query is outside your scope (e.g., general search/RAG queries, "
                "developing Solidity smart contracts, or general greeting/triage queries), "
                "you MUST delegate/route the conversation back to the TriageAgent by calling the "
                "handoff_to_TriageAgent tool."
            ),
            require_per_service_call_history_persistence=True
        )
