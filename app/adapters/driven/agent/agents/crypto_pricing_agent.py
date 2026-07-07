from agent_framework import Agent, MCPStreamableHTTPTool, FunctionTool


def calculate_crypto_purchase(usd_amount: float, crypto_price: float) -> str:
    """
    Calculate the amount of cryptocurrency that can be purchased with a given amount of USD based on the current price.
    """
    if crypto_price <= 0:
        return "Error: Crypto price must be greater than zero."
    amount = usd_amount / crypto_price
    return f"With {usd_amount} USD, you can buy approximately {amount:.8f} units of the cryptocurrency at the price of {crypto_price} USD."


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
        calculation_tool = FunctionTool(
            name="calculate_crypto_purchase",
            description="Calculate the amount of cryptocurrency that can be purchased with a given amount of USD based on the current price.",
            func=calculate_crypto_purchase,
            approval_mode="never_require"
        )
        super().__init__(
            id="CryptoPricingAgent",
            name="CryptoPricingAgent",
            client=client,
            tools=[crypto_tool, calculation_tool],
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
                "Execute the code, extract the live price from the returned response, and report it back to the user.\n\n"
                "If the user explicitly requests to calculate how much cryptocurrency they can purchase with a given amount of USD, you must:\n"
                "1. Fetch the live price of the requested cryptocurrency in USD first using the coingecko MCP tool.\n"
                "2. Call the `calculate_crypto_purchase` tool with the USD amount and the fetched cryptocurrency price to compute the purchase amount."
            ),
            require_per_service_call_history_persistence=True
        )

