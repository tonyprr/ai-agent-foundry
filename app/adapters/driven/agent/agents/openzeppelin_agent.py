from agent_framework import Agent, MCPStdioTool

class OpenZeppelinAgent(Agent):
    """
    Specialist agent that uses the OpenZeppelin MCP tool to generate Solidity smart contracts.
    Requires human-in-the-loop (HITL) approval for tool invocation.
    """
    def __init__(self, client):
        oz_tool = MCPStdioTool(
            name="openzeppelin",
            command="npx",
            args=["-y", "@openzeppelin/contracts-mcp"],
            description="Develops Solidity smart contracts using OpenZeppelin",
            approval_mode="always_require"
        )
        super().__init__(
            id="OpenZeppelinAgent",
            name="OpenZeppelinAgent",
            client=client,
            tools=[oz_tool],
            instructions=(
                "You are an OpenZeppelin Agent. Use the openzeppelin tools to develop, write, "
                "or customize Solidity contracts. Every time you invoke these tools, human-in-the-loop "
                "approval is strictly required."
            ),
            require_per_service_call_history_persistence=True
        )
