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
                "approval is strictly required.\n"
                "If the user's query is outside your scope (e.g., asking for live crypto prices, "
                "general search/RAG queries, or general greeting/triage queries), "
                "you MUST delegate/route the conversation back to the TriageAgent by calling the "
                "handoff_to_TriageAgent tool."
            ),
            require_per_service_call_history_persistence=True
        )
