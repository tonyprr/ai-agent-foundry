from agent_framework import WorkflowBuilder
from agent_framework._workflows._agent_executor import AgentExecutor

from .data_fetcher_agent import DataFetcherAgent
from .sentiment_search_agent import SentimentSearchAgent
from .financial_analyst_agent import FinancialAnalystAgent

def build_market_analysis_workflow_agent(client):
    """
    Builds the Market Analysis Sub-Workflow and wraps it as a single Agent
    so it can be used inside another workflow.
    """
    # 1. Initialize sub-agents
    data_fetcher = DataFetcherAgent(client=client)
    sentiment_search = SentimentSearchAgent(client=client)
    financial_analyst = FinancialAnalystAgent(client=client)

    # 2. Wrap them in executors for the sub-workflow
    data_exec = AgentExecutor(data_fetcher, id="DataFetcher", context_mode="full")
    sentiment_exec = AgentExecutor(sentiment_search, id="SentimentSearch", context_mode="full")
    analyst_exec = AgentExecutor(financial_analyst, id="FinancialAnalyst", context_mode="full")

    # 3. Build the internal routing (Sequential: Fetch Data -> Fetch Sentiment -> Analyst)
    builder = (
        WorkflowBuilder(
            name="MarketAnalysisSubWorkflow",
            start_executor=data_exec,
            output_from=[analyst_exec]
        )
        .add_edge(data_exec, sentiment_exec)
        .add_edge(sentiment_exec, analyst_exec)
    )
    
    workflow = builder.build()
    
    # 4. Transform the workflow into a single Agent
    return workflow.as_agent(
        name="MarketAnalysisAgent",
        description="A specialized workflow-agent that performs deep financial market analysis."
    )
