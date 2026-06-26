import os
import pytest
from fastapi.testclient import TestClient

# Ensure test settings are correct
os.environ["MOCK_MODE"] = "True"
os.environ["SESSION_STORE_TYPE"] = "memory"
os.environ["AZURE_SEARCH_VECTOR_FIELD"] = ""
os.environ["AZURE_SEARCH_SEMANTIC_CONFIG"] = ""

from app.main import app
from app.adapters.driven.agent.agent_adapter import AgentAdapter

def test_singleton_agents():
    """
    Verify that specialists and triage agents are instantiated as singletons.
    """
    with TestClient(app) as client:
        # Resolve from app state (or instantiate two adapter configs)
        rag_service = app.state.rag_service
        agent_port = rag_service._agent_port
        
        # Access agents via class variables or through workflow retrieval
        agent_port._get_or_create_workflow()
        
        triage1 = AgentAdapter._triage_agent
        rag1 = AgentAdapter._rag_search_agent
        crypto1 = AgentAdapter._crypto_pricing_agent
        oz1 = AgentAdapter._openzeppelin_agent
        
        assert triage1 is not None
        assert rag1 is not None
        assert crypto1 is not None
        assert oz1 is not None
        
        # Build again to prove singleton instance equality
        agent_port._get_or_create_workflow()
        assert AgentAdapter._triage_agent is triage1
        assert AgentAdapter._rag_search_agent is rag1
        assert AgentAdapter._crypto_pricing_agent is crypto1
        assert AgentAdapter._openzeppelin_agent is oz1


def test_triage_routing_to_rag():
    """
    Test routing from Triage to RAG Agent.
    """
    with TestClient(app) as client:
        payload = {
            "message": "Can you search the blockchain RAG documents?",
            "thread_id": "thread_rag_routing"
        }
        response = client.post("/api/v1/chat", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "response_text" in data
        assert "Microsoft Agent Framework" in data["response_text"]


def test_triage_routing_to_crypto():
    """
    Test routing from Triage to Crypto Pricing Agent.
    """
    with TestClient(app) as client:
        payload = {
            "message": "What is the price of Bitcoin?",
            "thread_id": "thread_crypto_routing"
        }
        response = client.post("/api/v1/chat", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "response_text" in data
        assert "65,000" in data["response_text"] or "pricing service" in data["response_text"]
