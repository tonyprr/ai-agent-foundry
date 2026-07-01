from fastapi.testclient import TestClient
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
        summarizer1 = AgentAdapter._summarizer_agent
        
        assert triage1 is not None
        assert rag1 is not None
        assert crypto1 is not None
        assert oz1 is not None
        assert summarizer1 is not None
        
        # Build again to prove singleton instance equality
        agent_port._get_or_create_workflow()
        assert AgentAdapter._triage_agent is triage1
        assert AgentAdapter._rag_search_agent is rag1
        assert AgentAdapter._crypto_pricing_agent is crypto1
        assert AgentAdapter._openzeppelin_agent is oz1
        assert AgentAdapter._summarizer_agent is summarizer1


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


def test_token_usage_logging(caplog):
    """
    Verify that token usage summary is printed as a log.
    """
    import logging
    with caplog.at_level(logging.INFO):
        with TestClient(app) as client:
            payload = {
                "message": "Can you search the blockchain RAG documents?",
                "thread_id": "thread_token_logging"
            }
            response = client.post("/api/v1/chat", json=payload)
            assert response.status_code == 200
            
            # Verify the token usage summary is present in the logged text
            assert "Token Usage & Processing Time Summary" in caplog.text
            assert "Agent: TriageAgent" in caplog.text
            assert "Duration:" in caplog.text
            assert "TOTAL -> Input:" in caplog.text


def test_multiturn_handoff_routing():
    """
    Test multi-turn routing where subsequent queries are correctly routed
    between different specialist agents via Triage Agent handoffs.
    """
    with TestClient(app) as client:
        thread_id = "thread_multiturn_handoff"
        
        # Turn 1: Query for RAG
        payload1 = {
            "message": "Can you search the blockchain RAG documents?",
            "thread_id": thread_id
        }
        response1 = client.post("/api/v1/chat", json=payload1)
        assert response1.status_code == 200
        data1 = response1.json()
        assert "response_text" in data1
        assert "Microsoft Agent Framework" in data1["response_text"]
        
        # Turn 2: Query for Crypto pricing (should route back to Triage, then to CryptoPricingAgent)
        payload2 = {
            "message": "What is the price of Bitcoin?",
            "thread_id": thread_id
        }
        response2 = client.post("/api/v1/chat", json=payload2)
        assert response2.status_code == 200
        data2 = response2.json()
        assert "response_text" in data2
        assert "65,000" in data2["response_text"] or "pricing service" in data2["response_text"]
        
        # Turn 3: Query for RAG again (should route back to Triage, then to RAGSearchAgent)
        payload3 = {
            "message": "Search the blockchain documents once more.",
            "thread_id": thread_id
        }
        response3 = client.post("/api/v1/chat", json=payload3)
        assert response3.status_code == 200
        data3 = response3.json()
        assert "response_text" in data3
        assert "Microsoft Agent Framework" in data3["response_text"]


def test_single_turn_multi_intent_routing():
    """
    Test that a multi-intent query in a single turn executes both specialist
    agents (RAG search and crypto pricing) sequentially and ends at the Triage agent.
    """
    with TestClient(app) as client:
        payload = {
            "message": "Can you search the blockchain RAG documents and also tell me the current price of Bitcoin?",
            "thread_id": "thread_single_turn_multi_intent"
        }
        response = client.post("/api/v1/chat", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "response_text" in data
        
        # Verify both specialist agent responses are compiled in the final output
        response_text = data["response_text"]
        assert "Microsoft Agent Framework" in response_text or "decentralized digital currency" in response_text
        assert "65,000" in response_text or "pricing service" in response_text
        assert "bullet summary" in response_text.lower()
        assert "- Search results:" in response_text
        assert "- Crypto price:" in response_text


