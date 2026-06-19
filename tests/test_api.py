import os
os.environ["MOCK_MODE"] = "True"
# Clear search fields that trigger SDK validation requirements in tests
os.environ["AZURE_SEARCH_VECTOR_FIELD"] = ""
os.environ["AZURE_SEARCH_SEMANTIC_CONFIG"] = ""

import pytest
from fastapi.testclient import TestClient
from app.main import app


def test_health_check():
    """
    Test the health check endpoint to verify server startup and basic metadata.
    """
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "Microsoft Agent Framework" in data["framework"]

def test_rag_chat_endpoint_with_overrides():
    """
    Test the main chat endpoint. Verifies that user queries are processed,
    custom thread IDs are honored, and dynamic search overrides are registered.
    """
    with TestClient(app) as client:
        payload = {
            "message": "Tell me about Microsoft Agent Framework",
            "thread_id": "test_thread_999",
            "search_overrides": {
                "index_name": "special-index",
                "mode": "semantic",
                "top_k": 3
            }
        }
        
        response = client.post("/api/v1/chat", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert "response_text" in data
        assert data["thread_id"] == "test_thread_999"
        
        # Verify metadata reflection of the applied overrides
        metadata = data["metadata"]
        assert metadata["search_index"] == "special-index"
        assert metadata["search_mode"] == "semantic"
        assert metadata["search_top_k"] == 3
        assert metadata["mock_mode"] is True

def test_rag_chat_session_id_generation():
    """
    Test that the system automatically generates a unique thread ID 
    if the request does not provide one.
    """
    with TestClient(app) as client:
        payload = {
            "message": "What is hexagonal architecture?"
        }
        
        response = client.post("/api/v1/chat", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert "response_text" in data
        assert "thread_id" in data
        assert data["thread_id"].startswith("thread_")
