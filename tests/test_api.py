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
    Test the main chat endpoint. Verifies that passing search_overrides
    is rejected with a 400 Bad Request.
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
        assert response.status_code == 400
        data = response.json()
        assert "search_overrides are not permitted" in data["detail"]


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


def test_openzeppelin_hitl_approval_workflow():
    """
    Test the complete OpenZeppelin human-in-the-loop (HITL) workflow:
    1. Send a request about smart contracts, routing to the OpenZeppelin agent.
    2. Check that the workflow suspends, returning a pending approval request.
    3. Call the /chat/approve endpoint to approve the tool execution.
    4. Verify that execution resumes and the final contract response is returned.
    """
    with TestClient(app) as client:
        # Step 1: Send request requiring OpenZeppelin Solidity agent
        payload = {
            "message": "Develop a Solidity contract using OpenZeppelin",
            "thread_id": "oz_thread_123"
        }
        
        response = client.post("/api/v1/chat", json=payload)
        assert response.status_code == 200
        data = response.json()
        
        # Verify the workflow suspended with a pending approval
        assert data["approval_request"] is not None
        req_info = data["approval_request"]
        assert req_info["tool_name"] == "openzeppelin_develop_contract"
        request_id = req_info["request_id"]
        
        # Step 2: Approve the request
        approve_payload = {
            "thread_id": "oz_thread_123",
            "request_id": request_id,
            "approved": True
        }
        
        approve_response = client.post("/api/v1/chat/approve", json=approve_payload)
        assert approve_response.status_code == 200
        approve_data = approve_response.json()
        
        # The execution should have resumed and returned the Solidity code
        assert "pragma solidity" in approve_data["response_text"]
        assert approve_data["approval_request"] is None
