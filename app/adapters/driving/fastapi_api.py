import logging
from fastapi import APIRouter, Request, HTTPException, status
from app.domain.models import RAGQueryRequest, RAGQueryResponse, ApproveRequestPayload
from app.ports.inputs import RAGUseCasePort

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["RAG Chat"])

@router.post(
    "/chat",
    response_model=RAGQueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Send a message to the RAG Agent",
    description="Processes user input asynchronously, retrieves matching context from Azure AI Search, and returns the response."
)
async def chat(request: Request, query_request: RAGQueryRequest):
    """
    Driving Adapter Route Handler.
    Retrieves the RAGUseCasePort instance from app state and executes it.
    """
    # Retrieve the dependency injected service from FastAPI app state
    rag_service: RAGUseCasePort = getattr(request.app.state, "rag_service", None)
    
    if not rag_service:
        logger.error("RAGUseCasePort was not registered in app state.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="System configuration error: RAG Service is unavailable."
        )

    try:
        logger.info(f"Received API query. message len={len(query_request.message)}")
        response = await rag_service.process_query(query_request)
        return response
    except ValueError as e:
        logger.warning(f"Invalid request parameters in query processing: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error processing RAG query: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while executing the agent: {str(e)}"
        )

@router.post(
    "/chat/approve",
    response_model=RAGQueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Approve or deny a pending tool call",
    description="Resumes the agent workflow execution after a tool execution was suspended for human-in-the-loop validation."
)
async def approve(request: Request, payload: ApproveRequestPayload):
    """
    Resumes a paused workflow run after receiving user approval/denial for a pending tool call.
    """
    rag_service: RAGUseCasePort = getattr(request.app.state, "rag_service", None)
    
    if not rag_service:
        logger.error("RAGUseCasePort was not registered in app state.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="System configuration error: RAG Service is unavailable."
        )

    try:
        logger.info(f"Received approval callback for thread '{payload.thread_id}', request '{payload.request_id}', approved={payload.approved}")
        response = await rag_service.approve_request(
            thread_id=payload.thread_id,
            request_id=payload.request_id,
            approved=payload.approved
        )
        return response
    except ValueError as e:
        logger.warning(f"Invalid parameters in approval process: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error processing approval: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while executing the agent: {str(e)}"
        )
