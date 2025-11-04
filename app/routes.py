"""
API routes for MEU Chatbot
"""
from fastapi import APIRouter, HTTPException, status
from app.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    StatsResponse,
    ErrorResponse
)
from app.services.chatbot import chatbot_service
from app.config import settings
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create API router
router = APIRouter(prefix="/api", tags=["MEU Chatbot"])

# Add OPTIONS handler for CORS preflight
@router.options("/chat")
async def chat_options():
    """Handle CORS preflight for chat endpoint"""
    return {}

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check if the API is running and healthy"
)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="MEU Chatbot API is running"
    )

@router.get(
    "/stats",
    response_model=StatsResponse,
    summary="Get Statistics",
    description="Get chatbot statistics including document count and model info"
)
async def get_stats():
    """Get chatbot statistics"""
    try:
        stats = chatbot_service.get_stats()
        return StatsResponse(
            status="success",
            data=stats
        )
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve statistics: {str(e)}"
        )

@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Chat with AI",
    description="Send a message and receive an AI-generated response based on MEU documents",
    responses={
        200: {"description": "Successful response"},
        400: {"model": ErrorResponse, "description": "Bad request"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def chat(request: ChatRequest):
    """
    Chat endpoint - main conversational interface
    
    The chatbot uses RAG (Retrieval Augmented Generation) to:
    1. Search relevant documents based on the question
    2. Use OpenAI GPT to generate contextual responses
    """
    try:
        user_message = request.message.strip()
        
        if not user_message:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Message cannot be empty"
            )
        
        logger.info(f"Received chat request: {user_message[:100]}...")
        
        # Generate response using RAG
        bot_response = chatbot_service.chat(user_message)
        
        logger.info(f"Generated response: {bot_response[:100]}...")
        
        return ChatResponse(
            status="success",
            response=bot_response
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing your message. Please try again."
        )

@router.post(
    "/reload",
    summary="Reload Documents",
    description="Reload all documents from the data directory (Admin only in production)"
)
async def reload_documents():
    """
    Reload all documents
    
    WARNING: This endpoint should be protected with authentication in production!
    """
    try:
        logger.info("Reloading documents...")
        chatbot_service.reload_documents(settings.data_dir)
        
        return {
            "status": "success",
            "message": "Documents reloaded successfully",
            "document_count": chatbot_service.vector_store.count()
        }
    except Exception as e:
        logger.error(f"Error reloading documents: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reload documents: {str(e)}"
        )