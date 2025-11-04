"""
Pydantic models for request/response validation
"""
from pydantic import BaseModel, Field
from typing import Dict, Any

class ChatRequest(BaseModel):
    """Chat request model"""
    message: str = Field(
        ..., 
        min_length=1,
        max_length=1000,
        description="User's question or message"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [{
                "message": "What is MEU's mission?"
            }]
        }
    }

class ChatResponse(BaseModel):
    """Chat response model"""
    status: str = Field(description="Response status")
    response: str = Field(description="AI generated response")
    
    model_config = {
        "json_schema_extra": {
            "examples": [{
                "status": "success",
                "response": "The mission of Modern Education University (MEU) is to provide high-quality and accessible education..."
            }]
        }
    }

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    message: str

class StatsResponse(BaseModel):
    """Statistics response"""
    status: str
    data: Dict[str, Any]

class ErrorResponse(BaseModel):
    """Error response model"""
    status: str = "error"
    message: str
    detail: str = None