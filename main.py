"""
FastAPI application entry point - OpenAI RAG Chatbot
Modern Education University (MEU) Chatbot Backend
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()
print(f"API Key loaded: {os.getenv('OPENAI_API_KEY')[:20]}...")

# Now import settings (after load_dotenv)
from app.config import settings
from app.routes import router
from app.services.chatbot import chatbot_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    print("\n" + "="*60)
    print("🚀 MEU Chatbot Backend Starting...")
    print("="*60)
    chatbot_service.initialize()
    print("="*60)
    print("✅ Ready to serve requests!")
    print(f"📍 API: http://localhost:8000")
    print(f"📚 Docs: http://localhost:8000/docs")
    print("="*60 + "\n")
    
    yield
    
    # Shutdown
    print("\n👋 Shutting down gracefully...")

# Create FastAPI application
app = FastAPI(
    title="MEU Chatbot API",
    description="RAG-powered chatbot for Modern Education University using OpenAI GPT",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "MEU Chatbot API",
        "version": "1.0.0",
        "description": "RAG-powered chatbot for Modern Education University",
        "model": settings.openai_model,
        "endpoints": {
            "health": "/api/health",
            "chat": "/api/chat",
            "stats": "/api/stats",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug
    )
    