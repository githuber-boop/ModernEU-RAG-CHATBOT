"""
Configuration management using Pydantic Settings
"""
from pydantic_settings import BaseSettings
from typing import List
import os

class Settings(BaseSettings):
    """Application settings"""
    
    # App configuration
    app_name: str = "MEU Chatbot API"
    debug: bool = True
    
    # OpenAI configuration
    openai_api_key: str
    openai_model: str = "gpt-4o-mini"  # gpt-4o-mini or gpt-4o
    
    # Vector store configuration
    vector_store_path: str = "./data/vector_store"
    embedding_model: str = "all-MiniLM-L6-v2"
    
    # Document processing
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k_results: int = 3
    
    # CORS configuration
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:3001"]
    
    # Data directory
    data_dir: str = './data/client_docs'
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "allow"

# Create settings instance
settings = Settings()