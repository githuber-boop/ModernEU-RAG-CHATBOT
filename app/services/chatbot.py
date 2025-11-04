"""
Chatbot Service - Main business logic for MEU Chatbot
Handles document loading, RAG retrieval, and AI response generation
"""
from app.utils.vector_store import VectorStore
from app.utils.llm_handler import LLMHandler
from app.utils.document_processor import DocumentProcessor
from app.config import settings
import logging

logger = logging.getLogger(__name__)

class ChatbotService:
    """Singleton chatbot service"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ChatbotService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def initialize(self):
        """Initialize the chatbot service (called once on startup)"""
        if self._initialized:
            logger.info("Chatbot already initialized")
            return
        
        logger.info("Starting chatbot initialization...")
        
        try:
            # Initialize vector store
            logger.info("Setting up vector store...")
            self.vector_store = VectorStore(
                persist_directory=settings.vector_store_path,
                embedding_model=settings.embedding_model
            )
            
            # Initialize LLM handler
            logger.info("Setting up OpenAI LLM handler...")
            self.llm_handler = LLMHandler(
                api_key=settings.openai_api_key,
                model=settings.openai_model
            )
            
            # Initialize document processor
            logger.info("Setting up document processor...")
            self.document_processor = DocumentProcessor(
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap
            )
            
            # Load documents if vector store is empty
            if self.vector_store.count() == 0:
                logger.warning("Vector store is empty. Loading documents...")
                self.load_documents(settings.data_dir)
            else:
                logger.info(f"Vector store has {self.vector_store.count()} documents")
            
            self.top_k = settings.top_k_results
            self._initialized = True
            
            logger.info("✅ Chatbot initialization complete!")
            
        except Exception as e:
            logger.error(f"Failed to initialize chatbot: {str(e)}")
            raise
    
    def load_documents(self, data_dir: str):
        """Load and process documents from directory"""
        try:
            logger.info(f"Loading documents from: {data_dir}")
            documents = self.document_processor.process_directory(data_dir)
            
            if documents:
                self.vector_store.add_documents(documents)
                logger.info(f"✅ Successfully loaded {len(documents)} document chunks")
            else:
                logger.warning("⚠️  No documents found to load")
                
        except Exception as e:
            logger.error(f"Error loading documents: {str(e)}")
            raise
    
    def chat(self, message: str) -> str:
        """
        Process a chat message and return AI response
        
        Args:
            message: User's question
            
        Returns:
            AI-generated response based on relevant documents
        """
        try:
            # Search for relevant context
            logger.debug(f"Searching for context: {message[:50]}...")
            context = self.vector_store.search(message, top_k=self.top_k)
            
            logger.debug(f"Found {len(context)} relevant chunks")
            
            # Generate response using LLM
            response = self.llm_handler.generate_response(message, context)
            
            return response
            
        except Exception as e:
            logger.error(f"Error in chat: {str(e)}")
            raise
    
    def reload_documents(self, data_dir: str):
        """Clear and reload all documents"""
        try:
            logger.info("Clearing vector store...")
            self.vector_store.clear()
            
            logger.info("Reloading documents...")
            self.load_documents(data_dir)
            
            logger.info("✅ Documents reloaded successfully")
            
        except Exception as e:
            logger.error(f"Error reloading documents: {str(e)}")
            raise
    
    def get_stats(self) -> dict:
        """Get chatbot statistics"""
        return {
            'document_count': self.vector_store.count(),
            'model': self.llm_handler.model,
            'embedding_model': settings.embedding_model,
            'chunk_size': settings.chunk_size,
            'top_k': self.top_k
        }

# Create singleton instance
chatbot_service = ChatbotService()