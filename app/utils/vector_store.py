"""
Vector Store using ChromaDB for document storage and retrieval
"""
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class VectorStore:
    """Vector database for semantic search over documents"""
    
    def __init__(self, persist_directory: str, embedding_model: str):
        """
        Initialize ChromaDB vector store
        
        Args:
            persist_directory: Path to persist the database
            embedding_model: Name of sentence transformer model
        """
        logger.info(f"Initializing vector store at: {persist_directory}")
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model)
        logger.info(f"Loaded embedding model: {embedding_model}")
        
        # Initialize ChromaDB client
        self.client = chromadb.Client(ChromaSettings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name="meu_documents")
            logger.info(f"✓ Loaded existing collection with {self.collection.count()} documents")
        except:
            self.collection = self.client.create_collection(name="meu_documents")
            logger.info("✓ Created new collection")
    
    def add_documents(self, documents: List[Dict[str, str]]):
        """
        Add documents to vector store
        
        Args:
            documents: List of dicts with 'content' and optional 'metadata'
        """
        if not documents:
            logger.warning("No documents to add")
            return
        
        embeddings = []
        contents = []
        metadatas = []
        ids = []
        
        logger.info(f"Processing {len(documents)} documents...")
        
        for i, doc in enumerate(documents):
            content = doc['content']
            
            # Generate embedding
            embedding = self.embedding_model.encode(content).tolist()
            
            embeddings.append(embedding)
            contents.append(content)
            metadatas.append(doc.get('metadata', {}))
            ids.append(f"doc_{self.collection.count() + i}")
        
        # Add to collection
        self.collection.add(
            embeddings=embeddings,
            documents=contents,
            metadatas=metadatas,
            ids=ids
        )
        
        logger.info(f"✓ Added {len(documents)} documents to vector store")
    
    def search(self, query: str, top_k: int = 3) -> List[str]:
        """
        Search for similar documents
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of relevant document contents
        """
        if self.collection.count() == 0:
            logger.warning("Vector store is empty")
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query).tolist()
        
        # Search in collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count())
        )
        
        if results and results['documents']:
            relevant_docs = results['documents'][0]
            logger.debug(f"Found {len(relevant_docs)} relevant documents")
            return relevant_docs
        
        return []
    
    def clear(self):
        """Clear all documents from the collection"""
        logger.info("Clearing vector store...")
        self.client.delete_collection(name="meu_documents")
        self.collection = self.client.create_collection(name="meu_documents")
        logger.info("✓ Vector store cleared")
    
    def count(self) -> int:
        """Return number of documents in collection"""
        return self.collection.count()