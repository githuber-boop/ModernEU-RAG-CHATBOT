"""
Document Processing Utilities
Handles loading and chunking of various file formats
"""
import os
from typing import List, Dict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Process and chunk documents for RAG"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize document processor
        
        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        logger.info(f"Document processor initialized (chunk_size={chunk_size}, overlap={chunk_overlap})")
    
    def load_text_file(self, file_path: str) -> str:
        """Load content from a text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading text file {file_path}: {str(e)}")
            raise
    
    
    def load_file(self, file_path: str) -> str:
        """
        Load content from any supported file type
        
        Args:
            file_path: Path to the file
            
        Returns:
            File content as string
        """
        ext = Path(file_path).suffix.lower()
        
        if ext == '.txt':
            return self.load_text_file(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + self.chunk_size
            
            # Get chunk
            chunk = text[start:end]
            
            # Try to break at natural boundaries if not at end
            if end < text_length:
                # Look for sentence, paragraph, or word boundaries
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                last_space = chunk.rfind(' ')
                
                # Choose the best break point
                break_point = max(last_period, last_newline, last_space)
                
                # Only use break point if it's not too early in the chunk
                if break_point > self.chunk_size * 0.5:
                    chunk = chunk[:break_point + 1]
                    end = start + break_point + 1
            
            # Add non-empty chunks
            chunk = chunk.strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - self.chunk_overlap
        
        return chunks
    
    def process_directory(self, directory: str) -> List[Dict[str, str]]:
        """
        Process all files in a directory
        
        Args:
            directory: Path to directory containing documents
            
        Returns:
            List of document chunks with metadata
        """
        if not os.path.exists(directory):
            logger.error(f"Directory not found: {directory}")
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        documents = []
        processed_files = 0
        
        logger.info(f"Processing documents in: {directory}")
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                
                # Skip hidden files and non-document files
                if file.startswith('.'):
                    continue
                
                try:
                    logger.info(f"Processing: {file}")
                    
                    # Load file content
                    content = self.load_file(file_path)
                    
                    if not content.strip():
                        logger.warning(f"  ⚠️  File is empty: {file}")
                        continue
                    
                    # Chunk the content
                    chunks = self.chunk_text(content)
                    
                    # Create document objects
                    for i, chunk in enumerate(chunks):
                        documents.append({
                            'content': chunk,
                            'metadata': {
                                'source': file,
                                'chunk_index': i,
                                'total_chunks': len(chunks),
                                'file_path': file_path
                            }
                        })
                    
                    logger.info(f"  ✓ Created {len(chunks)} chunks")
                    processed_files += 1
                    
                except Exception as e:
                    logger.error(f"  ✗ Error processing {file}: {str(e)}")
        
        logger.info(f"Processed {processed_files} files, created {len(documents)} chunks")
        
        return documents