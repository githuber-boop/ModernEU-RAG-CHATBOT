"""
LLM Handler for OpenAI API
Handles all interactions with OpenAI GPT models
"""
from openai import OpenAI
from typing import List
import logging

logger = logging.getLogger(__name__)

class LLMHandler:
    """Handler for OpenAI Language Model"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        """
        Initialize OpenAI LLM handler
        
        Args:
            api_key: OpenAI API key
            model: Model name (gpt-4o-mini, gpt-4o, gpt-3.5-turbo)
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        logger.info(f"✓ Initialized OpenAI LLM with model: {model}")
    
    def generate_response(self, query: str, context: List[str]) -> str:
        """
        Generate response using RAG context
        
        Args:
            query: User's question
            context: List of relevant document chunks
            
        Returns:
            AI-generated response
        """
        # Combine context chunks
        context_text = "\n\n".join(context) if context else "No relevant context found in documents."
        
        # Create system and user messages
        system_message = """You are a helpful AI assistant for Modern Education University (MEU). 
Your role is to provide accurate, friendly, and helpful information about MEU's courses, mission, and programs.

Guidelines:
- Answer based on the provided context
- Be concise but informative
- If the context doesn't contain the answer, politely say so
- Maintain a professional yet friendly tone
- Focus on MEU's certificate courses in tourism"""

        user_message = f"""Context from MEU documents:
{context_text}

User Question: {query}

Please provide a helpful answer based on the context above."""

        try:
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7,
                max_tokens=500,
                top_p=0.9
            )
            
            answer = response.choices[0].message.content
            
            logger.debug(f"Generated response: {answer[:100]}...")
            
            return answer
            
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise Exception(f"Failed to generate response: {str(e)}")