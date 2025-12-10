"""
Gemini embeddings integration for semantic similarity.

Provides embedding generation and semantic similarity calculations
using Google's Gemini embedding models.
"""

import logging
from typing import List, Optional, Dict, Any
import numpy as np


logger = logging.getLogger(__name__)


class GeminiEmbeddings:
    """
    Gemini embeddings client for generating and comparing semantic vectors.
    """
    
    def __init__(
        self,
        model_name: str = "models/text-embedding-004",
        api_key: Optional[str] = None
    ):
        self.model_name = model_name
        self.api_key = api_key
        self._client = None
        
    def _initialize_client(self):
        """Initialize Gemini client for embedding generation."""
        if self._client is None:
            try:
                import google.generativeai as genai
                if self.api_key:
                    genai.configure(api_key=self.api_key)
                self._client = genai
                logger.info(f"Gemini embeddings initialized: {self.model_name}")
            except ImportError:
                logger.error("google-generativeai not installed. Install with: pip install google-generativeai")
                raise
            except Exception as e:
                logger.error(f"Failed to initialize Gemini client: {e}")
                raise
                
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding vector for given text.
        
        Args:
            text: Text content to embed
            
        Returns:
            Embedding vector as list of floats, or None if failed
        """
        if not text or not text.strip():
            return None
            
        self._initialize_client()
        
        try:
            result = self._client.embed_content(
                model=self.model_name,
                content=text.strip(),
                task_type="SEMANTIC_SIMILARITY"
            )
            
            if hasattr(result, 'embedding') and result.embedding:
                return result.embedding
            else:
                logger.warning(f"No embedding returned for text: {text[:100]}...")
                return None
                
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return None
            
    def calculate_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float]
    ) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0.0 to 1.0)
        """
        if not embedding1 or not embedding2:
            return 0.0
            
        try:
            # Convert to numpy arrays
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            similarity = dot_product / (norm1 * norm2)
            
            # Ensure result is in [0, 1] range
            return max(0.0, min(1.0, float(similarity)))
            
        except Exception as e:
            logger.error(f"Similarity calculation failed: {e}")
            return 0.0
            
    def compare_texts(
        self, 
        text1: str, 
        text2: str
    ) -> Optional[float]:
        """
        Generate embeddings for two texts and compare similarity.
        
        Args:
            text1: First text to compare
            text2: Second text to compare
            
        Returns:
            Similarity score (0.0 to 1.0), or None if failed
        """
        embedding1 = self.generate_embedding(text1)
        embedding2 = self.generate_embedding(text2)
        
        if embedding1 is None or embedding2 is None:
            return None
            
        return self.calculate_similarity(embedding1, embedding2)