"""
Core content similarity checker with hybrid approach.

Combines character n-gram shingles with semantic embeddings for accurate 
content similarity detection in blog workflows.
"""

import hashlib
from typing import List, Set, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SimilarityResult:
    """Result of similarity check between two content pieces."""
    similarity_score: float
    shingle_similarity: float
    semantic_similarity: Optional[float]
    is_duplicate: bool
    threshold_used: float


class ContentSimilarityChecker:
    """
    Hybrid content similarity checker using:
    1. Character n-gram shingles (exact matching)
    2. Semantic embeddings (contextual similarity) 
    """
    
    def __init__(
        self,
        shingle_size: int = 5,
        similarity_threshold: float = 0.8,
        enable_semantic: bool = True
    ):
        self.shingle_size = shingle_size
        self.similarity_threshold = similarity_threshold
        self.enable_semantic = enable_semantic
        self._embeddings_client = None
        
    def _create_shingles(self, text: str) -> Set[str]:
        """Create character n-gram shingles from text."""
        if not text or len(text) < self.shingle_size:
            return set()
            
        # Normalize text
        normalized = text.lower().replace('\n', ' ').replace('\t', ' ')
        while '  ' in normalized:
            normalized = normalized.replace('  ', ' ')
            
        # Create shingles
        shingles = set()
        for i in range(len(normalized) - self.shingle_size + 1):
            shingle = normalized[i:i + self.shingle_size]
            shingles.add(shingle)
            
        return shingles
    
    def _calculate_jaccard_similarity(self, set1: Set[str], set2: Set[str]) -> float:
        """Calculate Jaccard similarity between two sets."""
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0
            
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def check_similarity(
        self, 
        content1: str, 
        content2: str,
        custom_threshold: Optional[float] = None
    ) -> SimilarityResult:
        """
        Check similarity between two content pieces.
        
        Args:
            content1: First content to compare
            content2: Second content to compare  
            custom_threshold: Optional custom threshold for this comparison
            
        Returns:
            SimilarityResult with similarity metrics
        """
        threshold = custom_threshold or self.similarity_threshold
        
        # Calculate shingle similarity
        shingles1 = self._create_shingles(content1)
        shingles2 = self._create_shingles(content2)
        shingle_sim = self._calculate_jaccard_similarity(shingles1, shingles2)
        
        # Semantic similarity (placeholder for future implementation)
        semantic_sim = None
        if self.enable_semantic and self._embeddings_client:
            # TODO: Implement semantic similarity using embeddings
            pass
            
        # Use shingle similarity as primary score for now
        final_score = shingle_sim
        is_duplicate = final_score >= threshold
        
        return SimilarityResult(
            similarity_score=final_score,
            shingle_similarity=shingle_sim,
            semantic_similarity=semantic_sim,
            is_duplicate=is_duplicate,
            threshold_used=threshold
        )
        
    def check_content_similarity(
        self,
        new_content: str,
        existing_contents: List[str],
        return_details: bool = False
    ) -> Tuple[bool, float]:
        """
        Check if new content is similar to any existing content.
        
        Args:
            new_content: Content to check for similarity
            existing_contents: List of existing content to compare against
            return_details: Whether to return detailed similarity info
            
        Returns:
            Tuple of (is_duplicate, max_similarity_score)
        """
        if not existing_contents:
            return False, 0.0
            
        max_similarity = 0.0
        
        for existing in existing_contents:
            result = self.check_similarity(new_content, existing)
            max_similarity = max(max_similarity, result.similarity_score)
            
            if result.is_duplicate:
                return True, result.similarity_score
                
        return False, max_similarity