"""
Batch similarity processing for workflow optimization.

Provides efficient batch processing of multiple content pieces
with in-memory caching and optimized similarity checks.
"""

import logging
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from .similarity_checker import ContentSimilarityChecker, SimilarityResult


logger = logging.getLogger(__name__)


@dataclass
class ContentItem:
    """Represents a content item for batch processing."""
    id: str
    content: str
    metadata: Optional[Dict] = None


@dataclass
class BatchResult:
    """Result of batch similarity processing."""
    processed_count: int
    duplicates_found: int
    unique_contents: List[ContentItem]
    duplicate_pairs: List[Tuple[str, str, float]]


class BatchSimilarityProcessor:
    """
    Optimized batch processor for content similarity detection.
    
    Features:
    - In-memory caching for performance
    - Batch processing optimization
    - Duplicate detection and removal
    - Workflow integration support
    """
    
    def __init__(
        self,
        similarity_checker: Optional[ContentSimilarityChecker] = None,
        similarity_threshold: float = 0.8,
        cache_size: int = 1000
    ):
        self.checker = similarity_checker or ContentSimilarityChecker(
            similarity_threshold=similarity_threshold
        )
        self.cache_size = cache_size
        self._processed_contents: Dict[str, ContentItem] = {}
        self._similarity_cache: Dict[Tuple[str, str], SimilarityResult] = {}
        
    def add_content(self, content_id: str, content: str, metadata: Optional[Dict] = None):
        """Add content to the batch processor."""
        item = ContentItem(
            id=content_id,
            content=content,
            metadata=metadata
        )
        self._processed_contents[content_id] = item
        
        # Maintain cache size limit
        if len(self._processed_contents) > self.cache_size:
            # Remove oldest entries
            oldest_keys = list(self._processed_contents.keys())[:-self.cache_size]
            for key in oldest_keys:
                del self._processed_contents[key]
                
    def check_similarity_batch(
        self,
        new_contents: List[ContentItem],
        return_details: bool = False
    ) -> BatchResult:
        """
        Process a batch of content items for similarity detection.
        
        Args:
            new_contents: List of new content items to process
            return_details: Whether to return detailed similarity information
            
        Returns:
            BatchResult with processing summary
        """
        unique_contents = []
        duplicate_pairs = []
        duplicates_found = 0
        
        for new_item in new_contents:
            is_duplicate = False
            max_similarity = 0.0
            duplicate_with = None
            
            # Check against existing processed contents
            for existing_id, existing_item in self._processed_contents.items():
                cache_key = self._get_cache_key(new_item.id, existing_id)
                
                # Check cache first
                if cache_key in self._similarity_cache:
                    result = self._similarity_cache[cache_key]
                else:
                    # Perform similarity check
                    result = self.checker.check_similarity(
                        new_item.content,
                        existing_item.content
                    )
                    # Cache result
                    self._similarity_cache[cache_key] = result
                    
                if result.is_duplicate:
                    is_duplicate = True
                    duplicate_with = existing_id
                    max_similarity = result.similarity_score
                    break
                    
                if result.similarity_score > max_similarity:
                    max_similarity = result.similarity_score
                    
            # Check against other new items
            if not is_duplicate:
                for other_item in unique_contents:
                    result = self.checker.check_similarity(
                        new_item.content,
                        other_item.content
                    )
                    
                    if result.is_duplicate:
                        is_duplicate = True
                        duplicate_with = other_item.id
                        max_similarity = result.similarity_score
                        break
                        
            if is_duplicate:
                duplicates_found += 1
                if return_details and duplicate_with:
                    duplicate_pairs.append((new_item.id, duplicate_with, max_similarity))
            else:
                unique_contents.append(new_item)
                self.add_content(new_item.id, new_item.content, new_item.metadata)
                
        return BatchResult(
            processed_count=len(new_contents),
            duplicates_found=duplicates_found,
            unique_contents=unique_contents,
            duplicate_pairs=duplicate_pairs
        )
        
    def _get_cache_key(self, id1: str, id2: str) -> Tuple[str, str]:
        """Generate cache key for similarity pair."""
        # Ensure consistent ordering for cache key
        return tuple(sorted([id1, id2]))
        
    def clear_cache(self):
        """Clear all cached similarity results."""
        self._similarity_cache.clear()
        
    def clear_processed_contents(self):
        """Clear all processed content items."""
        self._processed_contents.clear()
        
    def get_stats(self) -> Dict[str, int]:
        """Get processor statistics."""
        return {
            "processed_contents": len(self._processed_contents),
            "cached_similarities": len(self._similarity_cache),
            "cache_size_limit": self.cache_size
        }