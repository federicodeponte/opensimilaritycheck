"""
OpenSimilarityCheck - Content Similarity Detection for Blog Workflows

A standalone package for detecting content similarity using hybrid approaches:
- Character n-gram shingles for exact text matching
- Semantic embeddings for contextual similarity
- Batch processing for workflow optimization

🔧 Generated with Claude Code
"""

__version__ = "1.0.0"

from .similarity_checker import ContentSimilarityChecker
from .embeddings import GeminiEmbeddings
from .batch_processor import BatchSimilarityProcessor

__all__ = [
    "ContentSimilarityChecker",
    "GeminiEmbeddings", 
    "BatchSimilarityProcessor"
]