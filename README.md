# OpenSimilarityCheck

Content similarity detection for blog workflows using hybrid approaches.

## Features

- **Hybrid Similarity Detection**: Character n-gram shingles + semantic embeddings
- **Batch Processing**: Optimized for workflow integration with in-memory caching
- **Gemini Integration**: Semantic similarity using Google's embedding models
- **Production Ready**: Battle-tested in OpenBlog content generation pipeline

## Installation

```bash
pip install opensimilaritycheck
```

## Quick Start

```python
from opensimilaritycheck import ContentSimilarityChecker

# Initialize checker
checker = ContentSimilarityChecker(
    shingle_size=5,
    similarity_threshold=0.8
)

# Check similarity between two texts
result = checker.check_similarity(
    "This is the first article about AI.",
    "This is the second article about AI."
)

print(f"Similarity: {result.similarity_score:.2f}")
print(f"Is duplicate: {result.is_duplicate}")
```

## Batch Processing

```python
from opensimilaritycheck import BatchSimilarityProcessor, ContentItem

# Initialize batch processor
processor = BatchSimilarityProcessor(similarity_threshold=0.8)

# Process multiple content items
new_contents = [
    ContentItem(id="1", content="First article content..."),
    ContentItem(id="2", content="Second article content..."),
]

result = processor.check_similarity_batch(new_contents)
print(f"Processed: {result.processed_count}")
print(f"Duplicates found: {result.duplicates_found}")
```

## Semantic Similarity

```python
from opensimilaritycheck import GeminiEmbeddings

# Initialize with Gemini API key
embeddings = GeminiEmbeddings(api_key="your-api-key")

# Compare texts semantically
similarity = embeddings.compare_texts(
    "Machine learning is powerful",
    "AI algorithms are effective"
)
print(f"Semantic similarity: {similarity:.2f}")
```

## Configuration

### Similarity Thresholds

- `0.8+`: Very similar content (likely duplicates)
- `0.6-0.8`: Moderately similar content
- `0.4-0.6`: Some similarity
- `<0.4`: Different content

### Shingle Size

- `3-5`: Good for short content (titles, descriptions)
- `5-10`: Optimal for article content
- `10+`: Better for long-form content

## Integration with OpenBlog

This package was extracted from the OpenBlog pipeline for content similarity detection:

```python
from opensimilaritycheck import BatchSimilarityProcessor

# In your content pipeline
processor = BatchSimilarityProcessor()

# Before generating new content
is_duplicate, score = processor.checker.check_content_similarity(
    new_content=generated_article,
    existing_contents=previous_articles
)

if is_duplicate:
    print(f"Duplicate detected (similarity: {score:.2f})")
    # Handle duplicate...
```

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Generated with Claude Code

🔧 This package was generated and optimized using Claude Code.

Co-Authored-By: Claude <noreply@anthropic.com>