from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

class RegexChunking:
    def __init__(self, patterns=None):
        self.patterns = patterns or [r'\n\n'] 

    def chunk(self, text):
        paragraphs = [text]
        for pattern in self.patterns:
            paragraphs = [seg for p in paragraphs for seg in re.split(pattern, p)]
        return paragraphs

class SlidingWindowChunking:
    def __init__(self, window_size=100, step=50):
        self.window_size = window_size
        self.step = step

    def chunk(self, text):
        words = text.split()
        chunks = []
        for i in range(0, len(words) - self.window_size + 1, self.step):
            chunks.append(' '.join(words[i:i + self.window_size]))
        return chunks
    
class MultiLevelChunking:
    def __init__(self, chunkers, min_chunk_size=None):
        self.chunkers = chunkers
        self.min_chunk_size = min_chunk_size
        
    def chunk(self, text):
        current_chunks = [text]
        
        for i, chunker in enumerate(self.chunkers):
            next_level_chunks = []
            
            for chunk in current_chunks:
                if self.min_chunk_size and i < len(self.chunkers) - 1:
                    if len(chunk.split()) < self.min_chunk_size:
                        next_level_chunks.append(chunk)
                        continue
                
                new_chunks = chunker.chunk(chunk)
                next_level_chunks.extend(new_chunks)
            
            current_chunks = next_level_chunks
            
        return current_chunks
    
class CosineSimilarityExtractor:
    def __init__(self, query):
        self.query = query
        self.vectorizer = TfidfVectorizer()

    def find_relevant_chunks(self, chunks):
        vectors = self.vectorizer.fit_transform([self.query] + chunks)
        similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
        return [(chunks[i], similarities[i]) for i in range(len(chunks))]


if __name__ == "__main__":
    # Example Workflow
    text = """# Document Title

    ## Section 1: Introduction
    This is the first section of a sample document. It has multiple sentences.
    We are testing chunking and similarity algorithms. The document continues with more text.

    ## Section 2: Methods
    This section contains information about different topics.
    We can use different chunking strategies to split this document.
    - First method uses regex patterns
    - Second method uses sliding windows
    - Third method combines multiple approaches

    ## Section 3: Results
    The third section discusses testing strategies and provides examples of chunking methods.
    Properly chunked text helps with semantic similarity comparisons.
    These techniques can be applied to various NLP tasks."""

    # Create individual chunkers
    heading_chunker = RegexChunking(patterns=[r'#{1,3}.*?\n'])  
    paragraph_chunker = RegexChunking(patterns=[r'\n\n'])  
    sentence_chunker = RegexChunking(patterns=[r'(?<=[.!?])\s'])  

    sliding_chunker = SlidingWindowChunking(window_size=10, step=5)

    # Create multi-level chunker with various configurations
    chunkers = [heading_chunker, paragraph_chunker, sliding_chunker]
    multi_chunker = MultiLevelChunking(chunkers, min_chunk_size=20)
        
    # Get chunks from multi-level chunker
    multi_chunks = multi_chunker.chunk(text)

    query = "testing chunking"

    extractor = CosineSimilarityExtractor(query)
    relevant_chunks = extractor.find_relevant_chunks(multi_chunks)

    # Sort by similarity score (descending)
    relevant_chunks.sort(key=lambda x: x[1], reverse=True)
        
    # Print top 5 most relevant chunks
    print(f"Found {len(multi_chunks)} chunks, showing top 5 most relevant:")
    for i, (chunk, similarity) in enumerate(relevant_chunks[:5]):
        print(f"{i+1}. Similarity: {similarity:.4f} - '{chunk[:75]}...'")