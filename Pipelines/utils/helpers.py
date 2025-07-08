"""
Utility functions for RAG Pipeline
"""
from typing import List, Tuple
import numpy as np
from utils.logger import get_logger
logger = get_logger(__name__)

# from together import Together
# client = Together()

def rrf(lists: List[List[int]], k: int = 60) -> List[int]:
    """
    Reciprocal Rank Fusion algorithm for combining multiple ranked lists
    
    Args:
        lists: List of ranked lists (each containing indices)
        k: RRF parameter (default: 60)
    
    Returns:
        Combined ranked list of indices
    """
    scores = {}
    for lst in lists:
        for rank, idx in enumerate(lst):
            scores[idx] = scores.get(idx, 0.0) + 1.0/(k + rank + 1)
    return sorted(scores, key=lambda x: scores[x], reverse=True)


def rerank_chunks(
    query: str, 
    contextual_chunks: List[str], 
    chunk_indices: List[int], 
    top_n: int = 3
) -> Tuple[List[str], List[int]]:
    """
    Rerank chunks using a reranking model (currently using simple fallback)
    
    Args:
        query: Search query
        contextual_chunks: List of all available chunks
        chunk_indices: Indices of chunks to rerank
        top_n: Number of top chunks to return
    
    Returns:
        Tuple of (reranked_chunks, reranked_indices)
    """
    chunks_to_rerank = [contextual_chunks[i] for i in chunk_indices]

    # Simple fallback - return top_n chunks without reranking
    # Disable reranking due to Together.AI credit limits
    try:
        logger.info(f"Reranking {len(chunks_to_rerank)} chunks for query: {query}")
        
        if True :
            from together import Together
            client = Together()

            # Implement reranking logic here if you have a reranker
            response = client.rerank.create(
                model="Salesforce/Llama-Rank-V1",
                query=query,
                documents=chunks_to_rerank,
                top_n=top_n 
            )
            logger.info(f"Reranking response: {[result.index for result in response.results]}")
            retreived_chunks = []
            retrieved_index = []
            for result in response.results:
                retreived_chunks.append(chunks_to_rerank[result.index])
                retrieved_index.append(chunk_indices[result.index])
            return retreived_chunks, retrieved_index

    except Exception as e:
        logger.info(f"Reranking failed: {e}")
        # Fallback to returning the top_n chunks without reranking
        # This is a simple fallback due to credit limits or other issues
        logger.info("Using fallback method for reranking")    
        return chunks_to_rerank[:top_n], chunk_indices[:top_n]


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    Normalize embeddings using L2 normalization
    
    Args:
        embeddings: Input embeddings array
    
    Returns:
        Normalized embeddings
    """
    # Ensure the array is in the correct format
    if len(embeddings.shape) == 1:
        embeddings = embeddings.reshape(1, -1)
    
    # L2 normalization
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
    return embeddings / norms


def format_context(chunks: List[str], numbered: bool = False) -> str:
    """
    Format chunks into a context string
    
    Args:
        chunks: List of text chunks
        numbered: Whether to add numbers to chunks
    
    Returns:
        Formatted context string
    """
    if numbered:
        return "\n\n".join(f"[{i+1}] {chunk}" for i, chunk in enumerate(chunks))
    else:
        return "\n\n".join(chunks)


def validate_search_results(bm25_ids: List[int], dense_ids: List[int], max_chunks: int) -> Tuple[List[int], List[int]]:
    """
    Validate and clean search results
    
    Args:
        bm25_ids: BM25 search result indices
        dense_ids: Dense search result indices  
        max_chunks: Maximum number of chunks available
    
    Returns:
        Tuple of (cleaned_bm25_ids, cleaned_dense_ids)
    """
    # Remove invalid indices (negative or beyond chunk count)
    cleaned_bm25 = [idx for idx in bm25_ids if 0 <= idx < max_chunks]
    cleaned_dense = [idx for idx in dense_ids if 0 <= idx < max_chunks]
    
    return cleaned_bm25, cleaned_dense