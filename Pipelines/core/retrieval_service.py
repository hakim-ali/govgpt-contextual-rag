"""
Retrieval Service for RAG Pipeline
"""
import os
import json
import pickle
import numpy as np
import faiss
from typing import List, Tuple, Dict, Any
from rank_bm25 import BM25Okapi

from config.pipeline_config import PipelineConfig
from utils.helpers import rrf, rerank_chunks, validate_search_results


class RetrievalService:
    """Service for document retrieval using BM25 and FAISS"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.chunks_data = None
        self.bm25_index = None
        self.faiss_index = None
        self.is_loaded = False
    
    def load_artifacts(self) -> bool:
        """
        Load precomputed artifacts (chunks, BM25 index, FAISS index)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load contextual chunks
            with open(self.config.chunks_path) as fp:
                chunks = json.load(fp)
                self.chunks_data = [chunk_info["original_text"] for chunk_info in chunks['chunk_metadata']]
        
            # Load BM25 index
            with open(self.config.bm25_path, 'rb') as f:
                self.bm25_index = pickle.load(f)
            
            # Load FAISS index  
            self.faiss_index = faiss.read_index(self.config.faiss_path)
            
            print(f"✅ Loaded {len(self.chunks_data)} contextual chunks for model {self.config.EMBED_MODEL}")
            print(f"✅ Loaded BM25 index with {len(self.chunks_data)} documents")
            print(f"✅ Loaded FAISS index with dimension {self.faiss_index.d}")
            
            self.is_loaded = True
            return True
            
        except FileNotFoundError as e:
            print(f"❌ Model-specific artifacts not found for {self.config.EMBED_MODEL}")
            print(f"❌ Missing file: {e.filename}")
            print(f"💡 Please run preprocessing with EMBED_MODEL={self.config.EMBED_MODEL}")
            return False
                
        except Exception as e:
            print(f"❌ Error loading artifacts: {e}")
            return False
    
    def bm25_search(self, query: str) -> List[int]:
        """
        Perform BM25 keyword search
        
        Args:
            query: Search query
            
        Returns:
            List of document indices ranked by BM25 score
        """
        if not self.is_loaded or self.bm25_index is None:
            raise ValueError("Artifacts not loaded. Call load_artifacts() first.")
        
        try:
            bm25_scores = self.bm25_index.get_scores(query.split())
            bm25_ids = list(np.argsort(bm25_scores)[::-1][:self.config.BM25_K])
            return bm25_ids
        except Exception as e:
            print(f"Warning: Lexical (BM25) search failed: {e}")
            return []
    
    def vector_search(self, query_embedding: np.ndarray) -> List[int]:
        """
        Perform dense vector search using FAISS
        
        Args:
            query_embedding: Query embedding vector
            
        Returns:
            List of document indices ranked by similarity
        """
        if not self.is_loaded or self.faiss_index is None:
            raise ValueError("Artifacts not loaded. Call load_artifacts() first.")
        
        try:
            _, dense_ids = self.faiss_index.search(query_embedding, self.config.VECTOR_K)
            return dense_ids[0].tolist()
        except Exception as e:
            print(f"Warning: Vector search failed: {e}")
            return []
    
    def hybrid_search(self, query: str, query_embedding: np.ndarray) -> List[int]:
        """
        Perform hybrid search combining BM25 and vector search using RRF
        
        Args:
            query: Search query text
            query_embedding: Query embedding vector
            
        Returns:
            List of document indices ranked by hybrid score
        """
        if not self.is_loaded:
            raise ValueError("Artifacts not loaded. Call load_artifacts() first.")
        
        # BM25 search
        bm25_ids = self.bm25_search(query)
        
        # Vector search
        dense_ids = self.vector_search(query_embedding)
        
        # Fallback to BM25 if vector search fails
        if not dense_ids:
            dense_ids = bm25_ids[:self.config.VECTOR_K]
        
        # Validate search results
        max_chunks = len(self.chunks_data)
        bm25_ids, dense_ids = validate_search_results(bm25_ids, dense_ids, max_chunks)
        
        # Reciprocal Rank Fusion
        fused_ids = rrf([dense_ids, bm25_ids], k=self.config.RRF_K)[:self.config.TOP_K]
        
        print(f"🔀 RRF combined results: {len(fused_ids)} final chunks")
        return fused_ids
    
    def get_chunks(self, indices: List[int]) -> List[str]:
        """
        Get text chunks by indices
        
        Args:
            indices: List of chunk indices
            
        Returns:
            List of text chunks
        """
        if not self.is_loaded or self.chunks_data is None:
            raise ValueError("Artifacts not loaded. Call load_artifacts() first.")
        
        return [self.chunks_data[i] for i in indices if 0 <= i < len(self.chunks_data)]
    
    def retrieve_and_rerank(self, query: str, query_embedding: np.ndarray) -> Tuple[List[str], List[int]]:
        """
        Complete retrieval pipeline with reranking
        
        Args:
            query: Search query
            query_embedding: Query embedding vector
            
        Returns:
            Tuple of (final_chunks, final_indices)
        """
        # Hybrid search
        fused_indices = self.hybrid_search(query, query_embedding)
        
        # Reranking
        final_chunks_text, final_indices = rerank_chunks(
            query, self.chunks_data, fused_indices, self.config.TOP_N
        )
        
        print(f"🔄 Reranked to {len(final_chunks_text)} chunks after reranking")
        return final_chunks_text, final_indices
    
    def get_status(self) -> Dict[str, Any]:
        """Get status information about the retrieval service"""
        return {
            "loaded": self.is_loaded,
            "chunk_count": len(self.chunks_data) if self.chunks_data else 0,
            "bm25_loaded": self.bm25_index is not None,
            "faiss_loaded": self.faiss_index is not None,
            "faiss_dimension": self.faiss_index.d if self.faiss_index else None,
            "config": self.config.to_dict()
        }