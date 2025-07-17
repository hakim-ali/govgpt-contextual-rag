"""
Retrieval Service for RAG Pipeline
"""
import os
import json
import pickle
import numpy as np
from typing import List, Tuple, Dict, Any
from rank_bm25 import BM25Okapi
from utils.logger import get_logger
from config.pipeline_config import PipelineConfig
from utils.helpers import rrf, rerank_chunks, validate_search_results

# Import PGVector KnowledgeBaseManager
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'govgpt-kb', 'src'))
from database import get_kb_manager

logger = get_logger(__name__)

# Remove FAISS GPU references since we're using PGVector
# faiss.GpuIndexIVFFlat = None
# faiss.StandardGpuResources = None
# faiss.GpuIndexFlatIP = None

class RetrievalService:
    """Service for document retrieval using BM25 and PGVector"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.chunks_data = None
        self.enriched_chunks_data = None
        self.bm25_index = None
        self.kb_manager = None
        self.is_loaded = False
        
        # BM25 cache settings
        self.bm25_cache_dir = os.path.join(os.path.dirname(config.chunks_path), 'bm25_cache')
        self.bm25_cache_path = os.path.join(self.bm25_cache_dir, 'bm25_enriched.pkl')
        os.makedirs(self.bm25_cache_dir, exist_ok=True)
    
    def load_artifacts(self) -> bool:
        """
        Load precomputed artifacts (chunks, BM25 index, PGVector KB)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Initialize PGVector KnowledgeBaseManager
            self.kb_manager = get_kb_manager()
            if not self.kb_manager:
                logger.error("❌ KnowledgeBaseManager not available")
                return False
            
            # Get chunk count to verify KB is populated
            chunk_count = self.kb_manager.get_kb_embeddings_count()
            if chunk_count == 0:
                logger.error("❌ No chunks found in knowledge base")
                return False
            
            # Load enriched chunk texts for BM25 (try cache first)
            if self._load_cached_bm25():
                logger.info(f"✅ Loaded cached BM25 index with {len(self.enriched_chunks_data)} documents")
            else:
                logger.info("🔄 Building BM25 index from knowledge base...")
                self._build_bm25_from_kb()
                self._cache_bm25_index()
                logger.info(f"✅ Built and cached BM25 index with {len(self.enriched_chunks_data)} documents")
            
            # Use enriched chunks as primary data source (no backward compatibility)
            self.chunks_data = self.enriched_chunks_data
            logger.info(f"✅ Using enriched chunks as primary data source: {len(self.chunks_data)} documents")
            
            logger.info(f"✅ Loaded PGVector knowledge base with {chunk_count} chunks")
            logger.info(f"✅ Loaded BM25 index with {len(self.enriched_chunks_data)} enriched documents")
            
            self.is_loaded = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading artifacts: {e}")
            return False
    
    def _load_cached_bm25(self) -> bool:
        """Load cached BM25 index if available"""
        try:
            if os.path.exists(self.bm25_cache_path):
                with open(self.bm25_cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.bm25_index = cache_data['bm25_index']
                    self.enriched_chunks_data = cache_data['enriched_chunks_data']
                return True
            return False
        except Exception as e:
            logger.warning(f"Failed to load cached BM25 index: {e}")
            return False
    
    def _build_bm25_from_kb(self):
        """Build BM25 index from knowledge base enriched_chunk_text"""
        try:
            # Get all enriched chunk texts from knowledge base
            metadata = self.kb_manager.get_kb_chunk_metadata()
            
            # Use enriched_chunk_text if available, fallback to chunk_text
            self.enriched_chunks_data = []
            for chunk_meta in metadata:
                enriched_text = chunk_meta.get('enriched_chunk_text', '').strip()
                if enriched_text:
                    self.enriched_chunks_data.append(enriched_text)
                else:
                    # Fallback to chunk_text
                    self.enriched_chunks_data.append(chunk_meta.get('chunk_text', ''))
            
            # Build BM25 index
            tokenized_docs = [doc.split() for doc in self.enriched_chunks_data]
            self.bm25_index = BM25Okapi(tokenized_docs)
            
            logger.info(f"✅ Built BM25 index from {len(self.enriched_chunks_data)} enriched chunks")
            
        except Exception as e:
            logger.error(f"❌ Failed to build BM25 from knowledge base: {e}")
            raise
    
    def _cache_bm25_index(self):
        """Cache BM25 index for future use"""
        try:
            cache_data = {
                'bm25_index': self.bm25_index,
                'enriched_chunks_data': self.enriched_chunks_data,
                'created_at': os.path.getmtime(self.bm25_cache_path) if os.path.exists(self.bm25_cache_path) else None
            }
            
            with open(self.bm25_cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
                
            logger.info(f"✅ Cached BM25 index to {self.bm25_cache_path}")
            
        except Exception as e:
            logger.warning(f"Failed to cache BM25 index: {e}")
    
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
            logger.info(f"Warning: Lexical (BM25) search failed: {e}")
            return []
    
    def vector_search(self, query_embedding: np.ndarray) -> List[int]:
        """
        Perform dense vector search using PGVector
        
        Args:
            query_embedding: Query embedding vector
            
        Returns:
            List of document indices ranked by similarity
        """
        if not self.is_loaded or self.kb_manager is None:
            raise ValueError("Artifacts not loaded. Call load_artifacts() first.")
        
        try:
            # Convert numpy array to list for PGVector
            query_embedding_list = query_embedding.flatten().tolist()
            
            # Search using PGVector cosine similarity
            results = self.kb_manager.search_kb_similar_vectors(
                query_embedding=query_embedding_list,
                limit=self.config.VECTOR_K
            )
            
            # Convert results to indices
            # Results format: List[Tuple[chunk_id, chunk_text, distance]]
            # Map chunk_id to index in enriched_chunks_data
            indices = []
            for chunk_id, chunk_text, distance in results:
                try:
                    # Find index of this chunk in enriched_chunks_data
                    if chunk_text in self.enriched_chunks_data:
                        idx = self.enriched_chunks_data.index(chunk_text)
                        indices.append(idx)
                    else:
                        # Fallback: use chunk_id as index if numeric
                        if chunk_id.isdigit():
                            idx = int(chunk_id) - 1  # Convert 1-based to 0-based
                            if 0 <= idx < len(self.enriched_chunks_data):
                                indices.append(idx)
                except (ValueError, IndexError):
                    continue
            
            return indices
            
        except Exception as e:
            logger.info(f"Warning: Vector search failed: {e}")
            return []
    
    def hybrid_search(self, query: str, query_embedding: np.ndarray) -> List[int]:
        """
        Perform hybrid search combining BM25 and PGVector search using RRF
        
        Args:
            query: Search query text
            query_embedding: Query embedding vector
            
        Returns:
            List of document indices ranked by hybrid score
        """
        if not self.is_loaded:
            raise ValueError("Artifacts not loaded. Call load_artifacts() first.")
        
        # BM25 search on enriched chunks
        bm25_ids = self.bm25_search(query)
        
        # PGVector search
        dense_ids = self.vector_search(query_embedding)
        
        # Fallback to BM25 if vector search fails
        if not dense_ids:
            dense_ids = bm25_ids[:self.config.VECTOR_K]
        
        # Validate search results against enriched chunks
        max_chunks = len(self.enriched_chunks_data)
        bm25_ids, dense_ids = validate_search_results(bm25_ids, dense_ids, max_chunks)
        
        # Reciprocal Rank Fusion
        fused_ids = rrf([dense_ids, bm25_ids], k=self.config.RRF_K)[:self.config.TOP_K]
        
        logger.info(f"🔀 RRF combined results: {len(fused_ids)} final chunks (PGVector + BM25)")
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
        
        logger.info(f"🔄 Reranked to {len(final_chunks_text)} chunks after reranking")
        return final_chunks_text, final_indices
    
    def get_status(self) -> Dict[str, Any]:
        """Get status information about the retrieval service"""
        kb_stats = {}
        if self.kb_manager:
            try:
                kb_stats = self.kb_manager.get_kb_stats()
            except Exception as e:
                logger.warning(f"Could not get KB stats: {e}")
                
        return {
            "loaded": self.is_loaded,
            "chunk_count": len(self.chunks_data) if self.chunks_data else 0,
            "enriched_chunk_count": len(self.enriched_chunks_data) if self.enriched_chunks_data else 0,
            "bm25_loaded": self.bm25_index is not None,
            "bm25_cached": os.path.exists(self.bm25_cache_path),
            "pgvector_loaded": self.kb_manager is not None,
            "knowledge_base_stats": kb_stats,
            "config": self.config.to_dict()
        }