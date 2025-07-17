"""
Embedding Service for RAG Pipeline
"""
import os
import numpy as np
import openai
from typing import List, Optional
from tenacity import retry, wait_exponential, stop_after_attempt
from config.pipeline_config import PipelineConfig
from utils.logger import get_logger
logger = get_logger(__name__)

class EmbeddingService:
    """Service for generating embeddings using various providers"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.client = openai.OpenAI(
            api_key=config.OPENAI_API_KEY,
            base_url=config.OPENAI_API_BASE
        )
    
    @retry(wait=wait_exponential(multiplier=1, min=2, max=1024), stop=stop_after_attempt(10))
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        embs = []
        
        if self.config.EMBED_MODEL == "azure_ai/embed-v-4-0":
            # Batch requests for azure_ai/embed-v-4-0
            batch_size = 96
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                response = self.client.embeddings.create(
                    model=self.config.EMBED_MODEL, 
                    input=batch_texts, 
                    encoding_format="float"
                )
                embs.extend(item.embedding for item in response.data)
                
        elif self.config.EMBED_MODEL == "huggingface/Qwen/Qwen3-Embedding-8B":
            # Batch requests for Qwen embedding model
            batch_size = 32
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                response = self.client.embeddings.create(
                    model=self.config.EMBED_MODEL,
                    input=batch_texts,
                    encoding_format="float"
                )
                embs.extend(item.embedding for item in response.data)
        else:
            # Default: single request for other models
            response = self.client.embeddings.create(
                model=self.config.EMBED_MODEL, 
                input=texts, 
                encoding_format="float"
            )
            embs.extend(item.embedding for item in response.data)

        return embs
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query
        
        Args:
            query: Query string to embed
            
        Returns:
            Query embedding as numpy array
        """
        try:
            embeddings = self.generate_embeddings([query])
            query_emb = np.array(embeddings, dtype=np.float32).reshape(1, -1)
            
            # Normalize using FAISS normalization
            # import faiss
            # faiss.GpuIndexIVFFlat = None
            # faiss.StandardGpuResources = None
            # faiss.GpuIndexFlatIP = None
            # faiss.normalize_L2(query_emb)
            
            return query_emb
        except Exception as e:
            logger.info(f"Warning: Embedding generation failed: {e}")
            raise e
    
    def get_model_info(self) -> dict:
        """Get information about the current embedding model"""
        return {
            "model": self.config.EMBED_MODEL,
            "model_suffix": self.config.MODEL_SUFFIX,
            "provider": "openai" if "openai" in self.config.EMBED_MODEL else "custom"
        }