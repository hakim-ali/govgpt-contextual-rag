"""
Main RAG Engine for OpenWebUI Pipeline
"""
import openai
from typing import Dict, Any, Generator
from tenacity import retry, wait_exponential, stop_after_attempt

from config.pipeline_config import PipelineConfig
from .embedding_service import EmbeddingService
from .retrieval_service import RetrievalService
from utils.helpers import format_context
from utils.logger import get_logger
logger = get_logger(__name__)


class RAGEngine:
    """Main RAG Engine coordinating all components"""
    
    def __init__(self, config: PipelineConfig = None):
        if config is None:
            config = PipelineConfig()
        
        self.config = config
        self.embedding_service = EmbeddingService(config)
        self.retrieval_service = RetrievalService(config)
        
        # LLM client
        self.llm_client = openai.OpenAI(
            api_key=config.OPENAI_API_KEY,
            base_url=config.OPENAI_API_BASE
        )
        
        self.is_ready = False
    
    def initialize(self) -> bool:
        """
        Initialize the RAG engine by loading all artifacts
        
        Returns:
            True if successful, False otherwise
        """
        try:
            success = self.retrieval_service.load_artifacts()
            if success:
                self.is_ready = True
                logger.info("✅ RAG Engine initialized successfully")
            else:
                logger.info("❌ RAG Engine initialization failed")
            return success
        except Exception as e:
            logger.info(f"❌ RAG Engine initialization error: {e}")
            return False
    
    @retry(wait=wait_exponential(multiplier=1, min=2, max=1024), stop=stop_after_attempt(10))
    def generate_response(self, prompt: str, model: str = None) -> str:
        """
        Generate response using LLM
        
        Args:
            prompt: Input prompt
            model: Model to use (optional)
            
        Returns:
            Generated response
        """
        selected_model = model or self.config.RAG_MODEL
        
        response = self.llm_client.chat.completions.create(
            model=selected_model,
            messages=[{
                "role": "user",
                "content": prompt
            }],
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
    
    @retry(wait=wait_exponential(multiplier=1, min=2, max=1024), stop=stop_after_attempt(10))
    def generate_response_streaming(self, prompt: str, model: str = None) -> Generator[str, None, None]:
        """
        Generate streaming response using LLM
        
        Args:
            prompt: Input prompt
            model: Model to use (optional)
            
        Yields:
            Response chunks
        """
        selected_model = model or self.config.RAG_MODEL
        
        response = self.llm_client.chat.completions.create(
            model=selected_model,
            messages=[{
                "role": "user",
                "content": prompt
            }],
            temperature=0.7,
            stream=True
        )
        
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
    
    def query(self, query: str, model: str = None, stream: bool = False) -> Dict[str, Any]:
        """
        Process a RAG query
        
        Args:
            query: User query
            model: Model to use (optional)
            stream: Whether to return streaming response
            
        Returns:
            Dict containing response and metadata
        """
        if not self.is_ready:
            raise ValueError("RAG Engine not initialized. Call initialize() first.")
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_service.embed_query(query)
            
            # Retrieve and rerank relevant chunks
            final_chunks, final_indices = self.retrieval_service.retrieve_and_rerank(
                query, query_embedding
            )
            
            # Format context
            context = format_context(final_chunks)
            
            # Create prompt
            prompt = self.config.RAG_PROMPT_TEMPLATE.format(
                context=context,
                query=query
            )
            
            # Generate response
            if stream:
                response_generator = self.generate_response_streaming(prompt, model)
                return {
                    "question": query,
                    "response_stream": response_generator,
                    "context": context,
                    "chunks_used": len(final_chunks),
                    "metadata": {
                        "chunk_indices": final_indices,
                        "embedding_model": self.config.EMBED_MODEL,
                        "rag_model": model or self.config.RAG_MODEL
                    }
                }
            else:
                answer = self.generate_response(prompt, model)
                return {
                    "question": query,
                    "answer": answer,
                    "context": context,
                    "chunks_used": len(final_chunks),
                    "metadata": {
                        "chunk_indices": final_indices,
                        "embedding_model": self.config.EMBED_MODEL,
                        "rag_model": model or self.config.RAG_MODEL
                    }
                }
                
        except Exception as e:
            raise Exception(f"Error processing RAG query: {e}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the RAG engine"""
        retrieval_status = self.retrieval_service.get_status()
        embedding_info = self.embedding_service.get_model_info()
        
        return {
            "ready": self.is_ready,
            "config_valid": self.config.validate(),
            "retrieval_service": retrieval_status,
            "embedding_service": embedding_info,
            "llm_configured": bool(self.config.OPENAI_API_KEY)
        }