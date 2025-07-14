"""
Configuration for the Contextual RAG Pipeline
"""
import os
from typing import Dict, Any


class PipelineConfig:
    """Configuration management for RAG Pipeline"""
    
    def __init__(self, artifact_dir: str = None):
        # Set artifact directory - default to Pipeline root's artifacts folder
        if artifact_dir is None:
            # Pipeline folder is the root, so artifacts are in ./artifacts
            # pipeline_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            # artifact_dir = os.path.join(pipeline_root, "artifacts")
            artifact_dir = os.getenv("ARTIFACT_DIR", "./artifacts")
        
        self.ARTIFACT_DIR = artifact_dir
        
        # RAG Model Configuration
        self.RAG_MODEL = os.getenv("RAG_MODEL", "gpt-4.1")
        self.EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")
        
        # Retrieval Parameters
        self.VECTOR_K = int(os.getenv("VECTOR_K", 75))
        self.BM25_K = int(os.getenv("BM25_K", 75))
        self.RRF_K = int(os.getenv("RRF_K", 60))
        self.TOP_K = int(os.getenv("TOP_K", 20))
        self.TOP_N = int(os.getenv("TOP_N", 5))
        
        # API Configuration
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
        
        # Model-specific artifact naming
        self.MODEL_SUFFIX = self.EMBED_MODEL.replace("/", "_").replace("-", "_")
        
        # File paths
        self.chunks_path = os.path.join(self.ARTIFACT_DIR, 'enriched_chunks.json')
        self.bm25_path = os.path.join(self.ARTIFACT_DIR, 'bm25.pkl')
        self.faiss_path = os.path.join(self.ARTIFACT_DIR, f'faiss_{self.MODEL_SUFFIX}.idx')
        
        # RAG Prompt Template
        self.RAG_PROMPT_TEMPLATE = """
You are an expert assistant providing formal, accurate, and context-based answers. 

Use only the information from the context below to respond to the question. 
Do not reference or cite documents. Do not include assumptions or external knowledge.
If the answer is not directly available in the context, state that based on the current information.

### Context:
{context}

### Question:
{query}

### Answer:
"""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "artifact_dir": self.ARTIFACT_DIR,
            "rag_model": self.RAG_MODEL,
            "embed_model": self.EMBED_MODEL,
            "vector_k": self.VECTOR_K,
            "bm25_k": self.BM25_K,
            "rrf_k": self.RRF_K,
            "top_k": self.TOP_K,
            "top_n": self.TOP_N,
            "model_suffix": self.MODEL_SUFFIX
        }
    
    def validate(self) -> bool:
        """Validate configuration and required files"""
        required_files = [self.chunks_path, self.bm25_path, self.faiss_path]
        
        for file_path in required_files:
            if not os.path.exists(file_path):
                print(f"❌ Missing required file: {file_path}")
                return False
        
        if not self.OPENAI_API_KEY:
            print("⚠️  Warning: OPENAI_API_KEY not set")
            
        return True