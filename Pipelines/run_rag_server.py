#!/usr/bin/env python3
"""
Local RAG Server for Pipeline Testing

This server uses the Pipeline's local components (core/, config/, utils/)
to provide the same API as the main RAG server for testing purposes.
"""

import os
import sys
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

# Add current directory to path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from config.pipeline_config import PipelineConfig
from core.rag_engine import RAGEngine

# Load environment variables
load_dotenv()

# Global RAG engine instance
rag_engine: Optional[RAGEngine] = None

# Request/Response models (matching main server API)
class QueryRequest(BaseModel):
    query: str
    model: str = None

class Query(BaseModel):
    query: str

class RAGResponse(BaseModel):
    question: str
    answer: str
    context: str

# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_engine
    
    # Startup
    print("🚀 Starting Local RAG Server...")
    
    try:
        # Initialize configuration
        config = PipelineConfig()
        
        # Validate configuration
        if not config.validate():
            print("❌ Configuration validation failed")
            raise Exception("Configuration validation failed")
        
        # Initialize RAG engine
        rag_engine = RAGEngine(config)
        
        # Load artifacts
        if not rag_engine.initialize():
            print("❌ RAG engine initialization failed")
            raise Exception("RAG engine initialization failed")
        
        print("✅ Local RAG Server ready!")
        print(f"📊 Configuration: {config.to_dict()}")
        
    except Exception as e:
        print(f"❌ Server startup failed: {e}")
        raise e
    
    yield
    
    # Shutdown
    print("🛑 Shutting down Local RAG Server...")

# FastAPI app
app = FastAPI(
    title="Local RAG Server",
    description="UAE Information Assurance RAG System (Local Testing)",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "UAE Information Assurance Local RAG Server",
        "status": "ready",
        "mode": "local_testing"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    global rag_engine
    
    if not rag_engine:
        return {
            "status": "unhealthy",
            "message": "RAG engine not initialized",
            "artifacts_loaded": False,
            "chunk_count": 0
        }
    
    try:
        health_status = rag_engine.get_health_status()
        return {
            "status": "healthy" if health_status["ready"] else "unhealthy",
            "artifacts_loaded": health_status["retrieval_service"]["loaded"],
            "chunk_count": health_status["retrieval_service"]["chunk_count"],
            "faiss_loaded": health_status["retrieval_service"]["faiss_loaded"],
            "bm25_loaded": health_status["retrieval_service"]["bm25_loaded"],
            "config": health_status["retrieval_service"]["config"]
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"Health check failed: {e}",
            "artifacts_loaded": False,
            "chunk_count": 0
        }

@app.post("/query", response_model=RAGResponse)
async def query_rag(request: QueryRequest):
    """
    Query the RAG system (non-streaming)
    Compatible with main server API
    """
    global rag_engine
    
    if not rag_engine:
        raise HTTPException(status_code=500, detail="RAG engine not initialized")
    
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        # Process query using RAG engine
        result = rag_engine.query(
            query=request.query,
            model=request.model,
            stream=False
        )
        
        return RAGResponse(
            question=result["question"],
            answer=result["answer"],
            context=result["context"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing failed: {e}")

@app.post("/retrieve")
async def retrieve_streaming(q: Query):
    """
    OpenWebUI compatible streaming endpoint
    Compatible with main server API
    """
    global rag_engine
    
    if not rag_engine:
        raise HTTPException(status_code=500, detail="RAG engine not initialized")
    
    if not q.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        # Process query with streaming
        result = rag_engine.query(
            query=q.query,
            model=None,
            stream=True
        )
        
        def stream_generator():
            """Generator for streaming response"""
            try:
                for chunk in result["response_stream"]:
                    yield chunk
            except Exception as e:
                yield f"Error: {e}"
        
        return StreamingResponse(
            stream_generator(),
            media_type="text/plain; charset=utf-8"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Streaming query failed: {e}")

@app.get("/status")
async def get_status():
    """Get detailed server status"""
    global rag_engine
    
    if not rag_engine:
        return {
            "server_status": "not_initialized",
            "rag_engine": None,
            "config": None
        }
    
    try:
        health_status = rag_engine.get_health_status()
        return {
            "server_status": "ready",
            "rag_engine_status": health_status,
            "config": health_status["retrieval_service"]["config"]
        }
    except Exception as e:
        return {
            "server_status": "error",
            "error": str(e),
            "rag_engine": None
        }

# Interactive endpoint for testing
@app.get("/query/{query_text}")
async def query_get(query_text: str, model: str = None):
    """GET endpoint for quick testing"""
    global rag_engine
    
    if not rag_engine:
        raise HTTPException(status_code=500, detail="RAG engine not initialized")
    
    try:
        result = rag_engine.query(
            query=query_text,
            model=model,
            stream=False
        )
        
        return {
            "question": result["question"],
            "answer": result["answer"],
            "context": result["context"][:500] + "..." if len(result["context"]) > 500 else result["context"],
            "chunks_used": result["chunks_used"],
            "metadata": result["metadata"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")

if __name__ == "__main__":
    import uvicorn
    
    # Server configuration
    host = "0.0.0.0"
    port = 8100
    
    print(f"🚀 Starting Local RAG Server on {host}:{port}")
    print("📝 Endpoints:")
    print(f"  - Health: http://{host}:{port}/health")
    print(f"  - Query: http://{host}:{port}/query")
    print(f"  - Retrieve: http://{host}:{port}/retrieve")
    print(f"  - Status: http://{host}:{port}/status")
    
    # Run the server
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        reload=False  # Disable reload for Docker
    )