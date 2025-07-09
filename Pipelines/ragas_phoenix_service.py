#!/usr/bin/env python3
"""
RAGAS + Phoenix Evaluation Service for GovGPT RAG Pipeline

This service provides real-time evaluation using RAGAS metrics with Phoenix UI integration.
Supports ground truth matching with fuzzy search fallback.

Author: GovGPT Team
Version: 1.0.0
"""

import os
import sys
import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import difflib

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# RAGAS imports
try:
    from ragas.metrics import (
        ContextPrecision,
        ContextRecall, 
        ContextEntityRecall,
        AnswerRelevancy,
        Faithfulness
    )
    from ragas.dataset_schema import SingleTurnSample
    from ragas import evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    RAGAS_AVAILABLE = True
except ImportError as e:
    RAGAS_AVAILABLE = False
    print(f"⚠️ RAGAS not available: {e}")

# Phoenix imports
try:
    import phoenix as px
    from phoenix.otel import register
    from phoenix.trace import DocumentEvaluations, SpanEvaluations
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk import trace as trace_sdk
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    PHOENIX_AVAILABLE = True
except ImportError as e:
    PHOENIX_AVAILABLE = False
    print(f"⚠️ Phoenix not available: {e}")

# LLM and Embedding imports
try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_community.embeddings import HuggingFaceEmbeddings
    LLM_AVAILABLE = True
except ImportError as e:
    LLM_AVAILABLE = False
    print(f"⚠️ LangChain not available: {e}")

# Ground truth samples
try:
    from ground_truth_samples import GROUND_TRUTH_SAMPLES
    GROUND_TRUTH_AVAILABLE = True
    print(f"✅ Loaded {len(GROUND_TRUTH_SAMPLES)} ground truth samples")
except ImportError:
    GROUND_TRUTH_AVAILABLE = False
    GROUND_TRUTH_SAMPLES = {}
    print("⚠️ Ground truth samples not available")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="RAGAS + Phoenix Evaluation Service",
    description="Real-time RAG evaluation with RAGAS metrics and Phoenix UI integration",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class EvaluationRequest(BaseModel):
    query: str = Field(..., description="User query/question")
    response: str = Field(..., description="Generated response/answer")
    context: str = Field(..., description="Retrieved context")
    user_id: Optional[str] = Field(None, description="Optional user identifier")

class EvaluationResponse(BaseModel):
    status: str
    message: str
    evaluation_id: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None
    ground_truth_found: Optional[bool] = None
    ground_truth_match_score: Optional[float] = None

# Global service state
class ServiceState:
    def __init__(self):
        self.llm = None
        self.embeddings = None
        self.phoenix_session = None
        self.tracer = None
        self.ragas_metrics = None
        self.ragas_initialized = False
        self.phoenix_initialized = False
    
    async def initialize(self):
        """Initialize RAGAS, Phoenix, and LLM components"""
        await self._initialize_llm()
        await self._initialize_embeddings()
        await self._initialize_phoenix()
        await self._initialize_ragas()
    
    async def _initialize_llm(self):
        """Initialize LLM for RAGAS evaluation"""
        if not LLM_AVAILABLE:
            logger.error("LangChain not available for LLM initialization")
            return
        
        try:
            api_key = os.getenv("EVALUATION_API_KEY")
            api_base = os.getenv("EVALUATION_API_BASE", "https://litellm.sandbox.dge.gov.ae/v1")
            model = os.getenv("EVALUATION_MODEL", "gpt-4.1")
            
            if not api_key:
                raise ValueError("EVALUATION_API_KEY not configured")
            
            self.llm = ChatOpenAI(
                model=model,
                openai_api_key=api_key,
                openai_api_base=api_base,
                temperature=0.3,
                request_timeout=600
            )
            
            logger.info(f"✅ LLM initialized: {model}")
            
        except Exception as e:
            logger.error(f"❌ LLM initialization failed: {e}")
            self.llm = None
    
    async def _initialize_embeddings(self):
        """Initialize embeddings for ResponseRelevancy metric"""
        try:
            embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
            
            if "ada-002" in embedding_model or "text-embedding" in embedding_model:
                # Use OpenAI embeddings
                api_key = os.getenv("EVALUATION_API_KEY")
                api_base = os.getenv("EVALUATION_API_BASE")
                
                self.embeddings = OpenAIEmbeddings(
                    model=embedding_model,
                    openai_api_key=api_key,
                    openai_api_base=api_base
                )
            else:
                # Use HuggingFace embeddings as fallback
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
            
            logger.info(f"✅ Embeddings initialized: {embedding_model}")
            
        except Exception as e:
            logger.error(f"❌ Embeddings initialization failed: {e}")
            # Use local embeddings as fallback
            try:
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
                logger.info("✅ Fallback embeddings initialized")
            except Exception as fallback_e:
                logger.error(f"❌ Fallback embeddings failed: {fallback_e}")
                self.embeddings = None
    
    async def _initialize_phoenix(self):
        """Initialize Phoenix observability"""
        if not PHOENIX_AVAILABLE:
            logger.error("Phoenix not available")
            return
        
        try:
            # Use hosted Phoenix endpoint
            phoenix_endpoint = os.getenv("PHOENIX_ENDPOINT", "http://74.162.37.71:8010/v1/traces")
            
            # Register OpenTelemetry to send traces to hosted Phoenix
            tracer_provider = register(
                project_name="govgpt-ragas-evaluation",
                endpoint=phoenix_endpoint
            )
            
            self.tracer = trace.get_tracer(__name__)
            self.phoenix_initialized = True
            
            logger.info(f"✅ Phoenix initialized with hosted endpoint: {phoenix_endpoint}")
            
        except Exception as e:
            logger.error(f"❌ Phoenix initialization failed: {e}")
            self.phoenix_initialized = False
    
    async def _initialize_ragas(self):
        """Initialize RAGAS metrics"""
        if not RAGAS_AVAILABLE:
            logger.error("RAGAS not available")
            return
        
        if not self.llm:
            logger.error("LLM required for RAGAS initialization")
            return
        
        try:
            # Wrap LLM and embeddings for RAGAS
            ragas_llm = LangchainLLMWrapper(self.llm)
            ragas_embeddings = LangchainEmbeddingsWrapper(self.embeddings) if self.embeddings else None
            
            # Configure RAGAS metrics with our LLM using new API
            self.ragas_metrics = [
                ContextPrecision(llm=ragas_llm),
                ContextRecall(llm=ragas_llm),
                ContextEntityRecall(llm=ragas_llm),
                Faithfulness(llm=ragas_llm)
            ]
            
            # Add AnswerRelevancy if embeddings available
            if ragas_embeddings:
                self.ragas_metrics.append(
                    AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)
                )
                logger.info("✅ All 5 RAGAS metrics initialized (including AnswerRelevancy)")
            else:
                logger.warning("⚠️ Only 4 RAGAS metrics initialized (no embeddings for AnswerRelevancy)")
            
            self.ragas_initialized = True
            
        except Exception as e:
            logger.error(f"❌ RAGAS initialization failed: {e}")
            self.ragas_initialized = False

# Global service instance
service = ServiceState()

def find_ground_truth(query: str, threshold: float = 0.6) -> tuple[Optional[str], float]:
    """
    Find ground truth answer using fuzzy string matching
    
    Args:
        query: User query to match
        threshold: Minimum similarity score (0.0 to 1.0)
    
    Returns:
        Tuple of (ground_truth_answer, match_score)
    """
    if not GROUND_TRUTH_AVAILABLE or not GROUND_TRUTH_SAMPLES:
        return None, 0.0
    
    query_normalized = query.strip().lower()
    best_match = None
    best_score = 0.0
    
    for gt_query, gt_answer in GROUND_TRUTH_SAMPLES.items():
        gt_query_normalized = gt_query.strip().lower()
        
        # Calculate similarity score
        similarity = difflib.SequenceMatcher(None, query_normalized, gt_query_normalized).ratio()
        
        if similarity > best_score and similarity >= threshold:
            best_score = similarity
            best_match = gt_answer
    
    return best_match, best_score

async def evaluate_with_ragas(
    query: str, 
    response: str, 
    context: str,
    ground_truth: Optional[str] = None
) -> Dict[str, float]:
    """
    Evaluate using RAGAS metrics
    
    Args:
        query: User query
        response: Generated response
        context: Retrieved context
        ground_truth: Optional ground truth answer
    
    Returns:
        Dictionary of metric scores
    """
    if not service.ragas_initialized:
        raise HTTPException(status_code=503, detail="RAGAS not initialized")
    
    try:
        # Create RAGAS sample
        sample = SingleTurnSample(
            user_input=query,
            response=response,
            retrieved_contexts=[context],
            reference=ground_truth  # Will be None if no ground truth found
        )
        
        # Debug logging
        logger.info(f"🔍 RAGAS Sample Debug:")
        logger.info(f"  - Query: {query[:100]}...")
        logger.info(f"  - Response: {response[:100]}...")
        logger.info(f"  - Context: {context[:100]}...")
        logger.info(f"  - Ground Truth: {ground_truth[:100] if ground_truth else 'None'}...")
        logger.info(f"  - Retrieved Contexts Length: {len([context])}")
        logger.info(f"  - Has Reference: {ground_truth is not None}")
        
        # Create EvaluationDataset for RAGAS
        from ragas import EvaluationDataset
        import threading
        import asyncio
        
        # Evaluate with RAGAS in a separate thread with its own event loop
        eval_dataset = EvaluationDataset(samples=[sample])
        
        # Run evaluation in a separate thread with event loop
        result_container = [None]
        exception_container = [None]
        
        def run_evaluation():
            try:
                # Create new event loop for this thread
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                
                try:
                    result_container[0] = evaluate(
                        dataset=eval_dataset,
                        metrics=service.ragas_metrics
                    )
                finally:
                    new_loop.close()
            except Exception as e:
                exception_container[0] = e
        
        thread = threading.Thread(target=run_evaluation)
        thread.start()
        thread.join()
        
        if exception_container[0]:
            raise exception_container[0]
        
        result = result_container[0]
        
        # Extract scores from result dataframe
        scores = {}
        if hasattr(result, 'to_pandas'):
            df = result.to_pandas()
            logger.info(f"📊 RAGAS Result DataFrame columns: {df.columns.tolist()}")
            logger.info(f"📊 RAGAS Result DataFrame shape: {df.shape}")
            logger.info(f"📊 RAGAS Result DataFrame head:\n{df.head()}")
            
            for col in df.columns:
                if col not in ['user_input', 'response', 'retrieved_contexts', 'reference']:
                    score_value = float(df[col].iloc[0]) if len(df) > 0 else 0.0
                    scores[col] = score_value
                    logger.info(f"  - {col}: {score_value}")
        else:
            # Fallback for older API
            logger.info(f"📊 RAGAS Result (dict): {result}")
            for metric_name, score in result.items():
                if isinstance(score, list) and len(score) > 0:
                    scores[metric_name] = float(score[0])
                elif isinstance(score, (int, float)):
                    scores[metric_name] = float(score)
        
        logger.info(f"📊 Final scores: {scores}")
        return scores
        
    except Exception as e:
        logger.error(f"RAGAS evaluation error: {e}")
        raise HTTPException(status_code=500, detail=f"RAGAS evaluation failed: {e}")

async def log_to_phoenix(
    query: str,
    response: str, 
    context: str,
    scores: Dict[str, float],
    ground_truth: Optional[str] = None,
    evaluation_id: str = None
):
    """Log evaluation results to Phoenix UI"""
    if not service.phoenix_initialized:
        logger.warning("Phoenix not initialized, skipping logging")
        return
    
    try:
        with service.tracer.start_as_current_span("ragas_evaluation") as span:
            # Add span attributes
            span.set_attribute("evaluation.query", query)
            span.set_attribute("evaluation.response", response)
            span.set_attribute("evaluation.context", context)
            span.set_attribute("evaluation.id", evaluation_id or "unknown")
            
            if ground_truth:
                span.set_attribute("evaluation.ground_truth", ground_truth)
            
            # Add metric scores as span attributes and events
            for metric_name, score in scores.items():
                span.set_attribute(f"evaluation.metric.{metric_name}", score)
            
            # Log evaluations to Phoenix as events
            span.add_event(
                name="ragas_metrics",
                attributes={
                    "evaluation.metrics": str(scores),
                    "evaluation.timestamp": datetime.now().isoformat(),
                    **{f"metric.{name}": score for name, score in scores.items()}
                }
            )
        
        logger.info(f"✅ Logged evaluation to Phoenix: {evaluation_id}")
        
    except Exception as e:
        logger.error(f"Phoenix logging error: {e}")

# FastAPI endpoints
@app.on_event("startup")
async def startup_event():
    """Initialize service on startup"""
    logger.info("🚀 Starting RAGAS + Phoenix Evaluation Service...")
    await service.initialize()
    logger.info("✅ Service initialization complete")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "RAGAS + Phoenix Evaluation Service",
        "version": "1.0.0",
        "ragas_available": RAGAS_AVAILABLE,
        "ragas_initialized": service.ragas_initialized,
        "phoenix_available": PHOENIX_AVAILABLE,
        "phoenix_initialized": service.phoenix_initialized,
        "llm_available": service.llm is not None,
        "embeddings_available": service.embeddings is not None,
        "ground_truth_available": GROUND_TRUTH_AVAILABLE,
        "ground_truth_samples": len(GROUND_TRUTH_SAMPLES),
        "evaluation_model": os.getenv("EVALUATION_MODEL", "gpt-4.1"),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002"),
        "phoenix_endpoint": os.getenv("PHOENIX_ENDPOINT", "http://74.162.37.71:8010/v1/traces"),
        "phoenix_ui_url": "http://74.162.37.71:8010"
    }

@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_rag(request: EvaluationRequest, background_tasks: BackgroundTasks):
    """
    Evaluate RAG response using RAGAS metrics
    """
    try:
        evaluation_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Find ground truth with fuzzy matching
        ground_truth, match_score = find_ground_truth(request.query)
        ground_truth_found = ground_truth is not None
        
        logger.info(f"🔍 Evaluating query: {request.query[:50]}...")
        if ground_truth_found:
            logger.info(f"📋 Ground truth found (score: {match_score:.2f})")
        else:
            logger.info("📋 No ground truth found, using without reference")
        
        # Evaluate with RAGAS
        scores = await evaluate_with_ragas(
            query=request.query,
            response=request.response,
            context=request.context,
            ground_truth=ground_truth
        )
        
        # Log to Phoenix in background
        background_tasks.add_task(
            log_to_phoenix,
            request.query,
            request.response,
            request.context,
            scores,
            ground_truth,
            evaluation_id
        )
        
        logger.info(f"✅ Evaluation complete: {evaluation_id}")
        
        return EvaluationResponse(
            status="success",
            message="Evaluation completed successfully",
            evaluation_id=evaluation_id,
            metrics=scores,
            ground_truth_found=ground_truth_found,
            ground_truth_match_score=match_score if ground_truth_found else None
        )
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_available_metrics():
    """Get list of available RAGAS metrics"""
    metrics_info = {
        "context_precision": {
            "description": "Measures the precision of retrieved context",
            "requires_ground_truth": True,
            "available": service.ragas_initialized
        },
        "context_recall": {
            "description": "Measures the recall of retrieved context", 
            "requires_ground_truth": True,
            "available": service.ragas_initialized
        },
        "context_entity_recall": {
            "description": "Measures entity recall in retrieved context",
            "requires_ground_truth": True, 
            "available": service.ragas_initialized
        },
        "faithfulness": {
            "description": "Measures faithfulness of response to context",
            "requires_ground_truth": False,
            "available": service.ragas_initialized
        },
        "answer_relevancy": {
            "description": "Measures relevancy of response to query",
            "requires_ground_truth": False,
            "requires_embeddings": True,
            "available": service.ragas_initialized and service.embeddings is not None
        }
    }
    
    return {
        "metrics": metrics_info,
        "total_available": sum(1 for m in metrics_info.values() if m["available"]),
        "ground_truth_samples": len(GROUND_TRUTH_SAMPLES)
    }

if __name__ == "__main__":
    # Load environment variables
    port = int(os.getenv("RAGAS_SERVICE_PORT", "8300"))
    host = os.getenv("RAGAS_SERVICE_HOST", "0.0.0.0")
    
    logger.info(f"Starting RAGAS + Phoenix service on {host}:{port}")
    
    uvicorn.run(
        "ragas_phoenix_service:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )