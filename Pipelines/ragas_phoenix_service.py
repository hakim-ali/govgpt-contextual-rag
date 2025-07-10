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
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime
import difflib
import threading
from contextlib import contextmanager
from typing import Union 
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
    from opentelemetry import trace, context as otel_context
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk import trace as trace_sdk
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.trace import SpanKind, Status, StatusCode
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
    from opentelemetry.semconv.trace import SpanAttributes
    # OpenInference semantic conventions
    from openinference.semconv.trace import (
        OpenInferenceSpanKindValues,
        SpanAttributes as OpenInferenceSpanAttributes
    )
    PHOENIX_AVAILABLE = True
except ImportError as e:
    PHOENIX_AVAILABLE = False
    print(f"⚠️ Phoenix not available: {e}")
    # Fallback values if OpenInference not available
    class OpenInferenceSpanKindValues:
        EVALUATOR = "EVALUATOR"
        CHAIN = "CHAIN"
    
    class OpenInferenceSpanAttributes:
        INPUT_VALUE = "input.value"
        OUTPUT_VALUE = "output.value"
        EMBEDDING_MODEL_NAME = "embedding.model_name"
        LLM_MODEL_NAME = "llm.model_name"

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
    ground_truth: Optional[str] = Field(None, description="Optional ground truth answer")
    user_id: Optional[str] = Field(None, description="Optional user identifier")

class EvaluationResponse(BaseModel):
    status: str
    message: str
    evaluation_id: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None
    metrics: Optional[Dict[str, Union[float, str]]] = None                                                                                                                                     
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
            embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
            
            if "cohere" in embedding_model or "text-embedding" in embedding_model:
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
        """Initialize Phoenix observability with API key authentication"""
        if not PHOENIX_AVAILABLE:
            logger.error("Phoenix not available")
            return
        
        try:
            # Get Phoenix configuration from environment
            phoenix_api_key = os.getenv("PHOENIX_API_KEY")
            phoenix_endpoint = os.getenv("PHOENIX_ENDPOINT", "https://monitoring.sandbox.dge.gov.ae")
            
            if not phoenix_api_key:
                raise ValueError("PHOENIX_API_KEY environment variable is required for Phoenix authentication")
            
            # Configure Phoenix authentication
            os.environ["PHOENIX_CLIENT_HEADERS"] = f"api_key={phoenix_api_key}"
            os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = phoenix_endpoint
            
            # Construct traces endpoint
            traces_endpoint = f"{phoenix_endpoint.rstrip('/')}/v1/traces"
            
            # Register OpenTelemetry to send traces to Phoenix
            tracer_provider = register(
                project_name="govgpt-ragas-evaluation",
                endpoint=traces_endpoint
            )
            
            self.tracer = trace.get_tracer(__name__)
            self.phoenix_initialized = True
            
            logger.info(f"✅ Phoenix initialized successfully")
            logger.info(f"📊 Phoenix UI available at: {phoenix_endpoint}")
            logger.info(f"🔗 Traces endpoint: {traces_endpoint}")
            
        except Exception as e:
            logger.error(f"❌ Phoenix initialization failed: {e}")
            if "PHOENIX_API_KEY" in str(e):
                logger.error("💡 Please set PHOENIX_API_KEY environment variable with your Phoenix API key")
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

def get_final_ground_truth(explicit_ground_truth: Optional[str], query: str) -> tuple[Optional[str], str]:
    """
    Get final ground truth with priority: explicit parameter > fuzzy matching > None
    
    Args:
        explicit_ground_truth: Ground truth provided as parameter
        query: User query for fuzzy matching fallback
    
    Returns:
        Tuple of (final_ground_truth, source)
    """
    # Priority 1: Explicit parameter
    if explicit_ground_truth and explicit_ground_truth.strip():
        return explicit_ground_truth.strip(), "explicit"
    
    # Priority 2: Fuzzy matching fallback
    fuzzy_gt, match_score = find_ground_truth(query)
    if fuzzy_gt:
        return fuzzy_gt, "fuzzy"
    
    # Priority 3: None
    return None, "none"

async def evaluate_with_ragas(
    query: str, 
    response: str, 
    context: str,
    ground_truth: Optional[str] = None
) -> Dict[str, any]:
    """
    Evaluate using RAGAS metrics with graceful handling for missing ground truth
    
    Args:
        query: User query
        response: Generated response
        context: Retrieved context
        ground_truth: Optional ground truth answer
    
    Returns:
        Dictionary of metric scores (numeric) or status messages (string)
    """
    if not service.ragas_initialized:
        raise HTTPException(status_code=503, detail="RAGAS not initialized")
    
    # Determine what metrics we can actually run
    has_ground_truth = ground_truth is not None
    has_embeddings = service.embeddings is not None
    
    logger.info(f"🔍 RAGAS Evaluation Debug:")
    logger.info(f"  - Query: {query[:100]}...")
    logger.info(f"  - Response: {response[:100]}...")
    logger.info(f"  - Context: {context[:100]}...")
    logger.info(f"  - Ground Truth: {ground_truth[:100] if ground_truth else 'None'}...")
    logger.info(f"  - Has Ground Truth: {has_ground_truth}")
    logger.info(f"  - Has Embeddings: {has_embeddings}")
    
    # Initialize complete scores with status messages
    complete_scores = {
        "faithfulness": "Evaluation Failed",
        "answer_relevancy": "Embeddings Not Available" if not has_embeddings else "Evaluation Failed",
        "context_precision": "Ground Truth Not Available" if not has_ground_truth else "Evaluation Failed",
        "context_recall": "Ground Truth Not Available" if not has_ground_truth else "Evaluation Failed",
        "context_entity_recall": "Ground Truth Not Available" if not has_ground_truth else "Evaluation Failed"
    }
    
    try:
        # Build metrics list for what we can actually evaluate
        runnable_metrics = []
        
        # Always try faithfulness (no ground truth needed)
        from ragas.metrics import Faithfulness
        runnable_metrics.append(Faithfulness(llm=LangchainLLMWrapper(service.llm)))
        
        # Add answer relevancy if embeddings available
        if has_embeddings:
            from ragas.metrics import AnswerRelevancy
            runnable_metrics.append(AnswerRelevancy(
                llm=LangchainLLMWrapper(service.llm),
                embeddings=LangchainEmbeddingsWrapper(service.embeddings)
            ))
        
        # Add ground truth dependent metrics if available
        if has_ground_truth:
            from ragas.metrics import ContextPrecision, ContextRecall, ContextEntityRecall
            runnable_metrics.extend([
                ContextPrecision(llm=LangchainLLMWrapper(service.llm)),
                ContextRecall(llm=LangchainLLMWrapper(service.llm)),
                ContextEntityRecall(llm=LangchainLLMWrapper(service.llm))
            ])
        
        logger.info(f"📊 Running {len(runnable_metrics)} metrics: {[type(m).__name__ for m in runnable_metrics]}")
        
        if runnable_metrics:  # Only run if we have metrics to evaluate
            # Create RAGAS sample
            sample = SingleTurnSample(
                user_input=query,
                response=response,
                retrieved_contexts=[context],
                reference=ground_truth if has_ground_truth else None
            )
            
            # Create EvaluationDataset for RAGAS
            from ragas import EvaluationDataset
            import threading
            import asyncio
            
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
                            metrics=runnable_metrics
                        )
                    finally:
                        new_loop.close()
                except Exception as e:
                    exception_container[0] = e
            
            thread = threading.Thread(target=run_evaluation)
            thread.start()
            thread.join()
            
            if exception_container[0]:
                logger.error(f"RAGAS evaluation error: {exception_container[0]}")
                # Don't fail completely, just log the error and return status messages
            else:
                result = result_container[0]
                
                # Extract scores from result dataframe
                if hasattr(result, 'to_pandas'):
                    df = result.to_pandas()
                    logger.info(f"📊 RAGAS Result DataFrame columns: {df.columns.tolist()}")
                    
                    for col in df.columns:
                        if col not in ['user_input', 'response', 'retrieved_contexts', 'reference']:
                            try:
                                score_value = float(df[col].iloc[0]) if len(df) > 0 else 0.0
                                complete_scores[col] = score_value
                                logger.info(f"  ✅ {col}: {score_value:.3f}")
                            except (ValueError, IndexError) as e:
                                logger.warning(f"  ⚠️ {col}: Could not extract score - {e}")
                
                elif isinstance(result, dict):
                    # Fallback for older API
                    for metric_name, score in result.items():
                        try:
                            if isinstance(score, list) and len(score) > 0:
                                complete_scores[metric_name] = float(score[0])
                            elif isinstance(score, (int, float)):
                                complete_scores[metric_name] = float(score)
                            logger.info(f"  ✅ {metric_name}: {complete_scores[metric_name]:.3f}")
                        except (ValueError, IndexError) as e:
                            logger.warning(f"  ⚠️ {metric_name}: Could not extract score - {e}")
        
        # Log final status
        evaluated_count = sum(1 for v in complete_scores.values() if isinstance(v, (int, float)))
        unavailable_count = len(complete_scores) - evaluated_count
        
        logger.info(f"📊 Evaluation Summary:")
        logger.info(f"  - Evaluated: {evaluated_count} metrics")
        logger.info(f"  - Unavailable: {unavailable_count} metrics")
        logger.info(f"  - Final scores: {complete_scores}")
        
        return complete_scores
        
    except Exception as e:
        logger.error(f"RAGAS evaluation error: {e}")
        # Return status messages instead of failing
        logger.info(f"📊 Returning status messages due to evaluation error")
        return complete_scores

# async def log_to_phoenix(
#     query: str,
#     response: str, 
#     context: str,
#     scores: Dict[str, float],
#     ground_truth: Optional[str] = None,
#     evaluation_id: str = None
# ):
#     """Log evaluation results to Phoenix UI"""
#     if not service.phoenix_initialized:
#         logger.warning("Phoenix not initialized, skipping logging")
#         return
    
#     try:
#         with service.tracer.start_as_current_span("ragas_evaluation") as span:
#             # Add span attributes
#             span.set_attribute("evaluation.query", query)
#             span.set_attribute("evaluation.response", response)
#             span.set_attribute("evaluation.context", context)
#             span.set_attribute("evaluation.id", evaluation_id or "unknown")
            
#             if ground_truth:
#                 span.set_attribute("evaluation.ground_truth", ground_truth)
            
#             # Add metric scores as span attributes and events
#             for metric_name, score in scores.items():
#                 span.set_attribute(f"evaluation.metric.{metric_name}", score)
            
#             # Log evaluations to Phoenix as events
#             span.add_event(
#                 name="ragas_metrics",
#                 attributes={
#                     "evaluation.metrics": str(scores),
#                     "evaluation.timestamp": datetime.now().isoformat(),
#                     **{f"metric.{name}": score for name, score in scores.items()}
#                 }
#             )
        
#         logger.info(f"✅ Logged evaluation to Phoenix: {evaluation_id}")
        
#     except Exception as e:
#         logger.error(f"Phoenix logging error: {e}")

class PhoenixLogger:
    """Enhanced Phoenix logger following CrewAI patterns with OpenInference semantic conventions"""
    
    def __init__(self, tracer, session_id: str = None):
        self.tracer = tracer
        self.session_id = session_id or str(uuid.uuid4())
        self.thread_local = threading.local()
    
    @contextmanager
    def evaluation_session(self, evaluation_id: str):
        """Context manager for evaluation session tracking"""
        session_span_name = f"evaluation_session_{self.session_id}"
        
        with self.tracer.start_as_current_span(
            session_span_name,
            kind=SpanKind.INTERNAL,
            attributes={
                "session.id": self.session_id,
                "evaluation.session.id": evaluation_id,
                "evaluation.framework": "ragas",
                "evaluation.version": "1.0.0",
                "service.name": "govgpt-ragas-evaluation",
                "service.version": "1.0.0"
            }
        ) as session_span:
            self.thread_local.session_span = session_span
            try:
                yield session_span
            finally:
                session_span.set_status(Status(StatusCode.OK))
                session_span.add_event(
                    "evaluation_session_completed",
                    attributes={
                        "timestamp": datetime.now().isoformat(),
                        "session.id": self.session_id
                    }
                )
    
    def log_evaluation_request(self, query: str, response: str, context: str, ground_truth: Optional[str] = None):
        """Log the initial evaluation request using OpenInference conventions with proper Phoenix UI sections"""
        span_name = "ragas_content_processing"
        
        with self.tracer.start_as_current_span(
            span_name,
            kind=SpanKind.INTERNAL,
            attributes={
                # OpenInference semantic conventions
                OpenInferenceSpanAttributes.INPUT_VALUE: query,
                OpenInferenceSpanAttributes.OUTPUT_VALUE: response,
                "openinference.span.kind": OpenInferenceSpanKindValues.EVALUATOR,
                
                # Input Section
                "input.value": query,
                "input.preview": query[:100] + "..." if len(query) > 100 else query,
                "input.character_count": len(query),
                
                # Context Section (dedicated section for retrieved context)
                "context.value": context,
                "context.preview": context[:200] + "..." if len(context) > 200 else context,
                "context.character_count": len(context),
                "context.source": "govgpt_rag_pipeline",
                "context.type": "retrieved_documents",
                
                # Additional retrieval metadata (for Phoenix UI organization)
                "retrieval.context.content": context,
                "retrieval.context.preview": context[:200] + "..." if len(context) > 200 else context,
                "retrieval.context.character_count": len(context),
                "retrieval.context.source": "govgpt_rag_pipeline",
                
                # Output Section
                "output.value": response,
                "output.preview": response[:100] + "..." if len(response) > 100 else response,
                "output.character_count": len(response),
                
                # Evaluation metadata
                "evaluation.session.id": self.session_id,
                "evaluation.request.timestamp": datetime.now().isoformat(),
                "evaluation.ground_truth.available": ground_truth is not None,
                "evaluation.ground_truth.content": ground_truth or "N/A"
            }
        ) as request_span:
            
            # Add structured event for content processing
            request_span.add_event(
                "content_captured",
                attributes={
                    "input_preview": query[:50] + "..." if len(query) > 50 else query,
                    "context_preview": context[:100] + "..." if len(context) > 100 else context,
                    "context_length": len(context),
                    "output_length": len(response),
                    "ground_truth_available": ground_truth is not None,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            # Add dedicated context event for better UI organization
            request_span.add_event(
                "context_processed",
                attributes={
                    "context_type": "retrieved_documents",
                    "context_source": "govgpt_rag_pipeline",
                    "context_length": len(context),
                    "context_summary": context[:150] + "..." if len(context) > 150 else context,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            request_span.set_status(Status(StatusCode.OK))
            return request_span
    
    def log_metric_evaluation(self, metric_name: str, score_value: any, query: str, response: str, context: str, ground_truth: Optional[str] = None):
        """Log individual metric evaluation with 2-decimal formatting and proper sections"""
        span_name = f"ragas_metric_{metric_name}"
        
        # Handle both numeric scores and status messages
        if isinstance(score_value, str):
            # Status message (e.g., "Ground Truth Not Available")
            formatted_score = score_value
            score_interpretation = "unavailable"
            span_status = Status(StatusCode.OK)  # Not an error, just unavailable
            score_raw = None
            is_numeric = False
        else:
            # Numeric score
            formatted_score = f"{score_value:.2f}"
            score_interpretation = self._get_score_interpretation(metric_name, score_value)
            span_status = Status(StatusCode.OK) if score_value >= 0 else Status(StatusCode.ERROR, "Invalid score")
            score_raw = score_value
            is_numeric = True
        
        with self.tracer.start_as_current_span(
            span_name,
            kind=SpanKind.INTERNAL,
            attributes={
                # OpenInference semantic conventions
                "openinference.span.kind": OpenInferenceSpanKindValues.EVALUATOR,
                OpenInferenceSpanAttributes.INPUT_VALUE: query,
                OpenInferenceSpanAttributes.OUTPUT_VALUE: f"Score: {formatted_score}" + (f" ({score_interpretation})" if is_numeric else ""),
                
                # Metric-specific attributes
                "evaluation.metric.name": metric_name,
                "evaluation.metric.value": formatted_score,
                "evaluation.metric.value_raw": score_raw,  # Keep raw for calculations
                "evaluation.metric.status": score_interpretation,
                "evaluation.metric.available": is_numeric,
                "evaluation.metric.type": "ragas",
                "evaluation.metric.timestamp": datetime.now().isoformat(),
                "evaluation.session.id": self.session_id,
                
                # Reference to content sections (for UI organization)
                "input.reference": query[:50] + "..." if len(query) > 50 else query,
                "output.reference": response[:50] + "..." if len(response) > 50 else response,
                "context.reference": context[:50] + "..." if len(context) > 50 else context,
                "retrieval.context.reference": context[:50] + "..." if len(context) > 50 else context,
                
                # Metadata
                "evaluation.metric.requires_ground_truth": metric_name in ["context_precision", "context_recall", "context_entity_recall"],
                "evaluation.metric.has_ground_truth": ground_truth is not None
            }
        ) as metric_span:
            
            # Add metric evaluation event
            event_attributes = {
                "metric": metric_name,
                "value": formatted_score,
                "status": score_interpretation,
                "available": is_numeric,
                "timestamp": datetime.now().isoformat()
            }
            
            if is_numeric:
                event_attributes["score_range"] = "0.00-1.00"
            
            metric_span.add_event(
                "metric_evaluated",
                attributes=event_attributes
            )
            
            # Set span status
            metric_span.set_status(span_status)
                
            return metric_span
    
    def log_evaluation_summary(self, evaluation_id: str, scores: Dict[str, any], ground_truth_found: bool, match_score: Optional[float] = None, ground_truth_source: str = "none"):
        """Log evaluation summary with individual scores (no aggregation) formatted to 2 decimals"""
        span_name = "ragas_evaluation_results"
        
        # Process mixed score types
        formatted_scores = {}
        numeric_scores = {}
        status_messages = {}
        
        for name, score_value in scores.items():
            if isinstance(score_value, str):
                # Status message
                formatted_scores[name] = score_value
                status_messages[name] = score_value
            else:
                # Numeric score
                formatted_scores[name] = f"{score_value:.2f}"
                numeric_scores[name] = score_value
        
        numeric_count = len(numeric_scores)
        unavailable_count = len(status_messages)
        
        with self.tracer.start_as_current_span(
            span_name,
            kind=SpanKind.INTERNAL,
            attributes={
                # OpenInference semantic conventions
                "openinference.span.kind": OpenInferenceSpanKindValues.EVALUATOR,
                OpenInferenceSpanAttributes.OUTPUT_VALUE: f"Evaluation Complete - {numeric_count} evaluated, {unavailable_count} unavailable",
                
                # Summary attributes
                "evaluation.id": evaluation_id,
                "evaluation.session.id": self.session_id,
                "evaluation.completed.timestamp": datetime.now().isoformat(),
                "evaluation.metrics.total_count": len(scores),
                "evaluation.metrics.evaluated_count": numeric_count,
                "evaluation.metrics.unavailable_count": unavailable_count,
                
                # Ground truth information
                "evaluation.ground_truth.found": ground_truth_found,
                "evaluation.ground_truth.source": ground_truth_source,
                "evaluation.ground_truth.match_score": f"{match_score:.2f}" if match_score else "0.00",
                
                # Individual metric values (mixed types)
                **{f"evaluation.metric.{name}.value": formatted_score for name, formatted_score in formatted_scores.items()},
                
                # Score interpretations (only for numeric scores)
                **{f"evaluation.metric.{name}.level": self._get_score_interpretation(name, score) 
                   for name, score in numeric_scores.items()},
                   
                # Availability status
                **{f"evaluation.metric.{name}.available": isinstance(scores[name], (int, float)) 
                   for name in scores.keys()}
            }
        ) as summary_span:
            
            # Add completion event with mixed scores
            summary_span.add_event(
                "evaluation_completed",
                attributes={
                    "evaluation_id": evaluation_id,
                    "total_metrics": len(scores),
                    "evaluated_metrics": numeric_count,
                    "unavailable_metrics": unavailable_count,
                    "ground_truth_source": ground_truth_source,
                    "ground_truth_found": ground_truth_found,
                    "scores_summary": str(formatted_scores),
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            summary_span.set_status(Status(StatusCode.OK))
            return summary_span
    
    def _get_score_interpretation(self, metric_name: str, score: float) -> str:
        """Provide human-readable interpretation of metric scores"""
        if score < 0:
            return "invalid"
        elif score < 0.3:
            return "poor"
        elif score < 0.6:
            return "fair"
        elif score < 0.8:
            return "good"
        else:
            return "excellent"

# Global Phoenix logger instance
phoenix_logger: Optional[PhoenixLogger] = None

async def log_to_phoenix(
    query: str,
    response: str, 
    context: str,
    scores: Dict[str, any],
    ground_truth: Optional[str] = None,
    evaluation_id: str = None,
    ground_truth_source: str = "none"
):
    """Enhanced Phoenix logging with support for mixed metric types"""
    global phoenix_logger
    
    if not service.phoenix_initialized:
        logger.warning("Phoenix not initialized, skipping logging")
        return
    
    if not phoenix_logger:
        phoenix_logger = PhoenixLogger(service.tracer)
    
    try:
        eval_id = evaluation_id or f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Find ground truth match score if applicable
        ground_truth_found = ground_truth is not None
        match_score = None
        if ground_truth_found and ground_truth_source == "fuzzy":
            # Calculate match score from the original fuzzy matching
            _, match_score = find_ground_truth(query)
        
        with phoenix_logger.evaluation_session(eval_id) as session:
            # Log evaluation request
            phoenix_logger.log_evaluation_request(query, response, context, ground_truth)
            
            # Log individual metric evaluations (handles both numeric and status messages)
            for metric_name, score_value in scores.items():
                phoenix_logger.log_metric_evaluation(
                    metric_name, score_value, query, response, context, ground_truth
                )
            
            # Log evaluation summary with ground truth source
            phoenix_logger.log_evaluation_summary(
                eval_id, scores, ground_truth_found, match_score, ground_truth_source
            )
        
        logger.info(f"✅ Enhanced Phoenix logging completed for evaluation: {eval_id}")
        
    except Exception as e:
        logger.error(f"Enhanced Phoenix logging error: {e}")
        # Fallback to basic logging if enhanced logging fails
        try:
            with service.tracer.start_as_current_span("ragas_evaluation_fallback") as fallback_span:
                fallback_span.set_attribute("evaluation.id", evaluation_id or "unknown")
                fallback_span.set_attribute("evaluation.error", str(e))
                fallback_span.set_attribute("evaluation.metrics", str(scores))
                fallback_span.set_status(Status(StatusCode.ERROR, f"Enhanced logging failed: {e}"))
        except Exception as fallback_error:
            logger.error(f"Fallback Phoenix logging also failed: {fallback_error}")
 
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
        "phoenix_enabled": bool(os.getenv("PHOENIX_API_KEY")),
        "phoenix_api_key_configured": bool(os.getenv("PHOENIX_API_KEY")),
        "phoenix_endpoint": os.getenv("PHOENIX_ENDPOINT", "https://monitoring.sandbox.dge.gov.ae"),
        "phoenix_ui_url": os.getenv("PHOENIX_ENDPOINT", "https://monitoring.sandbox.dge.gov.ae")
    }

@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_rag(request: EvaluationRequest, background_tasks: BackgroundTasks):
    """
    Evaluate RAG response using RAGAS metrics with optional ground truth
    """
    try:
        evaluation_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Ground truth priority: explicit parameter > fuzzy matching > None
        final_ground_truth, gt_source = get_final_ground_truth(
            request.ground_truth, 
            request.query
        )
        
        # Get match score for fuzzy matches
        match_score = None
        if gt_source == "fuzzy":
            _, match_score = find_ground_truth(request.query)
        
        ground_truth_found = final_ground_truth is not None
        
        logger.info(f"🔍 Evaluating query: {request.query[:50]}...")
        logger.info(f"📋 Ground truth source: {gt_source}")
        if ground_truth_found:
            if gt_source == "explicit":
                logger.info(f"📋 Explicit ground truth provided")
            elif gt_source == "fuzzy":
                logger.info(f"📋 Fuzzy ground truth found (score: {match_score:.2f})")
        else:
            logger.info("📋 No ground truth available - will show 'Ground Truth Not Available' for relevant metrics")
        
        # Evaluate with RAGAS (always returns all 5 metrics)
        scores = await evaluate_with_ragas(
            query=request.query,
            response=request.response,
            context=request.context,
            ground_truth=final_ground_truth
        )
        
        # Count evaluated vs unavailable metrics
        evaluated_count = sum(1 for v in scores.values() if isinstance(v, (int, float)))
        unavailable_count = len(scores) - evaluated_count
        
        # Log to Phoenix in background (with all 5 metrics)
        background_tasks.add_task(
            log_to_phoenix,
            request.query,
            request.response,
            request.context,
            scores,
            final_ground_truth,
            evaluation_id,
            gt_source
        )
        
        logger.info(f"✅ Evaluation complete: {evaluation_id}")
        logger.info(f"📊 Metrics: {evaluated_count} evaluated, {unavailable_count} unavailable")
        
        # Create response message based on ground truth availability
        if gt_source == "explicit":
            message = "Evaluation completed with explicit ground truth"
        elif gt_source == "fuzzy":
            message = f"Evaluation completed with fuzzy-matched ground truth (score: {match_score:.2f})"
        else:
            message = "Evaluation completed - Ground truth dependent metrics show 'Ground Truth Not Available'"
        
        return EvaluationResponse(
            status="success",
            message=message,
            evaluation_id=evaluation_id,
            metrics=scores,
            ground_truth_found=ground_truth_found,
            ground_truth_match_score=match_score if gt_source == "fuzzy" else None
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