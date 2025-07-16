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
import queue

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

# Enhanced Pydantic models for production observability
class UserPreferences(BaseModel):
    language: Optional[str] = None
    timezone: Optional[str] = None
    location: Optional[str] = None

class SessionStats(BaseModel):
    total_sessions: Optional[int] = None
    total_queries: Optional[int] = None
    current_session_duration: Optional[str] = None
    last_interaction: Optional[str] = None

class UserProfile(BaseModel):
    user_id: Optional[str] = None
    name: Optional[str] = None
    email: Optional[str] = None
    role: Optional[str] = None
    preferences: Optional[UserPreferences] = None
    session_stats: Optional[SessionStats] = None

class QualityScores(BaseModel):
    avg_faithfulness: Optional[float] = None
    avg_relevancy: Optional[float] = None
    pending_evaluation: Optional[bool] = None
    avg_response_time_ms: Optional[int] = None

class SessionMetadata(BaseModel):
    session_id: Optional[str] = None
    chat_id: Optional[str] = None
    message_id: Optional[str] = None
    start_time: Optional[str] = None
    last_interaction: Optional[str] = None
    duration_seconds: Optional[int] = None
    message_count: Optional[int] = None
    conversation_type: Optional[str] = None
    topics: Optional[List[str]] = None
    features: Optional[Dict[str, Any]] = None
    quality_scores: Optional[QualityScores] = None
    variables: Optional[Dict[str, Any]] = None

class MessageMetadata(BaseModel):
    content_length: Optional[int] = None
    message_index: Optional[int] = None
    has_timestamp: Optional[bool] = None
    model: Optional[str] = None
    response_time_ms: Optional[int] = None
    context_used: Optional[bool] = None
    context_length: Optional[int] = None
    evaluation_pending: Optional[bool] = None
    streaming_used: Optional[bool] = None
    query_length: Optional[int] = None
    intent: Optional[str] = None
    context_needed: Optional[bool] = None
    ragas_scores: Optional[Dict[str, float]] = None

class ConversationMessage(BaseModel):
    message_id: Optional[str] = None
    timestamp: Optional[str] = None
    role: Optional[str] = None
    content: Optional[str] = None
    metadata: Optional[MessageMetadata] = None

class SystemContext(BaseModel):
    pipeline_name: Optional[str] = None
    pipeline_version: Optional[str] = None
    model: Optional[str] = None
    request_id: Optional[str] = None
    timestamp: Optional[str] = None
    response_time_ms: Optional[int] = None
    token_count: Optional[int] = None
    context_length: Optional[int] = None
    retrieval_score: Optional[float] = None
    streaming_enabled: Optional[bool] = None

class TraceData(BaseModel):
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    parent_span_id: Optional[str] = None
    operation_name: Optional[str] = None
    tags: Optional[Dict[str, Any]] = None
    attributes: Optional[Dict[str, Any]] = None

class ProductionMetadata(BaseModel):
    environment: Optional[str] = None
    service_name: Optional[str] = None
    version: Optional[str] = None
    logged_at: Optional[str] = None

class EvaluationData(BaseModel):
    query: str = Field(..., description="User query/question")
    response: str = Field(..., description="Generated response/answer")
    context: str = Field(..., description="Retrieved context")
    ground_truth: Optional[str] = Field(None, description="Optional ground truth answer")

# Legacy model for backward compatibility
class EvaluationRequest(BaseModel):
    query: str = Field(..., description="User query/question")
    response: str = Field(..., description="Generated response/answer")
    context: str = Field(..., description="Retrieved context")
    ground_truth: Optional[str] = Field(None, description="Optional ground truth answer")
    user_id: Optional[str] = Field(None, description="Optional user identifier")
    
    # Enhanced production fields
    evaluation_data: Optional[EvaluationData] = None
    user_profile: Optional[UserProfile] = None
    session_metadata: Optional[SessionMetadata] = None
    system_context: Optional[SystemContext] = None
    trace_data: Optional[TraceData] = None
    production_metadata: Optional[ProductionMetadata] = None
    
    # Additional context fields for backward compatibility
    user_context: Optional[Dict[str, Any]] = None
    session_data: Optional[Dict[str, Any]] = None
    request_id: Optional[str] = None
    timestamp: Optional[str] = None
    pipeline: Optional[str] = None

class UserSessionRequest(BaseModel):
    event_type: str = Field(default="user_session")
    request_id: str
    timestamp: str
    user_context: Dict[str, Any]
    session_data: Dict[str, Any]
    session_stats: Dict[str, Any]

class SystemMetricsRequest(BaseModel):
    event_type: str = Field(default="system_metrics")
    request_id: str
    timestamp: str
    metrics: Dict[str, Any]

class EvaluationResponse(BaseModel):
    status: str
    message: str
    evaluation_id: Optional[str] = None
    metrics: Optional[Dict[str, Union[float, str]]] = None
    ground_truth_found: Optional[bool] = None
    ground_truth_match_score: Optional[float] = None
    user_profile: Optional[UserProfile] = None
    session_metadata: Optional[SessionMetadata] = None
    system_context: Optional[SystemContext] = None
    production_metadata: Optional[ProductionMetadata] = None

class SessionLogResponse(BaseModel):
    status: str
    message: str
    session_id: Optional[str] = None
    logged_at: str

class MetricsLogResponse(BaseModel):
    status: str
    message: str
    metrics_logged: int
    logged_at: str

# === CrewAI-style Conversation Tracking (Non-Trace) ===
class ConversationTracker:
    """CrewAI-style conversation tracking without traces"""
    
    def __init__(self):
        self.conversations = {}  # user_id -> conversation_data
        self.chat_sessions = {}  # chat_id -> session_data
    
    def track_conversation(self, user_id: str, chat_id: str, query: str, response: str, 
                          ragas_scores: Dict[str, Any], session_metadata: Optional[SessionMetadata] = None):
        """Track conversation following CrewAI patterns"""
        timestamp = datetime.now().isoformat()
        
        # Initialize user conversation if not exists
        if user_id not in self.conversations:
            self.conversations[user_id] = {
                "user_id": user_id,
                "total_interactions": 0,
                "quality_scores": [],
                "first_interaction": timestamp,
                "last_interaction": timestamp,
                "chat_sessions": set()
            }
        
        # Initialize chat session if not exists
        if chat_id not in self.chat_sessions:
            self.chat_sessions[chat_id] = {
                "chat_id": chat_id,
                "user_id": user_id,
                "session_id": session_metadata.session_id if session_metadata else None,
                "message_count": 0,
                "start_time": timestamp,
                "last_activity": timestamp,
                "avg_quality_score": 0.0,
                "interactions": []
            }
        
        # Extract numeric RAGAS scores
        numeric_scores = {k: v for k, v in ragas_scores.items() if isinstance(v, (int, float))}
        current_score = sum(numeric_scores.values()) / len(numeric_scores) if numeric_scores else 0.0
        
        # Update user-level tracking
        user_data = self.conversations[user_id]
        user_data["total_interactions"] += 1
        user_data["quality_scores"].append(current_score)
        user_data["last_interaction"] = timestamp
        user_data["chat_sessions"].add(chat_id)
        
        # Update chat-level tracking
        chat_data = self.chat_sessions[chat_id]
        chat_data["message_count"] += 1
        chat_data["last_activity"] = timestamp
        
        # Calculate running average quality score for chat
        existing_interactions = len(chat_data["interactions"])
        if existing_interactions > 0:
            total_score = chat_data["avg_quality_score"] * existing_interactions + current_score
            chat_data["avg_quality_score"] = total_score / (existing_interactions + 1)
        else:
            chat_data["avg_quality_score"] = current_score
        
        # Add interaction record (minimal data)
        interaction = {
            "timestamp": timestamp,
            "query_length": len(query),
            "response_length": len(response),
            "quality_score": current_score,
            "ragas_metrics": numeric_scores
        }
        chat_data["interactions"].append(interaction)
        
        # Log minimal conversation tracking (CrewAI style)
        logger.info(f"📝 Conversation tracked: User={user_id} | Chat={chat_id} | Quality={current_score:.3f} | Total_interactions={user_data['total_interactions']}")
    
    def get_user_analytics(self, user_id: str) -> Dict[str, Any]:
        """Get user analytics (CrewAI-style)"""
        if user_id not in self.conversations:
            return {"status": "not_found", "user_id": user_id}
        
        user_data = self.conversations[user_id]
        quality_scores = user_data["quality_scores"]
        
        return {
            "user_id": user_id,
            "total_interactions": user_data["total_interactions"],
            "total_chat_sessions": len(user_data["chat_sessions"]),
            "avg_quality_score": sum(quality_scores) / len(quality_scores) if quality_scores else 0.0,
            "quality_trend": "improving" if len(quality_scores) >= 2 and quality_scores[-1] > quality_scores[0] else "stable",
            "first_interaction": user_data["first_interaction"],
            "last_interaction": user_data["last_interaction"]
        }
    
    def get_chat_analytics(self, chat_id: str) -> Dict[str, Any]:
        """Get chat session analytics"""
        if chat_id not in self.chat_sessions:
            return {"status": "not_found", "chat_id": chat_id}
        
        chat_data = self.chat_sessions[chat_id]
        
        return {
            "chat_id": chat_id,
            "user_id": chat_data["user_id"],
            "session_id": chat_data["session_id"],
            "message_count": chat_data["message_count"],
            "avg_quality_score": chat_data["avg_quality_score"],
            "duration_minutes": self._calculate_duration_minutes(chat_data["start_time"], chat_data["last_activity"]),
            "start_time": chat_data["start_time"],
            "last_activity": chat_data["last_activity"]
        }
    
    def _calculate_duration_minutes(self, start_time: str, end_time: str) -> float:
        """Calculate duration between timestamps in minutes"""
        try:
            start = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            end = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
            return (end - start).total_seconds() / 60.0
        except:
            return 0.0

# Global conversation tracker instance
conversation_tracker = ConversationTracker()

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

@app.on_event("startup")
async def startup_event():
    """Initialize all services on startup"""
    logger.info("🚀 Initializing RAGAS + Phoenix service...")
    await service.initialize()
    
    # Initialize Phoenix manager if Phoenix is available
    if service.phoenix_initialized:
        get_phoenix_manager()
    
    logger.info("✅ Service initialization complete")

# Background processing queue
background_queue = queue.Queue()

# Phoenix session manager for comprehensive user tracking
class ProductionPhoenixManager:
    """Production-ready Phoenix manager for comprehensive user session tracking"""
    
    def __init__(self, tracer):
        self.tracer = tracer
        self.active_sessions = {}  # Track active user sessions
        self.conversation_cache = {}  # Cache conversation history
    
    def start_user_session(self, user_profile: UserProfile, session_metadata: SessionMetadata) -> str:
        """Start comprehensive user session tracking"""
        session_key = f"{user_profile.user_id}_{session_metadata.session_id}"
        
        if session_key not in self.active_sessions:
            session_span = self.tracer.start_span(
                name="user_session_comprehensive",
                attributes={
                    "user.id": user_profile.user_id,
                    "user.name": user_profile.name,
                    "user.email": user_profile.email,
                    "user.role": user_profile.role,
                    "session.id": session_metadata.session_id,
                    "session.chat_id": session_metadata.chat_id,
                    "session.start_time": session_metadata.start_time,
                    "session.type": session_metadata.conversation_type,
                    "user.preferences.language": user_profile.preferences.language if user_profile.preferences else None,
                    "user.preferences.timezone": user_profile.preferences.timezone if user_profile.preferences else None,
                    "user.preferences.location": user_profile.preferences.location if user_profile.preferences else None,
                    "session.topics": ",".join(session_metadata.topics or []),
                    "environment": "production",
                    "service": "govgpt_rag_evaluation"
                }
            )
            
            self.active_sessions[session_key] = {
                "span": session_span,
                "start_time": datetime.now(),
                "user_profile": user_profile,
                "session_metadata": session_metadata,
                "message_count": 0,
                "quality_metrics": []
            }
        
        return session_key
    
    def log_conversation_turn(self, session_key: str, conversation_history: List[ConversationMessage], evaluation_scores: Dict[str, Any]):
        """Log complete conversation turn with evaluation scores"""
        if session_key not in self.active_sessions:
            return
        
        session_info = self.active_sessions[session_key]
        session_span = session_info["span"]
        
        # Create conversation turn span
        with self.tracer.start_as_current_span(
            "conversation_turn",
            attributes={
                "conversation.turn_number": len(conversation_history),
                "conversation.message_count": len(conversation_history),
                "evaluation.completed": True,
                "session.key": session_key
            }
        ) as turn_span:
            
            # Log each message in the conversation
            for i, message in enumerate(conversation_history[-2:]):  # Last user message and response
                self._log_message_with_context(turn_span, message, i, session_key)
            
            # Log evaluation scores
            self._log_evaluation_scores(turn_span, evaluation_scores, session_key)
            
            # Update session statistics
            session_info["message_count"] += len(conversation_history)
            session_info["quality_metrics"].append(evaluation_scores)
            
            # Add session update event
            session_span.add_event(
                "conversation_turn_completed",
                attributes={
                    "turn_number": len(conversation_history),
                    "total_messages": session_info["message_count"],
                    "session_duration_seconds": (datetime.now() - session_info["start_time"]).total_seconds(),
                    "quality_scores_available": bool(evaluation_scores),
                    "timestamp": datetime.now().isoformat()
                }
            )
    
    def _log_message_with_context(self, parent_span, message: ConversationMessage, index: int, session_key: str):
        """Log individual message with comprehensive context"""
        with self.tracer.start_as_current_span(
            f"message_{message.role}",
            attributes={
                "message.id": message.message_id,
                "message.role": message.role,
                "message.content_preview": message.content[:100] + "..." if len(message.content) > 100 else message.content,
                "message.content_length": message.metadata.content_length if message.metadata else len(message.content),
                "message.timestamp": message.timestamp,
                "message.index": index,
                "session.key": session_key
            }
        ) as message_span:
            
            # Add message-specific metadata
            if message.metadata:
                if message.role == "assistant" and message.metadata.model:
                    message_span.set_attribute("message.model", message.metadata.model)
                    message_span.set_attribute("message.response_time_ms", message.metadata.response_time_ms or 0)
                    message_span.set_attribute("message.context_used", message.metadata.context_used or False)
                    message_span.set_attribute("message.streaming_used", message.metadata.streaming_used or False)
                
                if message.metadata.ragas_scores:
                    for metric, score in message.metadata.ragas_scores.items():
                        message_span.set_attribute(f"ragas.{metric}", score)
            
            # Add message event
            message_span.add_event(
                "message_processed",
                attributes={
                    "message_role": message.role,
                    "content_length": len(message.content),
                    "has_evaluation": bool(message.metadata and message.metadata.ragas_scores),
                    "timestamp": datetime.now().isoformat()
                }
            )
    
    def _log_evaluation_scores(self, parent_span, scores: Dict[str, Any], session_key: str):
        """Log comprehensive evaluation scores"""
        with self.tracer.start_as_current_span(
            "evaluation_scores",
            attributes={
                "evaluation.session_key": session_key,
                "evaluation.metrics_count": len(scores),
                "evaluation.timestamp": datetime.now().isoformat()
            }
        ) as eval_span:
            
            # Log individual metrics
            for metric_name, score_value in scores.items():
                if isinstance(score_value, (int, float)):
                    eval_span.set_attribute(f"metric.{metric_name}.value", score_value)
                    eval_span.set_attribute(f"metric.{metric_name}.available", True)
                else:
                    eval_span.set_attribute(f"metric.{metric_name}.status", str(score_value))
                    eval_span.set_attribute(f"metric.{metric_name}.available", False)
            
            # Add evaluation event
            eval_span.add_event(
                "metrics_evaluated",
                attributes={
                    "total_metrics": len(scores),
                    "numeric_metrics": sum(1 for v in scores.values() if isinstance(v, (int, float))),
                    "status_metrics": sum(1 for v in scores.values() if isinstance(v, str)),
                    "timestamp": datetime.now().isoformat()
                }
            )
    
    def update_session_quality(self, session_key: str, quality_update: Dict[str, float]):
        """Update session-level quality metrics"""
        if session_key in self.active_sessions:
            session_info = self.active_sessions[session_key]
            session_span = session_info["span"]
            
            # Update span attributes with quality metrics
            for metric, value in quality_update.items():
                session_span.set_attribute(f"session.quality.{metric}", value)
            
            # Add quality update event
            session_span.add_event(
                "session_quality_updated",
                attributes={
                    **{f"quality.{k}": v for k, v in quality_update.items()},
                    "timestamp": datetime.now().isoformat()
                }
            )

# Global Phoenix manager
production_phoenix_manager: Optional[ProductionPhoenixManager] = None

def get_phoenix_manager() -> Optional[ProductionPhoenixManager]:
    """Get or create the production Phoenix manager"""
    global production_phoenix_manager
    if production_phoenix_manager is None and service.phoenix_initialized and service.tracer:
        production_phoenix_manager = ProductionPhoenixManager(service.tracer)
        logger.info("✅ ProductionPhoenixManager initialized for user session tracking")
    return production_phoenix_manager

def _classify_query_intent(query: str) -> str:
    """Classify query intent for analytics"""
    query_lower = query.lower()
    
    if any(word in query_lower for word in ['how', 'steps', 'process', 'procedure']):
        return "process_guidance"
    elif any(word in query_lower for word in ['what', 'define', 'explain', 'meaning']):
        return "information_request"
    elif any(word in query_lower for word in ['where', 'when', 'contact', 'office']):
        return "location_inquiry"
    elif any(word in query_lower for word in ['requirement', 'need', 'document', 'certificate']):
        return "requirement_check"
    elif any(word in query_lower for word in ['fee', 'cost', 'price', 'charge']):
        return "pricing_inquiry"
    elif any(word in query_lower for word in ['renew', 'extend', 'update']):
        return "renewal_request"
    else:
        return "general_inquiry"

def _calculate_query_complexity(query: str) -> str:
    """Calculate query complexity for analytics"""
    word_count = len(query.split())
    question_marks = query.count('?')
    compound_indicators = query.count(' and ') + query.count(' or ') + query.count(',')
    
    complexity_score = word_count * 0.1 + question_marks * 2 + compound_indicators * 1.5
    
    if complexity_score >= 8:
        return "high"
    elif complexity_score >= 4:
        return "medium"
    else:
        return "low"

def _categorize_response_time(response_time_ms: int) -> str:
    """Categorize response time for analytics"""
    if response_time_ms < 1000:
        return "fast"
    elif response_time_ms < 3000:
        return "moderate"
    elif response_time_ms < 10000:
        return "slow"
    else:
        return "very_slow"

def _categorize_business_query(query: str) -> str:
    """Categorize query for business analytics"""
    query_lower = query.lower()
    
    if any(word in query_lower for word in ['license', 'permit', 'authorization']):
        return "licensing"
    elif any(word in query_lower for word in ['visa', 'immigration', 'resident']):
        return "immigration"
    elif any(word in query_lower for word in ['business', 'company', 'trade']):
        return "business_services"
    elif any(word in query_lower for word in ['health', 'medical', 'hospital']):
        return "healthcare"
    elif any(word in query_lower for word in ['education', 'school', 'university']):
        return "education"
    elif any(word in query_lower for word in ['tax', 'customs', 'duty']):
        return "taxation"
    elif any(word in query_lower for word in ['investment', 'entrepreneur', 'startup']):
        return "investment"
    else:
        return "general_government"

def _score_to_grade(score: float) -> str:
    """Convert numerical score to letter grade"""
    if score >= 0.9:
        return "A"
    elif score >= 0.8:
        return "B"
    elif score >= 0.7:
        return "C"
    elif score >= 0.6:
        return "D"
    else:
        return "F"

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

async def log_to_phoenix_production(
    query: str, response: str, context: str, scores: Dict[str, any],
    ground_truth: Optional[str], evaluation_id: str, gt_source: str,
    user_profile: Optional[UserProfile], session_metadata: Optional[SessionMetadata],
    system_context: Optional[SystemContext], trace_data: Optional[TraceData],
    production_metadata: Optional[ProductionMetadata]
):
    """Production-ready Phoenix logging with latest Arize AI patterns and CrewAI implementation"""
    if not service.phoenix_initialized:
        logger.warning("Phoenix not initialized, skipping production logging")
        return
    
    try:
        eval_id = evaluation_id or f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Find ground truth match score
        ground_truth_found = ground_truth is not None
        match_score = None
        if ground_truth_found and gt_source == "fuzzy":
            _, match_score = find_ground_truth(query)
        
        # Beautiful evaluator naming following Phoenix best practices
        user_id = user_profile.user_id if user_profile else "system"
        evaluator_name = f"RAG_Quality_Evaluator_{user_id}"
        
        # Production-focused minimal span (CrewAI-style)
        with service.tracer.start_as_current_span(
            evaluator_name,
            kind=SpanKind.INTERNAL,  # Using INTERNAL with openinference.span.kind=EVALUATOR
            attributes={
                # === Core OpenInference ===
                "openinference.span.kind": OpenInferenceSpanKindValues.EVALUATOR,
                OpenInferenceSpanAttributes.LLM_MODEL_NAME: system_context.model if system_context else "unknown",
                
                # === Phoenix UI Visible Differentiation ===
                "user_id": user_profile.user_id if user_profile else None,  # Phoenix UI friendly
                "chat_id": session_metadata.chat_id if session_metadata else None,  # Phoenix UI friendly
                "session_id": session_metadata.session_id if session_metadata else None,  # Phoenix UI friendly
                "message_id": session_metadata.message_id if session_metadata else None,  # Phoenix UI friendly
                
                # === OpenInference Standard Attributes ===
                "user.id": user_profile.user_id if user_profile else None,
                "session.id": session_metadata.session_id if session_metadata else None,
                
                # === Core Input/Output ===
                OpenInferenceSpanAttributes.INPUT_VALUE: query,
                OpenInferenceSpanAttributes.OUTPUT_VALUE: response,
                
                # === Essential Evaluation Data ===
                "evaluation.id": eval_id,
                "evaluation.framework": "ragas",
                "evaluation.timestamp": datetime.now().isoformat(),
                "evaluation.ground_truth_available": ground_truth_found,
                
                # === Core System Info ===
                "system.pipeline": "govgpt_rag_v2",
                "system.environment": "production",
                "system.response_time_ms": system_context.response_time_ms if system_context else None,
                
                # === Phoenix UI Display Enhancement ===
                "display.user_chat": f"User:{user_profile.user_id if user_profile else 'unknown'}|Chat:{session_metadata.chat_id if session_metadata else 'unknown'}",
                "display.evaluation_type": "RAGAS_Quality_Assessment",
            }
        ) as main_span:
            
            # === RAGAS Quality Metrics (Phoenix semantic conventions) ===
            evaluated_count = 0
            total_score = 0.0
            
            for metric_name, score_value in scores.items():
                if isinstance(score_value, (int, float)):
                    # Phoenix-style metric attributes
                    main_span.set_attribute(f"evaluation.score.{metric_name}", score_value)
                    main_span.set_attribute(f"evaluation.score.{metric_name}.formatted", f"{score_value:.3f}")
                    main_span.set_attribute(f"evaluation.score.{metric_name}.status", "success")
                    evaluated_count += 1
                    total_score += score_value
                else:
                    # Handle unavailable metrics
                    main_span.set_attribute(f"evaluation.score.{metric_name}.status", str(score_value))
                    main_span.set_attribute(f"evaluation.score.{metric_name}.available", False)
            
            # === Essential Quality Assessment ===
            avg_score = total_score / evaluated_count if evaluated_count > 0 else 0.0
            main_span.set_attribute("evaluation.overall_score", avg_score)
            main_span.set_attribute("evaluation.metrics_evaluated", evaluated_count)
            main_span.set_attribute("evaluation.success", True)
            
            # === Single Essential Event: RAGAS Evaluation Completed ===
            main_span.add_event(
                "ragas_evaluation_completed",
                attributes={
                    # Phoenix UI visible differentiation
                    "user_id": user_profile.user_id if user_profile else None,
                    "chat_id": session_metadata.chat_id if session_metadata else None,
                    "session_id": session_metadata.session_id if session_metadata else None,
                    
                    # OpenInference standard
                    "user.id": user_profile.user_id if user_profile else None,
                    "session.id": session_metadata.session_id if session_metadata else None,
                    
                    # Evaluation results
                    "evaluation.overall_score": avg_score,
                    "evaluation.metrics_count": evaluated_count,
                    "evaluation.quality_level": "excellent" if avg_score >= 0.9 else "good" if avg_score >= 0.8 else "needs_improvement",
                    
                    # Phoenix UI display
                    "display.summary": f"User {user_profile.user_id if user_profile else 'unknown'} | Chat {session_metadata.chat_id if session_metadata else 'unknown'} | Score {avg_score:.3f}",
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            # Set span status
            main_span.set_status(Status(StatusCode.OK))
        
        # Minimal production logging (CrewAI-style)
        user_id = user_profile.user_id if user_profile else "anonymous"
        chat_id = session_metadata.chat_id if session_metadata else "none"
        
        logger.info(f"✅ RAGAS evaluation: {eval_id} | User: {user_id} | Chat: {chat_id} | Score: {avg_score:.3f} | Metrics: {evaluated_count}/{len(scores)}")
        
    except Exception as e:
        logger.error(f"❌ Phoenix logging failed: {eval_id} | User: {user_profile.user_id if user_profile else 'unknown'} | Error: {e}")
        
        # Minimal fallback trace
        try:
            with service.tracer.start_as_current_span(
                f"ragas_eval_fallback_{eval_id}",
                attributes={
                    "evaluation.id": eval_id,
                    "evaluation.error": str(e),
                    "user.id": user_profile.user_id if user_profile else None,
                    "chat.id": session_metadata.chat_id if session_metadata else None,
                    "timestamp": datetime.now().isoformat()
                }
            ):
                logger.warning(f"⚠️  Fallback trace: {eval_id}")
        except Exception:
            logger.critical(f"💥 Critical logging failure: {eval_id}")


async def log_session_to_phoenix(request_id: str, user_context: Dict, session_data: Dict, session_stats: Dict):
    """Log user session data to Phoenix for journey tracking"""
    if not service.phoenix_initialized:
        return
    
    try:
        with service.tracer.start_as_current_span(
            "user_session_tracking",
            attributes={
                "session.request_id": request_id,
                "session.user_id": user_context.get("user_id"),
                "session.session_id": session_data.get("session_id"),
                "session.chat_id": session_data.get("chat_id"),
                "session.message_count": session_data.get("message_count", 0),
                "session.user_name": user_context.get("name"),
                "session.user_role": user_context.get("role"),
                "session.stats_count": session_stats.get("session_count", 0),
                "session.total_queries": session_stats.get("total_queries", 0),
                "event.type": "user_session",
                "timestamp": datetime.now().isoformat()
            }
        ) as session_span:
            
            session_span.add_event(
                "user_session_logged",
                attributes={
                    "user_journey_tracked": True,
                    "session_correlation_enabled": True,
                    "user_context_available": bool(user_context.get("user_id")),
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            session_span.set_status(Status(StatusCode.OK))
        
        logger.info(f"✅ Session logged to Phoenix: {request_id}")
        
    except Exception as e:
        logger.error(f"Session Phoenix logging error: {e}")

async def log_metrics_to_phoenix(request_id: str, metrics: Dict):
    """Log system performance metrics to Phoenix"""
    if not service.phoenix_initialized:
        return
    
    try:
        with service.tracer.start_as_current_span(
            "system_performance_metrics",
            attributes={
                "metrics.request_id": request_id,
                "metrics.response_time_ms": metrics.get("response_time_ms", 0),
                "metrics.response_length": metrics.get("response_length", 0),
                "metrics.context_length": metrics.get("context_length", 0),
                "metrics.model": metrics.get("model", "unknown"),
                "metrics.streaming_enabled": metrics.get("streaming_enabled", False),
                "metrics.pipeline_version": metrics.get("pipeline_version", "unknown"),
                "event.type": "system_metrics",
                "timestamp": datetime.now().isoformat()
            }
        ) as metrics_span:
            
            metrics_span.add_event(
                "performance_metrics_logged",
                attributes={
                    "performance_tracking_enabled": True,
                    "metrics_count": len(metrics),
                    "response_time_tracked": bool(metrics.get("response_time_ms")),
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            metrics_span.set_status(Status(StatusCode.OK))
        
        logger.info(f"✅ Metrics logged to Phoenix: {request_id}")
        
    except Exception as e:
        logger.error(f"Metrics Phoenix logging error: {e}")
 
# FastAPI endpoints
@app.on_event("startup")
async def startup_event():
    """Initialize service on startup"""
    logger.info("🚀 Starting RAGAS + Phoenix Evaluation Service...")
    await service.initialize()
    logger.info("✅ Service initialization complete")

@app.get("/health")
async def health_check():
    """Enhanced health check endpoint with production features"""
    return {
        "status": "healthy",
        "service": "RAGAS + Phoenix Evaluation Service",
        "version": "2.0.0",
        "mode": "production",
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
        "phoenix_url": os.getenv("PHOENIX_ENDPOINT", "https://monitoring.sandbox.dge.gov.ae"),
        "production_features": {
            "user_session_tracking": True,
            "system_metrics_logging": True,
            "enhanced_phoenix_integration": True,
            "comprehensive_payload_support": True,
            "production_observability": True
        },
        "endpoints": {
            "evaluate": "/evaluate",
            "log_session": "/log_session",
            "log_metrics": "/log_metrics",
            "health": "/health",
            "metrics": "/metrics"
        }
    }

@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_rag_comprehensive(request: EvaluationRequest, background_tasks: BackgroundTasks):
    """
    Enhanced evaluate endpoint supporting both legacy and production payloads
    """
    try:
        evaluation_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Handle both legacy and production payload formats
        if request.evaluation_data:
            # Production format
            query = request.evaluation_data.query
            response = request.evaluation_data.response
            context = request.evaluation_data.context
            ground_truth = request.evaluation_data.ground_truth
        else:
            # Legacy format
            query = request.query
            response = request.response
            context = request.context
            ground_truth = request.ground_truth
        
        # Extract production context
        user_profile = request.user_profile
        session_metadata = request.session_metadata
        system_context = request.system_context
        trace_data = request.trace_data
        production_metadata = request.production_metadata
        
        # Log production context if available
        if user_profile or session_metadata or system_context:
            logger.info(f"🏭 PRODUCTION EVALUATION: {evaluation_id}")
            if user_profile and user_profile.user_id:
                logger.info(f"  👤 User: {user_profile.name} ({user_profile.user_id})")
            if session_metadata and session_metadata.session_id:
                logger.info(f"  💬 Session: {session_metadata.session_id}")
            if system_context and system_context.response_time_ms:
                logger.info(f"  ⏱️ Response time: {system_context.response_time_ms}ms")
        
        # Ground truth priority: explicit parameter > fuzzy matching > None
        final_ground_truth, gt_source = get_final_ground_truth(ground_truth, query)
        
        # Get match score for fuzzy matches
        match_score = None
        if gt_source == "fuzzy":
            _, match_score = find_ground_truth(query)
        
        ground_truth_found = final_ground_truth is not None
        
        logger.info(f"🔍 Evaluating query: {query[:50]}...")
        logger.info(f"📋 Ground truth source: {gt_source}")
        
        # Evaluate with RAGAS (always returns all 5 metrics)
        scores = await evaluate_with_ragas(
            query=query,
            response=response,
            context=context,
            ground_truth=final_ground_truth
        )
        
        # Count evaluated vs unavailable metrics
        evaluated_count = sum(1 for v in scores.values() if isinstance(v, (int, float)))
        unavailable_count = len(scores) - evaluated_count
        
        # Enhanced Phoenix logging with production context
        background_tasks.add_task(
            log_to_phoenix_production,
            query, response, context, scores, final_ground_truth,
            evaluation_id, gt_source, user_profile, session_metadata, 
            system_context, trace_data, production_metadata
        )
        
        # Additional user session tracking if manager available
        # CrewAI-style conversation tracking (non-trace)
        if user_profile and session_metadata:
            user_id = user_profile.user_id
            chat_id = session_metadata.chat_id
            
            # Track conversation in background
            background_tasks.add_task(
                conversation_tracker.track_conversation,
                user_id, chat_id, query, response, scores, session_metadata
            )
        
        logger.info(f"✅ Evaluation complete: {evaluation_id}")
        logger.info(f"📊 Metrics: {evaluated_count} evaluated, {unavailable_count} unavailable")
        
        # Create response message
        if gt_source == "explicit":
            message = "Production evaluation completed with explicit ground truth"
        elif gt_source == "fuzzy":
            message = f"Production evaluation completed with fuzzy-matched ground truth (score: {match_score:.2f})"
        else:
            message = "Production evaluation completed - Ground truth dependent metrics show status messages"
        
        return EvaluationResponse(
            status="success",
            message=message,
            evaluation_id=evaluation_id,
            metrics=scores,
            ground_truth_found=ground_truth_found,
            ground_truth_match_score=match_score if gt_source == "fuzzy" else None,
            user_profile=user_profile,
            session_metadata=session_metadata,
            system_context=system_context,
            production_metadata=production_metadata
        )
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/log_session", response_model=SessionLogResponse)
async def log_user_session(request: UserSessionRequest, background_tasks: BackgroundTasks):
    """
    Log user session data for journey tracking
    """
    try:
        logger.info(f"💬 SESSION LOG: {request.request_id}")
        
        # Extract user info
        user_context = request.user_context
        session_data = request.session_data
        
        if user_context.get("user_id"):
            logger.info(f"  👤 User: {user_context.get('name')} ({user_context.get('user_id')})")
        
        if session_data.get("session_id"):
            logger.info(f"  💬 Session: {session_data.get('session_id')}")
            logger.info(f"  📚 Messages: {session_data.get('message_count', 0)}")
        
        # Log to Phoenix for session tracking
        background_tasks.add_task(
            log_session_to_phoenix,
            request.request_id, user_context, session_data, request.session_stats
        )
        
        return SessionLogResponse(
            status="success",
            message="User session logged successfully",
            session_id=session_data.get("session_id"),
            logged_at=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"❌ Session logging failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/log_metrics", response_model=MetricsLogResponse)
async def log_system_metrics(request: SystemMetricsRequest, background_tasks: BackgroundTasks):
    """
    Log system performance metrics
    """
    try:
        logger.info(f"📊 METRICS LOG: {request.request_id}")
        
        metrics = request.metrics
        logger.info(f"  ⏱️ Response time: {metrics.get('response_time_ms', 0)}ms")
        logger.info(f"  📄 Response length: {metrics.get('response_length', 0)} chars")
        logger.info(f"  🔍 Context length: {metrics.get('context_length', 0)} chars")
        logger.info(f"  🤖 Model: {metrics.get('model', 'unknown')}")
        
        # Log to Phoenix for performance monitoring
        background_tasks.add_task(
            log_metrics_to_phoenix,
            request.request_id, metrics
        )
        
        return MetricsLogResponse(
            status="success",
            message="System metrics logged successfully",
            metrics_logged=len(metrics),
            logged_at=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"❌ Metrics logging failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search_sessions")
async def search_user_sessions(request: dict):
    """Search user sessions and chat history for analytics"""
    try:
        # Extract search parameters
        user_id = request.get("user_id")
        session_id = request.get("session_id")
        search_query = request.get("query", "")
        date_from = request.get("date_from")
        date_to = request.get("date_to")
        limit = request.get("limit", 50)
        
        # Build search response (in production, this would query a database)
        search_results = {
            "user_sessions": [],
            "conversation_history": [],
            "quality_analytics": {},
            "user_journey": []
        }
        
        # Add mock data structure for reference
        if user_id:
            search_results["user_sessions"] = [
                {
                    "session_id": f"session_{user_id}_001",
                    "user_id": user_id,
                    "start_time": "2025-07-14T10:00:00Z",
                    "duration": "45.2s",
                    "message_count": 12,
                    "avg_quality_score": 0.87,
                    "topics": ["government", "regulations", "permits"]
                }
            ]
            
            search_results["quality_analytics"] = {
                "avg_faithfulness": 0.89,
                "avg_relevancy": 0.84,
                "total_evaluations": 156,
                "improvement_trend": "+2.3%"
            }
        
        return {
            "status": "success",
            "search_params": {
                "user_id": user_id,
                "session_id": session_id,
                "query": search_query,
                "date_range": f"{date_from} to {date_to}" if date_from and date_to else None
            },
            "results": search_results,
            "total_found": len(search_results.get("user_sessions", [])),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Session search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/user/{user_id}")
async def get_user_analytics(user_id: str):
    """Get user conversation analytics (CrewAI-style)"""
    try:
        analytics = conversation_tracker.get_user_analytics(user_id)
        
        if analytics.get("status") == "not_found":
            raise HTTPException(status_code=404, detail=f"User {user_id} not found")
        
        return {
            "status": "success",
            "data": analytics,
            "generated_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"User analytics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/chat/{chat_id}")
async def get_chat_analytics(chat_id: str):
    """Get chat session analytics"""
    try:
        analytics = conversation_tracker.get_chat_analytics(chat_id)
        
        if analytics.get("status") == "not_found":
            raise HTTPException(status_code=404, detail=f"Chat {chat_id} not found")
        
        return {
            "status": "success",
            "data": analytics,
            "generated_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat analytics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/summary")
async def get_analytics_summary():
    """Get overall analytics summary"""
    try:
        total_users = len(conversation_tracker.conversations)
        total_chats = len(conversation_tracker.chat_sessions)
        
        # Calculate overall metrics
        all_quality_scores = []
        total_interactions = 0
        
        for user_data in conversation_tracker.conversations.values():
            all_quality_scores.extend(user_data["quality_scores"])
            total_interactions += user_data["total_interactions"]
        
        avg_quality = sum(all_quality_scores) / len(all_quality_scores) if all_quality_scores else 0.0
        
        summary = {
            "total_users": total_users,
            "total_chat_sessions": total_chats,
            "total_interactions": total_interactions,
            "avg_quality_score": round(avg_quality, 3),
            "quality_distribution": {
                "excellent": len([s for s in all_quality_scores if s >= 0.9]),
                "good": len([s for s in all_quality_scores if 0.8 <= s < 0.9]),
                "satisfactory": len([s for s in all_quality_scores if 0.7 <= s < 0.8]),
                "needs_improvement": len([s for s in all_quality_scores if s < 0.7])
            }
        }
        
        return {
            "status": "success",
            "data": summary,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Analytics summary error: {e}")
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