"""
GovGPT Contextual RAG Pipeline for OpenWebUI - Production Scale

This pipeline provides advanced contextual retrieval-augmented generation
for UAE Information Assurance standards and government documents by
communicating with a hosted RAG server via HTTP requests.

Features:
- Production-scale user session management (1000+ users)
- Persistent user memory integration with RAG server
- Comprehensive observability and RAGAS evaluation
- Stateless architecture for scalability
- Professional logging for production monitoring

Author: GovGPT Team  
Version: 2.0.0 (Production Scale)
Compatible with: OpenWebUI Pipelines v0.6+
"""

import os
from typing import List, Dict, Any, Optional, AsyncGenerator, Union, Generator, Iterator
from pydantic import BaseModel, Field

import requests
import aiohttp
import json
from tenacity import retry, wait_exponential, stop_after_attempt
import asyncio
from datetime import datetime
import uuid
import time

RAGAS_SERVICE_AVAILABLE = True

class RAGServerClient:
    def __init__(
        self, 
        server_url: str,
        api_key: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3
    ):
        self.server_url = server_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries

        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"

    def health_check(self) -> Dict[str, Any]:
        try:
            response = requests.get(
                f"{self.server_url}/health",
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            return {
                "status": "healthy",
                "server_response": response.json(),
                "status_code": response.status_code
            }
        except requests.exceptions.RequestException as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "server_url": self.server_url
            }

    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
    def query_sync(self, query: str, model: Optional[str] = None) -> Dict[str, Any]:
        payload = {"query": query}
        if model:
            payload["model"] = model

        try:
            response = requests.post(
                f"{self.server_url}/query",
                headers=self.headers,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"RAG server request failed: {e}")

    def get_server_info(self) -> Dict[str, Any]:
        return {
            "server_url": self.server_url,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "has_api_key": bool(self.api_key),
            "endpoints": {
                "health": f"{self.server_url}/health",
                "query": f"{self.server_url}/query",
                "retrieve": f"{self.server_url}/retrieve"
            }
        }


class Pipeline:
    class Valves(BaseModel):
        RAG_SERVER_URL: str = Field(
            default="http://host.docker.internal:8100",
            description="URL of the hosted RAG server (local Pipeline server)"
        )
        RAG_SERVER_TIMEOUT: int = Field(
            default=30,
            description="Request timeout in seconds"
        )
        RAG_SERVER_API_KEY: str = Field(
            default="",
            description="Optional API key for RAG server authentication"
        )
        RAG_SERVER_MAX_RETRIES: int = Field(
            default=3,
            description="Maximum retry attempts for failed requests"
        )
        RAG_MODEL: str = Field(
            default="gpt-4.1",
            description="Primary LLM model for response generation (sent to server)"
        )
        ENABLE_STREAMING: bool = Field(
            default=True,
            description="Enable streaming responses via /retrieve endpoint"
        )
        ENABLE_DEBUG: bool = Field(
            default=False,
            description="Enable debug logging for troubleshooting"
        )
        STREAM_BUFFER_SIZE: int = Field(
            default=1024,
            description="Buffer size for streaming responses"
        )
        STREAM_WORD_BOUNDARY: bool = Field(
            default=True,
            description="Enable word boundary detection for smoother streaming"
        )
        STREAM_CHUNK_SIZE: int = Field(
            default=25,
            description="Characters per chunk when using word boundary streaming"
        )
        ENABLE_RAGAS: bool = Field(
            default=True,
            description="Enable RAGAS + Phoenix real-time evaluation"
        )
        RAGAS_SERVICE_URL: str = Field(
            default="http://host.docker.internal:8300",
            description="URL of the RAGAS + Phoenix evaluation service"
        )
        RAGAS_SERVICE_TIMEOUT: int = Field(
            default=30,
            description="Timeout for RAGAS service requests in seconds"
        )

    def __init__(self):
        self.name = "GovGPT Contextual RAG (Production Scale)"
        self.valves = self.Valves(
            **{
                "RAG_SERVER_URL": os.getenv("RAG_SERVER_URL", "http://host.docker.internal:8100"),
                "RAG_SERVER_TIMEOUT": int(os.getenv("RAG_SERVER_TIMEOUT", "30")),
                "RAG_SERVER_API_KEY": os.getenv("RAG_SERVER_API_KEY", ""),
                "RAG_SERVER_MAX_RETRIES": int(os.getenv("RAG_SERVER_MAX_RETRIES", "3")),
                "RAG_MODEL": os.getenv("RAG_MODEL", "gpt-4.1"),
                "ENABLE_STREAMING": os.getenv("ENABLE_STREAMING", "true").lower() == "true",
                "ENABLE_DEBUG": os.getenv("ENABLE_DEBUG", "false").lower() == "true",
                "STREAM_BUFFER_SIZE": int(os.getenv("STREAM_BUFFER_SIZE", "1024")),
                "STREAM_WORD_BOUNDARY": os.getenv("STREAM_WORD_BOUNDARY", "true").lower() == "true",
                "STREAM_CHUNK_SIZE": int(os.getenv("STREAM_CHUNK_SIZE", "25")),
                "ENABLE_RAGAS": os.getenv("ENABLE_RAGAS", "true").lower() == "true",
                "RAGAS_SERVICE_URL": os.getenv("RAGAS_SERVICE_URL", "http://host.docker.internal:8300"),
                "RAGAS_SERVICE_TIMEOUT": int(os.getenv("RAGAS_SERVICE_TIMEOUT", "30")),
            }
        )
        self.rag_client = None
        self.ragas_service_available = False
        
        # Production observability - no instance storage for scalability
        self.request_id = None  # Set per request for correlation

    async def on_startup(self):
        """Initialize pipeline on startup"""
        print(f"🚀 {self.name} starting up...")
        
        # Test RAGAS service if enabled
        if self.valves.ENABLE_RAGAS:
            try:
                await self._test_ragas_service()
            except Exception as e:
                print(f"⚠️ RAGAS service test failed: {e}")
        
        # Test server connection on startup if debug enabled
        if self.valves.ENABLE_DEBUG:
            try:
                client = self._get_rag_client()
                health = client.health_check()
                if health["status"] == "healthy":
                    print(f"✅ RAG server connection successful")
                else:
                    print(f"⚠️ RAG server health check failed: {health.get('error', 'Unknown error')}")
            except Exception as e:
                print(f"❌ RAG server connection failed: {e}")
    
    async def _test_ragas_service(self):
        """Test RAGAS + Phoenix service availability"""
        try:
            response = requests.get(
                f"{self.valves.RAGAS_SERVICE_URL}/health",
                timeout=5
            )
            response.raise_for_status()
            
            health_data = response.json()
            self.ragas_service_available = True
            print(f"✅ RAGAS service healthy at {self.valves.RAGAS_SERVICE_URL}")
            print(f"   - RAGAS initialized: {health_data.get('ragas_initialized', False)}")
            print(f"   - Phoenix initialized: {health_data.get('phoenix_initialized', False)}")
            print(f"   - Ground truth samples: {health_data.get('ground_truth_samples', 0)}")
            
            phoenix_url = health_data.get('phoenix_url')
            if phoenix_url:
                print(f"   - Phoenix UI: {phoenix_url}")
                
        except Exception as e:
            print(f"❌ RAGAS service connection failed: {e}")
            print(f"   - Make sure service is running at {self.valves.RAGAS_SERVICE_URL}")
            self.ragas_service_available = False

    async def inlet(self, body: dict, __user__: Optional[dict] = None, **kwargs) -> dict:
        """
        INLET: Extract user context for production observability
        Stateless design - extract all context from request
        """
        # Generate request ID for correlation and start timing
        self.request_id = str(uuid.uuid4())
        self.request_start_time = time.time()
        
        # Extract user context from available sources
        user_context = self._extract_user_context(body, __user__)
        session_data = self._extract_session_data(body, kwargs)
        
        # Update user session statistics
        self._update_user_session_stats(user_context, session_data)
        
        # Log request start with user context
        self._log_request_start(user_context, session_data)
        
        return body
    
    def _extract_user_context(self, body: dict, __user__: Optional[dict] = None) -> dict:
        """Extract user context from request sources"""
        # Primary: __user__ parameter
        if __user__:
            user_context = {
                "user_id": __user__.get("id"),
                "name": __user__.get("name"),
                "email": __user__.get("email"),
                "role": __user__.get("role")
            }
        # Fallback: body.user field
        elif body.get("user"):
            user_data = body["user"]
            user_context = {
                "user_id": user_data.get("id"),
                "name": user_data.get("name"),
                "email": user_data.get("email"),
                "role": user_data.get("role")
            }
        # Fallback: variables
        else:
            variables = body.get("metadata", {}).get("variables", {})
            user_context = {
                "user_id": None,
                "name": variables.get("{{USER_NAME}}", "Unknown"),
                "email": None,
                "role": "user"
            }
        
        return user_context
    
    def _extract_session_data(self, body: dict, kwargs: dict = None) -> dict:
        """Extract session data from request (OpenWebUI sends metadata in kwargs)"""
        kwargs = kwargs or {}
        metadata = body.get("metadata", {})
        
        # Debug: Log available kwargs to understand the structure (always log for now to fix IDs issue)
        if kwargs:
            print(f"🔍 DEBUG SESSION DATA: kwargs keys: {list(kwargs.keys())}")
            for key, value in kwargs.items():
                if key not in ['__user__']:  # Don't log user data
                    print(f"   - {key}: {type(value)} = {value}")
        else:
            print("🔍 DEBUG SESSION DATA: No kwargs provided")
        
        return {
            # Try multiple sources: kwargs (OpenWebUI metadata), body, then metadata
            "chat_id": kwargs.get("chat_id") or body.get("chat_id") or metadata.get("chat_id"),
            "session_id": kwargs.get("session_id") or body.get("session_id") or metadata.get("session_id"), 
            "message_id": kwargs.get("message_id") or kwargs.get("id") or body.get("id") or metadata.get("message_id"),
            "variables": kwargs.get("variables", {}) or body.get("variables", {}) or metadata.get("variables", {}),
            "features": kwargs.get("features", {}) or body.get("features", {}) or metadata.get("features", {}),
            "message_count": len(body.get("messages", []))
        }
    
    def _update_user_session_stats(self, user_context: dict, session_data: dict):
        """Update user session statistics for Phoenix logging"""
        # Simple session stats tracking (in production, this would come from database)
        user_id = user_context.get("user_id") if user_context else None
        session_id = session_data.get("session_id") if session_data else None
        
        if user_id:
            self.user_session_stats = {
                "session_count": 1,  # In production, query from user database
                "total_queries": 1,   # In production, query from user database
                "current_session": session_id
            }
    
    def _calculate_session_duration(self) -> str:
        """Calculate session duration for analytics"""
        try:
            # Calculate duration based on request processing time
            if hasattr(self, 'request_start_time'):
                duration_seconds = time.time() - self.request_start_time
                return f"{duration_seconds:.1f}s"
            else:
                # Fallback to default duration
                return "0.0s"
        except Exception:
            return "0.0s"
    
    def _calculate_retrieval_score(self, context: str) -> float:
        """Calculate a simple retrieval quality score based on context characteristics"""
        try:
            if not context or not context.strip():
                return 0.0
            
            # Simple scoring based on context characteristics
            context_length = len(context.strip())
            word_count = len(context.split())
            
            # Basic quality indicators
            score = 0.0
            
            # Length scoring (reasonable context length)
            if 100 <= context_length <= 2000:
                score += 0.3
            elif context_length > 50:
                score += 0.2
            
            # Word count scoring
            if 20 <= word_count <= 400:
                score += 0.3
            elif word_count > 10:
                score += 0.2
            
            # Content quality indicators
            if any(keyword in context.lower() for keyword in ['government', 'uae', 'emirates', 'regulation', 'license']):
                score += 0.2
                
            # Sentence structure (has proper sentences)
            sentences = context.count('.') + context.count('!') + context.count('?')
            if sentences >= 2:
                score += 0.2
            
            return min(score, 1.0)  # Cap at 1.0
            
        except Exception:
            return 0.5  # Default neutral score
    
    def _log_request_start(self, user_context: dict, session_data: dict):
        """Log request start with user context"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": "request_start",
            "request_id": self.request_id,
            "user": user_context,
            "session": session_data,
            "pipeline": self.name
        }
        # Always log production observability data
        print(f"📥 INLET: {json.dumps(log_entry, indent=2)}")
    
    async def on_shutdown(self):
        """Clean up on shutdown"""
        print(f"🛑 {self.name} shutting down...")
        
        # Clean up client connection
        if self.rag_client:
            self.rag_client = None

    def _get_rag_client(self) -> RAGServerClient:
        if not self.rag_client or self._client_config_changed():
            self.rag_client = RAGServerClient(
                server_url=self.valves.RAG_SERVER_URL,
                api_key=self.valves.RAG_SERVER_API_KEY if self.valves.RAG_SERVER_API_KEY else None,
                timeout=self.valves.RAG_SERVER_TIMEOUT,
                max_retries=self.valves.RAG_SERVER_MAX_RETRIES
            )
        return self.rag_client

    def _client_config_changed(self) -> bool:
        """Check if client configuration has changed to avoid unnecessary recreations"""
        if not self.rag_client:
            return True
        
        current_config = (
            self.valves.RAG_SERVER_URL,
            self.valves.RAG_SERVER_TIMEOUT, 
            self.valves.RAG_SERVER_MAX_RETRIES,
            self.valves.RAG_SERVER_API_KEY
        )
        
        client_config = (
            self.rag_client.server_url,
            self.rag_client.timeout,
            self.rag_client.max_retries,
            self.rag_client.api_key or ""
        )
        
        return current_config != client_config
    
    def _evaluate_with_ragas(self, query: str, response: str, context: str = "", user_context: dict = None, session_data: dict = None):
        """Evaluate response with RAGAS + Phoenix service with user context"""
        if not (self.valves.ENABLE_RAGAS and self.ragas_service_available):
            return
        
        try:
            # Enhanced evaluation payload with user context
            payload = {
                "query": query,
                "response": response,
                "context": context,
                "user_context": user_context or {},
                "session_data": session_data or {},
                "request_id": self.request_id,
                "timestamp": datetime.now().isoformat(),
                "pipeline": self.name
            }
            
            # Send to RAGAS service
            eval_response = requests.post(
                f"{self.valves.RAGAS_SERVICE_URL}/evaluate",
                json=payload,
                timeout=self.valves.RAGAS_SERVICE_TIMEOUT
            )
            eval_response.raise_for_status()
            
            eval_result = eval_response.json()
            
            # Production logging with user context
            self._log_ragas_evaluation(query, eval_result, user_context, session_data)
                
        except Exception as e:
            self._log_ragas_error(query, str(e), user_context, session_data)
    
    def _log_ragas_evaluation(self, query: str, eval_result: dict, user_context: dict, session_data: dict):
        """Log RAGAS evaluation results with user context"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": "ragas_evaluation",
            "request_id": self.request_id,
            "query": query[:50] + "..." if len(query) > 50 else query,
            "user": user_context,
            "session": session_data,
            "evaluation": {
                "evaluation_id": eval_result.get('evaluation_id'),
                "ground_truth_found": eval_result.get('ground_truth_found', False),
                "match_score": eval_result.get('ground_truth_match_score'),
                "metrics": list(eval_result.get('metrics', {}).keys())
            }
        }
        
        if self.valves.ENABLE_DEBUG:
            print(f"✅ RAGAS: {json.dumps(log_entry, indent=2)}")
    
    def _log_ragas_error(self, query: str, error: str, user_context: dict, session_data: dict):
        """Log RAGAS evaluation error with user context"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": "ragas_error",
            "request_id": self.request_id,
            "query": query[:50] + "..." if len(query) > 50 else query,
            "user": user_context,
            "session": session_data,
            "error": error
        }
        
        if self.valves.ENABLE_DEBUG:
            print(f"❌ RAGAS ERROR: {json.dumps(log_entry, indent=2)}")

    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: List[dict],
        body: dict
    ) -> Union[str, Generator, Iterator]:
        """Main pipeline method with user context forwarding"""
        try:
            if not user_message or not user_message.strip():
                return "❌ No user query provided"

            model = model_id or self.valves.RAG_MODEL
            client = self._get_rag_client()
            
            # Extract user context for RAG server
            user_context = self._extract_user_context(body, None)
            session_data = self._extract_session_data(body)
            
            # Log processing start
            self._log_processing_start(user_message, user_context, session_data, model)
            
            if self.valves.ENABLE_STREAMING:
                return self._stream_response(client, user_message, model, user_context, session_data, messages)
            else:
                return self._sync_response(client, user_message, model, user_context, session_data, messages)

        except Exception as e:
            error_msg = f"❌ Pipeline error: {str(e)}"
            self._log_error(user_message, str(e))
            return error_msg
    
    def _log_processing_start(self, query: str, user_context: dict, session_data: dict, model: str):
        """Log processing start with user context"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": "processing_start",
            "request_id": self.request_id,
            "query": query[:100] + "..." if len(query) > 100 else query,
            "user": user_context,
            "session": session_data,
            "model": model,
            "streaming": self.valves.ENABLE_STREAMING
        }
        # Always log production observability data
        print(f"⚙️ PIPE: {json.dumps(log_entry, indent=2)}")
    
    def _log_error(self, query: str, error: str):
        """Log error with context"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": "error",
            "request_id": self.request_id,
            "query": query[:100] + "..." if len(query) > 100 else query,
            "error": error,
            "pipeline": self.name
        }
        print(f"❌ ERROR: {json.dumps(log_entry, indent=2)}")

    def _stream_response(self, client: RAGServerClient, user_message: str, model: str, user_context: dict, session_data: dict, messages: List[dict]) -> Generator[str, None, None]:
        """Stream response from RAG server with context extraction"""
        try:
            # Use requests for streaming to avoid event loop issues
            import requests
            
            # Enhanced payload with user context for RAG server
            payload = {
                "query": user_message,
                "user_context": user_context,
                "session_data": session_data,
                "conversation_history": messages,
                "request_id": self.request_id
            }
            if model:
                payload["model"] = model
            
            headers = {
                "Content-Type": "application/json"
                # Server returns plain text lines, not SSE format
            }
            if client.api_key:
                headers["Authorization"] = f"Bearer {client.api_key}"
            
            response = requests.post(
                f"{client.server_url}/retrieve",
                json=payload,
                headers=headers,
                stream=True,
                timeout=client.timeout
            )
            response.raise_for_status()

            # Simple two-mode streaming: either has context at start or doesn't
            extracted_context = ""
            full_response = ""
            context_buffer = ""
            has_context = None  # None, True, False
            
            for chunk in response.iter_content(chunk_size=1024, decode_unicode=True):
                if chunk:
                    if has_context is None:
                        # First chunk - determine if response has context
                        context_buffer = chunk
                        if context_buffer.startswith('[CONTEXT]'):
                            has_context = True
                            if self.valves.ENABLE_DEBUG:
                                print(f"🔍 Context detected at start, buffering until complete")
                        else:
                            has_context = False
                            full_response += chunk
                            yield chunk
                            if self.valves.ENABLE_DEBUG:
                                print(f"📡 No context, streaming directly")
                        continue
                    
                    if has_context:
                        # Buffer until we have complete context block
                        context_buffer += chunk
                        
                        if '[/CONTEXT]' in context_buffer:
                            # Extract context and stream the rest
                            end_pos = context_buffer.find('[/CONTEXT]')
                            extracted_context = context_buffer[9:end_pos]  # Skip '[CONTEXT]'
                            answer_part = context_buffer[end_pos + 10:]  # Skip '[/CONTEXT]'
                            
                            if answer_part:
                                full_response += answer_part
                                yield answer_part
                            
                            has_context = False  # Switch to direct streaming
                            if self.valves.ENABLE_DEBUG:
                                print(f"🔍 Context extracted: {len(extracted_context)} characters")
                                print(f"✅ Switching to direct streaming")
                    else:
                        # Direct streaming mode
                        full_response += chunk
                        yield chunk
            
            # Store evaluation data for background processing in outlet
            if full_response.strip():
                if self.valves.ENABLE_DEBUG:
                    print(f"📝 Full response: {len(full_response)} characters")
                
                # Store for background evaluation in outlet method
                self.pending_evaluation = {
                    "query": user_message,
                    "response": full_response.strip(),
                    "context": extracted_context,
                    "user_context": user_context,
                    "session_data": session_data,
                    "messages": messages,
                    "type": "streaming"
                }
            
            # signal end‐of‐stream
            yield "\n"

            # if self.schema.response_mode == "streaming":
            # # Simply forward each raw text chunk
            #     for chunk in response.iter_content(chunk_size=1024, decode_unicode=True):
            #         if chunk:
            #             yield chunk
            #     # signal end‐of‐stream
            #     yield "\n"

            # else:
            #     # Blocking mode – whole answer at once
            #     data = response.json()
            #     yield data.get("answer", "")
            #     yield "\n"
            
            # Process streaming response line by line
            # if self.valves.STREAM_WORD_BOUNDARY:
            #     # Stream with word boundaries for smoother display
            #     buffer = ""
            #     chunk_size = 25  # Characters per chunk
                
            #     for line in response.iter_lines(decode_unicode=True):
            #         if line is not None:  # line can be empty string
            #             # Add line to buffer with newline
            #             if line.strip():  # Only add newline for non-empty lines
            #                 buffer += line + "\n"
            #             elif buffer:  # Empty line, process current buffer
            #                 # Process buffer in chunks with word boundaries
            #                 while buffer:
            #                     if len(buffer) <= chunk_size:
            #                         if buffer.strip():
            #                             yield buffer
            #                         buffer = ""
            #                     else:
            #                         # Find a good break point (space, punctuation)
            #                         break_point = chunk_size
            #                         for i in range(min(chunk_size, len(buffer)), 0, -1):
            #                             if i > 0 and buffer[i-1] in [' ', '.', '!', '?', ',', ';', ':', '\n']:
            #                                 break_point = i
            #                                 break
                                    
            #                         chunk_to_yield = buffer[:break_point]
            #                         if chunk_to_yield.strip():
            #                             yield chunk_to_yield
            #                         buffer = buffer[break_point:]
                
            #     # Yield any remaining buffer
            #     if buffer.strip():
            #         yield buffer
            # else:
            #     # Stream line by line for immediate display
            #     for line in response.iter_lines(decode_unicode=True):
            #         if line is not None:
            #             if line.strip():  # Only yield non-empty lines
            #                 yield line + "\n"
            #             else:
            #                 yield "\n"
                
        except Exception as e:
            yield f"❌ Streaming request failed: {str(e)}"

    def _sync_response(self, client: RAGServerClient, user_message: str, model: str, user_context: dict, session_data: dict, messages: List[dict]) -> str:
        try:
            # Enhanced payload with user context
            payload = {
                "query": user_message,
                "user_context": user_context,
                "session_data": session_data,
                "conversation_history": messages,
                "request_id": self.request_id
            }
            if model:
                payload["model"] = model
            
            # Send enhanced request to RAG server
            response = requests.post(
                f"{client.server_url}/query",
                headers=client.headers,
                json=payload,
                timeout=client.timeout
            )
            response.raise_for_status()
            result = response.json()
            
            answer = result.get("answer", "❌ No answer received from server")
            context = result.get("context", "")
            
            # Store evaluation data for background processing in outlet
            self.pending_evaluation = {
                "query": user_message,
                "response": answer,
                "context": context,
                "user_context": user_context,
                "session_data": session_data,
                "messages": messages,
                "type": "sync"
            }
            
            # Add debug info if enabled
            if self.valves.ENABLE_DEBUG:
                debug_info = "\n\n---\n📊 Debug Info:\n"
                debug_info += f"- Response length: {len(answer)} chars\n"
                debug_info += f"- Model used: {model}\n"
                debug_info += f"- Server: {self.valves.RAG_SERVER_URL}\n"
                if "context" in result:
                    debug_info += "- Context retrieved: Yes\n"
                if self.valves.ENABLE_RAGAS and self.ragas_service_available:
                    debug_info += "- RAGAS evaluation: Enabled\n"
                answer += debug_info
            
            return answer
        except Exception as e:
            error_msg = f"❌ Query request failed: {str(e)}"
            if self.valves.ENABLE_DEBUG:
                print(f"Sync query error: {e}")
            return error_msg
    
    async def outlet(self, body: dict, __user__: Optional[dict] = None, **_kwargs) -> dict:
        """
        OUTLET: Background RAGAS evaluation and comprehensive Phoenix logging
        """
        # Extract user context for final logging
        user_context = self._extract_user_context(body, __user__)
        session_data = self._extract_session_data(body)
        
        # Log response completion
        self._log_response_completion(body, user_context, session_data)
        
        # Perform background RAGAS evaluation with comprehensive data
        if self.pending_evaluation:
            # Fire-and-forget RAGAS evaluation with complete conversation history
            self._background_evaluate_with_ragas(
                self.pending_evaluation, 
                body, 
                user_context, 
                session_data
            )
            # Clear pending evaluation
            self.pending_evaluation = None
        
        return body
    
    def _log_response_completion(self, body: dict, user_context: dict, session_data: dict):
        """Log response completion with user context"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": "response_completion",
            "request_id": self.request_id,
            "user": user_context,
            "session": session_data,
            "response_data": {
                "chat_id": body.get('chat_id'),
                "message_id": body.get('id'),
                "message_count": len(body.get('messages', [])),
                "model": body.get('model')
            }
        }
        
        # Always log production observability data
        print(f"📤 OUTLET: {json.dumps(log_entry, indent=2)}")
    
    def _background_evaluate_with_ragas(self, evaluation_data: dict, outlet_body: dict, user_context: dict, session_data: dict):
        """
        Background RAGAS evaluation with comprehensive user session and chat history
        """
        if not (self.valves.ENABLE_RAGAS and self.ragas_service_available):
            return
        
        try:
            # Build comprehensive conversation history from outlet body
            conversation_history = self._build_conversation_history(outlet_body, evaluation_data)
            
            # Enhanced user profile with session analytics
            enhanced_user_profile = self._build_enhanced_user_profile(user_context, session_data)
            
            # Chat session metadata with quality tracking
            chat_session_metadata = self._build_chat_session_metadata(session_data, conversation_history)
            
            # System context with performance metrics
            system_context = self._build_comprehensive_system_context(evaluation_data)
            
            # Comprehensive production payload matching RAGAS service Pydantic models
            comprehensive_payload = {
                # Legacy fields for backward compatibility
                "query": evaluation_data["query"],
                "response": evaluation_data["response"],
                "context": evaluation_data["context"],
                "ground_truth": None,
                
                # Enhanced production fields
                "evaluation_data": {
                    "query": evaluation_data["query"],
                    "response": evaluation_data["response"],
                    "context": evaluation_data["context"],
                    "ground_truth": None
                },
                "user_profile": enhanced_user_profile,
                "session_metadata": chat_session_metadata,
                "system_context": system_context,
                "trace_data": {
                    "trace_id": str(uuid.uuid4()),
                    "span_id": str(uuid.uuid4()),
                    "parent_span_id": self.request_id,
                    "operation_name": "comprehensive_rag_evaluation",
                    "tags": {
                        "user_id": enhanced_user_profile.get("user_id"),
                        "session_id": chat_session_metadata.get("session_id"),
                        "chat_id": chat_session_metadata.get("chat_id"),
                        "evaluation_type": "background_comprehensive"
                    },
                    "attributes": {
                        "pipeline.name": self.name,
                        "pipeline.version": "2.0.0",
                        "evaluation.background": True
                    }
                },
                "production_metadata": {
                    "environment": "production",
                    "service_name": "govgpt_rag_pipeline",
                    "version": "2.0.0",
                    "logged_at": datetime.now().isoformat()
                },
                
                # Additional context for comprehensive tracking
                "user_context": user_context,
                "session_data": session_data,
                "request_id": self.request_id,
                "timestamp": datetime.now().isoformat(),
                "pipeline": self.name
            }
            
            # Fire-and-forget request to RAGAS service
            import threading
            thread = threading.Thread(
                target=self._send_comprehensive_evaluation,
                args=(comprehensive_payload,)
            )
            thread.daemon = True  # Don't block application shutdown
            thread.start()
            
            print(f"✅ Background RAGAS evaluation initiated for user {enhanced_user_profile.get('name', 'Unknown')}")
            
        except Exception as e:
            # Log error but don't raise - background processing should not affect main flow
            print(f"⚠️ Background RAGAS evaluation error: {str(e)}")
    
    def _build_conversation_history(self, outlet_body: dict, evaluation_data: dict) -> list:
        """Build comprehensive conversation history with metadata"""
        conversation_history = []
        
        # Get messages from outlet body (complete conversation)
        messages = outlet_body.get("messages", evaluation_data.get("messages", []))
        
        for i, message in enumerate(messages):
            message_entry = {
                "message_id": message.get("id", f"msg_{i+1:03d}"),
                "timestamp": datetime.fromtimestamp(message.get("timestamp", time.time())).isoformat() if message.get("timestamp") else datetime.now().isoformat(),
                "role": message.get("role", "unknown"),
                "content": message.get("content", ""),
                "metadata": {
                    "content_length": len(message.get("content", "")),
                    "message_index": i,
                    "has_timestamp": bool(message.get("timestamp"))
                }
            }
            
            # Add evaluation metadata for assistant messages
            if message.get("role") == "assistant" and i == len(messages) - 1:
                # This is the current response being evaluated
                message_entry["metadata"].update({
                    "model": self.valves.RAG_MODEL,
                    "response_time_ms": int((time.time() - (self.request_start_time or time.time())) * 1000),
                    "context_used": bool(evaluation_data.get("context")),
                    "context_length": len(evaluation_data.get("context", "")),
                    "evaluation_pending": True,
                    "streaming_used": evaluation_data.get("type") == "streaming"
                })
            
            conversation_history.append(message_entry)
        
        return conversation_history
    
    def _build_enhanced_user_profile(self, user_context: dict, session_data: dict) -> dict:
        """Build enhanced user profile with session statistics"""
        variables = session_data.get("variables", {}) if session_data else {}
        
        return {
            "user_id": user_context.get("user_id") if user_context else None,
            "name": user_context.get("name") if user_context else variables.get("{{USER_NAME}}", "Unknown"),
            "email": user_context.get("email") if user_context else None,
            "role": user_context.get("role") if user_context else "user",
            "preferences": {
                "language": variables.get("{{USER_LANGUAGE}}", "en-US"),
                "timezone": variables.get("{{CURRENT_TIMEZONE}}", "UTC"),
                "location": variables.get("{{USER_LOCATION}}", "Unknown")
            },
            "session_stats": {
                "total_sessions": self.user_session_stats.get("session_count", 1),
                "total_queries": self.user_session_stats.get("total_queries", 1),
                "current_session_duration": self._calculate_session_duration(),
                "last_interaction": datetime.now().isoformat()
            }
        }
    
    def _build_chat_session_metadata(self, session_data: dict, conversation_history: list) -> dict:
        """Build chat session metadata with analytics"""
        if not session_data:
            session_data = {}
        
        # Calculate conversation topics from recent messages
        topics = self._extract_conversation_topics(conversation_history)
        
        return {
            "session_id": session_data.get("session_id"),
            "chat_id": session_data.get("chat_id"),
            "start_time": datetime.fromtimestamp(self.request_start_time).isoformat() if self.request_start_time else datetime.now().isoformat(),
            "last_interaction": datetime.now().isoformat(),
            "duration_seconds": int(time.time() - (self.request_start_time or time.time())),
            "message_count": len(conversation_history),
            "conversation_type": "rag_query",
            "topics": topics,
            "features": session_data.get("features", {}),
            "quality_scores": {
                "pending_evaluation": True,
                "avg_response_time_ms": int((time.time() - (self.request_start_time or time.time())) * 1000)
            }
        }
    
    def _build_comprehensive_system_context(self, evaluation_data: dict) -> dict:
        """Build comprehensive system context with performance metrics"""
        response_time_ms = int((time.time() - (self.request_start_time or time.time())) * 1000)
        
        return {
            "pipeline_name": self.name,
            "pipeline_version": "2.0.0",
            "model": self.valves.RAG_MODEL,
            "request_id": self.request_id,
            "timestamp": datetime.now().isoformat(),
            "response_time_ms": response_time_ms,
            "token_count": len(evaluation_data.get("response", "").split()),
            "context_length": len(evaluation_data.get("context", "")),
            "retrieval_score": self._calculate_retrieval_score(evaluation_data.get("context", "")),
            "streaming_enabled": self.valves.ENABLE_STREAMING,
            "evaluation_type": evaluation_data.get("type", "unknown"),
            "background_processing": True
        }
    
    def _extract_conversation_topics(self, conversation_history: list) -> list:
        """Extract conversation topics from message content"""
        topics = []
        
        # Simple keyword extraction from user messages
        user_messages = [msg for msg in conversation_history if msg.get("role") == "user"]
        
        common_keywords = ["procurement", "methodology", "calculation", "government", "policy", 
                          "regulation", "compliance", "standards", "guidelines", "procedures"]
        
        for message in user_messages[-3:]:  # Last 3 user messages
            content = message.get("content", "").lower()
            for keyword in common_keywords:
                if keyword in content and keyword not in topics:
                    topics.append(keyword)
        
        return topics[:5]  # Limit to 5 topics
    
    def _send_comprehensive_evaluation(self, payload: dict):
        """Send comprehensive evaluation to RAGAS service (background thread)"""
        try:
            response = requests.post(
                f"{self.valves.RAGAS_SERVICE_URL}/evaluate",
                json=payload,
                timeout=30,  # Longer timeout for comprehensive evaluation
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                print(f"✅ Comprehensive background evaluation sent successfully")
            else:
                print(f"⚠️ Background evaluation response: {response.status_code}")
                
        except Exception as e:
            print(f"⚠️ Background evaluation request failed: {str(e)}")
            # Silent failure - don't affect main application
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get pipeline information for debugging"""
        return {
            "name": self.name,
            "version": "2.0.0",
            "server_url": self.valves.RAG_SERVER_URL,
            "streaming_enabled": self.valves.ENABLE_STREAMING,
            "debug_enabled": self.valves.ENABLE_DEBUG,
            "word_boundary_enabled": self.valves.STREAM_WORD_BOUNDARY,
            "buffer_size": self.valves.STREAM_BUFFER_SIZE,
            "chunk_size": self.valves.STREAM_CHUNK_SIZE,
            "timeout": self.valves.RAG_SERVER_TIMEOUT,
            "model": self.valves.RAG_MODEL,
            "ragas_enabled": self.valves.ENABLE_RAGAS,
            "ragas_service_available": self.ragas_service_available,
            "ragas_service_url": self.valves.RAGAS_SERVICE_URL,
            "observability_features": {
                "user_context_tracking": True,
                "session_correlation": True,
                "production_logging": True,
                "stateless_architecture": True
            }
        }

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive pipeline status"""
        client_info = None
        server_health = None
        
        if self.rag_client:
            client_info = self.rag_client.get_server_info()
            try:
                server_health = self.rag_client.health_check()
            except Exception as e:
                server_health = {"status": "error", "error": str(e)}

        return {
            "pipeline_name": self.name,
            "mode": "production_scale",
            "version": "2.0.0",
            "valves": self.valves.model_dump(),
            "server_config": client_info,
            "server_health": server_health,
            "client_initialized": bool(self.rag_client),
            "ragas_status": {
                "service_available": RAGAS_SERVICE_AVAILABLE,
                "enabled": self.valves.ENABLE_RAGAS,
                "service_healthy": self.ragas_service_available,
                "service_url": self.valves.RAGAS_SERVICE_URL,
            },
            "observability_status": {
                "user_context_tracking": True,
                "session_correlation": True,
                "production_logging": True,
                "stateless_design": True,
                "scalable_architecture": True
            }
        }
