"""
GovGPT Contextual RAG Pipeline for OpenWebUI

This pipeline provides advanced contextual retrieval-augmented generation
for UAE Information Assurance standards and government documents by
communicating with a hosted RAG server via HTTP requests.

Author: GovGPT Team  
Version: 1.1.0 (Server Client Mode)
Compatible with: OpenWebUI Pipelines v0.5+
"""

import os
from typing import List, Dict, Any, Optional, AsyncGenerator, Union, Generator, Iterator
from pydantic import BaseModel, Field

import requests
import aiohttp
import json
from tenacity import retry, wait_exponential, stop_after_attempt
import asyncio

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
            description="URL of the hosted RAG server"
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
        self.name = "GovGPT Contextual RAG (GPT-4.1)"
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
            phoenix_url = health_data.get('phoenix_url')
            if phoenix_url:
                print(f"   - Phoenix UI: {phoenix_url}")
                
        except Exception as e:
            print(f"❌ RAGAS service connection failed: {e}")
            print(f"   - Make sure service is running at {self.valves.RAGAS_SERVICE_URL}")
            self.ragas_service_available = False

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
    
    def _evaluate_with_ragas(self, query: str, response: str, context: str = ""):
        """Evaluate response with RAGAS + Phoenix service"""
        if not (self.valves.ENABLE_RAGAS and self.ragas_service_available):
            return
        
        try:
            
            # Prepare evaluation request
            payload = {
                "query": query,
                "response": response,
                "context": context
            }
            
            # Send to RAGAS service
            response = requests.post(
                f"{self.valves.RAGAS_SERVICE_URL}/evaluate",
                json=payload,
                timeout=self.valves.RAGAS_SERVICE_TIMEOUT
            )
            response.raise_for_status()
            
            eval_result = response.json()
            
            if self.valves.ENABLE_DEBUG:
                print(f"✅ RAGAS evaluation completed for query: {query[:50]}...")
                print(f"   - Evaluation ID: {eval_result.get('evaluation_id')}")
           
                if eval_result.get('metrics'):
                    print(f"   - Metrics: {list(eval_result['metrics'].keys())}")
                
        except Exception as e:
            if self.valves.ENABLE_DEBUG:
                print(f"❌ RAGAS evaluation error: {e}")
            # Don't raise - evaluation failure shouldn't break the pipeline

    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: List[dict],
        body: dict
    ) -> Union[str, Generator, Iterator]:
        """Main pipeline method called by OpenWebUI"""
        # Note: messages and body parameters are available but not used in this implementation
        # They contain the full chat context and request body if needed for advanced processing
        try:
            if not user_message or not user_message.strip():
                return "❌ No user query provided"

            model = model_id or self.valves.RAG_MODEL
            client = self._get_rag_client()
            
            if self.valves.ENABLE_DEBUG:
                print(f"🔍 Processing query: {user_message[:100]}...")
                print(f"🤖 Using model: {model}")
                print(f"📡 Streaming mode: {self.valves.ENABLE_STREAMING}")
                print(f"🌐 Server URL: {self.valves.RAG_SERVER_URL}")

            if self.valves.ENABLE_STREAMING:
                return self._stream_response(client, user_message, model)
            else:
                return self._sync_response(client, user_message, model)

        except Exception as e:
            error_msg = f"❌ Pipeline error: {str(e)}"
            if self.valves.ENABLE_DEBUG:
                print(f"Pipeline error: {e}")
            return error_msg

    def _stream_response(self, client: RAGServerClient, user_message: str, model: str) -> Generator[str, None, None]:
        """Stream response from RAG server with proper formatting"""
        try:
            # Use requests for streaming to avoid event loop issues
            import requests
            
            payload = {"query": user_message}
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

            # Collect full response for DeepEval monitoring
            full_response = ""
            for chunk in response.iter_content(chunk_size=1024, decode_unicode=True):
                if chunk:
                    full_response += chunk
                    yield chunk
            
            # Evaluate with RAGAS after streaming is complete
            # Note: We don't have context in streaming mode, so pass empty string
            if full_response.strip():
                self._evaluate_with_ragas(user_message, full_response.strip(), "")
            
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

    def _sync_response(self, client: RAGServerClient, user_message: str, model: str) -> str:
        try:
            result = client.query_sync(user_message, model)
            answer = result.get("answer", "❌ No answer received from server")
            context = result.get("context", "")
            
            # Evaluate with RAGAS before adding debug info
            self._evaluate_with_ragas(user_message, answer, context)
            
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
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get pipeline information for debugging"""
        return {
            "name": self.name,
            "version": "1.3.0",
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
            "mode": "server_client",
            "version": "1.3.0",
            "valves": self.valves.model_dump(),
            "server_config": client_info,
            "server_health": server_health,
            "client_initialized": bool(self.rag_client),
            "ragas_status": {
                "service_available": RAGAS_SERVICE_AVAILABLE,
                "enabled": self.valves.ENABLE_RAGAS,
                "service_healthy": self.ragas_service_available,
                "service_url": self.valves.RAGAS_SERVICE_URL,
            }
        }
