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

def detect_language(text: str, threshold: float = 0.3) -> str:
    """
    Production-ready language detection - returns only 'arabic' or 'english'.
    
    Args:
        text: Input string to analyze
        threshold: Minimum ratio of Arabic characters to classify as Arabic (default: 0.3)
        
    Returns:
        "arabic" if Arabic character ratio > threshold
        "english" for everything else (including mixed, empty, numbers-only)
    
    Logic:
    - Counts Arabic vs Latin characters in the text
    - If Arabic ratio exceeds threshold → "arabic"
    - Everything else → "english" (safe fallback for production)
    - Lower threshold (0.3) makes detection more sensitive to Arabic content
    """
    if not text or not text.strip():
        return "english"  # Default fallback for empty text
    
    # Character range checks
    def is_arabic_char(c):
        # Basic Arabic Unicode blocks (covers most common Arabic script)
        code = ord(c)
        return any([
            0x0600 <= code <= 0x06FF,    # Arabic
            0x0750 <= code <= 0x077F,    # Arabic Supplement  
            0x08A0 <= code <= 0x08FF,    # Arabic Extended-A
            0xFB50 <= code <= 0xFDFF,    # Arabic Presentation Forms-A
            0xFE70 <= code <= 0xFEFF     # Arabic Presentation Forms-B
        ])
    
    def is_latin_char(c):
        # Basic Latin, Latin-1 Supplement, Latin Extended (includes English)
        code = ord(c)
        return any([
            0x0020 <= code <= 0x007F,    # Basic Latin (ASCII)
            0x00A0 <= code <= 0x00FF,    # Latin-1 Supplement
            0x0100 <= code <= 0x017F,    # Latin Extended-A
            0x0180 <= code <= 0x024F     # Latin Extended-B
        ])
    
    # Count characters
    arabic_count = 0
    latin_count = 0
    
    for char in text:
        if is_arabic_char(char):
            arabic_count += 1
        elif is_latin_char(char):
            latin_count += 1
        # Ignore other characters (punctuation, numbers, symbols)
    
    # Calculate ratio based on relevant characters only
    total_relevant = arabic_count + latin_count
    if total_relevant == 0:
        return "english"  # No relevant characters, default to English
    
    arabic_ratio = arabic_count / total_relevant
    
    # Binary decision: Arabic if above threshold, English otherwise
    return "arabic" if arabic_ratio > threshold else "english"

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
            default="http://40.119.184.8:8100",
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
            description="Primary LLM model for English responses"
        )
        ARABIC_RAG_MODEL: str = Field(
            default="azure_ai/cohere-command-a",
            description="LLM model for Arabic responses"
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
            default="http://74.162.37.71:8300",
            description="URL of the RAGAS + Phoenix evaluation service"
        )
        RAGAS_SERVICE_TIMEOUT: int = Field(
            default=30,
            description="Timeout for RAGAS service requests in seconds"
        )
        ARABIC_RAG_SERVER_URL: str = Field(
            default="http://74.162.37.71:8200",
            description="URL of the hosted Arabic RAG server (production)"
        )
        ARABIC_RAG_SERVER_TIMEOUT: int = Field(
            default=30,
            description="Request timeout for Arabic RAG server in seconds"
        )
        ARABIC_RAG_SERVER_API_KEY: str = Field(
            default="",
            description="Optional API key for Arabic RAG server authentication"
        )
        ARABIC_RAG_SERVER_MAX_RETRIES: int = Field(
            default=3,
            description="Maximum retry attempts for Arabic RAG server requests"
        )
        LANGUAGE_DETECTION_THRESHOLD: float = Field(
            default=0.70,
            description="Threshold for Arabic language detection (0.0-1.0, lower = more sensitive)"
        )

    def __init__(self):
        self.name = "GovGPT RAG"
        self.valves = self.Valves(
            **{
                "RAG_SERVER_URL": os.getenv("RAG_SERVER_URL", "http://40.119.184.8:8100"),
                "RAG_SERVER_TIMEOUT": int(os.getenv("RAG_SERVER_TIMEOUT", "30")),
                "RAG_SERVER_API_KEY": os.getenv("RAG_SERVER_API_KEY", ""),
                "RAG_SERVER_MAX_RETRIES": int(os.getenv("RAG_SERVER_MAX_RETRIES", "3")),
                "RAG_MODEL": os.getenv("RAG_MODEL", "gpt-4.1"),
                "ARABIC_RAG_MODEL": os.getenv("ARABIC_RAG_MODEL", "azure_ai/cohere-command-a"),
                "ENABLE_STREAMING": os.getenv("ENABLE_STREAMING", "true").lower() == "true",
                "ENABLE_DEBUG": os.getenv("ENABLE_DEBUG", "false").lower() == "true",
                "STREAM_BUFFER_SIZE": int(os.getenv("STREAM_BUFFER_SIZE", "1024")),
                "STREAM_WORD_BOUNDARY": os.getenv("STREAM_WORD_BOUNDARY", "true").lower() == "true",
                "STREAM_CHUNK_SIZE": int(os.getenv("STREAM_CHUNK_SIZE", "25")),
                "ENABLE_RAGAS": os.getenv("ENABLE_RAGAS", "true").lower() == "true",
                "RAGAS_SERVICE_URL": os.getenv("RAGAS_SERVICE_URL", "http://74.162.37.71:8300"),
                "RAGAS_SERVICE_TIMEOUT": int(os.getenv("RAGAS_SERVICE_TIMEOUT", "30")),
                "ARABIC_RAG_SERVER_URL": os.getenv("ARABIC_RAG_SERVER_URL", "http://74.162.37.71:8200"),
                "ARABIC_RAG_SERVER_TIMEOUT": int(os.getenv("ARABIC_RAG_SERVER_TIMEOUT", "30")),
                "ARABIC_RAG_SERVER_API_KEY": os.getenv("ARABIC_RAG_SERVER_API_KEY", ""),
                "ARABIC_RAG_SERVER_MAX_RETRIES": int(os.getenv("ARABIC_RAG_SERVER_MAX_RETRIES", "3")),
                "LANGUAGE_DETECTION_THRESHOLD": float(os.getenv("LANGUAGE_DETECTION_THRESHOLD", "0.3")),
            }
        )
        self.rag_client_english = None
        self.rag_client_arabic = None
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
        
        # Test server connections on startup if debug enabled
        if self.valves.ENABLE_DEBUG:
            # Test English RAG server
            try:
                client = self._get_rag_client("english")
                health = client.health_check()
                if health["status"] == "healthy":
                    print(f"✅ English RAG server connection successful")
                else:
                    print(f"⚠️ English RAG server health check failed: {health.get('error', 'Unknown error')}")
            except Exception as e:
                print(f"❌ English RAG server connection failed: {e}")
            
            # Test Arabic RAG server
            try:
                client = self._get_rag_client("arabic")
                health = client.health_check()
                if health["status"] == "healthy":
                    print(f"✅ Arabic RAG server connection successful")
                else:
                    print(f"⚠️ Arabic RAG server health check failed: {health.get('error', 'Unknown error')}")
            except Exception as e:
                print(f"❌ Arabic RAG server connection failed: {e}")

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

    async def on_shutdown(self):
        """Clean up on shutdown"""
        print(f"🛑 {self.name} shutting down...")
        
        # Clean up client connections
        if self.rag_client_english:
            self.rag_client_english = None
        if self.rag_client_arabic:
            self.rag_client_arabic = None

    def _get_rag_client(self, server_type: str = "english") -> RAGServerClient:
        """Get RAG client for specified server type (english or arabic)"""
        if server_type == "arabic":
            if not self.rag_client_arabic or self._client_config_changed("arabic"):
                self.rag_client_arabic = RAGServerClient(
                    server_url=self.valves.ARABIC_RAG_SERVER_URL,
                    api_key=self.valves.ARABIC_RAG_SERVER_API_KEY if self.valves.ARABIC_RAG_SERVER_API_KEY else None,
                    timeout=self.valves.ARABIC_RAG_SERVER_TIMEOUT,
                    max_retries=self.valves.ARABIC_RAG_SERVER_MAX_RETRIES
                )
            return self.rag_client_arabic
        else:  # Default to English
            if not self.rag_client_english or self._client_config_changed("english"):
                self.rag_client_english = RAGServerClient(
                    server_url=self.valves.RAG_SERVER_URL,
                    api_key=self.valves.RAG_SERVER_API_KEY if self.valves.RAG_SERVER_API_KEY else None,
                    timeout=self.valves.RAG_SERVER_TIMEOUT,
                    max_retries=self.valves.RAG_SERVER_MAX_RETRIES
                )
            return self.rag_client_english

    def _client_config_changed(self, server_type: str = "english") -> bool:
        """Check if client configuration has changed to avoid unnecessary recreations"""
        if server_type == "arabic":
            if not self.rag_client_arabic:
                return True
            
            current_config = (
                self.valves.ARABIC_RAG_SERVER_URL,
                self.valves.ARABIC_RAG_SERVER_TIMEOUT, 
                self.valves.ARABIC_RAG_SERVER_MAX_RETRIES,
                self.valves.ARABIC_RAG_SERVER_API_KEY
            )
            
            client_config = (
                self.rag_client_arabic.server_url,
                self.rag_client_arabic.timeout,
                self.rag_client_arabic.max_retries,
                self.rag_client_arabic.api_key or ""
            )
        else:  # English
            if not self.rag_client_english:
                return True
            
            current_config = (
                self.valves.RAG_SERVER_URL,
                self.valves.RAG_SERVER_TIMEOUT, 
                self.valves.RAG_SERVER_MAX_RETRIES,
                self.valves.RAG_SERVER_API_KEY
            )
            
            client_config = (
                self.rag_client_english.server_url,
                self.rag_client_english.timeout,
                self.rag_client_english.max_retries,
                self.rag_client_english.api_key or ""
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
                print(f"   - Ground truth found: {eval_result.get('ground_truth_found', False)}")
                if eval_result.get('ground_truth_match_score'):
                    print(f"   - Match score: {eval_result['ground_truth_match_score']:.2f}")
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

            # 1. Detect query language (binary decision: arabic or english)
            detected_language = detect_language(user_message, self.valves.LANGUAGE_DETECTION_THRESHOLD)
            
            # 2. Set server type (binary routing)
            server_type = detected_language  # Will be either "arabic" or "english"
            
            # 3. Get appropriate client and model
            if server_type == "arabic":
                model = self.valves.ARABIC_RAG_MODEL
            else:
                model = self.valves.RAG_MODEL
            client = self._get_rag_client(server_type)
            
            if self.valves.ENABLE_DEBUG:
                print(f"🔍 Processing query: {user_message[:100]}...")
                print(f"🌐 Detected language: {detected_language} (threshold: {self.valves.LANGUAGE_DETECTION_THRESHOLD})")
                print(f"🎯 Routing to: {server_type} server")
                print(f"🤖 Using model: {model} ({'Arabic-optimized' if server_type == 'arabic' else 'English-optimized'})")
                print(f"📡 Streaming mode: {self.valves.ENABLE_STREAMING}")
                server_url = self.valves.ARABIC_RAG_SERVER_URL if server_type == "arabic" else self.valves.RAG_SERVER_URL
                print(f"🌐 Server URL: {server_url}")

            if self.valves.ENABLE_STREAMING:
                return self._stream_response(client, user_message, model, server_type)
            else:
                return self._sync_response(client, user_message, model, server_type)

        except Exception as e:
            error_msg = f"❌ Pipeline error: {str(e)}"
            if self.valves.ENABLE_DEBUG:
                print(f"Pipeline error: {e}")
            return error_msg

    def _stream_response(self, client: RAGServerClient, user_message: str, model: str, server_type: str = "english") -> Generator[str, None, None]:
        """Stream response from RAG server with context extraction"""
        try:
            # Use requests for streaming to avoid event loop issues
            import requests
            import re
            
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
            
            # Evaluate with RAGAS after streaming is complete with extracted context
            if full_response.strip():
                if self.valves.ENABLE_DEBUG:
                    print(f"📝 Full response: {len(full_response)} characters")
                
                self._evaluate_with_ragas(user_message, full_response.strip(), extracted_context)
            
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

    def _sync_response(self, client: RAGServerClient, user_message: str, model: str, server_type: str = "english") -> str:
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
                debug_info += f"- Server type: {server_type}\n"
                server_url = self.valves.ARABIC_RAG_SERVER_URL if server_type == "arabic" else self.valves.RAG_SERVER_URL
                debug_info += f"- Server: {server_url}\n"
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
            "version": "1.5.0",
            "english_server_url": self.valves.RAG_SERVER_URL,
            "arabic_server_url": self.valves.ARABIC_RAG_SERVER_URL,
            "streaming_enabled": self.valves.ENABLE_STREAMING,
            "debug_enabled": self.valves.ENABLE_DEBUG,
            "word_boundary_enabled": self.valves.STREAM_WORD_BOUNDARY,
            "buffer_size": self.valves.STREAM_BUFFER_SIZE,
            "chunk_size": self.valves.STREAM_CHUNK_SIZE,
            "english_timeout": self.valves.RAG_SERVER_TIMEOUT,
            "arabic_timeout": self.valves.ARABIC_RAG_SERVER_TIMEOUT,
            "english_model": self.valves.RAG_MODEL,
            "arabic_model": self.valves.ARABIC_RAG_MODEL,
            "language_detection": {
                "threshold": self.valves.LANGUAGE_DETECTION_THRESHOLD,
                "binary_mode": True,
                "supported_languages": ["arabic", "english"]
            },
            "ragas_enabled": self.valves.ENABLE_RAGAS,
            "ragas_service_available": self.ragas_service_available,
            "ragas_service_url": self.valves.RAGAS_SERVICE_URL,
        }

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive pipeline status"""
        english_client_info = None
        english_server_health = None
        arabic_client_info = None
        arabic_server_health = None
        
        if self.rag_client_english:
            english_client_info = self.rag_client_english.get_server_info()
            try:
                english_server_health = self.rag_client_english.health_check()
            except Exception as e:
                english_server_health = {"status": "error", "error": str(e)}
        
        if self.rag_client_arabic:
            arabic_client_info = self.rag_client_arabic.get_server_info()
            try:
                arabic_server_health = self.rag_client_arabic.health_check()
            except Exception as e:
                arabic_server_health = {"status": "error", "error": str(e)}

        return {
            "pipeline_name": self.name,
            "mode": "dual_server_multilingual",
            "version": "1.5.0",
            "valves": self.valves.model_dump(),
            "english_server": {
                "config": english_client_info,
                "health": english_server_health,
                "client_initialized": bool(self.rag_client_english)
            },
            "arabic_server": {
                "config": arabic_client_info,
                "health": arabic_server_health,
                "client_initialized": bool(self.rag_client_arabic)
            },
            "language_detection": {
                "threshold": self.valves.LANGUAGE_DETECTION_THRESHOLD,
                "binary_mode": True,
                "supported_languages": ["arabic", "english"]
            },
            "ragas_status": {
                "service_available": RAGAS_SERVICE_AVAILABLE,
                "enabled": self.valves.ENABLE_RAGAS,
                "service_healthy": self.ragas_service_available,
                "service_url": self.valves.RAGAS_SERVICE_URL,
            }
        }
