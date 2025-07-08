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
from typing import List, Dict, Any, Optional, AsyncGenerator
from pydantic import BaseModel, Field
from fastapi import Request

from utils.api_client import RAGServerClient, RAGServerError, test_rag_server_connection


class Pipeline:
    """
    GovGPT Contextual RAG Pipeline - Server Client Mode
    
    This pipeline acts as an HTTP client that communicates with your hosted
    RAG server to provide intelligent document retrieval and question answering
    for UAE government standards and policies.
    """
    
    class Valves(BaseModel):
        """Configuration valves for the pipeline"""
        
        # Server Configuration
        RAG_SERVER_URL: str = Field(
            default="http://localhost:8100",
            description="URL of the hosted RAG server"
        )
        RAG_SERVER_TIMEOUT: int = Field(
            default=30,
            description="Request timeout in seconds",
            ge=5, le=300
        )
        RAG_SERVER_API_KEY: str = Field(
            default="",
            description="Optional API key for RAG server authentication"
        )
        RAG_SERVER_MAX_RETRIES: int = Field(
            default=3,
            description="Maximum retry attempts for failed requests",
            ge=1, le=10
        )
        
        # Model Configuration (passed to server)
        RAG_MODEL: str = Field(
            default="gpt-4.1",
            description="Primary LLM model for response generation (sent to server)"
        )
        
        # Pipeline Configuration
        ENABLE_STREAMING: bool = Field(
            default=True,
            description="Enable streaming responses via /retrieve endpoint"
        )
        ENABLE_DEBUG: bool = Field(
            default=False,
            description="Enable debug logging and information"
        )
        ENABLE_SERVER_HEALTH_CHECK: bool = Field(
            default=True,
            description="Check server health before processing queries"
        )
        AUTO_TEST_CONNECTION: bool = Field(
            default=True,
            description="Automatically test server connection on initialization"
        )
        
        # Knowledge Scope (informational)
        KNOWLEDGE_SCOPE: str = Field(
            default="UAE Information Assurance Standards",
            description="Description of knowledge base scope (informational)"
        )
    
    def __init__(self):
        """Initialize the pipeline"""
        self.type = "pipe"  # OpenWebUI pipeline type
        self.id = "govgpt_contextual_rag_server"
        self.name = "GovGPT Contextual RAG (Server)"
        self.description = "Advanced RAG system for UAE government documents - communicates with hosted RAG server"
        
        # Initialize valves with environment variables or defaults
        self.valves = self.Valves(
            **{
                "RAG_SERVER_URL": os.getenv("RAG_SERVER_URL", "http://localhost:8100"),
                "RAG_SERVER_TIMEOUT": int(os.getenv("RAG_SERVER_TIMEOUT", 30)),
                "RAG_SERVER_API_KEY": os.getenv("RAG_SERVER_API_KEY", ""),
                "RAG_SERVER_MAX_RETRIES": int(os.getenv("RAG_SERVER_MAX_RETRIES", 3)),
                "RAG_MODEL": os.getenv("RAG_MODEL", "gpt-4.1"),
                "ENABLE_STREAMING": os.getenv("ENABLE_STREAMING", "true").lower() == "true",
                "ENABLE_DEBUG": os.getenv("ENABLE_DEBUG", "false").lower() == "true",
                "ENABLE_SERVER_HEALTH_CHECK": os.getenv("ENABLE_SERVER_HEALTH_CHECK", "true").lower() == "true",
                "AUTO_TEST_CONNECTION": os.getenv("AUTO_TEST_CONNECTION", "true").lower() == "true",
            }
        )
        
        # Initialize RAG server client
        self.rag_client = None
        self.server_ready = False
        self.last_health_check = None
        
        # Auto-test connection if enabled
        if self.valves.AUTO_TEST_CONNECTION:
            self._test_server_connection()
    
    def _get_rag_client(self) -> RAGServerClient:
        """Get or create RAG server client with current valve settings"""
        if not self.rag_client or self._client_config_changed():
            self.rag_client = RAGServerClient(
                server_url=self.valves.RAG_SERVER_URL,
                api_key=self.valves.RAG_SERVER_API_KEY if self.valves.RAG_SERVER_API_KEY else None,
                timeout=self.valves.RAG_SERVER_TIMEOUT,
                max_retries=self.valves.RAG_SERVER_MAX_RETRIES
            )
        return self.rag_client
    
    def _client_config_changed(self) -> bool:
        """Check if client configuration has changed"""
        if not self.rag_client:
            return True
        
        # Compare current valves with client config
        return (
            self.rag_client.server_url != self.valves.RAG_SERVER_URL or
            self.rag_client.timeout != self.valves.RAG_SERVER_TIMEOUT or
            self.rag_client.max_retries != self.valves.RAG_SERVER_MAX_RETRIES
        )
    
    def _test_server_connection(self) -> bool:
        """Test connection to RAG server"""
        try:
            client = self._get_rag_client()
            health = client.health_check()
            
            if health["status"] == "healthy":
                self.server_ready = True
                self.last_health_check = health
                
                if self.valves.ENABLE_DEBUG:
                    print(f"✅ RAG server connection successful: {self.valves.RAG_SERVER_URL}")
                    server_info = health.get("server_response", {})
                    if "chunk_count" in server_info:
                        print(f"📊 Server has {server_info['chunk_count']} chunks loaded")
                return True
            else:
                self.server_ready = False
                if self.valves.ENABLE_DEBUG:
                    print(f"❌ RAG server health check failed: {health.get('error', 'Unknown error')}")
                return False
                
        except Exception as e:
            self.server_ready = False
            if self.valves.ENABLE_DEBUG:
                print(f"❌ RAG server connection test failed: {e}")
            return False
    
    async def pipe(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __request__: Optional[Request] = None,
    ) -> str | AsyncGenerator[str, None]:
        """
        Main pipeline processing method
        
        Args:
            body: Request body containing messages and model info
            __user__: User information (optional)
            __request__: FastAPI request object (optional)
            
        Returns:
            Response string or async generator for streaming
        """
        
        # Check server health if enabled
        if self.valves.ENABLE_SERVER_HEALTH_CHECK and not self.server_ready:
            self._test_server_connection()
        
        if not self.server_ready:
            error_msg = f"❌ RAG server is not available at {self.valves.RAG_SERVER_URL}. Please check server status and configuration."
            if self.valves.ENABLE_STREAMING:
                async def error_stream():
                    yield error_msg
                return error_stream()
            else:
                return error_msg
        
        try:
            # Extract user query from messages
            messages = body.get("messages", [])
            if not messages:
                error_msg = "❌ No messages provided"
                if self.valves.ENABLE_STREAMING:
                    async def error_stream():
                        yield error_msg
                    return error_stream()
                else:
                    return error_msg
            
            # Get the latest user message
            user_message = ""
            for message in reversed(messages):
                if message.get("role") == "user":
                    user_message = message.get("content", "")
                    break
            
            if not user_message.strip():
                error_msg = "❌ No user query found in messages"
                if self.valves.ENABLE_STREAMING:
                    async def error_stream():
                        yield error_msg
                    return error_stream()
                else:
                    return error_msg
            
            # Get model from body or use default
            model = body.get("model", self.valves.RAG_MODEL)
            
            if self.valves.ENABLE_DEBUG:
                print(f"🔍 Processing query: {user_message[:100]}...")
                print(f"🤖 Using model: {model}")
                print(f"🌐 RAG server: {self.valves.RAG_SERVER_URL}")
                print(f"📡 Streaming mode: {self.valves.ENABLE_STREAMING}")
            
            # Get RAG client
            client = self._get_rag_client()
            
            # Process query with RAG server
            if self.valves.ENABLE_STREAMING:
                # Streaming response via /retrieve endpoint
                async def response_stream():
                    try:
                        if self.valves.ENABLE_DEBUG:
                            yield f"<!-- DEBUG: Requesting from {self.valves.RAG_SERVER_URL}/retrieve -->\n\n"
                        
                        async for chunk in client.query_streaming(user_message, model):
                            yield chunk
                            
                    except Exception as e:
                        error_msg = f"❌ Streaming request failed: {str(e)}"
                        if self.valves.ENABLE_DEBUG:
                            print(f"Streaming error: {e}")
                        yield error_msg
                
                return response_stream()
            
            else:
                # Non-streaming response via /query endpoint
                try:
                    result = await client.query_async(user_message, model)
                    
                    response = result.get("answer", "❌ No answer received from server")
                    
                    if self.valves.ENABLE_DEBUG:
                        debug_info = f"\n\n---\n🔍 Debug Info:\n"
                        debug_info += f"- Server: {self.valves.RAG_SERVER_URL}\n"
                        debug_info += f"- Model: {model}\n"
                        debug_info += f"- Response length: {len(response)} chars\n"
                        if "context" in result:
                            debug_info += f"- Context available: Yes\n"
                        response += debug_info
                    
                    return response
                    
                except Exception as e:
                    error_msg = f"❌ Query request failed: {str(e)}"
                    if self.valves.ENABLE_DEBUG:
                        print(f"Query error: {e}")
                    return error_msg
        
        except Exception as e:
            error_msg = f"❌ Pipeline error: {str(e)}"
            if self.valves.ENABLE_DEBUG:
                print(f"Pipeline error: {e}")
            
            if self.valves.ENABLE_STREAMING:
                async def error_stream():
                    yield error_msg
                return error_stream()
            else:
                return error_msg
    
    async def test_server_connection_async(self) -> Dict[str, Any]:
        """Async method to test server connection"""
        try:
            return await test_rag_server_connection(
                self.valves.RAG_SERVER_URL,
                self.valves.RAG_SERVER_API_KEY if self.valves.RAG_SERVER_API_KEY else None
            )
        except Exception as e:
            return {
                "success": False,
                "message": f"Connection test failed: {e}",
                "server_url": self.valves.RAG_SERVER_URL
            }
    
    def get_status(self) -> Dict[str, Any]:
        """Get pipeline status (for debugging/monitoring)"""
        client_info = None
        if self.rag_client:
            client_info = self.rag_client.get_server_info()
        
        return {
            "pipeline_id": self.id,
            "pipeline_name": self.name,
            "mode": "server_client",
            "server_ready": self.server_ready,
            "valves": self.valves.model_dump(),
            "server_config": client_info,
            "last_health_check": self.last_health_check
        }