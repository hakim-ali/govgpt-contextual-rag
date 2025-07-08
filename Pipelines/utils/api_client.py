"""
HTTP API Client for RAG Server Communication

This module provides a client interface to communicate with the hosted
RAG server (fastapi_rag_server_openwebui.py) via HTTP requests.
"""

import requests
import asyncio
import aiohttp
import json
from typing import Dict, Any, Optional, AsyncGenerator
from tenacity import retry, wait_exponential, stop_after_attempt


class RAGServerClient:
    """
    HTTP client for communicating with the RAG server
    
    Supports both regular and streaming requests to the hosted RAG server.
    """
    
    def __init__(
        self, 
        server_url: str,
        api_key: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3
    ):
        """
        Initialize the RAG server client
        
        Args:
            server_url: Base URL of the RAG server (e.g., "http://localhost:8100")
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.server_url = server_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Setup headers
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check if the RAG server is healthy and ready
        
        Returns:
            Dict containing health status information
        """
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
        """
        Synchronous query to RAG server using /query endpoint
        
        Args:
            query: User query text
            model: Optional model name to use
            
        Returns:
            Dict containing response from RAG server
        """
        payload = {
            "query": query
        }
        
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
    
    async def query_async(self, query: str, model: Optional[str] = None) -> Dict[str, Any]:
        """
        Asynchronous query to RAG server using /query endpoint
        
        Args:
            query: User query text
            model: Optional model name to use
            
        Returns:
            Dict containing response from RAG server
        """
        payload = {
            "query": query
        }
        
        if model:
            payload["model"] = model
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.post(
                    f"{self.server_url}/query",
                    headers=self.headers,
                    json=payload
                ) as response:
                    response.raise_for_status()
                    return await response.json()
                    
        except aiohttp.ClientError as e:
            raise Exception(f"RAG server async request failed: {e}")
    
    async def query_streaming(self, query: str, model: Optional[str] = None) -> AsyncGenerator[str, None]:
        """
        Streaming query to RAG server using /retrieve endpoint
        
        Args:
            query: User query text
            model: Optional model name to use
            
        Yields:
            Response chunks as they arrive from the server
        """
        payload = {
            "query": query
        }
        
        # Note: Based on your server code, /retrieve endpoint expects different format
        # Adjusting to match your Query model
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.post(
                    f"{self.server_url}/retrieve",
                    headers=self.headers,
                    json=payload
                ) as response:
                    response.raise_for_status()
                    
                    # Stream the response
                    async for chunk in response.content.iter_chunked(1024):
                        if chunk:
                            # Decode and yield the chunk
                            try:
                                chunk_text = chunk.decode('utf-8')
                                if chunk_text.strip():
                                    yield chunk_text
                            except UnicodeDecodeError:
                                # Skip malformed chunks
                                continue
                                
        except aiohttp.ClientError as e:
            raise Exception(f"RAG server streaming request failed: {e}")
    
    async def test_connection(self) -> bool:
        """
        Test if connection to RAG server is working
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            health = self.health_check()
            return health["status"] == "healthy"
        except Exception:
            return False
    
    def get_server_info(self) -> Dict[str, Any]:
        """
        Get information about the RAG server configuration
        
        Returns:
            Dict with server configuration info
        """
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


class RAGServerError(Exception):
    """Custom exception for RAG server communication errors"""
    
    def __init__(self, message: str, status_code: Optional[int] = None, server_response: Optional[Dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.server_response = server_response


# Utility functions for common operations

async def test_rag_server_connection(server_url: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Test connection to RAG server and return status
    
    Args:
        server_url: RAG server URL
        api_key: Optional API key
        
    Returns:
        Dict with connection test results
    """
    client = RAGServerClient(server_url, api_key)
    
    try:
        health = client.health_check()
        
        if health["status"] == "healthy":
            return {
                "success": True,
                "message": "RAG server is healthy and reachable",
                "server_info": health["server_response"],
                "server_url": server_url
            }
        else:
            return {
                "success": False,
                "message": f"RAG server health check failed: {health.get('error', 'Unknown error')}",
                "server_url": server_url
            }
            
    except Exception as e:
        return {
            "success": False,
            "message": f"Failed to connect to RAG server: {e}",
            "server_url": server_url
        }


async def quick_rag_query(server_url: str, query: str, api_key: Optional[str] = None) -> str:
    """
    Quick utility function to make a single RAG query
    
    Args:
        server_url: RAG server URL
        query: Query text
        api_key: Optional API key
        
    Returns:
        Answer text from RAG server
    """
    client = RAGServerClient(server_url, api_key)
    
    try:
        result = await client.query_async(query)
        return result.get("answer", "No answer received from server")
    except Exception as e:
        raise RAGServerError(f"Query failed: {e}")