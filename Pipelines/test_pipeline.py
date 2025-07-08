#!/usr/bin/env python3
"""
Test script for GovGPT Contextual RAG Pipeline (Server Client Mode)

This script tests the pipeline functionality by communicating with
your hosted RAG server via HTTP requests.
"""

import os
import sys
import asyncio
from typing import Dict, Any

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from govgpt_contextual_rag_pipeline import Pipeline
from utils.api_client import test_rag_server_connection, quick_rag_query


async def test_rag_server_connection_direct():
    """Test direct connection to RAG server"""
    print("🌐 Testing RAG Server Connection")
    print("=" * 40)
    
    # Default server URL (you can change this)
    server_url = "http://localhost:8100"
    
    print(f"📡 Testing connection to: {server_url}")
    
    try:
        result = await test_rag_server_connection(server_url)
        
        if result["success"]:
            print("✅ RAG server connection successful")
            print(f"📊 Server info: {result.get('server_info', {})}")
            return True
        else:
            print("❌ RAG server connection failed")
            print(f"❌ Error: {result['message']}")
            return False
            
    except Exception as e:
        print(f"❌ Connection test error: {e}")
        return False


async def test_pipeline_initialization():
    """Test pipeline initialization and server connectivity"""
    print("\n🔧 Testing Pipeline Initialization")
    print("=" * 40)
    
    # Initialize pipeline
    pipeline = Pipeline()
    
    print(f"📋 Pipeline ID: {pipeline.id}")
    print(f"📋 Pipeline Name: {pipeline.name}")
    print(f"📋 Server URL: {pipeline.valves.RAG_SERVER_URL}")
    
    # Enable debug for testing
    pipeline.valves.ENABLE_DEBUG = True
    pipeline.valves.AUTO_TEST_CONNECTION = True
    
    # Test server connection
    connection_ok = pipeline._test_server_connection()
    
    if connection_ok:
        print("✅ Pipeline initialization successful")
        print(f"📊 Server ready: {pipeline.server_ready}")
        
        # Get pipeline status
        status = pipeline.get_status()
        print(f"📊 Pipeline mode: {status.get('mode', 'unknown')}")
        
        return True
    else:
        print("❌ Pipeline initialization failed")
        print("💡 Make sure your RAG server is running at http://localhost:8100")
        return False


async def test_pipeline_query_non_streaming():
    """Test non-streaming query processing"""
    print("\n📝 Testing Non-Streaming Query")
    print("=" * 40)
    
    pipeline = Pipeline()
    
    # Configure for non-streaming
    pipeline.valves.ENABLE_STREAMING = False
    pipeline.valves.ENABLE_DEBUG = True
    
    # Test query
    test_body = {
        "messages": [
            {"role": "user", "content": "What are the key information assurance requirements for UAE government systems?"}
        ],
        "model": pipeline.valves.RAG_MODEL
    }
    
    try:
        print("📤 Sending non-streaming query...")
        response = await pipeline.pipe(test_body)
        
        if isinstance(response, str) and not response.startswith("❌"):
            print("✅ Non-streaming query successful")
            print(f"📝 Response length: {len(response)} characters")
            print(f"📝 Response preview: {response[:200]}...")
            return True
        else:
            print("❌ Non-streaming query failed")
            print(f"❌ Response: {response}")
            return False
            
    except Exception as e:
        print(f"❌ Non-streaming query error: {e}")
        return False


async def test_pipeline_query_streaming():
    """Test streaming query processing"""
    print("\n📡 Testing Streaming Query")
    print("=" * 40)
    
    pipeline = Pipeline()
    
    # Configure for streaming
    pipeline.valves.ENABLE_STREAMING = True
    pipeline.valves.ENABLE_DEBUG = True
    
    # Test query
    test_body = {
        "messages": [
            {"role": "user", "content": "Explain the procurement process requirements for UAE government entities."}
        ],
        "model": pipeline.valves.RAG_MODEL
    }
    
    try:
        print("📡 Starting streaming query...")
        response = await pipeline.pipe(test_body)
        
        if hasattr(response, '__aiter__'):
            print("✅ Streaming response generator created")
            
            # Collect response chunks
            chunks_received = 0
            total_content = ""
            
            async for chunk in response:
                chunks_received += 1
                total_content += chunk
                
                if chunks_received <= 5:  # Show first few chunks
                    print(f"📝 Chunk {chunks_received}: {chunk[:50]}...")
                
                if chunks_received >= 20:  # Limit for testing
                    break
            
            print(f"✅ Received {chunks_received} chunks")
            print(f"📝 Total content length: {len(total_content)} characters")
            return True
        else:
            print("❌ Expected streaming response but got:", type(response))
            if isinstance(response, str):
                print(f"❌ Response: {response}")
            return False
            
    except Exception as e:
        print(f"❌ Streaming query error: {e}")
        return False


async def test_direct_api_client():
    """Test the API client directly"""
    print("\n🔧 Testing Direct API Client")
    print("=" * 40)
    
    server_url = "http://localhost:8100"
    
    try:
        print("📤 Testing quick query function...")
        answer = await quick_rag_query(
            server_url, 
            "What is data classification?",
            api_key=None
        )
        
        print("✅ Direct API client query successful")
        print(f"📝 Answer: {answer[:200]}...")
        return True
        
    except Exception as e:
        print(f"❌ Direct API client error: {e}")
        return False


async def test_error_handling():
    """Test error handling with incorrect server URL"""
    print("\n⚠️  Testing Error Handling")
    print("=" * 40)
    
    pipeline = Pipeline()
    
    # Use invalid server URL
    pipeline.valves.RAG_SERVER_URL = "http://localhost:9999"  # Wrong port
    pipeline.valves.ENABLE_DEBUG = True
    pipeline.valves.ENABLE_STREAMING = False
    pipeline.valves.RAG_SERVER_TIMEOUT = 5  # Short timeout for testing
    
    test_body = {
        "messages": [
            {"role": "user", "content": "Test query"}
        ]
    }
    
    try:
        print("📤 Testing with invalid server URL...")
        response = await pipeline.pipe(test_body)
        
        if isinstance(response, str) and "not available" in response:
            print("✅ Error handling works correctly")
            print(f"📝 Error message: {response[:100]}...")
            return True
        else:
            print("❌ Expected error message but got:", response)
            return False
            
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


async def main():
    """Run all tests"""
    print("🚀 Starting GovGPT Pipeline Tests (Server Client Mode)")
    print("=" * 60)
    
    # Check prerequisites
    print("📋 Prerequisites Check:")
    print("- Ensure your RAG server is running:")
    print("  docker-compose up -d  # OR python fastapi_rag_server_openwebui.py")
    print("- Server should be accessible at http://localhost:8100")
    print("- Test with: curl http://localhost:8100/health")
    print()
    
    tests = [
        ("RAG Server Connection", test_rag_server_connection_direct),
        ("Pipeline Initialization", test_pipeline_initialization),
        ("Non-Streaming Query", test_pipeline_query_non_streaming),
        ("Streaming Query", test_pipeline_query_streaming),
        ("Direct API Client", test_direct_api_client),
        ("Error Handling", test_error_handling),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = await test_func()
            if success:
                print(f"✅ {test_name}: PASSED")
                passed += 1
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print("\n" + "="*60)
    print(f"🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Pipeline is ready for OpenWebUI integration.")
        print("\n📝 Next Steps:")
        print("1. Upload this Pipeline folder to your Git repository")
        print("2. Add the Pipeline to OpenWebUI using the Git URL")
        print("3. Configure the RAG_SERVER_URL valve in OpenWebUI")
        print("4. Start chatting with your RAG system!")
        return 0
    else:
        print("⚠️  Some tests failed. Please check:")
        print("1. RAG server is running and accessible")
        print("2. Server URL configuration is correct")
        print("3. Network connectivity between Pipeline and server")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())