#!/usr/bin/env python3
"""
Example usage of GovGPT Contextual RAG Pipeline

This script demonstrates how to use the pipeline programmatically
for testing and development purposes.
"""

import os
import asyncio
import sys

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from govgpt_contextual_rag_pipeline import Pipeline


async def example_single_query():
    """Example: Single query processing"""
    print("📝 Example 1: Single Query Processing")
    print("-" * 40)
    
    # Initialize pipeline
    pipeline = Pipeline()
    
    # Configure for testing (adjust these values)
    pipeline.valves.ENABLE_DEBUG = True
    pipeline.valves.ENABLE_STREAMING = False
    pipeline.valves.TOP_N = 5  # Use fewer chunks for faster testing
    
    # Example query
    query = "What are the information assurance requirements for government systems?"
    
    # Create request body (simulating OpenWebUI format)
    body = {
        "messages": [
            {"role": "user", "content": query}
        ],
        "model": pipeline.valves.RAG_MODEL
    }
    
    try:
        print(f"🔍 Query: {query}")
        print("⏳ Processing...")
        
        response = await pipeline.pipe(body)
        
        print("\n✅ Response:")
        print("-" * 20)
        print(response)
        
    except Exception as e:
        print(f"❌ Error: {e}")


async def example_streaming_query():
    """Example: Streaming query processing"""
    print("\n📡 Example 2: Streaming Query Processing")
    print("-" * 40)
    
    # Initialize pipeline
    pipeline = Pipeline()
    
    # Configure for streaming
    pipeline.valves.ENABLE_STREAMING = True
    pipeline.valves.ENABLE_DEBUG = True
    
    # Example query
    query = "Explain the procurement process requirements for UAE government entities."
    
    # Create request body
    body = {
        "messages": [
            {"role": "user", "content": query}
        ],
        "model": pipeline.valves.RAG_MODEL
    }
    
    try:
        print(f"🔍 Query: {query}")
        print("📡 Streaming response:")
        print("-" * 20)
        
        response = await pipeline.pipe(body)
        
        # Process streaming response
        if hasattr(response, '__aiter__'):
            async for chunk in response:
                print(chunk, end='', flush=True)
            print("\n")  # New line after streaming
        else:
            print("⚠️  Expected streaming response but got:", type(response))
            
    except Exception as e:
        print(f"❌ Error: {e}")


async def example_multiple_queries():
    """Example: Multiple queries with different configurations"""
    print("\n🔄 Example 3: Multiple Queries with Different Configurations")
    print("-" * 40)
    
    # Initialize pipeline
    pipeline = Pipeline()
    pipeline.valves.ENABLE_STREAMING = False
    pipeline.valves.ENABLE_DEBUG = False  # Less verbose for multiple queries
    
    # Test queries
    queries = [
        "What is data classification in government systems?",
        "How should security incidents be handled?", 
        "What are the backup and recovery requirements?",
        "Explain the access control standards.",
    ]
    
    print("Testing multiple queries...")
    
    for i, query in enumerate(queries, 1):
        try:
            print(f"\n📝 Query {i}: {query}")
            
            body = {
                "messages": [{"role": "user", "content": query}],
                "model": pipeline.valves.RAG_MODEL
            }
            
            response = await pipeline.pipe(body)
            
            # Show first 150 characters of response
            if isinstance(response, str):
                preview = response[:150] + "..." if len(response) > 150 else response
                print(f"✅ Response: {preview}")
            else:
                print(f"⚠️  Unexpected response type: {type(response)}")
                
        except Exception as e:
            print(f"❌ Query {i} failed: {e}")


async def example_configuration_testing():
    """Example: Testing different configurations"""
    print("\n⚙️  Example 4: Configuration Testing")
    print("-" * 40)
    
    # Initialize pipeline
    pipeline = Pipeline()
    
    # Test different TOP_N values
    test_configs = [
        {"TOP_N": 3, "description": "Minimal context"},
        {"TOP_N": 10, "description": "Moderate context"},
        {"TOP_N": 20, "description": "Rich context"},
    ]
    
    query = "What are the key security requirements?"
    
    for config in test_configs:
        try:
            print(f"\n🔧 Testing: {config['description']} (TOP_N={config['TOP_N']})")
            
            # Update configuration
            pipeline.valves.TOP_N = config["TOP_N"]
            pipeline.valves.ENABLE_DEBUG = False
            pipeline.valves.ENABLE_STREAMING = False
            
            body = {
                "messages": [{"role": "user", "content": query}],
                "model": pipeline.valves.RAG_MODEL
            }
            
            response = await pipeline.pipe(body)
            
            if isinstance(response, str):
                word_count = len(response.split())
                print(f"✅ Response generated: {word_count} words")
            else:
                print(f"⚠️  Unexpected response type: {type(response)}")
                
        except Exception as e:
            print(f"❌ Configuration test failed: {e}")


async def main():
    """Run all examples"""
    print("🎯 GovGPT Pipeline Usage Examples")
    print("=" * 50)
    
    # Check environment
    print("🔍 Checking environment...")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not set in environment")
        print("⚠️  Set this variable for full functionality testing")
        print("⚠️  Examples will run but may fail at response generation")
    
    # Run examples
    examples = [
        example_single_query,
        example_streaming_query,
        example_multiple_queries,
        example_configuration_testing,
    ]
    
    for example_func in examples:
        try:
            await example_func()
        except Exception as e:
            print(f"❌ Example failed: {e}")
        
        print("\n" + "="*50)
    
    print("✨ Examples completed!")
    print("\n💡 Tips:")
    print("- Set OPENAI_API_KEY environment variable for full functionality")
    print("- Adjust valves configuration based on your needs")
    print("- Enable debug mode to see retrieval statistics")
    print("- Use streaming for real-time user experience")


if __name__ == "__main__":
    asyncio.run(main())