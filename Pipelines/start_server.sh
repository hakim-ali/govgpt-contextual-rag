#!/bin/bash
# Startup script for RAG Server with PGVector initialization

echo "🚀 Starting RAG Server with PGVector Knowledge Base..."

# Initialize Knowledge Base (if needed)
echo "📊 Initializing Knowledge Base..."
python init_kb.py

# Check if initialization was successful
if [ $? -eq 0 ]; then
    echo "✅ Knowledge Base initialization completed"
    
    # Start the RAG server
    echo "🚀 Starting RAG Server..."
    python run_rag_server.py
else
    echo "❌ Knowledge Base initialization failed"
    exit 1
fi