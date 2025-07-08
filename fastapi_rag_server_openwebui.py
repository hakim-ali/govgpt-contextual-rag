# import time
# time.sleep(60*60)  # Delay to ensure all imports are loaded correctly
# ---------------------------------------------------------------------
import os
import json
import pickle
import numpy as np
import faiss
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from tenacity import retry, wait_exponential, stop_after_attempt
import litellm
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any
from contextlib import asynccontextmanager
from together import Together
import openai
load_dotenv()

client = Together()
llm_client = openai.OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),             # pass litellm proxy key, if you're using virtual keys
    base_url=os.getenv("OPENAI_API_BASE") # litellm-proxy-base url
)

# CONFIG
RAG_MODEL       = os.getenv("RAG_MODEL", "claude-3-haiku-20240307")
print(f"Using RAG model: {RAG_MODEL}")
ARTIFACT_DIR    = os.getenv("ARTIFACT_DIR", "./artifacts")
VECTOR_K        = int(os.getenv("VECTOR_K", 50))
BM25_K          = int(os.getenv("BM25_K", 50))
RRF_K           = int(os.getenv("RRF_K", 60))
TOP_K           = int(os.getenv("TOP_K", 20))
TOP_N          = int(os.getenv("TOP_N", 20))
EMBED_MODEL     = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

# Model-specific artifact naming
MODEL_SUFFIX = EMBED_MODEL.replace("/", "_").replace("-", "_")
print(f"Using embedding model: {EMBED_MODEL} (suffix: {MODEL_SUFFIX})")

# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 Starting Contextual RAG API...")
    load_artifacts()
    print("✅ API ready!")
    yield
    # Shutdown (if needed)
    print("🛑 Shutting down API...")

# FastAPI app
app = FastAPI(
    title="Contextual RAG API", 
    description="UAE Information Assurance RAG System",
    lifespan=lifespan
)

# Request/Response models
class QueryRequest(BaseModel):
    query: str
    model: str = None  # Optional model parameter

class Query(BaseModel):
    query: str

class RAGResponse(BaseModel):
    question: str
    answer: str
    # original_chunks: List[str]
    context: str
    # metadata: Dict[str, Any]

# Global variables for loaded artifacts
chunks_data = None
bm25_index = None
faiss_index = None

# Retry wrapper for chat API
@retry(wait=wait_exponential(multiplier=1, min=2, max=1024), stop=stop_after_attempt(10))
def call_chat_api(prompt: str, model: str = None) -> str:
    selected_model = model or RAG_MODEL
    response = llm_client.chat.completions.create(
                    model=selected_model,
                    messages = [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.7
                )
    
    # response = litellm.completion(
    #     model=RAG_MODEL,
    #     messages=[{"role": "user", "content": prompt}],
    #     temperature=0.7
    # )
    return response.choices[0].message.content.strip()

# Streaming chat API for OpenWebUI
@retry(wait=wait_exponential(multiplier=1, min=2, max=1024), stop=stop_after_attempt(10))
def call_chat_api_streaming(prompt: str, model: str = None):
    selected_model = model or RAG_MODEL
    response = llm_client.chat.completions.create(
                    model=selected_model,
                    messages = [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.7,
                    stream=True
                )
    
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content

# RRF fusion
def rrf(lists: List[List[int]], k: int = RRF_K) -> List[int]:
    scores = {}
    for lst in lists:
        for rank, idx in enumerate(lst):
            scores[idx] = scores.get(idx, 0.0) + 1.0/(k + rank + 1)
    return sorted(scores, key=lambda x: scores[x], reverse=True)

def rerank_chunks(query: str, contextual_chunks: List[str], chunk_indices: List[int], top_n: int = 3) -> List[str]:
    """Rerank chunks using a reranking model (if available)"""
    chunks_to_rerank = [contextual_chunks[i] for i in chunk_indices]
    retrieved_index = list(range(len(chunks_to_rerank)))

    # Disable reranking due to Together.AI credit limits
    # if client:
    #     # Implement reranking logic here if you have a reranker
    #     response = client.rerank.create(
    #     model="Salesforce/Llama-Rank-V1",
    #     query=query,
    #     documents=chunks_to_rerank,
    #     top_n=top_n 
    #     )

    #     for result in response.results:
    #         # retreived_chunks += hybrid_top_k_docs[result.index] + '\n\n'
    #         retreived_chunks.append(chunks_to_rerank[result.index])
    #         retrieved_index.append(result.index)
    #     return retreived_chunks, retrieved_index
    # else:
    #     # Simple fallback - return top_n chunks
    #     return chunks_to_rerank[:top_n], retrieved_index
    
    # Simple fallback - return top_n chunks without reranking
    return chunks_to_rerank[:top_n], retrieved_index[:top_n]


# Load precomputed artifacts
def load_artifacts():
    global chunks_data, bm25_index, faiss_index
    
    chunks_path = os.path.join(ARTIFACT_DIR, f'enriched_chunks.json')
    bm25_path = os.path.join(ARTIFACT_DIR, f'bm25.pkl')
    faiss_path = os.path.join(ARTIFACT_DIR, f'faiss_{MODEL_SUFFIX}.idx')
    
    try:
        # Load contextual chunks
        with open(chunks_path) as fp:
            chunks = json.load(fp)
            chunks_data = [chunk_info["original_text"] for chunk_info in chunks['chunk_metadata']]
    
        # Load BM25 index
        with open(bm25_path, 'rb') as f:
            bm25_index = pickle.load(f)
        
        # Load FAISS index  
        faiss_index = faiss.read_index(faiss_path)
        
        print(f"✅ Loaded {len(chunks_data)} contextual chunks for model {EMBED_MODEL}")
        print(f"✅ Loaded BM25 index with {len(chunks_data)} documents")
        print(f"✅ Loaded FAISS index with dimension {faiss_index.d}")
        
        return chunks_data, bm25_index, faiss_index
        
    except FileNotFoundError as e:
        print(f"❌ Model-specific artifacts not found for {EMBED_MODEL}")
        print(f"❌ Missing file: {e.filename}")
        print(f"💡 Please run preprocessing with EMBED_MODEL={EMBED_MODEL}")
        
        # Try fallback to old format
        try:
            print("🔄 Attempting to load legacy artifacts...")
            with open(os.path.join(ARTIFACT_DIR, 'enriched_chunks.json')) as f:
                data = json.load(f)
                chunks_data = [chunk_info["original_text"] for chunk_info in data['chunk_metadata']]
            
            with open(os.path.join(ARTIFACT_DIR, 'bm25.pkl'), 'rb') as f:
                bm25_index = pickle.load(f)
            
            faiss_index = faiss.read_index(os.path.join(ARTIFACT_DIR, 'faiss.idx'))
            
            print(f"⚠️  Loaded {len(chunks_data)} chunks (legacy format - embeddings may not match current model)")
            return chunks_data, bm25_index, faiss_index
            
        except Exception as e2:
            raise HTTPException(status_code=500, detail=f"Failed to load artifacts: {e2}")
            
    except Exception as e:
        print(f"❌ Error loading artifacts: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load artifacts: {e}")

# Main answer function
def answer_query(query: str, model: str = None) -> Dict[str, Any]:
    if not all([chunks_data, bm25_index, faiss_index]):
        raise HTTPException(status_code=500, detail="Artifacts not loaded")
    
    # Handle both old and new format
    if isinstance(chunks_data, list) and isinstance(chunks_data[0], str):
        # Old format - list of strings
        chunks = chunks_data
    else:
        # New format - list of enriched text
        chunks = chunks_data
    
    # 1) BM25 retrieval
    bm25_scores = bm25_index.get_scores(query.split())
    bm25_ids = list(np.argsort(bm25_scores)[::-1][:BM25_K])
    
    # 2) Dense retrieval via embedding
    try:
        # resp = litellm.embedding(
        #     model=EMBED_MODEL,
        #     input=[query]
        # )
        texts = [query] 
        embs = [] 
        
        if EMBED_MODEL == "azure_ai/embed-v-4-0":
            # Batch requests if using azure_ai/embed-v-4-0, else single call
            batch_size = 96
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                response = llm_client.embeddings.create(
                    model=EMBED_MODEL, input=batch_texts, encoding_format="float"
                )
                embs.extend(item.embedding for item in response.data)   
        elif EMBED_MODEL == "huggingface/Qwen/Qwen3-Embedding-8B":
            batch_size = 32
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                response = llm_client.embeddings.create(
                    model=EMBED_MODEL,
                    input=batch_texts,
                    encoding_format="float"
                )
                embs.extend(item.embedding for item in response.data)
        else:
            response = llm_client.embeddings.create(
                model=EMBED_MODEL, input=texts, encoding_format="float"
            )
            embs.extend(item.embedding for item in response.data)

        # q_emb = np.array([item['embedding'] for item in resp['data']], dtype=np.float32).reshape(1, -1)
        q_emb = np.array(embs, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(q_emb)
        _, dense_ids = faiss_index.search(q_emb, VECTOR_K)
        dense_ids = dense_ids[0].tolist()
    except Exception as e:
        print(f"Warning: Vector search failed: {e}")
        dense_ids = bm25_ids[:VECTOR_K]
    
    # 3) Reciprocal Rank Fusion
    fused = rrf([dense_ids, bm25_ids])[:TOP_K]
    print(f"🔀 RRF combined results: {len(fused)} final chunks")

    final_chunks_text, fused = rerank_chunks(query, chunks, fused, TOP_N)
    print(f"🔄 Reranked to {len(final_chunks_text)} chunks after reranking")

    # 4) Construct context
    # context = "\n\n".join(f"[{i}] {chunks[i]}" for i in fused)
    context = "\n\n".join(f"{final_chunks_text[i]}" for i in range(len(final_chunks_text)))
    
    
    # context = "\n\n".join(context_chunks)
    prompt = f"Use the following context to answer:\n\n{context}\n\nQuestion: {query}\nAnswer:"
    prompt = f"""
    You are an expert assistant providing formal, accurate, and context-based answers. 

    Use only the information from the context below to respond to the question. 
    Do not reference or cite documents. Do not include assumptions or external knowledge.
    If the answer is not directly available in the context, state that based on the current information.


    ### Context:
    {context}

    ### Question:
    {query}

    ### Answer:
    """

    # prompt = """You are an intelligent assistant that adapts your responses based on the nature of the question. 

    #             INSTRUCTIONS:
    #             1. Analyze the question type first
    #             2. For context-dependent questions: Use only provided context, never external knowledge.
    #             3. For general questions: Respond naturally with your general capabilities

    #             EXAMPLES:

    #             Context-dependent question example:
    #             Human: "What are the exemptions in this policy?"
    #             Assistant: [Searches context for exemption information, provides specific details if found, or states what information is actually available]

    #             General question example:
    #             Human: "How are you?"
    #             Assistant: "I'm doing well, thank you for asking! I'm here to help you with questions and tasks. I can assist with information from documents you provide, answer general questions, help with analysis, and much more. How can I help you today?"

    #             Mixed question example:
    #             Human: "Can you explain what this document is about and also tell me what you're capable of?"
    #             Assistant: [First addresses the document content using context, then explains general capabilities]
                 
    #             Available Context:
    #             {context}

    #             Question: 
    #             {query}

    #             Response:
    #         """
    

    # 5) Generate answer
    try:
        answer = call_chat_api(prompt, model)
        
        return {
            "question": query,
            "answer": answer,
            # "original_chunks": original_chunks,
            "context": context,
            # "metadata": {
            #     "bm25_candidates": len(bm25_ids),
            #     "vector_candidates": len(dense_ids),
            #     "final_chunks": len(fused),
            #     "chunk_indices": fused
            # }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {e}")

# Streaming answer function for OpenWebUI
def answer_query_streaming(query: str, model: str = None):
    if not all([chunks_data, bm25_index, faiss_index]):
        raise HTTPException(status_code=500, detail="Artifacts not loaded")
    
    # Handle both old and new format
    if isinstance(chunks_data, list) and isinstance(chunks_data[0], str):
        # Old format - list of strings
        chunks = chunks_data
    else:
        # New format - list of enriched text
        chunks = chunks_data
    
    # 1) BM25 retrieval
    bm25_scores = bm25_index.get_scores(query.split())
    bm25_ids = list(np.argsort(bm25_scores)[::-1][:BM25_K])
    
    # 2) Dense retrieval via embedding
    try:
        texts = [query] 
        embs = [] 
        
        if EMBED_MODEL == "azure_ai/embed-v-4-0":
            # Batch requests if using azure_ai/embed-v-4-0, else single call
            batch_size = 96
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                response = llm_client.embeddings.create(
                    model=EMBED_MODEL, input=batch_texts, encoding_format="float"
                )
                embs.extend(item.embedding for item in response.data)   
        elif EMBED_MODEL == "huggingface/Qwen/Qwen3-Embedding-8B":
            batch_size = 32
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                response = llm_client.embeddings.create(
                    model=EMBED_MODEL,
                    input=batch_texts,
                    encoding_format="float"
                )
                embs.extend(item.embedding for item in response.data)
        else:
            response = llm_client.embeddings.create(
                model=EMBED_MODEL, input=texts, encoding_format="float"
            )
            embs.extend(item.embedding for item in response.data)

        q_emb = np.array(embs, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(q_emb)
        _, dense_ids = faiss_index.search(q_emb, VECTOR_K)
        dense_ids = dense_ids[0].tolist()
    except Exception as e:
        print(f"Warning: Vector search failed: {e}")
        dense_ids = bm25_ids[:VECTOR_K]
    
    # 3) Reciprocal Rank Fusion
    fused = rrf([dense_ids, bm25_ids])[:TOP_K]
    final_chunks_text, fused = rerank_chunks(query, chunks, fused, TOP_N)

    # 4) Construct context
    context = "\n\n".join(f"{final_chunks_text[i]}" for i in range(len(final_chunks_text)))
    
    prompt = f"""
    You are an expert assistant providing formal, accurate, and context-based answers. 

    Use only the information from the context below to respond to the question. 
    Do not reference or cite documents. Do not include assumptions or external knowledge.
    If the answer is not directly available in the context, state that based on the current information.

    ### Context:
    {context}

    ### Question:
    {query}

    ### Answer:
    """

    # 5) Generate streaming answer
    try:
        for chunk in call_chat_api_streaming(prompt, model):
            yield chunk
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {e}")

# API Endpoints

@app.get("/")
async def root():
    return {"message": "UAE Information Assurance Contextual RAG API", "status": "ready"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "artifacts_loaded": all([chunks_data, bm25_index, faiss_index]),
        "chunk_count": len(chunks_data) if chunks_data else 0
    }

@app.post("/query", response_model=RAGResponse)
async def query_rag(request: QueryRequest):
    """
    Query the RAG system with a question about UAE Information Assurance Standards
    
    Returns:
    - question: The original query
    - answer: Generated answer based on context
    - original_chunks: List of original document chunks used for context (without numbering)
    - context: List of numbered context chunks used for generation
    - metadata: Retrieval statistics and chunk indices
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        result = answer_query(request.query, request.model)
        return RAGResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# OpenWebUI compatible streaming endpoint
@app.post("/retrieve")
async def retrieve_streaming(q: Query):
    """
    OpenWebUI compatible streaming endpoint for RAG retrieval
    
    Returns streaming text response for OpenWebUI integration
    """
    if not q.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        def stream_generator():
            for chunk in answer_query_streaming(q.query):
                yield chunk
        
        return StreamingResponse(stream_generator(), media_type="text/plain; charset=utf-8")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Interactive endpoint for testing
@app.get("/query/{query_text}")
async def query_get(query_text: str, model: str = None):
    """GET endpoint for quick testing"""
    result = answer_query(query_text, model)
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8100)