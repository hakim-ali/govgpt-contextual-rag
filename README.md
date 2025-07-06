# GovGPT - Contextual RAG System

A Retrieval-Augmented Generation (RAG) system designed for government document processing and question answering. This system uses hybrid search (vector similarity + BM25) with contextual chunking for improved accuracy.

## 🚀 Features

- **Hybrid Search**: Combines vector similarity and BM25 keyword search using Reciprocal Rank Fusion (RRF)
- **Contextual Chunking**: Enhanced document chunking with contextual information for better retrieval
- **Multiple Model Support**: Configurable LLM and embedding models via LiteLLM proxy
- **FastAPI Server**: RESTful API with health checks and query endpoints
- **Docker Support**: Containerized deployment with optimized builds
- **Model Evaluation**: Comprehensive evaluation system with winner selection logic

## 📋 Prerequisites

- Python 3.11+
- Docker and Docker Compose
- Access to LiteLLM proxy (configured in environment)
- Required Python packages (see `requirements.txt`)

## 🛠️ Installation

### Local Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd GovGPT
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   # Or use uv for faster installation
   pip install uv
   uv pip install -r requirements.txt
   ```

3. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

### Docker Setup

1. **Build and run with Docker Compose**
   ```bash
   docker-compose up --build -d
   ```

2. **Or build with simple Dockerfile**
   ```bash
   docker build -f Dockerfile -t govgpt .
   docker run --env-file .env -p 8000:8000 govgpt
   ```

## ⚙️ Configuration

### Environment Variables (.env)

```bash
# API Configuration
OPENAI_API_KEY=your-api-key
OPENAI_API_BASE=https://your-litellm-proxy/v1

# Model Configuration
RAG_MODEL=azure_ai/deepseek-v3-0324
EMBED_MODEL=huggingface/Qwen/Qwen3-Embedding-8B

# RAG Parameters
VECTOR_K=50          # Number of vector search results
BM25_K=50           # Number of BM25 search results
RRF_K=60            # RRF parameter for rank fusion
TOP_K=20            # Final number of chunks to use
TOP_N=20            # Number of chunks after reranking

# Paths
ARTIFACT_DIR=./artifacts
```

### Supported Models

**RAG Models (LLM):**
- `azure_ai/deepseek-v3-0324`
- `ollama/qwen3:14b`
- `azure_ai/cohere-command-a`
- `gpt-4.1`

**Embedding Models:**
- `text-embedding-3-large`
- `huggingface/Qwen/Qwen3-Embedding-8B`
- `azure_ai/embed-v-4-0`

## 📁 Project Structure

```
GovGPT/
├── fastapi_rag_server.py      # Main FastAPI server
├── preprocess_rag.py          # Document preprocessing and indexing
├── winner.py                  # Model evaluation and winner selection
├── requirements.txt           # Python dependencies
├── docker-compose.yml         # Docker Compose configuration
├── Dockerfile                 # Main Dockerfile
├── Dockerfile.simple          # Simplified Dockerfile
├── .env                      # Environment configuration
├── artifacts/                # Processed data and indices
│   ├── enriched_chunks.json  # Contextual chunks
│   ├── bm25.pkl              # BM25 index
│   └── faiss_*.idx           # FAISS vector indices
├── documents/                # Source documents (PDFs)
├── evals/                    # Model evaluation results
└── composite_model_evaluation.ipynb  # Evaluation notebook
```

## 🔧 Usage

### 1. Document Preprocessing

First, process your documents to create searchable indices:

```bash
python preprocess_rag.py
```

This will:
- Extract text from PDFs in the `documents/` folder
- Create contextual chunks with surrounding context
- Build BM25 and FAISS vector indices
- Save artifacts to the `artifacts/` folder

### 2. Start the Server

**Local:**
```bash
python fastapi_rag_server.py
```

**Docker:**
```bash
docker-compose up -d
```

The server will start on `http://localhost:8000`

### 3. API Usage

**Health Check:**
```bash
curl http://localhost:8000/health
```

**Query (POST):**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the main procurement policies?"}'
```

**Query (GET):**
```bash
curl "http://localhost:8000/query/What%20are%20the%20main%20procurement%20policies"
```

**Response Format:**
```json
{
  "question": "What are the main procurement policies?",
  "answer": "Based on the provided context...",
  "context": "Retrieved document chunks..."
}
```

## 📊 Model Evaluation

### Running Evaluations

1. **Single Model Evaluation:**
   ```bash
   python evaluate_rag_enhanced.py
   ```

2. **Composite Evaluation:**
   Open and run `composite_model_evaluation.ipynb` to:
   - Compare multiple model configurations
   - Generate winner/runner-up analysis
   - Create consolidated evaluation reports

### Evaluation Metrics

The system evaluates models on:
- **Answer Relevancy**: How relevant the answer is to the question
- **Faithfulness**: How faithful the answer is to the retrieved context
- **Contextual Precision**: Precision of retrieved context
- **Contextual Recall**: Recall of retrieved context

## 🐳 Docker Deployment

### Simple Deployment

```bash
# Build
docker build -f Dockerfile -t govgpt .

# Run
docker run --name govgpt-server \
  --env-file .env \
  -v $(pwd)/artifacts:/app/artifacts:ro \
  -v $(pwd)/documents:/app/documents:ro \
  -p 8000:8000 \
  -d govgpt
```

### Production Deployment

```bash
# Using Docker Compose
docker-compose up --build -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f govgpt-api
```