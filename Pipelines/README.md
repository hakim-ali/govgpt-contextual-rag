# GovGPT Contextual RAG Pipeline (Server Client Mode)

Advanced Retrieval-Augmented Generation system for UAE Information Assurance standards and government documents. This pipeline acts as an HTTP client that communicates with your hosted RAG server for scalable, production-ready deployment.

## Architecture

This pipeline uses a **client-server architecture**:
- **Pipeline (Client)**: Lightweight OpenWebUI integration that handles chat interface
- **RAG Server**: Your hosted `fastapi_rag_server_openwebui.py` that processes the actual RAG logic
- **Communication**: HTTP requests between Pipeline and RAG server

## Features

- **Server Client Mode**: Communicates with hosted RAG server via HTTP
- **Streaming Support**: Real-time response generation via `/retrieve` endpoint
- **Health Monitoring**: Automatic server health checks and connection testing
- **Configurable**: Adjustable server endpoints and parameters via Pipeline Valves
- **Error Handling**: Robust error handling with retry logic
- **Scalable**: Multiple pipeline instances can use the same RAG server

## Knowledge Base

This pipeline contains pre-processed UAE government documents including:
- Information Assurance Standards
- Procurement Business Processes  
- Human Resources Law
- Government Terms of Reference
- Department of Health Legislations

## Prerequisites

### Option 1: Local RAG Server (Recommended for Development)

Run the RAG server locally using Pipeline's components:

```bash
# Navigate to Pipeline directory
cd Pipelines

# Copy environment template and configure
cp .env.example .env
# Edit .env with your API keys and settings

# Start local RAG server with Docker
docker-compose up -d rag-server

# OR run directly with Python
python run_rag_server.py
```

### Option 2: External RAG Server

Use your main project's RAG server:

```bash
# Navigate to your main project directory
cd /path/to/govgpt-contextual-rag

# Start the RAG server (Docker or direct)
docker-compose up -d  # For Docker setup
# OR
python fastapi_rag_server_openwebui.py  # For direct execution
```

### Verify Server Health
Test your RAG server is working:
```bash
curl http://localhost:8100/health
```

## Installation

### Option 1: OpenWebUI Admin Panel
1. Copy the GitHub URL of this pipeline
2. Go to OpenWebUI Admin Panel → Pipelines  
3. Add new pipeline using the GitHub URL
4. Configure the valves as needed

### Option 2: Manual Installation
1. Clone or download this repository
2. Upload the entire folder to your OpenWebUI Pipelines directory
3. Restart OpenWebUI Pipelines service

## Configuration

Configure the pipeline through OpenWebUI's Valve interface:

### Required Settings
- **RAG_SERVER_URL**: URL of your RAG server (default: http://localhost:8100)
- **RAG_SERVER_TIMEOUT**: Request timeout in seconds (default: 30)

### Optional Settings
- **RAG_SERVER_API_KEY**: API key for server authentication (if needed)
- **RAG_SERVER_MAX_RETRIES**: Maximum retry attempts (default: 3)
- **RAG_MODEL**: LLM model name to send to server (default: gpt-4.1)

### Pipeline Settings
- **ENABLE_STREAMING**: Use streaming responses via /retrieve endpoint (default: true)
- **ENABLE_DEBUG**: Enable debug logging (default: false)
- **ENABLE_SERVER_HEALTH_CHECK**: Check server health before queries (default: true)
- **AUTO_TEST_CONNECTION**: Test server connection on startup (default: true)

## Usage

1. Install and configure the pipeline in OpenWebUI
2. Start a new chat and select the GovGPT model
3. Ask questions about UAE government standards and policies

### Example Queries
```
What are the information assurance requirements for government systems?

How should procurement processes be handled according to UAE standards?

What are the key elements of the human resources law?

Explain the data classification standards for government entities.
```

## Project Structure

```
Pipeline/
├── govgpt_contextual_rag_pipeline.py  # Main pipeline (HTTP client)
├── run_rag_server.py                 # Local RAG server for testing
├── Dockerfile                        # Docker image for local server
├── docker-compose.yml               # Docker services configuration
├── .env.example                     # Environment variables template
├── utils/
│   ├── api_client.py                 # HTTP client for RAG server
│   └── helpers.py                   # Utility functions
├── config/                           # Configuration (for local testing)
│   └── pipeline_config.py           
├── core/                             # Local RAG modules (for testing)
│   ├── rag_engine.py                
│   ├── embedding_service.py          
│   └── retrieval_service.py          
├── artifacts/                        # Local artifacts (for testing)
├── requirements.txt                  # Dependencies
├── test_pipeline.py                 # Testing script
└── README.md                        # Documentation
```

## How It Works

1. **Query Processing**: User query is received through OpenWebUI
2. **Server Communication**: Pipeline sends HTTP request to RAG server
3. **RAG Processing**: Server performs hybrid search and response generation
4. **Response Handling**: Pipeline receives and forwards response to user
5. **Streaming**: Real-time streaming via `/retrieve` endpoint

### Request Flow
```
User → OpenWebUI → Pipeline → HTTP → RAG Server (localhost:8100)
User ← OpenWebUI ← Pipeline ← HTTP ← RAG Server
```

## Local Development Workflow

### 1. Setup Environment
```bash
# Copy and configure environment
cp .env.example .env
# Edit .env with your API keys and settings
```

### 2. Start Local RAG Server
```bash
# Option A: Using Docker (Recommended)
docker-compose up -d rag-server

# Option B: Direct Python execution
python run_rag_server.py
```

### 3. Test Pipeline
```bash
# Test Pipeline → Server communication
python test_pipeline.py

# Test specific functionality
python example_usage.py
```

### 4. Development Commands
```bash
# View server logs
docker-compose logs -f rag-server

# Stop server
docker-compose down

# Rebuild after changes
docker-compose up --build -d rag-server

# Run integration tests
docker-compose --profile testing up pipeline-test
```

### 5. Local Server Endpoints
- **Health Check**: `http://localhost:8100/health`
- **Query**: `http://localhost:8100/query` (POST)
- **Streaming**: `http://localhost:8100/retrieve` (POST)
- **Status**: `http://localhost:8100/status`
- **Quick Test**: `http://localhost:8100/query/your-question-here`

## Troubleshooting

### Common Issues

**"RAG server is not available" error**
- Ensure your RAG server is running: `docker-compose up -d` or `python fastapi_rag_server_openwebui.py`
- Check server URL in Pipeline valves (default: http://localhost:8100)
- Verify server health: `curl http://localhost:8100/health`

**Connection timeout errors**
- Increase `RAG_SERVER_TIMEOUT` in Pipeline valves
- Check network connectivity between Pipeline and RAG server
- Verify firewall settings if using remote server

**Streaming not working**
- Ensure `ENABLE_STREAMING` is set to true in Pipeline valves
- Check that your RAG server supports the `/retrieve` endpoint
- Try disabling streaming temporarily to test basic functionality

**No relevant results**
- Check that your RAG server has loaded the knowledge base properly
- Verify server logs for any processing errors
- Enable debug mode in Pipeline valves to see request/response details

### Debug Mode
Enable `ENABLE_DEBUG` valve to see:
- Number of chunks retrieved
- Model information  
- Processing statistics
- Error details

## Requirements

- OpenWebUI Pipelines v0.5+
- Python 3.8+
- Required packages (see requirements.txt)
- API access to OpenAI-compatible endpoint

## License

This pipeline is designed for UAE government use and contains official government documents. Please respect copyright and usage restrictions.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review OpenWebUI Pipeline documentation
3. Contact the GovGPT development team