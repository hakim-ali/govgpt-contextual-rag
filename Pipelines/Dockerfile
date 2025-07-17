# Dockerfile for RAG Server in Pipeline Folder
# Based on the main project Dockerfile but adapted for Pipeline folder structure

FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for ultra-fast package management
RUN pip install uv

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies using uv (much faster than pip)
RUN uv pip install --system --no-cache -r requirements.txt

# Create app directory and user
RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /app
RUN chown appuser:appuser /app

# Copy application files (Pipeline folder structure)
COPY --chown=appuser:appuser run_rag_server.py .
COPY --chown=appuser:appuser init_kb.py .
COPY --chown=appuser:appuser start_server.sh .
COPY --chown=appuser:appuser core/ /app/core/
COPY --chown=appuser:appuser config/ /app/config/
COPY --chown=appuser:appuser utils/ /app/utils/
COPY --chown=appuser:appuser artifacts/ /app/artifacts/
COPY --chown=appuser:appuser govgpt-kb/ /app/govgpt-kb/

# Make startup script executable
RUN chmod +x start_server.sh

# Switch to non-root user
USER appuser

# Environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV ARTIFACT_DIR=/app/artifacts

# Health check with optimized timing
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8100/health || exit 1

# Expose port
EXPOSE 8100

# Run the startup script (initializes KB and starts server)
CMD ["./start_server.sh"]