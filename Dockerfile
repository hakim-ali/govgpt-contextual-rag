# Single-stage Dockerfile using UV for fast package installation
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

# Copy application files
COPY --chown=appuser:appuser fastapi_rag_server_openwebui.py .
COPY --chown=appuser:appuser artifacts/ /app/artifacts/

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

# Run the FastAPI server
CMD ["python", "fastapi_rag_server_openwebui.py"]