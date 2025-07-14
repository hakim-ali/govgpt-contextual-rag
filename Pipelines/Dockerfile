# Dockerfile for RAG Server in Pipeline Folder
# Based on the main project Dockerfile but adapted for Pipeline folder structure

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH=/app
ENV ARTIFACT_DIR=./artifacts

# Install system dependencies
RUN apt-get update && \
    apt-get install -y gcc g++ && \
    rm -rf /var/lib/apt/lists/*


# Create artifacts directory
RUN mkdir -p /app/artifacts
RUN mkdir -p /app/core
RUN mkdir -p /app/config
RUN mkdir -p /app/utils

# Copy application files
COPY .env /app/.env
COPY ./artifacts /app/artifacts
COPY  ./core /app/core
COPY ./config /app/config
COPY ./utils /app/utils
COPY requirements.txt /app
COPY run_rag_server.py /app

# Set working directory
WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8100

# Run the FastAPI server
CMD ["python", "run_rag_server.py"]