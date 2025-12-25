FROM python:3.10-slim

LABEL maintainer="Motorola MT Assignment"
LABEL description="Machine Translation Pipeline for EN-NL Software Domain"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create output directories
RUN mkdir -p /app/outputs/encoder_decoder \
    /app/outputs/decoder_only \
    /app/outputs/evaluation \
    /app/logs

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface

# Default command: run inference demo
CMD ["python", "scripts/inference_demo.py"]
