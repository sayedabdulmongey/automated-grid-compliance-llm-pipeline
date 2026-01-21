# =============================================================================
# Grid Compliance QA API - Dockerfile
# =============================================================================
# Simple Dockerfile for serving the FastAPI inference endpoint
# Requires NVIDIA GPU with CUDA support
# =============================================================================

FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS production

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install Python
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

# Set working directory
WORKDIR /app

# Copy and install dependencies
COPY src/serving/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/serving/app.py ./app.py

# Environment variables (defaults)
ENV MODEL_ID=sayedsalem/qwen2.5-7b-grid-compliance
ENV MAX_NEW_TOKENS=256
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the API
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
