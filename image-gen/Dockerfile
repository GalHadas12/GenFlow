# ~/GenFlow/image-gen/Dockerfile

FROM python:3.10-slim

# 1. Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      git \
      curl \
      libgl1 \
      libglib2.0-0 \
      libopenblas-base \
      libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. Upgrade pip and install Python packages (CPU-only PyTorch)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
      fastapi \
      uvicorn[standard] \
      torch --index-url https://download.pytorch.org/whl/cpu \
      diffusers \
      transformers \
      pillow \
      accelerate \
      safetensors

# 3. Copy your FastAPI app
WORKDIR /app
COPY app.py .

# 4. Expose and launch
EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "127.0.0.1", "--port", "7860"]
