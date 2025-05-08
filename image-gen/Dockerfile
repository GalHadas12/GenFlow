# ~/GenFlow/image-gen/Dockerfile

FROM python:3.10-slim

# 1. Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      git \
      curl \
      libgl1 \
      libglib2.0-0 \
      libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. Upgrade pip and install Python packages (CPU-only PyTorch wheels)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
      fastapi \
      "uvicorn[standard]" \
      diffusers \
      transformers \
      pillow \
      accelerate \
      safetensors

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

WORKDIR /app
COPY app.py .

EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "127.0.0.1", "--port", "7860"]
