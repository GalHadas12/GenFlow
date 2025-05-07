# ~/GenFlow/image-gen/Dockerfile

FROM python:3.10-slim

# 1. System deps + curl
RUN apt update && apt install -y --no-install-recommends \
      build-essential \
      libgl1 \
      libglib2.0-0 \
      curl \
    && rm -rf /var/lib/apt/lists/*

# 2. Install uv into /usr/local/bin
RUN curl -LsSf https://astral.sh/uv/install.sh \
    | UV_INSTALL_DIR="/usr/local/bin" sh

# 3. Ensure uv is on PATH
ENV PATH="/usr/local/bin:${PATH}"

# 4. Install all Python packages globally via uv pip
RUN uv pip install --system --no-cache-dir \
      fastapi uvicorn \
      torch --index-url https://download.pytorch.org/whl/cpu \
      diffusers transformers pillow

# 5. Copy your FastAPI app
WORKDIR /app
COPY app.py .

# 6. Launch with uv
ENTRYPOINT ["uv"]
CMD ["run", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", "7860"]