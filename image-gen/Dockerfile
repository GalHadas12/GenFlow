# ~/GenFlow/image-gen/Dockerfile

FROM python:3.10-slim

# 1. System deps
RUN apt update && apt install -y --no-install-recommends \
      build-essential \
      libgl1 \
      libglib2.0-0 \
      curl \
    && rm -rf /var/lib/apt/lists/*

# 2. Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Create a venv (fast!) and install Python libs
#    uv venv venv     → creates the venv/
#    uv pip install   → pip runs inside that venv
RUN uv venv venv && \
    uv pip install --no-cache-dir \
      fastapi uvicorn \
      torch --index-url https://download.pytorch.org/whl/cpu \
      diffusers transformers pillow

# 4. Copy in your FastAPI app
WORKDIR /app
COPY app.py .

# 5. Use uv to run uvicorn inside the venv
ENTRYPOINT ["uv"]
CMD ["run", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", "7860"]