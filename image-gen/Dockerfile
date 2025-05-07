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
    | env UV_INSTALL_DIR="/usr/local/bin" sh            # use UV_INSTALL_DIR to override install path :contentReference[oaicite:0]{index=0}

# 3. Ensure /usr/local/bin is on PATH
ENV PATH="/usr/local/bin:${PATH}"

# 4. Create a fast venv and install Python libs
RUN uv venv venv \
 && uv pip install --no-cache-dir \
      fastapi uvicorn \
      torch --index-url https://download.pytorch.org/whl/cpu \
      diffusers transformers pillow

WORKDIR /app
COPY app.py .

# 5. Use uv to launch the server inside the venv
ENTRYPOINT ["uv"]
CMD ["run", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", "7860"]
