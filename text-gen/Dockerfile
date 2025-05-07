FROM python:3.10-slim

# install dependencies
RUN pip install --no-cache-dir \
      fastapi uvicorn \
      torch transformers

# copy in our app
WORKDIR /app
COPY app.py .

# listen only on localhost
CMD ["uvicorn", "app:app", "--host", "127.0.0.1", "--port", "5000"]
