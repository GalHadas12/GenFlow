from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import torch

app = FastAPI()

# force CPU
device = "cpu"
generator = pipeline("text-generation", model="gpt2", device_map={"": device})

class GenRequest(BaseModel):
    inputs: str
    max_new_tokens: int = 50

@app.post("/v1/generate")
def generate(req: GenRequest):
    out = generator(req.inputs, max_new_tokens=req.max_new_tokens)
    return {"generated_text": out[0]["generated_text"]}
