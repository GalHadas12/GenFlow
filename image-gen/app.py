# ~/GenFlow/image-gen/app.py

from fastapi import FastAPI
from pydantic import BaseModel
from diffusers import StableDiffusionXLPipeline
import io, base64

app = FastAPI()

# Load SDXL base (CPU-only). If you have a GPU, add `.to("cuda")`.
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0"
)
pipe.to("cpu")

class ImgRequest(BaseModel):
    prompt: str
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    height: int = 1024
    width: int = 1024

@app.post("/v1/generate")
async def generate(req: ImgRequest):
    result = pipe(
        req.prompt,
        num_inference_steps=req.num_inference_steps,
        guidance_scale=req.guidance_scale,
        height=req.height,
        width=req.width,
    )
    image = result.images[0]
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return {"image_base64": b64}
