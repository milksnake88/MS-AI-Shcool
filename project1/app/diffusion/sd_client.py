# app/diffusion/sd_client.py
import os
from uuid import uuid4
from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image

from app.config import SD_MODEL_ID

_device = "cuda" if torch.cuda.is_available() else "cpu"
_pipe: StableDiffusionXLPipeline | None = None

BASE_DIR = Path(__file__).resolve().parents[2]
GENERATED_DIR = BASE_DIR / "app" / "static" / "generated"

def _get_pipeline() -> StableDiffusionXLPipeline:
    global _pipe

    if _pipe is None:
        if not SD_MODEL_ID:
            raise RuntimeError("SD_MODEL_ID is not set or empty")

        dtype = torch.float16 if _device == "cuda" else torch.float32

        print("[SDXL] Loading pipeline from:", SD_MODEL_ID)

        _pipe = StableDiffusionXLPipeline.from_pretrained(
            SD_MODEL_ID,
            torch_dtype=dtype,
            use_safetensors=True,
            variant="fp16",
        ).to(_device)

        try:
            _pipe.enable_attention_slicing()
        except Exception:
            pass

        try:
            _pipe.enable_vae_tiling()
        except Exception:
            pass

    return _pipe


def generate_image_from_prompt(
    prompt: str,
    num_inference_steps: int = 30,
    guidance_scale: float = 10.0,
    seed: int | None = None,
    width: int = 1024,
    height: int = 1024,
) -> str:
    """
    이미지를 생성해서 app/static/generated 하위에 저장하고,
    - 프론트 용 URL (url)
    - 백엔드 용 로컬 파일 경로 (path)
    반환
    """
    pipe = _get_pipeline()

    # 출력 디렉터리 보장
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    generator = None
    if seed is not None:
        generator = torch.Generator(device=_device).manual_seed(seed)

    negative_prompt = (
        "deformed face, distorted face, asymmetrical face, extra eyes, extra limbs, "
        "extra fingers, long neck, disfigured, mutated, low quality, blurry, distorted, "
        "text, cropped, ugly, disfigured, poor anatomy, missing limbs, malformed hands, "
        "poorly drawn eyes, unsettling, monochrome, grayscale, realistic, photography, photo"
    )

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        generator=generator,
    )

    image: Image.Image = result.images[0]

    filename = f"{uuid4().hex}.png"
    filepath = GENERATED_DIR / filename   # 로컬 경로
    image.save(str(filepath))
    
    url_path = f"/static/generated/{filename}"  # 프론트에서 쓸 URL

    return url_path

