# app/main.py
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.ocr.azure_ocr import extract_text_from_image
from app.llm.gemini_client import build_sd_prompt_from_text
from app.diffusion.sd_client import generate_image_from_prompt
from app.vision.azure_cv_client import detect_objects_from_image_path

import traceback

app = FastAPI()

# static / templates 설정
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """
    메인 페이지: 이미지 업로드 폼
    """
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "ocr_text": None,
            "sd_prompt": None,
            "generated_image_url": None,
            "detections": None,
            "error": None,
            "filename": None,
        },
    )


@app.post("/ocr", response_class=HTMLResponse)
async def ocr_image(request: Request, file: UploadFile = File(...)):
    """
    이미지 파일을 받아 Azure OCR을 수행하고 결과 텍스트만 보여준다.
    """
    try:
        image_bytes = await file.read()
        ocr_text = extract_text_from_image(image_bytes)

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "ocr_text": ocr_text,
                "sd_prompt": None,
                "generated_image_url": None,
                "detections": None,
                "error": None,
                "filename": file.filename,
            },
        )
    except Exception as e:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "ocr_text": None,
                "sd_prompt": None,
                "generated_image_url": None,
                "detections": None,
                "error": str(e),
                "filename": getattr(file, "filename", None),
            },
        )


@app.post("/generate", response_class=HTMLResponse)
async def generate_from_image(request: Request, file: UploadFile = File(...)):
    """
    이미지 파일을 받아:
      1) Azure OCR로 텍스트 추출
      2) Gemini로 Stable Diffusion용 프롬프트 생성
      3) Stable Diffusion(기본 모델)로 이미지 생성
      4) Azure CV Object Detection으로 생성 이미지 내 객체 검출
    """
    try:
        image_bytes = await file.read()

        # 1) OCR
        ocr_text = extract_text_from_image(image_bytes)

        # 2) Gemini 프롬프트 생성
        sd_prompt = build_sd_prompt_from_text(ocr_text)

        # 3) Stable Diffusion 이미지 생성
        generated_image_url, generated_image_path = generate_image_from_prompt(sd_prompt)

        # 4) Azure CV Object Detection
        detections = detect_objects_from_image_path(generated_image_path)

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "ocr_text": ocr_text,
                "sd_prompt": sd_prompt,
                "generated_image_url": generated_image_url,
                "detections": detections,
                "error": None,
                "filename": file.filename,
            },
        )
    except Exception as e:
        print("[/generate] ERROR:")
        traceback.print_exc()
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "ocr_text": None,
                "sd_prompt": None,
                "generated_image_url": None,
                "detections": None,
                "error": str(e),
                "filename": getattr(file, "filename", None),
            },
        )
