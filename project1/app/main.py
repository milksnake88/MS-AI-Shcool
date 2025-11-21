# app/main.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import json

from app.ocr.azure_ocr import extract_text_from_image
from app.llm.gemini_client import (
        build_sd_prompt_from_text, 
        build_ai_question,
        build_chat_reaction,
        summarize_chat_history,
        )
from app.diffusion.sd_client import generate_image_from_prompt
from app.vision.azure_cv_client import detect_objects_from_image_url

import traceback

app = FastAPI()

app.mount("/static", StaticFiles(directory="app/static"), name="static")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          
    allow_credentials=True,
    allow_methods=["*"],          
    allow_headers=["*"],          
)


# ---------------------------
# 1. 책 표지 분석 (OCR)
# ---------------------------
@app.post("/api/analyze-cover")
async def analyze_cover(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        ocr_text = extract_text_from_image(image_bytes)
        # 제목: OCR 텍스트의 첫 줄 또는 가장 긴 줄
        #lines = [line.strip() for line in ocr_text.split("\n") if line.strip()]
        #title = lines[0] if lines else ""

        return { "title": ocr_text }

    except Exception as e:
        return { "error": str(e) }


# ---------------------------
# 2. 페이지 전체 처리 파이프라인
#    OCR → Gemini → SDXL → Detection
# ---------------------------
@app.post("/api/process-page")
async def process_page(
        file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        ocr_text = extract_text_from_image(image_bytes)
        print("ocr_text: ", ocr_text)
        sd_prompt = build_sd_prompt_from_text(ocr_text)
        print("sd_prompt: ", sd_prompt)
        image_url = generate_image_from_prompt(sd_prompt)
        print("iamge_url: ", image_url)
        objects = detect_objects_from_image_url(image_url)
        print("objects detected")
        ai_question = build_ai_question(ocr_text)
        print("ai_question: ", ai_question)
        return {
            "ocrText": ocr_text,
            "sd_prompt": sd_prompt,
            "imageUrl": image_url,
            "objects": objects,
            "aiQuestion": ai_question
            }

    except Exception as e:
        print("[/api/process-page] ERROR:", repr(e))
        return { "error": str(e) }


# ---------------------------
# 3. 그림 재생성
# ---------------------------
@app.post("/api/regenerate-image")
async def regenerate_image(payload: dict):
    try:
        prompt = payload.get("prompt")
        if not prompt or not isinstance(prompt, str):
            return {"error": "prompt 필드는 문자열로 반드시 포함되어야 합니다."}
        image_url = generate_image_from_prompt(prompt)
        print("regenerated iamge_url: ", image_url)
        objects = detect_objects_from_image_url(image_url)
        print("objects detected")
        return {
            "imageUrl": image_url,
            "objects": objects,
            }

    except Exception as e:
        print("[/api/regenerate-image] ERROR:", repr(e))
        return { "error": str(e) }


# ---------------------------
# 4. 채팅 API (아이 답장 → 리액션)
#    POST /api/chat
#    Request: { "message": "...", "history": [ ... ] }
#    Response: { "reply": "..." }
# ---------------------------
@app.post("/api/chat")
async def chat_api(payload: dict):
    try:
        message = payload.get("message", "")
        history = payload.get("history") or []

        if not isinstance(history, list):
            history = []

        # Gemini에게 채팅 리액션 생성 요청
        reply = build_chat_reaction(message, history)

        return { "reply": reply }

    except Exception as e:
        print("[/api/chat] ERROR:", repr(e))
        return { "error": str(e) }


# ---------------------------
# 5. 채팅 요약 API
#    POST /api/chat-summary
#    Request: { "history": [ ... ] }
#    Response: { "summary": "..." }
# ---------------------------
@app.post("/api/chat-summary")
async def chat_summary_api(payload: dict):
    try:
        history = payload.get("history") or []
        if not isinstance(history, list):
            history = []

        summary = summarize_chat_history(history)
        return {"summary": summary}

    except Exception as e:
        print("[/api/chat-summary] ERROR:", repr(e))
        return {"error": str(e)}

