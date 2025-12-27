# 📘 Project1: AI Storybook Assistant — Backend

**Input:** A photo of a real children’s storybook page  
**Output:** AI-generated prompt · SDXL illustration · Object detection Game

> This backend service turns a storybook page into an **AI-generated illustration and structured scene understanding** using OCR, LLM, Stable Diffusion XL, and Object detection.

🎥 **Demo Video Attached**

---

## 🔗 End-to-End Pipeline

```
Image Upload
   ↓
Azure OCR (text extraction)
   ↓
Gemini (prompt generation for SDXL)
   ↓
Stable Diffusion XL (image generation, GPU)
   ↓
Azure Custom Vision (object detection)
   ↓
JSON Response
```

---

## 🧠 Tech Stack

**Backend** — FastAPI, Python 3.10, Uvicorn
**OCR** — Azure Computer Vision
**LLM** — Google Gemini API
**Image Generation** — Stable Diffusion XL (`diffusers`)
**Detection** — Azure Custom Vision
**Infra** — Azure VM (GPU), SSH tunneling during development

---

## 📡 API

### **① Analyze Book Cover**

Extracts title text from the cover.

```
POST /api/analyze-cover
form-data: file = image
```

**Response**

```json
{ "title": "The Three Little Pigs" }
```

---

### **② Process Story Page (Full Pipeline)**

```
POST /api/process-page
form-data: file = image
```

**Response**

```json
{
  "ocrText": "...",
  "imageUrl": "/static/generated/abcd.png",
  "objects": [
    {
      "name": "pig",
      "confidence": 0.98,
      "boundingBox": { "left":0.1,"top":0.2,"width":0.3,"height":0.3 }
    }
  ]
}
```

Generated images are saved under:

```
app/static/generated/
```

and served publicly.

---

## 📁 Project Structure

```
app/
 ├ main.py
 ├ ocr/azure_ocr.py
 ├ llm/gemini_client.py
 ├ diffusion/sd_client.py
 ├ vision/azure_cv_client.py
 └ static/generated/
```

---

## 🚀 Run Backend

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Environment variables (Azure · Gemini · SDXL model) are managed via `.env`.

---

## 🎯 Engineering Highlights

* Orchestrated **OCR → LLM → Diffusion → Detection** in a single request
* **GPU-optimized SDXL** (attention slicing / VAE tiling)
* Static image serving + clean URL ↔ path mapping
* Frontend-friendly **JSON contract design**
* CORS configured for browser-based clients

---

## 🔒 Production Considerations

* Restrict CORS
* Secure API keys
* Enable SDXL safety checker for public services

---

## 🌱 Future Work

* Improve face & artifact suppression
* Add async job handling / queueing
* Add semantic consistency checks

---

## 👩‍💻 Goal

To help children **visualize the stories they read**,
by turning book pages into **AI-generated interactive scenes**.

---

If you’d like, I can also provide:

✅ a **badge header section**
✅ a **short project summary paragraph**
✅ or **a matching Korean README**

Just tell me 👍
