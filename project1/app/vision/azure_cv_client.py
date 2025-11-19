# app/vision/azure_cv_client.py

import os
import requests

PREDICTION_URL = os.getenv("AZURE_CV_PREDICTION_URL")
PREDICTION_KEY = os.getenv("AZURE_CV_PREDICTION_KEY")


def detect_objects_from_image_path(image_path: str) -> list[dict]:
    """
    Custom Vision의 Prediction URL을 이용해
    로컬 이미지 파일에 대해 Object Detection을 수행한다.

    Returns:
        [
          {
            "label": str,
            "probability": float,
            "bbox": {
              "left": float, "top": float,
              "width": float, "height": float,
            },
          },
          ...
        ]
    """
    if not PREDICTION_URL or not PREDICTION_KEY:
        raise RuntimeError(
            "AZURE_CV_PREDICTION_URL or AZURE_CV_PREDICTION_KEY is not set"
        )

    # 1) 이미지 파일을 바이너리로 읽기
    with open(image_path, "rb") as f:
        image_data = f.read()

    # 2) 문서에서 알려준 대로 헤더 구성
    headers = {
        "Prediction-Key": PREDICTION_KEY,
        "Content-Type": "application/octet-stream",
    }

    # 3) REST API 호출 (Body = 이미지 바이너리)
    response = requests.post(
        PREDICTION_URL,
        headers=headers,
        data=image_data,
        timeout=30,
    )
    response.raise_for_status()
    result = response.json()

    # 4) 결과 파싱
    # 예상 응답 구조:
    # {
    #   "id": "...",
    #   "project": "...",
    #   "predictions": [
    #     {
    #       "probability": 0.95,
    #       "tagId": "...",
    #       "tagName": "bed",
    #       "boundingBox": {
    #         "left": 0.1, "top": 0.2,
    #         "width": 0.3, "height": 0.4
    #       }
    #     },
    #     ...
    #   ]
    # }
    detections: list[dict] = []

    for pred in result.get("predictions", []):
        box = pred.get("boundingBox", {}) or {}
        detections.append(
            {
                "label": pred.get("tagName"),
                "probability": float(pred.get("probability", 0.0)),
                "bbox": {
                    "left": float(box.get("left", 0.0)),
                    "top": float(box.get("top", 0.0)),
                    "width": float(box.get("width", 0.0)),
                    "height": float(box.get("height", 0.0)),
                },
            }
        )

    return detections

