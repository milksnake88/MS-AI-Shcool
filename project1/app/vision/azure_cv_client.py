# app/vision/azure_cv_client.py
import os
from msrest.authentication import ApiKeyCredentials
from azure.cognitiveservices.vision.customvision.prediction import (
    CustomVisionPredictionClient,
)

# .env 에 넣어두거나 app.config 에서 가져와도 됨
ENDPOINT = os.getenv("AZURE_CV_ENDPOINT")
PREDICTION_KEY = os.getenv("AZURE_CV_PREDICTION_KEY")
PROJECT_ID = os.getenv("AZURE_CV_PROJECT_ID")           # GUID
PUBLISHED_NAME = os.getenv("AZURE_CV_PUBLISHED_NAME")   # 배포된 모델 이름

prediction_credentials = ApiKeyCredentials(
    in_headers={"Prediction-Key": PREDICTION_KEY}
)
predictor = CustomVisionPredictionClient(ENDPOINT, prediction_credentials)


def detect_objects_from_image_path(image_path: str) -> list[dict]:
    """
    로컬 이미지 파일을 Azure Custom Vision Object Detection에 보내고
    [ {label, probability, bbox}, ... ] 형태로 결과를 반환.
    """
    with open(image_path, "rb") as image_data:
        results = predictor.detect_image(
            PROJECT_ID,
            PUBLISHED_NAME,
            image_data
        )

    detections: list[dict] = []
    for pred in results.predictions:
        detections.append(
            {
                "label": pred.tag_name,
                "probability": float(pred.probability),
                "bbox": {
                    "left": pred.bounding_box.left,
                    "top": pred.bounding_box.top,
                    "width": pred.bounding_box.width,
                    "height": pred.bounding_box.height,
                },
            }
        )

    return detections

