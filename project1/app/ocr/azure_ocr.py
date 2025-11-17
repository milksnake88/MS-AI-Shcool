# app/ocr/azure_ocr.py
from io import BytesIO
import time

from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials

from app.config import AZURE_CV_ENDPOINT, AZURE_CV_KEY


def get_cv_client() -> ComputerVisionClient:
    if not AZURE_CV_ENDPOINT or not AZURE_CV_KEY:
        raise RuntimeError("Azure Computer Vision endpoint/key not set. Check .env or env vars.")
    credentials = CognitiveServicesCredentials(AZURE_CV_KEY)
    client = ComputerVisionClient(AZURE_CV_ENDPOINT, credentials)
    return client


def extract_text_from_image(image_bytes: bytes) -> str:
    """
    Azure OCR(Read API)를 사용해서 이미지 바이트에서 텍스트를 추출한다.
    여러 줄을 '\n'으로 이어 붙여서 하나의 문자열로 반환.
    """
    client = get_cv_client()

    # Read API 호출 (스트림 형태로 전송)
    image_stream = BytesIO(image_bytes)
    read_response = client.read_in_stream(image_stream, raw=True)

    # 비동기 작업 결과를 polling
    operation_location = read_response.headers["Operation-Location"]
    operation_id = operation_location.split("/")[-1]

    while True:
        result = client.get_read_result(operation_id)
        if result.status not in ["notStarted", "running"]:
            break
        time.sleep(1)

    text_lines = []
    if result.status == OperationStatusCodes.succeeded:
        for page in result.analyze_result.read_results:
            for line in page.lines:
                text_lines.append(line.text)

    return "\n".join(text_lines)
