import os
import tempfile
import importlib.util
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))
AI_SECRET_TOKEN = os.getenv("AI_SECRET_TOKEN")

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != AI_SECRET_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_module(name, rel_path):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_BASE_DIR, rel_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# 전역 모델/모듈 변수
food_model = None
food_device = None
food_classes = None
yolo_model = None
predict_mod = None
ocr_mod = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global food_model, food_device, food_classes, yolo_model, predict_mod, ocr_mod

    predict_mod = _load_module("predict_v2m", "models/food_classifier/predict_V2_M.py")
    food_model, food_device, food_classes = predict_mod.load_food_model(
        os.path.join(_BASE_DIR, "best_food_model_v2_m_ver2.pth")
    )

    ocr_mod = _load_module("receipt_ocr", "models/receipt_ocr/receipt_ocr.py")

    yolo_model = YOLO(os.path.join(_BASE_DIR, "yolo_model", "yolov8n.pt"))

    yield


app = FastAPI(lifespan=lifespan)


def _tmp_file(upload: UploadFile, data: bytes) -> str:
    suffix = os.path.splitext(upload.filename)[1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(data)
        return f.name


# ──────────────────────────────────────────
# 1. 음식 분류
# ──────────────────────────────────────────
@app.post("/internal/v1/food-classification")
async def food_classification(file: UploadFile = File(...), _=Depends(verify_token)):
    tmp_path = _tmp_file(file, await file.read())
    try:
        result = predict_mod.predict_image(food_model, food_device, tmp_path, food_classes)
        if "error" in result:
            return JSONResponse(status_code=400, content={"error": result["error"]})
        return {
            "bestMatch": result["best_match"],
            "confidence": result["confidence"],
            "top3": result["top3"],
            "source": result["source"],
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        os.unlink(tmp_path)


# ──────────────────────────────────────────
# 2. 영수증 OCR
# ──────────────────────────────────────────
@app.post("/internal/v1/receipt-ocr")
async def receipt_ocr(file: UploadFile = File(...), _=Depends(verify_token)):
    tmp_path = _tmp_file(file, await file.read())
    try:
        raw_items = ocr_mod.process_receipt_raw(tmp_path)
        ingredients = ocr_mod.filter_with_gemini(raw_items) if raw_items else []
        return {"ingredients": ingredients}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        os.unlink(tmp_path)


# ──────────────────────────────────────────
# 3. 냉장고 물체 감지
# ──────────────────────────────────────────
@app.post("/internal/v1/fridge-detection")
async def fridge_detection(file: UploadFile = File(...), _=Depends(verify_token)):
    tmp_path = _tmp_file(file, await file.read())
    try:
        results = yolo_model.predict(source=tmp_path, conf=0.25, verbose=False)
        objects = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                objects.append({
                    "name": yolo_model.names[int(box.cls)],
                    "confidence": round(float(box.conf) * 100, 1),
                    "box": {
                        "x1": round(x1), "y1": round(y1),
                        "x2": round(x2), "y2": round(y2),
                    },
                })
        return {"objects": objects}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        os.unlink(tmp_path)
