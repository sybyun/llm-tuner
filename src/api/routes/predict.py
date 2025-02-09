from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

router = APIRouter()

class PredictRequest(BaseModel):
    text: str

# 모델 로드 (LoRA 등 파인튜닝 적용된 모델 포함)
MODEL_PATH = "models/finetuned_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

@router.get("/predict")
async def predict_text(request: PredictRequest):
    try:
        inputs = tokenizer(request.text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=-1).item()

        return {"text": request.text, "prediction": prediction}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
