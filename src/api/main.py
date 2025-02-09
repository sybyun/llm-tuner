from fastapi import FastAPI, HTTPException
import yaml
from pydantic import BaseModel
from src.core.model_manager import ModelManager
from src.core.trainer import TrainerModule
from src.core.utils import load_config
import pytorch_lightning as pl

from api.routes import train, predict, health, config

app = FastAPI(title="LLM Fine-tuning API", version="1.0")

# 라우트 등록
app.include_router(train.router, prefix="/api")
app.include_router(predict.router, prefix="/api")
app.include_router(health.router, prefix="/api")
app.include_router(config.router, prefix="/api")

@app.get("/")
async def root():
    return {"message": "Welcome to the LLM Fine-tuning API!"}

# 기본 설정 파일 로드
CONFIG_PATH = "configs/model_config.yaml"
config = load_config(CONFIG_PATH)

# 모델 매니저 로드
model_manager = ModelManager(config)
model = model_manager.model

# 요청 데이터 모델 정의
class TrainRequest(BaseModel):
    config_path: str = CONFIG_PATH

@app.post("/train")
async def train_model(request: TrainRequest):
    """모델 학습을 시작하는 API"""
    try:
        # 새로운 설정 파일 로드
        new_config = load_config(request.config_path)

        # 학습 데이터 로드
        trainer = pl.Trainer(max_epochs=new_config["training"]["epochs"])
        trainer.fit(model)

        return {"message": "Model training started!", "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predict")
async def predict(text: str):
    """주어진 텍스트에 대한 예측 수행"""
    try:
        tokenizer = model_manager.tokenizer
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model(**inputs)
        prediction = outputs.logits.argmax(dim=-1).item()

        return {"text": text, "prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Welcome to the LLM Fine-tuning API!"}
