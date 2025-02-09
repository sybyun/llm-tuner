from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.core.trainer import TrainerModule
from src.core.data_loader import create_data_loader
from datasets import Dataset
import pytorch_lightning as pl

router = APIRouter()

class TrainRequest(BaseModel):
    config_path: str = "configs/model_config.yaml"
    dataset: dict  # JSON 형식의 데이터셋 (예: {"text": [...], "label": [...]})

@router.post("/train")
async def train_model(request: TrainRequest):
    try:
        trainer_module = TrainerModule()
        
        # 데이터셋을 Dataset 객체로 변환 후 DataLoader 생성
        dataset = Dataset.from_dict(request.dataset)
        train_loader = create_data_loader(dataset, trainer_module.tokenizer, batch_size=trainer_module.config["training"]["batch_size"])

        trainer = pl.Trainer(max_epochs=trainer_module.config["training"]["epochs"])
        trainer.fit(trainer_module, train_dataloaders=train_loader)

        # 평가 결과 반환
        metrics = {metric_name: metric.compute() for metric_name, metric in trainer_module.metrics.items()}
        return {"message": "Model training completed!", "metrics": metrics}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
