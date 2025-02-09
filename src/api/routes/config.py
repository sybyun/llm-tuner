from fastapi import APIRouter, HTTPException
from src.core.utils import load_config

router = APIRouter()

@router.get("/config")
async def get_config():
    """현재 모델 설정 확인 API"""
    try:
        config = load_config("configs/model_config.yaml")
        return {"config": config}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
