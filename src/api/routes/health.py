from fastapi import APIRouter

router = APIRouter()

@router.get("/health")
async def health_check():
    """서버 상태 확인"""
    return {"status": "ok", "message": "API is running smoothly!"}
