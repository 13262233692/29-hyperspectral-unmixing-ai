from fastapi import APIRouter
from .endpoints import router as unmix_router

api_router = APIRouter()
api_router.include_router(unmix_router)

__all__ = ["api_router"]
