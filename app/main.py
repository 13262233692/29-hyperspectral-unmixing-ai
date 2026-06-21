"""
高光谱解编推理引擎 - FastAPI 主入口
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import api_router


app = FastAPI(
    title="高光谱数据解编推理引擎",
    description="基于 1D-CNN 自编码器的无监督高光谱解编服务\n"
                "应用于大洋钻探与深海地质勘探",
    version="1.0.0",
    contact={
        "name": "深海地质勘探项目组",
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.get("/", tags=["root"])
async def root():
    return {
        "name": "高光谱数据解编推理引擎",
        "version": "1.0.0",
        "status": "running",
        "api_docs": "/docs",
    }


@app.get("/health", tags=["root"])
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
