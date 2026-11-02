
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from backend.model_service import SQLModelService

service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global service
    service = SQLModelService()
    yield
    service = None


app = FastAPI(
    title="QueryForge AI API",
    version="1.1.0",
    lifespan=lifespan
)


@app.get("/")
def root():
    return {
        "name": "QueryForge AI API",
        "status": "running",
        "health": "/health",
        "docs": "/docs",
        "generate_sql": "POST /generate-sql"
    }


class SQLRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    schema_: str = Field(
        ...,
        alias="schema",
        min_length=1,
        max_length=12000
    )
    model: str = "finetuned"


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": service is not None,
        "supported_models": ["base", "finetuned"]
    }


@app.post("/generate-sql")
def generate_sql(request: SQLRequest):
    if request.model not in ("base", "finetuned"):
        raise HTTPException(
            status_code=400,
            detail="Supported models: base, finetuned"
        )

    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )

    try:
        sql = service.generate_sql(
            question=request.question,
            schema=request.schema_,
            model_type=request.model
        )

        return {
            "sql": sql,
            "model": request.model,
            "status": "success"
        }

    except Exception as exc:
        print(f"Inference error: {exc}")
        raise HTTPException(
            status_code=500,
            detail="SQL generation failed"
        )