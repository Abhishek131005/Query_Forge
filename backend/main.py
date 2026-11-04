
from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from backend.model_service import SQLModelService

logger = logging.getLogger(__name__)
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


class GenerationConfig(BaseModel):
    max_new_tokens: int
    do_sample: bool


class SQLResponse(BaseModel):
    sql: str
    raw_output: str
    cleaned_output: str
    extracted_sql: str
    model: str
    status: str
    inference_time_seconds: float = Field(ge=0)
    queue_wait_seconds: float = Field(ge=0)
    input_tokens: int = Field(ge=0)
    generated_tokens: int = Field(gt=0)
    device: str
    generation_config: GenerationConfig
    prompt_version: str
    postprocessed: bool


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": service is not None,
        "supported_models": ["base", "finetuned"]
    }


@app.post("/generate-sql", response_model=SQLResponse)
def generate_sql(request: SQLRequest) -> SQLResponse:
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
        result = service.generate_sql(
            question=request.question,
            schema=request.schema_,
            model_type=request.model
        )
        return SQLResponse(**result)
    except Exception:
        logger.exception("Inference error")
        raise HTTPException(
            status_code=500,
            detail="SQL generation failed"
        )