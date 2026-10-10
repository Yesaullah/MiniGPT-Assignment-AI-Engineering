"""FastAPI service for MiniGPT.

uvicorn src.serve.app:app --port 8000
Env: MODEL_PATH (default checkpoints/model.pt), TORCH_THREADS (default 1).
"""
from __future__ import annotations

import os
import threading
import time

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from src.generate import run_generation
from src.tokenizer import UnknownCharacterError
from src.utils import load_checkpoint, sha256_file

MODEL_PATH = os.environ.get("MODEL_PATH", "checkpoints/model.pt")
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "1")))

model, cfg, tokenizer, _state = load_checkpoint(MODEL_PATH, "cpu")
MODEL_SHA256 = sha256_file(MODEL_PATH)
_lock = threading.Lock()  # one generation at a time on a small CPU
STATIC = os.path.join(os.path.dirname(__file__), "static", "index.html")

app = FastAPI(title="MiniGPT", description="A small GPT trained from scratch on Tiny Shakespeare.")


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field("ROMEO:\n", max_length=1000)
    max_new_tokens: int = Field(200, ge=1, le=500)
    temperature: float = Field(0.8, ge=0.0, le=2.0, allow_inf_nan=False)
    top_k: int | None = Field(None, ge=1)
    seed: int = Field(0, ge=0, le=2**32 - 1)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_sha256": MODEL_SHA256}


@app.get("/model-info")
def model_info() -> dict:
    return {"parameters": model.num_params(), "vocab_size": tokenizer.vocab_size,
            "tokenizer": "character-level", "config": cfg.model.__dict__,
            "model_sha256": MODEL_SHA256, "limits": {"prompt_chars": 1000, "max_new_tokens": 500}}


@app.post("/generate")
def generate(req: GenerateRequest) -> dict:
    missing = tokenizer.unknown_chars(req.prompt)
    if missing:
        raise HTTPException(422, detail={"error": "unknown_characters", "characters": missing})
    if req.top_k is not None and req.top_k > tokenizer.vocab_size:
        raise HTTPException(422, detail=f"top_k must be <= {tokenizer.vocab_size}")
    queued = time.perf_counter()
    if not _lock.acquire(timeout=30):
        raise HTTPException(503, detail="busy, try again", headers={"Retry-After": "10"})
    try:
        result = run_generation(model, cfg, tokenizer, req.prompt, req.max_new_tokens,
                                req.temperature, req.top_k, req.seed, "cpu")
    except UnknownCharacterError as e:
        raise HTTPException(422, detail=str(e))
    finally:
        _lock.release()
    result["queue_seconds"] = round(time.perf_counter() - queued - result["seconds"], 3)
    result["model_sha256"] = MODEL_SHA256[:12]
    return result


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC)
