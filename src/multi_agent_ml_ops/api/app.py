from __future__ import annotations

import os
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from multi_agent_ml_ops.agents.classifier_agent import classify_text
from multi_agent_ml_ops.agents.sentiment_agent import analyze_sentiment
from multi_agent_ml_ops.agents.summarizer_agent import summarize_text
from multi_agent_ml_ops.agents.supervisor import build_graph
from multi_agent_ml_ops.models.model_registry import ModelRegistry


@asynccontextmanager
async def lifespan(app: FastAPI) -> Any:
    registry = app.state.registry
    for model_name in ("classifier", "sentiment"):
        registry.load_model(model_name)
    app.state.models_loaded = True
    yield
    app.state.models_loaded = False


app = FastAPI(title="Multi-Agent ML Ops", lifespan=lifespan)

registry = ModelRegistry()
app.state.registry = registry
app.state.graph = build_graph()
app.state.started_at = time.time()
app.state.last_execution_time = 0.0
app.state.success_count = 0
app.state.failure_count = 0
app.state.models_loaded = False
app.state.rate_limit_per_minute = int(os.getenv("RATE_LIMIT_PER_MINUTE", "100"))
app.state.request_history = defaultdict(deque)


class AnalyzeRequest(BaseModel):
    text: str
    tasks: list[str] = ["classify", "sentiment", "summarize"]


class AnalyzeResponse(BaseModel):
    request_id: str
    status: str
    results: dict[str, object]


class AgentResponse(BaseModel):
    request_id: str
    status: str
    result: dict[str, object]


class HealthResponse(BaseModel):
    status: str
    uptime: float
    models_loaded: bool
    rate_limit_per_minute: int


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _check_rate_limit(request: Request) -> None:
    client_ip = _client_ip(request)
    history = app.state.request_history[client_ip]
    now = time.monotonic()
    while history and now - history[0] > 60:
        history.popleft()
    if len(history) >= app.state.rate_limit_per_minute:
        raise HTTPException(status_code=429, detail="rate limit exceeded")
    history.append(now)


@app.get("/api/v1/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        uptime=time.time() - app.state.started_at,
        models_loaded=app.state.models_loaded,
        rate_limit_per_minute=app.state.rate_limit_per_minute,
    )


@app.post("/api/v1/analyze", response_model=AnalyzeResponse)
async def analyze(payload: AnalyzeRequest, request: Request) -> AnalyzeResponse:
    request_id = str(uuid4())
    _check_rate_limit(request)
    start = time.perf_counter()
    try:
        graph_result = await app.state.graph.ainvoke({
            "input_text": payload.text,
            "tasks": payload.tasks,
            "request_id": request_id,
        })
        app.state.last_execution_time = time.perf_counter() - start
        app.state.success_count += 1
        return AnalyzeResponse(request_id=request_id, status="ok", results=graph_result.get("results", {}))
    except Exception:
        app.state.last_execution_time = time.perf_counter() - start
        app.state.failure_count += 1
        return AnalyzeResponse(request_id=request_id, status="error", results={})


@app.post("/api/v1/classify", response_model=AgentResponse)
async def classify_agent(payload: AnalyzeRequest, request: Request) -> AgentResponse:
    request_id = str(uuid4())
    _check_rate_limit(request)
    try:
        result = classify_text(payload.text)
        return AgentResponse(request_id=request_id, status="ok", result=result)
    except Exception:
        return AgentResponse(request_id=request_id, status="error", result={})


@app.post("/api/v1/sentiment", response_model=AgentResponse)
async def sentiment_agent(payload: AnalyzeRequest, request: Request) -> AgentResponse:
    request_id = str(uuid4())
    _check_rate_limit(request)
    try:
        result = analyze_sentiment(payload.text)
        return AgentResponse(request_id=request_id, status="ok", result=result)
    except Exception:
        return AgentResponse(request_id=request_id, status="error", result={})


@app.post("/api/v1/summarize", response_model=AgentResponse)
async def summarize_agent(payload: AnalyzeRequest, request: Request) -> AgentResponse:
    request_id = str(uuid4())
    _check_rate_limit(request)
    try:
        result = summarize_text(payload.text)
        return AgentResponse(request_id=request_id, status="ok", result=result)
    except Exception:
        return AgentResponse(request_id=request_id, status="error", result={})


@app.get("/api/v1/agents/status")
async def agent_status() -> dict[str, object]:
    total = app.state.success_count + app.state.failure_count
    success_rate = app.state.success_count / total if total else 1.0
    return {
        "status": "ok",
        "last_execution_time": app.state.last_execution_time,
        "success_rate": success_rate,
        "models_loaded": app.state.models_loaded,
        "agents": ["classifier", "sentiment", "summarizer"],
    }
