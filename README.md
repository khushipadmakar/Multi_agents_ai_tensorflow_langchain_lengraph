# Multi-Agent ML Ops

This project implements a production-style multi-agent AI system using LangGraph, TensorFlow, FastAPI, pytest, and GitHub Actions.

## Architecture

- Supervisor agent orchestrates specialist agents.
- Specialist agents provide classification, sentiment, and summarization.
- FastAPI serves a REST API with tracing and health endpoints.
- A notebook demonstrates training and deployment integration.

## Quick Start

```bash
pip install -e .
uvicorn multi_agent_ml_ops.api.app:app --reload
```
