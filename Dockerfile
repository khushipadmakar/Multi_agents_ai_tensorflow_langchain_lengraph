FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e .
EXPOSE 8000
CMD ["uvicorn", "multi_agent_ml_ops.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
