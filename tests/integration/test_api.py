from fastapi.testclient import TestClient

from multi_agent_ml_ops.api.app import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200


def test_analyze_endpoint():
    response = client.post(
        "/api/v1/analyze",
        json={"text": "The market is rising", "tasks": ["classify", "sentiment", "summarize"]},
    )
    assert response.status_code == 200
    payload = response.json()
    assert "request_id" in payload
    assert payload["status"] == "ok"


def test_agent_status_endpoint():
    response = client.get("/api/v1/agents/status")
    assert response.status_code == 200


def test_classify_endpoint():
    response = client.post("/api/v1/classify", json={"text": "The stock market is rising"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["result"]["label"] == "Business"


def test_sentiment_endpoint():
    response = client.post("/api/v1/sentiment", json={"text": "I loved the service"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["result"]["label"] == "positive"


def test_summarize_endpoint():
    response = client.post("/api/v1/summarize", json={"text": "This is a short summary request for the API"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "summary" in payload["result"]
