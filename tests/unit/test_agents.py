import pytest

from multi_agent_ml_ops.agents.classifier_agent import classify_text
from multi_agent_ml_ops.agents.sentiment_agent import analyze_sentiment
from multi_agent_ml_ops.agents.summarizer_agent import summarize_text


@pytest.mark.parametrize("text", ["The stock market is rising", "A new product launched today", "Politics news is trending"])
def test_classifier_agent_returns_expected_shape(text):
    result = classify_text(text)
    assert "label" in result
    assert "confidence" in result
    assert 0 <= result["confidence"] <= 1


@pytest.mark.parametrize("text", ["I loved the service", "The experience was terrible", "It was okay"])
def test_sentiment_agent_returns_expected_shape(text):
    result = analyze_sentiment(text)
    assert "label" in result
    assert "score" in result
    assert -1 <= result["score"] <= 1


@pytest.mark.parametrize("text", ["Alpha beta gamma", "A short summary request", "This is a longer text for summarization"])
def test_summarizer_agent_returns_expected_shape(text):
    result = summarize_text(text)
    assert "summary" in result
    assert isinstance(result["summary"], str)
