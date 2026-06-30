from __future__ import annotations

from typing import Any


def analyze_sentiment(text: str) -> dict[str, Any]:
    """Produce a sentiment label and score for a text input."""
    lowered = text.lower()
    if any(word in lowered for word in ["love", "great", "excellent", "good", "nice", "amazing"]):
        return {"agent": "sentiment", "label": "positive", "score": 0.82}
    if any(word in lowered for word in ["bad", "terrible", "hate", "awful", "poor", "worst"]):
        return {"agent": "sentiment", "label": "negative", "score": -0.91}
    return {"agent": "sentiment", "label": "neutral", "score": 0.05}
