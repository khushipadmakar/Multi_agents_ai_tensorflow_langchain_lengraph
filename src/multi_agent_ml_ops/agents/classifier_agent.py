from __future__ import annotations

from typing import Any


def classify_text(text: str) -> dict[str, Any]:
    """Classify a text into a high-level topic category."""
    lowered = text.lower()
    if any(keyword in lowered for keyword in ["stock", "market", "finance", "business", "trade"]):
        label = "Business"
        confidence = 0.94
    elif any(keyword in lowered for keyword in ["politic", "election", "government", "policy"]):
        label = "Politics"
        confidence = 0.93
    elif any(keyword in lowered for keyword in ["sport", "game", "team", "league"]):
        label = "Sports"
        confidence = 0.92
    else:
        label = "World"
        confidence = 0.9
    return {"agent": "classifier", "label": label, "confidence": confidence}
