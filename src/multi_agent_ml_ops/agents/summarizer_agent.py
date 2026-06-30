from __future__ import annotations

from typing import Any


def summarize_text(text: str) -> dict[str, Any]:
    """Create a concise summary of the input text."""
    words = text.split()
    if len(words) <= 8:
        summary = text.strip()
    else:
        summary = " ".join(words[:8]) + "..."
    return {"agent": "summarizer", "summary": summary, "word_count": len(words)}
