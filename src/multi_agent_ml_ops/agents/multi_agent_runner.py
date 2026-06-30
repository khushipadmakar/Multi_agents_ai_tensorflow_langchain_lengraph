from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from multi_agent_ml_ops.agents.agent_state import AgentState
from multi_agent_ml_ops.agents.classifier_agent import classify_text
from multi_agent_ml_ops.agents.sentiment_agent import analyze_sentiment
from multi_agent_ml_ops.agents.summarizer_agent import summarize_text
from multi_agent_ml_ops.models.text_classifier import TextClassifierModel
from multi_agent_ml_ops.models.sentiment_model import SentimentModel


classifier_model = TextClassifierModel()
sentiment_model = SentimentModel()


def route_task(state: dict[str, Any]) -> str:
    tasks = state.get("tasks", [])
    if not tasks:
        return "summarize"
    if {"classify", "sentiment", "summarize"}.issubset(set(tasks)):
        return "finalize"
    if "classify" in tasks:
        return "classify"
    if "sentiment" in tasks:
        return "sentiment"
    return "summarize"


def supervisor_node(state: AgentState) -> AgentState:
    text = state.get("input_text", "")
    tasks = state.get("tasks", [])
    state.setdefault("results", {})
    if "classify" in tasks:
        state["results"]["classify"] = {
            **classify_text(text),
            "model_output": classifier_model.predict([text]).tolist(),
        }
    if "sentiment" in tasks:
        state["results"]["sentiment"] = {
            **analyze_sentiment(text),
            "model_output": sentiment_model.predict([text]).tolist(),
        }
    if "summarize" in tasks:
        state["results"]["summarize"] = summarize_text(text)
    state["final_output"] = str(state["results"])
    return state


def finalize_node(state: AgentState) -> AgentState:
    state["final_output"] = state.get("final_output", "") or str(state.get("results", {}))
    return state


def build_langgraph_workflow() -> Any:
    workflow = StateGraph(AgentState)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("finalize", finalize_node)
    workflow.set_entry_point("supervisor")
    workflow.add_conditional_edges(
        "supervisor",
        route_task,
        {"classify": "finalize", "sentiment": "finalize", "summarize": "finalize", "finalize": "finalize"},
    )
    workflow.add_edge("finalize", END)
    return workflow.compile()


def run_multi_agent_pipeline(text: str, tasks: list[str]) -> dict[str, Any]:
    workflow = build_langgraph_workflow()
    result = workflow.invoke({
        "input_text": text,
        "tasks": tasks,
        "request_id": "request",
    })
    return dict(result)
