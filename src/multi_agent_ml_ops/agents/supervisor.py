from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from multi_agent_ml_ops.agents.agent_state import AgentState
from multi_agent_ml_ops.agents.classifier_agent import classify_text
from multi_agent_ml_ops.agents.sentiment_agent import analyze_sentiment
from multi_agent_ml_ops.agents.summarizer_agent import summarize_text


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
        state["results"]["classify"] = classify_text(text)
    if "sentiment" in tasks:
        state["results"]["sentiment"] = analyze_sentiment(text)
    if "summarize" in tasks:
        state["results"]["summarize"] = summarize_text(text)
    state["final_output"] = str(state["results"])
    return state


def classify_node(state: AgentState) -> AgentState:
    return state


def sentiment_node(state: AgentState) -> AgentState:
    return state


def summarize_node(state: AgentState) -> AgentState:
    return state


def finalize_node(state: AgentState) -> AgentState:
    state["final_output"] = state.get("final_output", "") or str(state.get("results", {}))
    state.setdefault("results", {})
    if not state["results"]:
        state["results"] = {"summary": {"agent": "supervisor", "summary": "No specialist output generated."}}
    return state


def build_graph() -> Any:
    workflow = StateGraph(AgentState)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("classify", classify_node)
    workflow.add_node("sentiment", sentiment_node)
    workflow.add_node("summarizer", summarize_node)
    workflow.add_node("finalize", finalize_node)
    workflow.set_entry_point("supervisor")
    workflow.add_conditional_edges(
        "supervisor",
        route_task,
        {"classify": "classify", "sentiment": "sentiment", "summarize": "summarizer", "finalize": "finalize"},
    )
    workflow.add_edge("classify", "finalize")
    workflow.add_edge("sentiment", "finalize")
    workflow.add_edge("summarizer", "finalize")
    workflow.add_edge("finalize", END)
    return workflow.compile()
