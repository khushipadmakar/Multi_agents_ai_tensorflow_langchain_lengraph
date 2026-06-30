from multi_agent_ml_ops.agents.agent_state import AgentState
from multi_agent_ml_ops.agents.supervisor import build_graph, route_task


def test_route_task_routes_by_task_type():
    assert route_task({"tasks": ["classify"]}) == "classify"
    assert route_task({"tasks": ["sentiment"]}) == "sentiment"
    assert route_task({"tasks": ["summarize"]}) == "summarize"


def test_graph_compiles_and_has_expected_nodes():
    graph = build_graph()
    assert graph is not None
    assert "supervisor" in graph.nodes
    assert "summarizer" in graph.nodes


def test_agent_state_shape():
    state: AgentState = {
        "messages": [],
        "results": {},
        "final_output": "",
        "tasks": ["classify"],
        "input_text": "hello",
        "request_id": "req-1",
    }
    assert state["tasks"] == ["classify"]
