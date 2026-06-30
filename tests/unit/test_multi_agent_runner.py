from multi_agent_ml_ops.agents.multi_agent_runner import build_langgraph_workflow


def test_langgraph_workflow_compiles_and_runs():
    workflow = build_langgraph_workflow()
    result = workflow.invoke({
        "input_text": "The stock market is rising and the service was excellent",
        "tasks": ["classify", "sentiment", "summarize"],
        "request_id": "req-1",
    })

    assert workflow is not None
    assert "results" in result
    assert "classify" in result["results"]
    assert "sentiment" in result["results"]
    assert "summarize" in result["results"]
