from typing import TypedDict


class AgentState(TypedDict, total=False):
    messages: list[str]
    results: dict[str, dict[str, object]]
    final_output: str
    tasks: list[str]
    input_text: str
    request_id: str
