from langgraph.constants import END
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode

from app.core.prompts import SYSTEM_PROMPT
from app.core.state import AgentState


def build_graph(llm, tools):
    llm_with_tools = llm.bind_tools(tools)

    def call_model(state: AgentState) -> dict:
        messages = state["messages"]
        response = llm_with_tools.invoke([SYSTEM_PROMPT] + messages)
        return {"messages": [response]}

    def should_continue(state: AgentState) -> str:
        messages = state["messages"]
        last_message = messages[-1]

        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "call_tool"
        return END

    workflow = StateGraph(AgentState)
    workflow.add_node("call_model", call_model)
    workflow.add_node("call_tool", ToolNode(tools))
    workflow.add_edge(START, "call_model")
    workflow.add_conditional_edges(
        "call_model",
        should_continue,
        {
            "call_tool": "call_tool",
            END: END,
        },
    )
    workflow.add_edge("call_tool", "call_model")

    return workflow.compile()
