from langchain_core.messages import HumanMessage

from app.config import get_settings
from app.core.graph import build_reactive_graph, build_deliberative_graph, build_hybrid_graph
from app.core.llm import create_llm
from app.core.tools import get_tools


def create_agent():
    settings = get_settings()
    llm = create_llm(settings)
    execute_llm = create_llm(settings)
    tools = get_tools()
    return build_hybrid_graph(llm, execute_llm, tools)


def run_agent(message: str):
    agent = create_agent()
    print("LangGraph Mermaid流程图：")
    print(agent.get_graph().draw_mermaid())
    return agent.invoke({"messages": [HumanMessage(content=message)]})
