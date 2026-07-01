from langchain_core.messages import HumanMessage

from app.config import get_settings
from app.core.graph import build_graph
from app.core.llm import create_llm
from app.core.tools import get_tools


def create_agent():
    settings = get_settings()
    llm = create_llm(settings)
    tools = get_tools()
    return build_graph(llm, tools)


def run_agent(message: str):
    agent = create_agent()
    return agent.invoke({"messages": [HumanMessage(content=message)]}, {"recursion_limit":10})
