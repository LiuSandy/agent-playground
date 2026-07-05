import os

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from datetime import datetime

from app.config import get_settings
from app.core.graph import build_reactive_graph, build_deliberative_graph, build_hybrid_graph
from app.core.llm import create_llm
from app.core.tools import get_tools

from dotenv import load_dotenv

load_dotenv()

IS_LANGSMITH_TRACING_ENABLED = os.getenv('LANGSMITH_TRACING', '').lower() == "true"


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

    config = {}

    if IS_LANGSMITH_TRACING_ENABLED:
        config = RunnableConfig(
            metadata={
                "version": "v2",
                "model": os.getenv("DEEPSEEK_MODEL"),
                "prompt_version": "v3",
                "timestamp": datetime.now().isoformat()
            },
            tags=["experiment", "hybrid", 'agent-playground'],
        )

    if config:
        return agent.invoke({"messages": [HumanMessage(content=message)]}, config=config)

    return agent.invoke({"messages": [HumanMessage(content=message)]})
