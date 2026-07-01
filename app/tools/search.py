from langchain_core.tools import tool

@tool
def search_tool(query: str) -> str:
    """搜索信息。query 是要搜索的关键词或问题。"""
    mock_data = {
        "北京天气": "北京今天晴，气温 22°C ~ 32°C，空气质量良",
        "python star最多": "GitHub 上 star 最多的 Python 项目是 public-apis/public-apis，拥有超过 300k star",
        "langgraph": "LangGraph 是 LangChain 团队推出的图式 Agent 编排框架，支持有状态、多角色的 Agent 构建",
    }
    for key, value in mock_data.items():
        if key in query.lower():
            return value
    return f"未找到关于 '{query}' 的信息，请尝试更具体的关键词"