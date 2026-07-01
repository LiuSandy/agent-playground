from app.tools.calculator import calculator_tool
from app.tools.search import search_tool


def get_tools():
    return [calculator_tool, search_tool]


tools = get_tools()
