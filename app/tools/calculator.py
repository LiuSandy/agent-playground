import math
from langchain_core.tools import tool

@tool
def calculator_tool(expression: str) -> str:
    """执行数学计算。expression 是一个数学表达式，例如 '3 + 5 * 2' 或 'sqrt(16)'。"""
    try:
        allowed_names = {
            k: v for k, v in math.__dict__.items()
            if not k.startswith("__")
        }
        allowed_names["__builtins__"] = {}
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算出错: {str(e)}"

