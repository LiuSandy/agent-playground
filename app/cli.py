import sys

from app.core.agent import run_agent


def print_message(msg) -> None:
    if msg.type == "ai" and getattr(msg, "tool_calls", None):
        print(f"[{msg.type}] {msg.content}")
        for tool_call in msg.tool_calls:
            name = tool_call.get("name", "unknown_tool")
            args = tool_call.get("args", {})
            print(f"[tool_call:{name}] args={args}")
        return

    if msg.type == "tool":
        tool_name = getattr(msg, "name", "unknown_tool")
        print(f"[tool:{tool_name}] {msg.content}")
        return

    print(f"[{msg.type}] {msg.content}")


def get_user_message() -> str:
    if len(sys.argv) > 1:
        return " ".join(sys.argv[1:]).strip()

    return input("请输入问题: ").strip()


def main() -> None:
    message = get_user_message()
    if not message:
        print("问题不能为空。")
        return

    result = run_agent(message)

    for msg in result["messages"]:
        print_message(msg)
