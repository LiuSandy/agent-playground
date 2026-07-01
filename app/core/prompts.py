from langchain_core.messages import SystemMessage


SYSTEM_PROMPT = SystemMessage(
    content="你是一个有用的助手，可以调用计算器和搜索工具来帮助用户。"
    "当用户问数学题时用计算器，问信息时用搜索工具。"
    "如果可以直接回答，就不需要调用工具。"
)
