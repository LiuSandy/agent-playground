from dotenv import load_dotenv
from langsmith import Client


DATASET_NAME = "hybrid-agent-regression-v1"

CASES = [
    {
        "inputs": {"question": "帮我计算 15 + 23 × 7"},
        "outputs": {
            "answer_keywords": ["176"],
            "expected_mode": "reactive",
            "expected_tools": ["calculator_tool"],
        },
    },
    {
        "inputs": {"question": "帮我计算 10 ÷ 0，并说明结果"},
        "outputs": {
            "answer_keywords": ["零"],
            "expected_mode": "reactive",
            "expected_tools": ["calculator_tool"],
        },
    },
    {
        "inputs": {"question": "介绍一下 LangGraph"},
        "outputs": {
            "answer_keywords": ["LangGraph", "Agent"],
            "expected_mode": "reactive",
            "expected_tools": ["search_tool"],
        },
    },
    {
        "inputs": {"question": "请只回复：你好"},
        "outputs": {
            "answer_keywords": ["你好"],
            "expected_mode": "reactive",
            "expected_tools": [],
        },
    },
    {
        "inputs": {"question": "调研并比较 Django 和 Flask 的性能差异"},
        "outputs": {
            "answer_keywords": ["Django", "Flask", "性能"],
            "expected_mode": "deliberative",
            "expected_tools": ["search_tool"],
        },
    },
    {
        "inputs": {"question": "调研 Django、Flask 和 FastAPI，并给出选型建议"},
        "outputs": {
            "answer_keywords": ["Django", "Flask", "FastAPI"],
            "expected_mode": "deliberative",
            "expected_tools": ["search_tool"],
        },
    },
    {
        "inputs": {"question": "搜索量子烤面包机的可靠资料"},
        "outputs": {
            "answer_keywords": ["未找到"],
            "expected_mode": "reactive",
            "expected_tools": ["search_tool"],
        },
    },
]


def create_dataset(client: Client | None = None):
    """Create the versioned regression dataset in LangSmith."""
    if client is None:
        load_dotenv()
        langsmith_client = Client()
    else:
        langsmith_client = client
    dataset = langsmith_client.create_dataset(
        dataset_name=DATASET_NAME,
        description="Hybrid Agent 的路由、工具与答案回归测试",
    )
    langsmith_client.create_examples(
        dataset_id=dataset.id,
        examples=CASES,
    )
    return dataset


def main() -> None:
    dataset = create_dataset()
    print(f"Created dataset: {dataset.name}")


if __name__ == "__main__":
    main()
