
# agent-playground

这是一个记录我学习 LangGraph 过程的项目。这个过程由博客文章和代码两部分组成：博客用于记录概念理解、实践笔记和阶段性总结，代码用于把学习内容落到可运行的示例里。

博客地址：[https://lius-node.vercel.app/blogs](https://lius-node.vercel.app/blogs)

当前代码演示了一个基于 LangGraph + LangChain 的轻量 Agent，重点包括模型调用、工具绑定、工具路由、状态流转和终端交互。

## 项目结构

```text
app/
├── cli.py              # 终端入口逻辑
├── config.py           # 环境变量配置
├── core/
│   ├── agent.py        # Agent 创建与运行封装
│   ├── graph.py        # LangGraph workflow
│   ├── llm.py          # LLM 初始化
│   ├── prompts.py      # System Prompt
│   ├── state.py        # AgentState 定义
│   └── tools.py        # 工具注册
├── evaluation/
│   ├── target.py       # LangSmith Dataset 到 Agent State 的适配层
│   └── evaluators.py   # 答案、路由和工具评估器
└── tools/
    ├── calculator.py   # 计算器工具
    └── search.py       # 搜索工具示例

scripts/
├── create_dataset.py   # 创建版本化 LangSmith Dataset
└── run_evaluation.py   # 运行 Experiment 并检查质量门禁

tests/                  # 不依赖真实模型 API 的确定性单元测试
```

## 环境要求

- Python `>=3.11.7`
- `uv`
- DeepSeek 或其他 OpenAI-compatible API Key

安装依赖：

```bash
uv sync
```

创建本地 `.env` 文件：

```env
DEEPSEEK_MODEL=deepseek-v4-pro
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_API_KEY=your_api_key_here

LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_PROJECT=agent-playground
```

## 运行示例

交互式输入：

```bash
uv run python main.py
```

## LangSmith 自动化评估

首次运行前，在 LangSmith 中创建回归测试数据集：

```bash
uv run python -m scripts.create_dataset
```

运行 Hybrid Agent 的答案、路由和工具评估，并检查质量阈值：

```bash
uv run python -m scripts.run_evaluation
```

本地单元测试不会调用真实模型或 LangSmith API：

```bash
uv run pytest
```


## 说明

本仓库会随着 LangGraph 学习过程持续调整，因此部分能力保持为教学示例实现，例如 `search_tool` 使用 mock 数据。若要接入真实搜索、数据库或外部 API，建议先保持工具接口稳定，再替换内部实现。
