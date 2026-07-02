from langchain_core.tools import tool

@tool
def search_tool(query: str) -> str:
    """搜索信息。query 是要搜索的关键词或问题。"""
    mock_data = {
        "北京天气": "北京今天晴，气温 22°C ~ 32°C，空气质量良。",
        "上海天气": "上海今天多云，气温 25°C ~ 31°C，午后可能有阵雨。",
        "python star最多": "GitHub 上 star 较多的 Python 项目包括 public-apis、system-design-primer 和 awesome-python。",
        "langgraph": "LangGraph 是 LangChain 团队推出的图式 Agent 编排框架，支持有状态、多节点、多角色的 Agent 构建。",
        "deliberative agent": "Deliberative Agent 通常先规划任务，再执行步骤，并根据观察结果反思或重规划。",
        "reactive agent": "Reactive Agent 更强调基于当前观察即时选择动作，通常是 think/action/observation 的短循环。",
        "django 性能": "Django 功能完整、生态成熟，内置 ORM、Admin 和中间件；性能通常足够稳定，但框架层较重。",
        "django performance": "Django 性能特点：框架功能完整但请求链路较重；实际吞吐常受 ORM 查询、模板渲染、中间件和数据库访问影响。",
        "django benchmark": "Django benchmark mock：在纯 JSON/hello-world 场景下通常不是 Python 框架里最高吞吐；在完整业务场景中胜在稳定性和工程能力。",
        "django orm": "Django ORM 会带来便利性和一定抽象开销；性能优化常见手段包括 select_related、prefetch_related、索引优化和减少 N+1 查询。",
        "flask 性能": "Flask 核心轻量、启动和请求链路较短；复杂项目中性能更多取决于扩展、数据库访问和部署方式。",
        "flask performance": "Flask 性能特点：核心轻量，请求处理路径短；大型项目加入 ORM、认证、中间件后，与 Django 的差异会缩小。",
        "fastapi 性能": "FastAPI 基于 ASGI，原生支持异步，常见 I/O 密集接口场景下吞吐表现较好。",
        "fastapi performance": "FastAPI 性能特点：ASGI 与 async 友好，适合 I/O 密集 API；同步数据库调用仍可能成为瓶颈。",
        "techempower": "TechEmpower benchmarks mock：不同轮次、测试类型和实现方式差异很大，适合看趋势，不应直接等同真实业务性能。",
        "framework benchmarks": "框架 benchmark mock：hello-world、JSON serialization、database query 等测试项结论不同，不能只看单一 RPS。",
        "requests per second": "RPS/latency mock：基准测试里的 RPS 受硬件、worker 数、数据库、序列化、网络和测试类型影响，横向比较要看同一测试条件。",
        "rps latency": "RPS 与 latency mock：高 RPS 不代表低尾延迟；比较框架时应同时关注 p95/p99 latency、吞吐和资源占用。",
        "python web framework benchmark": "Python Web framework benchmark mock：FastAPI/Starlette 常在异步 API 场景表现更高；Flask 较轻量；Django 更偏全栈工程效率。",
        "django vs flask vs fastapi": "Django vs Flask vs FastAPI mock：Django 全栈能力强但框架较重；Flask 简洁轻量；FastAPI 在异步 API 和类型驱动开发上更有优势。",
        "performance comparison": "性能比较 mock：真实业务中数据库、缓存、序列化、部署参数和代码质量往往比框架本身更决定最终性能。",
        "deepseek": "DeepSeek 提供 OpenAI-compatible API，可通过 base_url、api_key 和模型名接入 LangChain ChatOpenAI。",
        "openai compatible": "OpenAI-compatible 接口通常复用 OpenAI SDK/协议，但需要配置兼容服务的 base_url 和模型名称。",
    }
    normalized_query = query.lower()
    matches = [
        value
        for key, value in mock_data.items()
        if key in normalized_query
    ]
    if matches:
        return "\n".join(matches)
    return f"未找到关于 '{query}' 的信息，请尝试更具体的关键词"
