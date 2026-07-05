import json

from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.constants import END
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode
from langsmith import traceable

from app.core.prompts import (
    SYSTEM_PROMPT,
    PLANNER_PROMPT,
    EXECUTOR_PROMPT,
    REPLANNER_PROMPT,
)
from app.core.state import AgentState, DeliberativeState, HybridState


# 01 Reactive mode
def build_reactive_graph(llm, tools, max_steps=10):
    llm_with_tools = llm.bind_tools(tools)

    def call_model(state: AgentState) -> dict:
        response = llm_with_tools.invoke(state["messages"])
        step = state.get("step_count", 0) + 1
        print(f"\n--- Step {step} ---")
        if hasattr(response, "tool_calls") and response.tool_calls:
            for tc in response.tool_calls:
                print(f"  Action: {tc['name']}({tc['args']})")
        else:
            print(f"  Response: {response.content[:100]}...")
        return {"messages": [response], "step_count": step}

    def should_continue(state: AgentState) -> str:
        last_message = state["messages"][-1]
        if state.get("step_count", 0) >= max_steps:
            return END
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "call_tool"
        return END

    def observe_result(state: AgentState) -> dict:
        last_message = state["messages"][-1]
        print(f"  Observation: {last_message.content[:100]}")
        return {}

    workflow = StateGraph(AgentState)
    workflow.add_node("call_model", call_model)
    workflow.add_node("call_tool", ToolNode(tools))
    workflow.add_node("observe", observe_result)

    workflow.add_edge(START, "call_model")
    workflow.add_conditional_edges("call_model", should_continue, {
        "call_tool": "call_tool",
        END: END,
    })
    workflow.add_edge("call_tool", "observe")
    workflow.add_edge("observe", "call_model")

    return workflow.compile()

# Shared deliberative helpers
def _parse_plan(content: str) -> list[str]:
    if "```" in content:
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    try:
        plan = json.loads(content)
        if isinstance(plan, list):
            return [str(step).strip() for step in plan if str(step).strip()]
    except json.JSONDecodeError:
        pass
    return [
        line.strip("- ").strip()
        for line in content.strip().split("\n")
        if line.strip()
    ]


def _current_tool_exchange(messages):
    for idx in range(len(messages) - 1, -1, -1):
        message = messages[idx]
        if hasattr(message, "tool_calls") and message.tool_calls:
            return messages[idx:]
    return []


def _current_tool_observations(messages) -> list[str]:
    return [
        message.content
        for message in _current_tool_exchange(messages)
        if isinstance(message, ToolMessage)
    ]


def _create_deliberative_nodes(plan_llm, llm_with_tools, max_tool_rounds_per_step):
    @traceable(run_type="chain", name="parse_plan")
    def planner(state) -> dict:
        user_msg = state["messages"][-1]
        response = plan_llm.invoke([PLANNER_PROMPT, user_msg])
        plan = _parse_plan(response.content)
        if not plan:
            plan = [user_msg.content]
        print(f"\n[Plan] {' → '.join(plan)}")
        return {
            "plan": plan,
            "current_step": 0,
            "step_results": {},
            "original_question": user_msg.content,
            "tool_rounds": 0,
        }

    def executor(state) -> dict:
        plan = state.get("plan", [])
        step_idx = state.get("current_step", 0)
        step_desc = plan[step_idx]
        print(f"\n[Execute] Step {step_idx + 1}/{len(plan)}: {step_desc}")

        previous_results = "\n".join(
            f"{plan[idx]}: {result}"
            for idx, result in sorted(state.get("step_results", {}).items())
        ) or "暂无"
        context_msg = HumanMessage(content=(
            f"用户原始问题：{state.get('original_question', '')}\n\n"
            f"已完成步骤结果：\n{previous_results}"
        ))
        task_msg = HumanMessage(content=f"请完成以下任务步骤：{step_desc}")
        messages = [EXECUTOR_PROMPT, context_msg, task_msg]
        if isinstance(state["messages"][-1], ToolMessage):
            messages.extend(_current_tool_exchange(state["messages"]))

        response = llm_with_tools.invoke(messages)
        if hasattr(response, "tool_calls") and response.tool_calls:
            for tool_call in response.tool_calls:
                print(f"  Action: {tool_call['name']}({tool_call['args']})")
        else:
            print(f"  Result: {response.content[:100]}")
        return {"messages": [response]}

    def observe_tool_result(state) -> dict:
        tool_messages = [
            message for message in state["messages"]
            if isinstance(message, ToolMessage)
        ]
        if tool_messages:
            observations = [message.content for message in tool_messages[-2:]]
            print(f"  Observation: {' | '.join(observations)[:160]}")
        return {"tool_rounds": state.get("tool_rounds", 0) + 1}

    def record_step_result(state) -> dict:
        step_idx = state.get("current_step", 0)
        response = state["messages"][-1]
        new_results = dict(state.get("step_results", {}))
        if isinstance(response, ToolMessage):
            observations = _current_tool_observations(state["messages"])
            content = "\n".join(observations) or response.content
            print(f"  Result: {content[:100]}")
        else:
            content = response.content
        new_results[step_idx] = content
        return {
            "step_results": new_results,
            "current_step": step_idx + 1,
            "tool_rounds": 0,
        }

    def replanner(state) -> dict:
        plan = state.get("plan", [])
        current_step = state.get("current_step", 0)
        if current_step >= len(plan):
            return {}

        remaining = plan[current_step:]
        results_str = "\n".join(
            f"Step {idx + 1}: {result}"
            for idx, result in state.get("step_results", {}).items()
        )
        prompt = HumanMessage(content=REPLANNER_PROMPT.content.format(
            step_results=results_str,
            remaining_steps="\n".join(remaining),
        ))
        response = plan_llm.invoke([prompt])
        new_plan = _parse_plan(response.content)
        if new_plan and new_plan != remaining:
            print(f"\n[Replan] 调整计划: {' → '.join(new_plan)}")
            return {"plan": plan[:current_step] + new_plan}
        return {}

    def should_continue_after_planning(state) -> str:
        if state.get("current_step", 0) >= len(state.get("plan", [])):
            return "summarize"
        return "executor"

    def should_continue_after_executor(state) -> str:
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "call_tool"
        return "record_step_result"

    def should_continue_after_observation(state) -> str:
        if state.get("tool_rounds", 0) >= max_tool_rounds_per_step:
            print("  Tool limit reached for this step; using current observations.")
            return "record_step_result"
        return "executor"

    def should_continue_after_step(state) -> str:
        current_step = state.get("current_step", 0)
        if current_step >= len(state.get("plan", [])):
            return "summarize"
        if current_step > 0 and current_step % 3 == 0:
            return "replanner"
        return "executor"

    def summarize(state) -> dict:
        plan = state.get("plan", [])
        results_str = "\n".join(
            f"{plan[idx]}: {result}"
            for idx, result in sorted(state.get("step_results", {}).items())
        )
        summary_prompt = HumanMessage(content=(
            "请根据以下执行结果，生成最终回答：\n\n" + results_str
        ))
        response = plan_llm.invoke([summary_prompt])
        print(f"\n[Final] {response.content[:100]}")
        return {"messages": [response]}

    return {
        "planner": planner,
        "executor": executor,
        "observe_tool_result": observe_tool_result,
        "record_step_result": record_step_result,
        "replanner": replanner,
        "summarize": summarize,
        "should_continue_after_planning": should_continue_after_planning,
        "should_continue_after_executor": should_continue_after_executor,
        "should_continue_after_observation": should_continue_after_observation,
        "should_continue_after_step": should_continue_after_step,
    }


def _add_deliberative_nodes(workflow, nodes, tools):
    workflow.add_node("planner", nodes["planner"])
    workflow.add_node("executor", nodes["executor"])
    workflow.add_node("call_tool", ToolNode(tools))
    workflow.add_node("observe_tool_result", nodes["observe_tool_result"])
    workflow.add_node("record_step_result", nodes["record_step_result"])
    workflow.add_node("replanner", nodes["replanner"])
    workflow.add_node("summarize", nodes["summarize"])


def _add_deliberative_edges(workflow, nodes):
    workflow.add_conditional_edges("planner", nodes["should_continue_after_planning"], {
        "executor": "executor",
        "summarize": "summarize",
    })
    workflow.add_conditional_edges("executor", nodes["should_continue_after_executor"], {
        "call_tool": "call_tool",
        "record_step_result": "record_step_result",
    })
    workflow.add_edge("call_tool", "observe_tool_result")
    workflow.add_conditional_edges(
        "observe_tool_result",
        nodes["should_continue_after_observation"],
        {
            "executor": "executor",
            "record_step_result": "record_step_result",
        },
    )
    workflow.add_conditional_edges("record_step_result", nodes["should_continue_after_step"], {
        "executor": "executor",
        "replanner": "replanner",
        "summarize": "summarize",
    })
    workflow.add_edge("replanner", "executor")
    workflow.add_edge("summarize", END)


# 02 Deliberative mode
def build_deliberative_graph(plan_llm, execute_llm, tools, max_tool_rounds_per_step=2):
    nodes = _create_deliberative_nodes(
        plan_llm,
        execute_llm.bind_tools(tools),
        max_tool_rounds_per_step,
    )
    workflow = StateGraph(DeliberativeState)
    _add_deliberative_nodes(workflow, nodes, tools)
    workflow.add_edge(START, "planner")
    _add_deliberative_edges(workflow, nodes)
    return workflow.compile()


# 03 Hybrid mode
def build_hybrid_graph(plan_llm, execute_llm, tools, max_steps=10, max_tool_rounds_per_step=2):
    reactive_llm_with_tools = execute_llm.bind_tools(tools)
    deliberative_nodes = _create_deliberative_nodes(
        plan_llm,
        execute_llm.bind_tools(tools),
        max_tool_rounds_per_step,
    )

    def router(state: HybridState) -> dict:
        user_msg = state["messages"][-1]
        content = user_msg.content.lower()
        complex_markers = [
            "调研", "研究", "分析", "比较", "对比", "规划", "计划",
            "总结", "多步", "步骤", "方案", "benchmark", "performance",
            "compare", "research", "analyze",
        ]
        is_complex = len(user_msg.content) > 40 or any(
            marker in content for marker in complex_markers
        )
        mode = "deliberative" if is_complex else "reactive"
        print(f"\n[Hybrid] Route: {mode}")
        return {"mode": mode, "original_question": user_msg.content}

    def choose_mode(state: HybridState) -> str:
        return state.get("mode", "reactive")

    def reactive_call_model(state: HybridState) -> dict:
        response = reactive_llm_with_tools.invoke([SYSTEM_PROMPT] + state["messages"])
        step = state.get("step_count", 0) + 1
        print(f"\n[Reactive] Step {step}")
        if hasattr(response, "tool_calls") and response.tool_calls:
            for tool_call in response.tool_calls:
                print(f"  Action: {tool_call['name']}({tool_call['args']})")
        else:
            print(f"  Response: {response.content[:100]}...")
        return {"messages": [response], "step_count": step}

    def reactive_observe(state: HybridState) -> dict:
        last_message = state["messages"][-1]
        print(f"  Observation: {last_message.content[:100]}")
        return {}

    def should_continue_reactive(state: HybridState) -> str:
        last_message = state["messages"][-1]
        if state.get("step_count", 0) >= max_steps:
            return END
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "reactive_call_tool"
        return END

    workflow = StateGraph(HybridState)
    workflow.add_node("router", router)
    workflow.add_node("reactive_call_model", reactive_call_model)
    workflow.add_node("reactive_call_tool", ToolNode(tools))
    workflow.add_node("reactive_observe", reactive_observe)
    _add_deliberative_nodes(workflow, deliberative_nodes, tools)

    workflow.add_edge(START, "router")
    workflow.add_conditional_edges("router", choose_mode, {
        "reactive": "reactive_call_model",
        "deliberative": "planner",
    })
    workflow.add_conditional_edges("reactive_call_model", should_continue_reactive, {
        "reactive_call_tool": "reactive_call_tool",
        END: END,
    })
    workflow.add_edge("reactive_call_tool", "reactive_observe")
    workflow.add_edge("reactive_observe", "reactive_call_model")
    _add_deliberative_edges(workflow, deliberative_nodes)

    return workflow.compile()
