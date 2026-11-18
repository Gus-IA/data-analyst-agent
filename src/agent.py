"""Data analyst agent built with LangGraph: the LLM writes Python, the sandbox runs it."""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.errors import GraphRecursionError
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from sandbox import run_code
from tools import describe_dataset as summarize_csv
from tools import format_result
from groq import BadRequestError
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

SYSTEM_PROMPT = (
    "You are a data analyst. The user has uploaded a CSV file. "
    "Always call describe_dataset first to learn the real column names; never guess them. "
    "To analyze data or draw charts, write Python and run it with run_python. "
    "In the sandbox the CSV is at /data/data.csv, pandas, numpy and matplotlib are available, "
    "there is no internet, and charts must be saved as PNG files under /out. "
    "Use print() to show results. If the code fails, read the error and fix it. "
    "Finish with a short summary of the findings, based only on the outputs you saw."
    "When comparing variables measured on different scales, standardize them first "
    "(for example, difference of means divided by the pooled standard deviation) "
    "and say which method you used. "
    "Write tables in valid markdown, with each row on its own line."
)

MAX_TOOL_RETRIES = 2


def make_tools(data_file: str):
    """Create the tools bound to one CSV. `runs` records every sandbox execution."""
    runs = []

    @tool
    def describe_dataset() -> str:
        """Describe the uploaded CSV: columns, types, missing values, first rows and numeric summary."""
        try:
            return summarize_csv(data_file)
        except Exception as exc:  # return the error to the model instead of crashing
            return f"Could not read the dataset: {exc}"

    @tool
    def run_python(code: str) -> str:
        """Run Python code in an isolated sandbox without internet.
        The CSV is at /data/data.csv. Save charts as PNG files in /out. Use print() for results.
        """
        result = run_code(code, data_file)
        runs.append(result)
        return format_result(result)

    return [describe_dataset, run_python], runs


def build_agent(data_file: str):
    tools, runs = make_tools(data_file)
    llm = ChatGroq(model=os.getenv("LLM_MODEL", "openai/gpt-oss-120b"), temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    def call_model(state: MessagesState):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        for attempt in range(MAX_TOOL_RETRIES + 1):
            try:
                return {"messages": [llm_with_tools.invoke(messages)]}
            except BadRequestError as exc:
                # The model sometimes writes malformed tool-call JSON (unquoted code)
                if "tool_use_failed" not in str(exc) or attempt == MAX_TOOL_RETRIES:
                    raise
                messages = messages + [
                    HumanMessage(
                        content=(
                            "Your last tool call was rejected because its arguments were not "
                            "valid JSON. Call the tool again and pass the Python code as one "
                            "JSON string in the 'code' argument, escaping newlines and quotes."
                        )
                    )
                ]

    graph = StateGraph(MessagesState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent", tools_condition
    )  # tool call -> "tools", else END
    graph.add_edge("tools", "agent")
    return graph.compile(), runs


def ask(agent, question: str, max_steps: int = 12):
    """Run the agent with a hard limit on steps. Returns (answer, messages)."""
    try:
        state = agent.invoke(
            {"messages": [("user", question)]},
            config={"recursion_limit": max_steps},
        )
        messages = state["messages"]
        return messages[-1].content, messages
    except GraphRecursionError:
        return "Stopped: the agent reached the maximum number of steps.", []


if __name__ == "__main__":
    csv = str(Path(__file__).resolve().parent.parent / "data" / "sample.csv")
    agent, runs = build_agent(csv)
    answer, _ = ask(...)
    print(answer)
    print("\nSandbox runs:", len(runs))
    for r in runs:
        print(" -", r["out_dir"], r["files"])
