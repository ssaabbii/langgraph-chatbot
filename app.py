from typing import Dict, List, TypedDict, Annotated, Literal
from datetime import datetime, timezone
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages 

load_dotenv()

@tool
def get_current_time() -> Dict:
    """Return the current UTC time in ISO-8601 format.
    Example → {"utc": "2025-05-21T06:42:00Z"}"""
    current_utc_time = datetime.now(timezone.utc).isoformat(timespec='seconds') + 'Z'
    return {"utc": current_utc_time}


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
).bind_tools([get_current_time])


prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the provided tools when needed."),
    ("placeholder", "{messages}"),
])


def call_model(state: AgentState) -> Dict:
    """
    This node invokes the LLM with the current messages from the state.
    It returns the AI's response, which might include tool calls.
    """
    messages = state["messages"]
    chain = prompt | llm
    response = chain.invoke({"messages": messages})
    return {"messages": [response]}

def call_tool(state: AgentState) -> Dict:
    """
    This node executes the tool calls suggested by the LLM.
    It iterates through all tool calls in the last AI message and appends
    ToolMessage objects with the results to the state.
    """
    last_message = state["messages"][-1]
    tool_messages = []
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            if tool_name == "get_current_time":
                result = get_current_time.invoke(tool_args)
                tool_messages.append(
                    ToolMessage(
                        content=str(result),
                        tool_call_id=tool_call["id"],
                    )
                )
            else:
                tool_messages.append(
                    ToolMessage(
                        content=f"Error: Tool '{tool_name}' not found.",
                        tool_call_id=tool_call["id"],
                    )
                )
    return {"messages": tool_messages}


def should_continue(state: AgentState) -> Literal["continue", "end"]:
    """
    Determines if the graph should continue processing (e.g., after a tool call)
    or end (if the LLM has provided a final answer).
    """
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "continue"
    elif isinstance(last_message, ToolMessage):
        return "continue"
    else:
        return "end"


workflow = StateGraph(AgentState)

workflow.add_node("llm_node", call_model)
workflow.add_node("tool_node", call_tool)

workflow.set_entry_point("llm_node")

workflow.add_conditional_edges(
    "llm_node",
    should_continue,
    {
        "continue": "tool_node",
        "end": END
    }
)

workflow.add_edge("tool_node", "llm_node")

graph = workflow.compile()

if __name__ == "__main__":
    from langgraph.cli import serve
    serve(graph)
