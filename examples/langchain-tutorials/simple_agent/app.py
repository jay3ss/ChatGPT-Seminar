from typing import Any

from dotenv import load_dotenv
from duckduckgo_search import DDGS
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

load_dotenv()


memory = MemorySaver()
model = ChatOpenAI(model="gpt-3.5-turbo", stream_usage=True)
search = DuckDuckGoSearchRun()
tools = [search]
model.bind_tools(tools)
agent_executor = create_react_agent(model, tools, checkpointer=memory)


def get_message(chunk: dict) -> list:
    return chunk.get("agent", {}).get("messages") or chunk.get("tools", {}).get(
        "messages"
    )


def print_message(message) -> None:
    if message.content:
        if isinstance(message, AIMessage) or isinstance(message, ToolMessage):
            print("AI:", message.content)
        elif isinstance(message, HumanMessage):
            print("Human:", message.content)


config = {"configurable": {"thread_id": "abc123"}}
for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="hi im bob! i live in sf")]}, config
):
    messages = get_message(chunk)

    for message in messages:
        print_message(message)

    print("----")

for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="whats the weather where I live?")]}, config
):
    messages = get_message(chunk)

    for message in messages:
        print_message(message)

    print("----")
