import pathlib
from operator import itemgetter

from dotenv import load_dotenv
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

load_dotenv()

MODEL = "gpt-4o-mini"


def get_messages(chunk: dict) -> list:
    return chunk.get("agent", {}).get("messages") or chunk.get("tools", {}).get(
        "messages"
    )


def print_messages(messages: list) -> None:
    for message in messages:
        print_message(message)


def print_message(message) -> None:
    if message.content:
        if isinstance(message, AIMessage):
            print("AI:", message.content)
            print("----")
        # elif isinstance(message, ToolMessage):
        #     print("Tool:", message.content)
        #     print("----")
        elif isinstance(message, HumanMessage):
            print(">>> :", message.content)
            print("----")


def has_data(path: pathlib.Path) -> bool:
    return path.exists() and any(path.iterdir())


llm = ChatOpenAI(model="gpt-4o-mini")
db = SQLDatabase.from_uri("sqlite:///Chinook.db")

toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()


SQL_PREFIX = """You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct SQLite query to, run then look at the results of the query and return the answer
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the below tools. Only use the information returned by the below tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP, etc.) to the database.

To start you should ALWAYS look at the tables in the database to see what you can query.
Do NOT skip this step.
Then you should query the schema of the most relevant tables."""

system_message = SystemMessage(content=SQL_PREFIX)
agent_executor = create_react_agent(llm, tools, messages_modifier=system_message)

for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="Which artist made the most money?")]}
):
    messages = get_messages(chunk)
    print_messages(messages)
