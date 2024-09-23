import pathlib

import bs4
from dotenv import load_dotenv
from langchain.tools.retriever import create_retriever_tool
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

load_dotenv()

MODEL = "gpt-4o-mini"


def get_message(chunk: dict) -> list:
    return chunk.get("agent", {}).get("messages") or chunk.get("tools", {}).get(
        "messages"
    )


def print_message(message) -> None:
    if message.content:
        if isinstance(message, AIMessage) or isinstance(message, ToolMessage):
            print("AI:", message.content)
        elif isinstance(message, HumanMessage):
            print(">>> :", message.content)


def has_data(path: pathlib.Path) -> bool:
    return path.exists() and any(path.iterdir())


config = {"configurable": {"thread_id": "abc123"}}

if __name__ == "__main__":
    memory = MemorySaver()
    llm = ChatOpenAI(model=MODEL, temperature=0.01)

    # construct retriever
    chromadb_path = pathlib.Path(".chroma/")
    if not has_data(chromadb_path):
        loader = WebBaseLoader(
            web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    class_=("post-content", "post-title", "post-header")
                )
            ),
        )
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        splits = text_splitter.split_documents(docs)
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=OpenAIEmbeddings(),
            persist_directory=chromadb_path.name,
        )
    else:
        vectorstore = Chroma(
            persist_directory=chromadb_path.name,
            embedding_function=OpenAIEmbeddings(model=MODEL),
        )

    # build retriever tool
    retriever = vectorstore.as_retriever()
    tool = create_retriever_tool(
        retriever,
        "blog_post_retriever",
        "Searches and returns excerpts from the Autonomous Agents blog post.",
    )
    tools = [tool]

    agent_executor = create_react_agent(llm, tools, checkpointer=memory)
    print("AI: How can I help you today?")
    while True:
        user_input = input(">>> ").strip().lower()
        if user_input in ["exit", "exit()", "quit", "quit()", "bye", "bye()"]:
            break

        for chunk in agent_executor.stream(
            {"messages": [HumanMessage(content=user_input)]}, config
        ):
            messages = get_message(chunk)

            for message in messages:
                print_message(message)

        print("----")
