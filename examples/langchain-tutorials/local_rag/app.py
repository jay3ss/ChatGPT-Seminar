import pathlib

from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

EMBEDDING_MODEL = "nomic-embed-text"
INFERENCE_MODEL = "llama3.1:8b"


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


chromadb_path = pathlib.Path(".chroma/")

if not has_data(chromadb_path):
    print("No data in database. Downloading data and processing it...")
    print("Downloading data...")
    loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
    data = loader.load()

    print("Processing data...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(data)

    local_embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    print("Storing data...")
    vectorstore = Chroma.from_documents(
        documents=all_splits,
        embedding=local_embeddings,
        persist_directory=chromadb_path.name,
    )

else:
    print("Data present in database. Loading...")
    vectorstore = Chroma(
        persist_directory=chromadb_path.name,
        embedding_function=OllamaEmbeddings(model=EMBEDDING_MODEL),
    )
print("Done!")
model = ChatOllama(model=INFERENCE_MODEL, temperature=1)


def format_docs(docs: list) -> str:
    """
    Convert loaded documents into strings by concatenating their content
    and ignoring metadata
    """
    return "\n\n".join(doc.page_content for doc in docs)


RAG_TEMPLATE = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved \
context to answer the question. If you don't know the answer, just say that you don't know. \
Use thirty sentences maximum and keep the answer concise and, if appropriate, include code \
examples in Python 3.

<context>
{context}
</context>

Answer the following question:

{question}"""

rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)

retriever = vectorstore.as_retriever()

qa_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | model
    | StrOutputParser()
)

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "abc123"}}
    print("AI: How can I help you today?")
    while True:
        user_input = input(">>> ").strip().lower()
        if user_input in ["exit", "exit()", "quit", "quit()", "bye", "bye()"]:
            break

        response = qa_chain.invoke(user_input)
        print(response)
        # for chunk in qa_chain.stream(user_input, config):
        #     messages = get_message(chunk)

        #     for message in messages:
        #         print_message(message)

        print("----")
