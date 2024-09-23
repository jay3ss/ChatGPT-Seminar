import os
import pathlib

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()


MODEL = "text-embedding-ada-002"


def srt_to_text(file: pathlib.Path) -> str:
    text = []
    with file.open("r", encoding="utf-8") as f:
        for line in f.readlines():
            stripped_line = line.strip()
            should_append = stripped_line and not (
                "-->" in stripped_line or stripped_line.isdigit()
            )
            if should_append:
                text.append(stripped_line)

    return "\n".join(text)


def get_all_files_of_ext(path: str, reverse: bool = True, ext: str = "srt"):
    for file_path in sorted(pathlib.Path(path).rglob(f"*.{ext}"), reverse=reverse):
        yield file_path


def has_data(path: pathlib.Path) -> bool:
    return path.exists() and any(path.iterdir())


message = """
Answer this question using the provided context only.

{question}

Context:
{context}
"""

if __name__ == "__main__":
    print("Welcome to the simple RAG App")
    path = "/home/josechung/Videos"
    chromadb_path = pathlib.Path(".chroma/")

    if not has_data(chromadb_path):
        print("Generating embeddings")
        documents = [
            Document(
                page_content=srt_to_text(file),
                metadata={"source": str(file.absolute())},
            )
            for file in get_all_files_of_ext(path)
        ]

        vectorstore = Chroma.from_documents(
            documents, embedding=OpenAIEmbeddings(), persist_directory=".chroma/"
        )
    else:
        print("Looks like we've already generated the embeddings. Let's use those!")
        vectorstore = Chroma(
            persist_directory=chromadb_path.name,
            embedding_function=OpenAIEmbeddings(model=MODEL),
        )
    # retriever = RunnableLambda(vectorstore.similarity_search).bind(k=1)
    #                   OR
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 1}
    )

    llm = ChatOpenAI(verbose=False)
    prompt = ChatPromptTemplate.from_messages([("human", message)])

    rag_chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm

    print("AI: how can I help you today?")
    while True:
        user_input = input(">>> ").strip().lower()
        if user_input in ["exit", "exit()", "quit", "quit()", "bye", "bye()"]:
            break
        response = rag_chain.invoke(user_input)
        print("AI: ", end="")
        print(response.content)
