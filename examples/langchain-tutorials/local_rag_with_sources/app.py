import pathlib
import pprint
import time
from typing import List, Union

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypedDict

EMBEDDING_MODEL = "nomic-embed-text"
INFERENCE_MODEL = "llama3.1:8b"

MessageType = Union[AIMessage, HumanMessage, ToolMessage]


def get_message(chunk: dict) -> List:
    return chunk.get("agent", {}).get("messages") or chunk.get("tools", {}).get(
        "messages"
    )


def print_messages(messages: list) -> None:
    for message in messages:
        print_message(message)


def print_message(message: MessageType) -> None:
    if message.content:
        if isinstance(message, AIMessage) or isinstance(message, ToolMessage):
            print("AI:", message.content)
        elif isinstance(message, HumanMessage):
            print(">>> :", message.content)


def has_data(path: pathlib.Path) -> bool:
    return path.exists() and any(path.iterdir())


chromadb_path = pathlib.Path(".chroma/")


def load_data(path: pathlib.Path) -> List[Document]:
    return PyPDFDirectoryLoader(path).load()


def split_documents(
    docs: List[Document], batch_size: int = 5, verbose: bool = False
) -> str:
    semantic_chunker = SemanticChunker(
        OllamaEmbeddings(model=EMBEDDING_MODEL),
        breakpoint_threshold_type="interquartile",
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1_000, chunk_overlap=200)

    all_chunks = []
    total_docs = len(docs)
    start_time = time.perf_counter()
    for i in range(0, total_docs, batch_size):
        batch = docs[i : i + batch_size]
        combined_text = "\n".join([doc.page_content for doc in batch])
        semantic_chunks = semantic_chunker.create_documents([combined_text])

        split_chunks = [
            sub_chunk
            for chunk in semantic_chunks
            for sub_chunk in text_splitter.split_documents([chunk])
        ]

        all_chunks.extend(split_chunks)

        if verbose:
            processed_docs = min(i + batch_size, total_docs)
            elapsed_time = time.perf_counter() - start_time
            percent_processed = processed_docs / total_docs
            eta = elapsed_time / percent_processed
            percent_processed *= 100
            print(
                (
                    f"\rProcessed {percent_processed:.4f}% docs "
                    f"and {elapsed_time:.2f}s have elapsed. "
                    f"ETA: {eta:.2f}s"
                ),
                end="",
                flush=True,
            )
    if verbose:
        print()
    return all_chunks


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# class Source(BaseModel):
#     """Information about a source"""

#     author: str = Field(..., description="The name of author of the source.")
#     title: str = Field(..., description="The title of the source.")
#     year: int = Field(..., description="The year that the source was published.")


class Source(TypedDict):
    """Information about a source"""

    author: Annotated[str, ..., "The author of the source"]
    title: Annotated[str, ..., "The title of the source"]
    year: Annotated[str, ..., "The year the source was published"]


class AnswerWithSources(TypedDict):
    """An answer with sources"""

    # answer: str = Field(..., description="The answer to the user's question")
    answer: str
    sources: Annotated[
        List[Source],
        ...,
        "List of sources (author + year + title) used to answer the question",
    ]
    # sources: List[Source] = Field(
    #     ...,
    #     description=(
    #         "List of sources used to answer the question. "
    #         "Populate each source with (author + title + year))"
    #     ),
    # )


# class AnswerWithSources(TypedDict):
#     """An answer to the question, with sources"""

#     answer: str
#     sources: Annotated[
#         List[str],
#         ...,
#         "List of sources (author + year ) used to answer the question",
#     ]


if __name__ == "__main__":
    if not has_data(chromadb_path):
        print("No data in database. Loading data and processing it...")
        pdf_path = pathlib.Path(
            "/home/josechung/study/school/gatech/chappity/readings/papers"
        )
        print("Loading data...", end=" ")
        docs = load_data(pdf_path)

        print("Done!\nProcessing documents...", end=" ")
        split_docs = split_documents(docs, batch_size=15, verbose=True)

        print("Done!\nCreating vector store...")
        vector_store = Chroma.from_documents(
            documents=split_docs,
            embedding=OllamaEmbeddings(model=EMBEDDING_MODEL),
            persist_directory=chromadb_path.name,
        )
    else:
        print("Data present in database. Loading...")
        vectorstore = Chroma(
            persist_directory=chromadb_path.name,
            embedding_function=OllamaEmbeddings(model=EMBEDDING_MODEL),
        )

    retriever = vectorstore.as_retriever()

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question thoroughly. Provide clear explanations, detailed examples, "
        "and if applicable, relevant code snippets in Python 3. "
        "Aim for a comprehensive answer that covers various aspects of the topic."
        "\n\n"
        "{context}"
    )
    llm = ChatOllama(model=INFERENCE_MODEL, temperature=0)

    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "{input}")]
    )
    # question_answer_chain = create_stuff_documents_chain(llm, prompt)
    # rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    rag_chain_from_docs = (
        {
            "input": lambda x: x["input"],
            "context": lambda x: format_docs(x["context"]),
        }
        | prompt
        # | llm.with_structured_output(AnswerWithSources)
        | llm
    )

    retrieve_docs = (lambda x: x["input"]) | retriever
    chain = RunnablePassthrough.assign(context=retrieve_docs).assign(
        answer=rag_chain_from_docs
    )

    print("AI: How can I help you today?")
    while True:
        user_input = input(">>> ").strip().lower()
        if user_input in ["exit", "exit()", "quit", "quit()", "bye", "bye()"]:
            break

        if user_input:
            try:
                response = chain.invoke({"input": user_input})
                sources_response = chain.invoke(
                    {
                        "input": (
                            "Based on the previous answer, can you list the "
                            "sources you used, with the author, title, and year?"
                        )
                    }
                )

                # for chunk in chain.stream(user_input):
                #     messages = get_message(chunk)
                #     print_messages(messages)
                # pprint.pprint(response, indent=4)
                # print(response["answer"]["answer"])
                # print("Sources:", response["answer"]["sources"])
                print(response["answer"].content)
            except Exception as e:
                print("Error invoking the chain:", e)
                # Optionally inspect the exception details
                if hasattr(e, "errors"):
                    print("Validation errors:", e.errors())
        else:
            print("AI: I didn't understand your question. Please try again.")

# How can I leverage Chain of Thought in my LLM-based applications to give the LLM better reasoning? Please give me a couple sample prompts
