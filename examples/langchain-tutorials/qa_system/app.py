import datetime
import pathlib
import pprint
from typing import List, Optional

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import YoutubeLoader
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

load_dotenv()

chromadb_path = pathlib.Path(".chroma/")
EMBEDDING_MODEL = "text-embedding-3-small"
INFERENCE_MODEL = "gpt-4o-mini"


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


class Search(BaseModel):
    """Search over a database of tutorial videos about a software library."""

    query: str = Field(
        ...,
        description="Similarity search query applied to video transcripts.",
    )
    publish_year: Optional[int] = Field(None, description="Year video was published.")


if not has_data(chromadb_path):

    urls = [
        "https://www.youtube.com/watch?v=HAn9vnJy6S4",
        "https://www.youtube.com/watch?v=dA1cHGACXCo",
        "https://www.youtube.com/watch?v=ZcEMLz27sL4",
        "https://www.youtube.com/watch?v=hvAPnpSfSGo",
        "https://www.youtube.com/watch?v=EhlPDL4QrWY",
        "https://www.youtube.com/watch?v=mmBo8nlu2j0",
        "https://www.youtube.com/watch?v=rQdibOsL1ps",
        "https://www.youtube.com/watch?v=28lC4fqukoc",
        "https://www.youtube.com/watch?v=es-9MgxB-uc",
        "https://www.youtube.com/watch?v=wLRHwKuKvOE",
        "https://www.youtube.com/watch?v=ObIltMaRJvY",
        "https://www.youtube.com/watch?v=DjuXACWYkkU",
        "https://www.youtube.com/watch?v=o7C9ld6Ln-M",
    ]
    docs = []
    for url in urls:
        docs.extend(YoutubeLoader.from_youtube_url(url, add_video_info=True).load())

    # add some additional metadata: what year the video was published
    for doc in docs:
        doc.metadata["publish_year"] = int(
            datetime.datetime.strptime(
                doc.metadata["publish_date"], "%Y-%m-%d %H:%M:%S"
            ).strftime("%Y")
        )

    # for doc in docs:
    #     pprint.pprint(doc.metadata, indent=2)

    # index the documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
    chunked_docs = text_splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(
        chunked_docs,
        embeddings,
        persist_directory=chromadb_path.name,
    )
else:
    vectorstore = Chroma(
        persist_directory=chromadb_path.name,
        embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL),
    )

system = """You are an expert at converting user questions into database queries. \
You have access to a database of tutorial videos about a software library for building LLM-powered applications \
Given a question, return a list of database queries optimized to retrieve the most relevant results.

If there are acronyms or words you are not familiar with, do not try to rephrase them."""
prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "{question}")])
llm = ChatOpenAI(model=INFERENCE_MODEL, temperature=0)
structured_llm = llm.with_structured_output(Search)
query_analyzer = {"question": RunnablePassthrough()} | prompt | structured_llm
print(query_analyzer.invoke("videos on RAG published in 2023"))


def retrieval(search: Search) -> List[Document]:
    if search.publish_year is not None:
        # Chroma-specific syntax
        _filter = {"publish_year": {"$eq": search.publish_year}}
    else:
        _filter = None
    return vectorstore.similarity_search(search.query, filter=_filter)


retrieval_chain = query_analyzer | retrieval
results = retrieval_chain.invoke("RAG tutorial published in 2023")
pprint.pprint(
    [(doc.metadata["title"], doc.metadata["publish_date"]) for doc in results]
)
