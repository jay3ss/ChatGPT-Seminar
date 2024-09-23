import chromadb
import numpy as np
from chromadb import Client, Collection
from embeddings import chunk_text, get_embeddings

MODEL = "text-embedding-ada-002"


def init_chroma(collection_name: str) -> Collection:
    client = chromadb.Client()
    return client.create_collection(collection_name)


def store_embeddings_in_chroma(
    embeddings: list[np.ndarray], chunks: list[np.ndarray], collection: Collection
) -> None:
    # assuming `embeddings` and `chunks` of lists of same length
    for i, (embedding, chunk) in enumerate(zip(embeddings, chunks)):
        # create a unique ID for each chunk
        doc_id = f"doc_{i}"
        collection.add(
            # store chunk, embedding and assign ID
            documents=[chunk],
            embeddings=[embedding],
            ids=[doc_id],
        )


def persist(collection: Collection) -> None:
    collection.persist()


def load_embeddings(collection_name: str) -> Collection:
    return Client().get_collection(collection_name)
