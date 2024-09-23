from typing import Callable

import numpy as np
import openai
import tiktoken

COST_PER_1000_TOKENS = 0.1 / 1000
COST_PER_TOKEN = COST_PER_1000_TOKENS / 1000
MODEL = "text-embedding-ada-002"


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute the cosine vector between two vectors"""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b) if norm_a and norm_b else 0


def find_similar_chunk(
    question_embedding: np.ndarray,
    embeddings: list[np.ndarray],
    measure: Callable = cosine_similarity,
) -> int:
    """Find the index of the most similar chunk based on a similarity measure"""
    similarities = [measure(question_embedding, e) for e in embeddings]
    most_similar_index = similarities.index(max(similarities))
    return most_similar_index


def get_embeddings(chunks: list[str], model: str = MODEL) -> list[np.ndarray]:
    """Generate embeddings for a list of text chunks"""
    return [
        openai.embeddings.create(input=chunk, model=model).data[0].embedding
        for chunk in chunks
    ]


def chunk_text(text: str, max_tokens: int = 512, model: str = MODEL) -> list[str]:
    """Chunk text into parts, ensuring each chunk is within the token limit."""
    tokenizer = tiktoken.encoding_for_model(model_name=model)
    tokens = tokenizer.encode(text)

    chunks = [
        tokenizer.decode(tokens[i : i + max_tokens])
        for i in range(0, len(tokens), max_tokens)
    ]

    return chunks


def count_tokens(text: str, model: str = MODEL) -> int:
    """Count the number of tokens in a given text."""
    tokenizer = tiktoken.encoding_for_model(model_name=model)
    return len(tokenizer.encode(text))


def estimate_embedding_cost(
    text: str,
    model: str = MODEL,
    cost_per_token: float = COST_PER_TOKEN,
) -> float:
    """Estimate the cost of generating embeddings for a given text."""
    num_tokens = count_tokens(text, model)
    return num_tokens * cost_per_token


def process_and_estimate(text: str, model: str = MODEL, max_tokens: int = 512) -> float:
    chunks = chunk_text(text, max_tokens, model)
    estimated_cost = sum([estimate_embedding_cost(chunk, model) for chunk in chunks])

    return estimated_cost


def confirm_and_generate_embeddings(
    text: str, model: str = MODEL, max_tokens: int = 512
) -> list[np.ndarray]:
    """Estimate the cost, ask user for confirmation, and generate embeddings."""
    estimated_cost = process_and_estimate(text, model, max_tokens)
    print(f"Estimated cost for embeddings: ${estimated_cost:4}")

    proceed = input("Do you want to proceed with generating embeddings? (yes/no): ")
    if proceed.lower() in ["yes", "y"]:
        chunks = chunk_text(text, max_tokens, model)
        return get_embeddings(chunks, model)
    else:
        print("Embedding process canceled")
        return []
