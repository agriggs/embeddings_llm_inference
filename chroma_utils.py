import pathlib
import hashlib
from time import monotonic

import chromadb
from more_itertools import batched
from embedding_utils import create_embedding_function

OUTER_BATCH_SIZE = 500


def embedding_collection_name(model: str) -> str:
    """Keep vectors from different models in separate collections."""
    model_id = hashlib.sha256(model.encode("utf-8")).hexdigest()[:16]
    return f"car_reviews_{model_id}"


def build_chroma_collection(
    chroma_path: pathlib.Path,
    collection_name: str,
    embedding_func_name: str,
    ids: list[str],
    documents: list[str],
    metadatas: list[dict],
    distance_func_name: str = "cosine",
):
    """Create a ChromaDB collection"""

    chroma_client = chromadb.PersistentClient(chroma_path)

    print(f"Loading embedding model: {embedding_func_name}", flush=True)
    embedding_func = create_embedding_function(embedding_func_name)

    collection = chroma_client.create_collection(
        name=collection_name,
        embedding_function=embedding_func,
        metadata={"hnsw:space": distance_func_name, "embedding_model": embedding_func_name},
    )

    document_indices = list(range(len(documents)))
    started = monotonic()
    total_batches = (len(documents) + OUTER_BATCH_SIZE - 1) // OUTER_BATCH_SIZE

    for batch_number, batch in enumerate(batched(document_indices, OUTER_BATCH_SIZE), start=1):
        start_idx = batch[0]
        end_idx = batch[-1] + 1
        print(
            f"Batch {batch_number}/{total_batches}: embedding reviews "
            f"{start_idx + 1}-{end_idx}/{len(documents)}...",
            flush=True,
        )
        embeddings = embedding_func(documents[start_idx:end_idx])
        print(f"Batch {batch_number}/{total_batches}: storing embeddings...", flush=True)

        collection.add(
            ids=ids[start_idx:end_idx],
            documents=documents[start_idx:end_idx],
            metadatas=metadatas[start_idx:end_idx],
            embeddings=embeddings,
        )
        print(f"Stored {end_idx}/{len(documents)} reviews ({monotonic() - started:.1f}s elapsed)", flush=True)

    stored = collection.count()
    if stored != len(documents):
        raise RuntimeError(f"Expected {len(documents)} stored reviews, found {stored}")
    print(f"Build complete: {stored} reviews stored in {collection_name}.", flush=True)
