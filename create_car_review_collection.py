import argparse

from car_data_etl import prepare_car_reviews_data
from chroma_utils import build_chroma_collection, embedding_collection_name
from llm_model_config import resolve_model

DATA_PATH = "data/archive/*"
CHROMA_PATH = "car_review_embeddings"


def main():
    parser = argparse.ArgumentParser(description="Build car review embeddings")
    parser.parse_args()
    try:
        embedding_model = resolve_model("embedding")
    except ValueError as exc:
        parser.error(str(exc))
    collection_name = embedding_collection_name(embedding_model)
    print(f"Using embedding model: {embedding_model}", flush=True)
    print(f"Creating collection: {collection_name}", flush=True)
    print("Preparing review data...", flush=True)
    chroma_car_reviews_dict = prepare_car_reviews_data(DATA_PATH)

    build_chroma_collection(
        CHROMA_PATH,
        collection_name,
        embedding_model,
        chroma_car_reviews_dict["ids"],
        chroma_car_reviews_dict["documents"],
        chroma_car_reviews_dict["metadatas"],
    )

if __name__ == "__main__":
    main()
