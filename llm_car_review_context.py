import argparse

from llm_model_config import MODEL_ENV_VARS, resolve_model
from chroma_utils import embedding_collection_name

import chromadb
from embedding_utils import create_embedding_function

# Optional model SDKs — import only if available to keep the script import-safe
try:
    from openai import OpenAI
    _has_openai = True
except Exception:
    OpenAI = None
    _has_openai = False

try:
    from google import genai
    from google.genai import types as genai_types
    _has_gemini = True
except Exception:
    genai = None
    genai_types = None
    _has_gemini = False

try:
    from anthropic import Anthropic
    _has_claude = True
except Exception:
    Anthropic = None
    _has_claude = False


# Configuration
DATA_PATH = "data/archive/*"
CHROMA_PATH = "car_review_embeddings"


def get_collection(embedding_model=None):
    embedding_model = resolve_model("embedding", embedding_model)
    client = chromadb.PersistentClient(CHROMA_PATH)
    collection_name = embedding_collection_name(embedding_model)
    try:
        collection = client.get_collection(name=collection_name, embedding_function=None)
    except chromadb.errors.NotFoundError as exc:
        raise ValueError(
            f"No collection for embedding model {embedding_model!r}. Run "
            "python create_car_review_collection.py first, using the same model configuration."
        ) from exc
    if (collection.metadata or {}).get("embedding_model") != embedding_model:
        raise ValueError("Collection embedding model does not match the selected model")
    embedding_func = create_embedding_function(embedding_model)

    collection = client.get_collection(
        name=collection_name, embedding_function=embedding_func
    )
    return collection


def query_good_reviews(collection, question, n_results=10):
    return collection.query(
        query_texts=[question],
        n_results=n_results,
        include=["documents"],
        where={"Rating": {"$gte": 3}},
    )


def query_poor_reviews(collection, question, n_results=5):
    return collection.query(
        query_texts=[question],
        n_results=n_results,
        include=["documents"],
        where={"Rating": {"$lte": 3}},
    )

def analyze_with_openai(context, question, model=None):
    model = resolve_model("openai", model)
    if not _has_openai:
        raise RuntimeError("OpenAI SDK not available. Install the official OpenAI Python SDK.")
    client = OpenAI()
    

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": context},
            {"role": "user", "content": question},
        ],
        n=1,
    )
    return resp.choices[0].message.content


def analyze_with_gemini(context, question, model=None):
    model = resolve_model("gemini", model)
    if not _has_gemini:
        raise RuntimeError("Gemini (google-genai) SDK not available. Install google-genai.")
    client = genai.Client()

    resp = client.models.generate_content(
        model=model,
        contents=context,
        config=genai_types.GenerateContentConfig(
            system_instruction=question,
            temperature=0,
        ),
    )
    # gemini's response text is available on .text
    return getattr(resp, "text", None) or str(resp)


def analyze_with_claude(context, question, model=None):
    model = resolve_model("claude", model)
    if not _has_claude:
        raise RuntimeError("Anthropic SDK not available. Install the anthropic package.")
    client = Anthropic()
    
    resp = client.messages.create(
        model=model,
        system=context,
        max_tokens=4096,
        messages=[{"role": "user", "content": question}],
    )
    # Reasoning models may return thinking blocks before the answer text.
    return "\n".join(block.text for block in resp.content if block.type == "text")


def run(provider: str, model: str | None = None, embedding_model: str | None = None):
    model = resolve_model(provider, model)
    embedding_model = resolve_model("embedding", embedding_model)
    print(f"Using {provider} model: {model}")
    print(f"Using embedding model: {embedding_model}")
    collection = get_collection(embedding_model)

    # Positive reviews question
    pos_question = "What's the key to great customer satisfaction based on detailed positive reviews?"
    good_reviews = query_good_reviews(collection, pos_question, n_results=10)
    reviews_str = ",".join(good_reviews["documents"][0])

    print("Good reviews (excerpt):")
    print(reviews_str[:1000])
    print("###########################################")

    # Negative reviews question
    poor_question = (
        "Which of these poor reviews has the worst implications about our dealership? Explain why."
    )
    poor = query_poor_reviews(collection, poor_question, n_results=5)
    poor_reviews_str = ",".join(poor["documents"][0])

    print("Worst review (excerpt):")
    print(poor["documents"][0][0][:1000])
    print("###########################################")

    ai_context = """
        You are a customer success employee at a large
        car dealership. Use the following car reviews
        to answer questions: {}
    """

    ai_context_good_reviews = ai_context.format(reviews_str)
    ai_context_poor_reviews = ai_context.format(poor_reviews_str)
    
    if provider == "openai":
        summary = analyze_with_openai(ai_context_good_reviews, pos_question, model=model)
        worst = analyze_with_openai(ai_context_poor_reviews, poor_question, model=model)
    elif provider == "gemini":
        summary = analyze_with_gemini(ai_context_good_reviews, pos_question, model=model)
        worst = analyze_with_gemini(ai_context_poor_reviews, poor_question, model=model)
    elif provider == "claude":
        summary = analyze_with_claude(ai_context_good_reviews, pos_question, model=model)
        worst = analyze_with_claude(ai_context_poor_reviews, poor_question, model=model)
    else:
        raise ValueError("Unknown provider: choose one of openai|gemini|claude")

    print(f"{provider} generated summary of good reviews:\n")
    print(summary)
    print("###########################################")
    print(f"{provider} generated analysis of worst review:\n")
    print(worst)


def main():
    parser = argparse.ArgumentParser(description="Unified car review context analyzer")
    parser.add_argument("provider", choices=list(MODEL_ENV_VARS), help="Which provider to use")
    args = parser.parse_args()
    try:
        run(args.provider)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
