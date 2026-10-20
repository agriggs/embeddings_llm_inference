import os
from contextlib import contextmanager
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import chromadb
import numpy as np

import chroma_utils
import llm_car_review_context as analyzer
import llm_model_config as config


@contextmanager
def chroma_test_directory():
    # Release Chroma's Windows file handles before removing the test directory.
    with tempfile.TemporaryDirectory() as directory:
        client = chromadb.PersistentClient(directory)
        try:
            yield directory
        finally:
            client._system.stop()
            chromadb.api.client.SharedSystemClient.clear_system_cache()


class LocalEmbedding(chromadb.EmbeddingFunction):
    """Deterministic vectors for testing collection wiring without downloads."""

    def __init__(self, model_name):
        self.model_name = model_name

    def __call__(self, input):
        return [np.array([len(text), 1, 0], dtype=np.float32) for text in input]

    @staticmethod
    def name():
        return "test_embedding"

    def get_config(self):
        return {"model_name": self.model_name}

    @staticmethod
    def build_from_config(config):
        return LocalEmbedding(**config)


class EmbeddingConfigTests(unittest.TestCase):
    def test_precedence_and_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "models.toml"
            path.write_text('[models]\nembedding = "config-model"\n', encoding="utf-8")
            with patch.object(config, "MODEL_CONFIG_PATH", path), patch.dict(os.environ, {}, clear=True):
                self.assertEqual(config.resolve_model("embedding"), "config-model")
                with patch.dict(os.environ, {"EMBEDDING_MODEL": "env-model"}):
                    self.assertEqual(config.resolve_model("embedding"), "env-model")
                    self.assertEqual(config.resolve_model("embedding", "cli-model"), "cli-model")
                    with self.assertRaises(ValueError):
                        config.resolve_model("embedding", " ")
                path.write_text("[models]\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "non-empty model ID"):
                    config.resolve_model("embedding")
        self.assertNotIn("embedding", config.MODEL_ENV_VARS)

    def test_build_and_query_use_same_model_and_preserve_every_document(self):
        with chroma_test_directory() as directory, patch.object(
            chroma_utils, "create_embedding_function", LocalEmbedding
        ), patch.object(analyzer, "create_embedding_function", LocalEmbedding), patch.object(analyzer, "CHROMA_PATH", directory):
            name = chroma_utils.embedding_collection_name("model-a")
            # Cross a batch boundary, including a final singleton batch.
            documents = [f"review {i}" for i in range(501)]
            chroma_utils.build_chroma_collection(
                directory, name, "model-a", [str(i) for i in range(501)],
                documents, [{"Rating": 5} for _ in documents],
            )
            collection = analyzer.get_collection("model-a")
            self.assertEqual(collection.count(), 501)
            self.assertEqual(collection.metadata["embedding_model"], "model-a")
            self.assertEqual(len(collection.query(query_texts=["good car"], n_results=1)["ids"][0]), 1)
            with self.assertRaisesRegex(ValueError, "create_car_review_collection.py"):
                analyzer.get_collection("model-b")
            collection.modify(metadata={"embedding_model": "different-model"})
            with self.assertRaisesRegex(ValueError, "does not match"):
                analyzer.get_collection("model-a")


if __name__ == "__main__":
    unittest.main()
