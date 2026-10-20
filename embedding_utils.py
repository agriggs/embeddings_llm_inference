"""Quiet embedding output while retaining useful model-loading warnings."""

import logging
import re

import numpy as np
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


class _PositionIdsReportFilter(logging.Filter):
    def filter(self, record):
        message = re.sub(r"\x1b\[[0-9;]*m", "", record.getMessage())
        if "BertModel LOAD REPORT from: BAAI/bge-small-en-v1.5" not in message:
            return True
        rows = re.findall(r"^\s*([^|\n]+?)\s*\|\s*UNEXPECTED\s*\|", message, re.MULTILINE)
        # BERT recreates this non-persistent buffer; do not hide other issues.
        return not (
            [key.strip() for key in rows] == ["embeddings.position_ids"]
            and not any(status in message for status in ("MISSING", "MISMATCH", "CONVERSION"))
        )


class QuietSentenceTransformerEmbeddingFunction(SentenceTransformerEmbeddingFunction):
    def __call__(self, input):
        embeddings = self._model.encode(
            list(input),
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )
        return [np.array(embedding, dtype=np.float32) for embedding in embeddings]


def create_embedding_function(model):
    from sentence_transformers.util import get_device_name
    from transformers.utils import logging as transformers_logging

    transformers_logging.disable_progress_bar()
    logger = logging.getLogger("transformers.modeling_utils")
    if not any(isinstance(item, _PositionIdsReportFilter) for item in logger.filters):
        logger.addFilter(_PositionIdsReportFilter())
    device = get_device_name()
    print(f"Embedding device: {device}", flush=True)
    return QuietSentenceTransformerEmbeddingFunction(model_name=model, device=device)
