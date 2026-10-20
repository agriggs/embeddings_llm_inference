import logging
import unittest
from unittest.mock import patch

from embedding_utils import _PositionIdsReportFilter, create_embedding_function


class EmbeddingOutputTests(unittest.TestCase):
    def test_detected_device_is_passed_to_embedding_model(self):
        for device in ("cuda:0", "mps", "xpu", "npu", "cpu"):
            with self.subTest(device=device), patch(
                "sentence_transformers.util.get_device_name", return_value=device
            ), patch("embedding_utils.QuietSentenceTransformerEmbeddingFunction") as constructor:
                self.assertIs(create_embedding_function("test-model"), constructor.return_value)
                constructor.assert_called_once_with(model_name="test-model", device=device)

    def test_only_known_position_buffer_report_is_suppressed(self):
        report = (
            "BertModel LOAD REPORT from: BAAI/bge-small-en-v1.5\n"
            "Key | Status | Details\n"
            "embeddings.position_ids | UNEXPECTED |\n"
            "Notes:\n- UNEXPECTED :can be ignored"
        )
        output_filter = _PositionIdsReportFilter()

        def allowed(message):
            return output_filter.filter(logging.LogRecord(
                "transformers.modeling_utils", logging.WARNING, "", 0, message, (), None
            ))

        self.assertFalse(allowed(report))
        self.assertFalse(allowed("\x1b[1m" + report + "\x1b[0m"))
        self.assertTrue(allowed(report + "\nencoder.weight | MISSING |"))
        self.assertTrue(allowed(report + "\nencoder.weight | UNEXPECTED |"))
        self.assertTrue(allowed(report.replace("BAAI/bge-small-en-v1.5", "another-model")))
        self.assertTrue(allowed("Model could not be loaded"))


if __name__ == "__main__":
    unittest.main()
