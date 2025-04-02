"""
Tests for Transformer-based NER module.
"""
import unittest
from unittest.mock import patch, MagicMock

# Mock the pipeline function from transformers before importing our module
mock_pipeline = MagicMock()

# Apply the patch globally or within the test class/methods
# Patching where 'pipeline' is looked up in transformer_ner.py
@patch('thainlp.ner.transformer_ner.pipeline', mock_pipeline)
class TestTransformerNER(unittest.TestCase):

    def setUp(self):
        # Reset the mock before each test
        mock_pipeline.reset_mock()
        # Clear the internal model cache in the module being tested
        # This is important if tests might use different mock setups
        from thainlp.ner import transformer_ner
        transformer_ner._LOADED_MODELS.clear()

    def test_tag_success(self):
        """Test successful NER tagging."""
        from thainlp.ner import tag_transformer, DEFAULT_TRANSFORMER_MODEL

        # Configure the mock pipeline's return value for a specific call
        mock_ner_instance = MagicMock()
        mock_pipeline.return_value = mock_ner_instance
        mock_ner_instance.return_value = [
            {'entity_group': 'PERSON', 'score': 0.99, 'word': 'สมชาย', 'start': 3, 'end': 8},
            {'entity_group': 'LOCATION', 'score': 0.98, 'word': 'กรุงเทพ', 'start': 20, 'end': 27}
        ]

        text = "คุณสมชายไปกรุงเทพเมื่อวาน"
        expected_output = [
            ('สมชาย', 'PERSON', 3, 8),
            ('กรุงเทพ', 'LOCATION', 20, 27)
        ]

        result = tag_transformer(text)

        # Assertions
        self.assertEqual(result, expected_output)
        # Check that the pipeline was called correctly
        mock_pipeline.assert_called_once_with(
            "ner",
            model="airesearch/wangchanberta-base-att-spm-uncased-ner", # Check default model ID
            tokenizer="airesearch/wangchanberta-base-att-spm-uncased-ner",
            device=-1,
            grouped_entities=True
        )
        # Check that the loaded pipeline instance was called with the text
        mock_ner_instance.assert_called_once_with(text)

    def test_tag_load_fail(self):
        """Test NER tagging when model loading fails."""
        from thainlp.ner import tag_transformer

        # Configure the mock pipeline to raise an exception
        mock_pipeline.side_effect = Exception("Model loading failed")

        text = "คุณสมชายไปกรุงเทพเมื่อวาน"
        result = tag_transformer(text)

        # Assertions
        self.assertEqual(result, []) # Expect empty list on failure
        mock_pipeline.assert_called_once() # Check pipeline was attempted

    def test_tag_predict_fail(self):
        """Test NER tagging when prediction fails."""
        from thainlp.ner import tag_transformer

        # Configure the mock pipeline instance to raise an exception on call
        mock_ner_instance = MagicMock()
        mock_pipeline.return_value = mock_ner_instance
        mock_ner_instance.side_effect = Exception("Prediction failed")

        text = "คุณสมชายไปกรุงเทพเมื่อวาน"
        result = tag_transformer(text)

        # Assertions
        self.assertEqual(result, []) # Expect empty list on failure
        mock_pipeline.assert_called_once()
        mock_ner_instance.assert_called_once_with(text)

    def test_tag_invalid_model(self):
        """Test using an invalid or non-NER model name."""
        from thainlp.ner import tag_transformer
        # Assuming 'hmm_pos' is registered but not a valid HF NER model
        # This test relies on ModelManager correctly identifying the model type

        text = "some text"
        # We don't need to mock pipeline here, as _load_ner_model should fail early
        result = tag_transformer(text, model_name="hmm_pos")

        self.assertEqual(result, [])
        # Check that pipeline was NOT called because loading checks failed
        mock_pipeline.assert_not_called()


if __name__ == '__main__':
    unittest.main()
