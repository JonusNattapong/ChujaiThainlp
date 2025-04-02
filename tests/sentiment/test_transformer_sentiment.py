"""
Tests for Transformer-based Sentiment Analysis module.
"""
import unittest
from unittest.mock import patch, MagicMock

# Mock the pipeline function from transformers
mock_pipeline = MagicMock()

# Patch where 'pipeline' is looked up in transformer_sentiment.py
@patch('thainlp.sentiment.transformer_sentiment.pipeline', mock_pipeline)
class TestTransformerSentiment(unittest.TestCase):

    def setUp(self):
        # Reset the mock before each test
        mock_pipeline.reset_mock()
        # Clear the internal model cache
        from thainlp.sentiment import transformer_sentiment
        transformer_sentiment._LOADED_MODELS.clear()

    def test_analyze_success(self):
        """Test successful sentiment analysis."""
        from thainlp.sentiment import analyze_transformer, DEFAULT_TRANSFORMER_MODEL

        # Configure the mock pipeline's return value
        mock_sentiment_instance = MagicMock()
        mock_pipeline.return_value = mock_sentiment_instance
        mock_sentiment_instance.return_value = [{'label': 'positive', 'score': 0.998}]

        text = "หนังเรื่องนี้ดีมาก สนุกสุดๆ"
        expected_output = {'label': 'positive', 'score': 0.998}

        result = analyze_transformer(text)

        # Assertions
        self.assertEqual(result, expected_output)
        # Check that the pipeline was called correctly
        mock_pipeline.assert_called_once_with(
            "sentiment-analysis",
            model="airesearch/wangchanberta-base-att-spm-uncased-sentiment", # Default model
            tokenizer="airesearch/wangchanberta-base-att-spm-uncased-sentiment",
            device=-1
        )
        # Check that the loaded pipeline instance was called with the text
        mock_sentiment_instance.assert_called_once_with(text)

    def test_analyze_load_fail(self):
        """Test sentiment analysis when model loading fails."""
        from thainlp.sentiment import analyze_transformer

        # Configure the mock pipeline to raise an exception
        mock_pipeline.side_effect = Exception("Model loading failed")

        text = "หนังเรื่องนี้ดีมาก สนุกสุดๆ"
        result = analyze_transformer(text)

        # Assertions
        self.assertIsNone(result) # Expect None on failure
        mock_pipeline.assert_called_once() # Check pipeline was attempted

    def test_analyze_predict_fail(self):
        """Test sentiment analysis when prediction fails."""
        from thainlp.sentiment import analyze_transformer

        # Configure the mock pipeline instance to raise an exception on call
        mock_sentiment_instance = MagicMock()
        mock_pipeline.return_value = mock_sentiment_instance
        mock_sentiment_instance.side_effect = Exception("Prediction failed")

        text = "หนังเรื่องนี้ดีมาก สนุกสุดๆ"
        result = analyze_transformer(text)

        # Assertions
        self.assertIsNone(result) # Expect None on failure
        mock_pipeline.assert_called_once()
        mock_sentiment_instance.assert_called_once_with(text)

    def test_analyze_no_result(self):
        """Test sentiment analysis when pipeline returns empty list."""
        from thainlp.sentiment import analyze_transformer

        # Configure the mock pipeline instance to return empty list
        mock_sentiment_instance = MagicMock()
        mock_pipeline.return_value = mock_sentiment_instance
        mock_sentiment_instance.return_value = [] # Empty result

        text = "..." # Some text
        result = analyze_transformer(text)

        # Assertions
        self.assertIsNone(result) # Expect None if no result from pipeline
        mock_pipeline.assert_called_once()
        mock_sentiment_instance.assert_called_once_with(text)

    def test_analyze_invalid_model(self):
        """Test using an invalid or non-Sentiment model name."""
        from thainlp.sentiment import analyze_transformer
        # Assuming 'hmm_pos' is registered but not a valid HF Sentiment model

        text = "some text"
        # We don't need to mock pipeline here, as _load_sentiment_model should fail early
        result = analyze_transformer(text, model_name="hmm_pos")

        self.assertIsNone(result)
        # Check that pipeline was NOT called
        mock_pipeline.assert_not_called()


if __name__ == '__main__':
    unittest.main()
