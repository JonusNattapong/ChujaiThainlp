"""
Transformer-based Sentiment Analysis using Model Hub.
"""
from typing import List, Dict, Optional, Any, Union
import logging
from thainlp.model_hub import ModelManager

# Configure logging
logger = logging.getLogger(__name__)

# Global cache for loaded models
_LOADED_MODELS: Dict[str, Any] = {}

DEFAULT_MODEL = "wangchanberta_sentiment" # Default Transformer Sentiment model

def _load_sentiment_model(model_name: str) -> Optional[Any]:
    """Loads Sentiment Analysis pipeline from Model Hub, caching it globally."""
    if model_name in _LOADED_MODELS:
        return _LOADED_MODELS[model_name]

    logger.info(f"Loading Sentiment model '{model_name}' from Model Hub...")
    manager = ModelManager()
    model_info = manager.get_model_info(model_name)

    if not model_info or model_info['task'] != 'sentiment' or model_info['source'] != 'huggingface':
        logger.error(f"Model '{model_name}' is not a valid Hugging Face Sentiment model in the registry.")
        return None

    # Ensure transformers is installed
    try:
        from transformers import pipeline
    except ImportError:
        logger.error("Transformers library not found. Please install it: pip install transformers torch tensorflow sentencepiece")
        # Or install with extra: pip install thainlp[hf]
        return None

    try:
        # Use pipeline for sentiment analysis task
        # device=0 for GPU if available, -1 for CPU
        sentiment_pipeline = pipeline("sentiment-analysis", model=model_info['hf_id'], tokenizer=model_info['hf_id'], device=-1)
        _LOADED_MODELS[model_name] = sentiment_pipeline # Cache the pipeline
        logger.info(f"Sentiment model '{model_name}' loaded successfully.")
        return sentiment_pipeline
    except Exception as e:
        logger.error(f"Failed to load Sentiment model '{model_name}': {e}")
        return None

def analyze(text: str, model_name: str = DEFAULT_MODEL) -> Optional[Dict[str, Union[str, float]]]:
    """
    Analyze sentiment of text using a specified Transformer model.

    Args:
        text (str): Input Thai text.
        model_name (str): Name of the Sentiment model from the Model Hub
                          (default: "wangchanberta_sentiment").

    Returns:
        Optional[Dict[str, Union[str, float]]]: Dictionary containing 'label' (e.g., 'positive',
                                                 'negative', 'neutral') and 'score' (confidence),
                                                 or None if loading/prediction fails.
                                                 Labels depend on the specific model.
    """
    sentiment_pipeline = _load_sentiment_model(model_name)
    if not sentiment_pipeline:
        return None

    try:
        # Pipeline returns a list, usually with one result for the whole text
        results = sentiment_pipeline(text)
        if results:
            # Return the first result (label and score)
            return results[0]
        else:
            logger.warning(f"Sentiment model '{model_name}' returned no results for the text.")
            return None
    except Exception as e:
        logger.error(f"Error during Sentiment prediction with model '{model_name}': {e}")
        return None

# Example usage:
if __name__ == "__main__":
    # Ensure dependencies are installed: pip install thainlp[hf]
    text_pos = "หนังเรื่องนี้สนุกมาก นักแสดงเก่งทุกคนเลย"
    text_neg = "อาหารร้านนี้ไม่อร่อยและบริการแย่มาก"
    text_neu = "วันนี้อากาศดี ท้องฟ้าแจ่มใส" # May be classified pos/neg depending on model bias

    print(f"Text: {text_pos}")
    sentiment_pos = analyze(text_pos) # Use default model
    print("Sentiment:", sentiment_pos)
    # Expected output might look like: {'label': 'positive', 'score': 0.99...}

    print(f"\nText: {text_neg}")
    sentiment_neg = analyze(text_neg)
    print("Sentiment:", sentiment_neg)
    # Expected output might look like: {'label': 'negative', 'score': 0.99...}

    print(f"\nText: {text_neu}")
    sentiment_neu = analyze(text_neu)
    print("Sentiment:", sentiment_neu)
    # Expected output might look like: {'label': 'neutral'/'positive', 'score': ...}

    # Example specifying a different (hypothetical) model
    # print("\nUsing hypothetical 'other_sentiment_model':")
    # sentiment_other = analyze(text_pos, model_name="other_sentiment_model") # Fails if not registered
    # print("Sentiment:", sentiment_other)
