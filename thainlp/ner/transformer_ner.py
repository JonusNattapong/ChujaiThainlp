"""
Transformer-based Named Entity Recognition (NER) using Model Hub.
"""
from typing import List, Tuple, Optional, Any, Dict # Added Dict
import logging
from thainlp.model_hub import ModelManager

# Configure logging
logger = logging.getLogger(__name__)

# Global cache for loaded models to avoid reloading
_LOADED_MODELS: Dict[str, Any] = {}

DEFAULT_MODEL = "wangchanberta_ner" # Default Transformer NER model

def _load_ner_model(model_name: str) -> Optional[Tuple[Any, Any]]:
    """Loads NER model (model, tokenizer) from Model Hub, caching it globally."""
    if model_name in _LOADED_MODELS:
        return _LOADED_MODELS[model_name]

    logger.info(f"Loading NER model '{model_name}' from Model Hub...")
    manager = ModelManager()
    model_info = manager.get_model_info(model_name)

    if not model_info or model_info['task'] != 'ner' or model_info['source'] != 'huggingface':
        logger.error(f"Model '{model_name}' is not a valid Hugging Face NER model in the registry.")
        return None

    # Ensure transformers is installed
    try:
        from transformers import pipeline # Use pipeline for easier NER
    except ImportError:
        logger.error("Transformers library not found. Please install it: pip install transformers torch tensorflow sentencepiece")
        # Or install with extra: pip install thainlp[hf]
        return None

    try:
        # Use pipeline for NER task
        # device=0 for GPU if available, -1 for CPU
        ner_pipeline = pipeline("ner", model=model_info['hf_id'], tokenizer=model_info['hf_id'], device=-1, grouped_entities=True)
        _LOADED_MODELS[model_name] = ner_pipeline # Cache the pipeline
        logger.info(f"NER model '{model_name}' loaded successfully.")
        return ner_pipeline
    except Exception as e:
        logger.error(f"Failed to load NER model '{model_name}': {e}")
        return None

def tag(text: str, model_name: str = DEFAULT_MODEL) -> List[Tuple[str, str, int, int]]:
    """
    Extract named entities from text using a specified Transformer model.

    Args:
        text (str): Input Thai text.
        model_name (str): Name of the NER model from the Model Hub
                          (default: "wangchanberta_ner").

    Returns:
        List[Tuple[str, str, int, int]]: List of tuples, where each tuple
                                         contains (entity_text, entity_type, start_char, end_char).
                                         Returns empty list if loading fails or no entities found.
    """
    ner_pipeline = _load_ner_model(model_name)
    if not ner_pipeline:
        return []

    try:
        results = ner_pipeline(text)
        # Convert pipeline output format to (text, type, start, end)
        formatted_results = [
            (entity['word'], entity['entity_group'], entity['start'], entity['end'])
            for entity in results
        ]
        return formatted_results
    except Exception as e:
        logger.error(f"Error during NER prediction with model '{model_name}': {e}")
        return []

# Example usage:
if __name__ == "__main__":
    # Ensure dependencies are installed: pip install thainlp[hf]
    text1 = "นายสมชาย สุดหล่อ เดินทางไปกรุงเทพมหานครเมื่อวานนี้"
    text2 = "บริษัท ไทยเอ็นแอลพี จำกัด ก่อตั้งเมื่อปี 2566"

    print(f"Text: {text1}")
    entities1 = tag(text1) # Use default model
    print("Entities:", entities1)
    # Expected output might look like:
    # Entities: [('สมชาย สุดหล่อ', 'PERSON', 3, 15), ('กรุงเทพมหานคร', 'LOCATION', 26, 39)]

    print(f"\nText: {text2}")
    entities2 = tag(text2)
    print("Entities:", entities2)
    # Expected output might look like:
    # Entities: [('บริษัท ไทยเอ็นแอลพี จำกัด', 'ORGANIZATION', 0, 26), ('2566', 'DATE', 40, 44)]

    # Example specifying a different (hypothetical) model
    # print("\nUsing hypothetical 'other_ner_model':")
    # entities3 = tag(text1, model_name="other_ner_model") # This would fail if not registered
    # print("Entities:", entities3)
