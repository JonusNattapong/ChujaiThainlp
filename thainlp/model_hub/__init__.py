"""
Model hub for Thai NLP models
"""

def get_model_info(model_name: str) -> dict:
    """Get model information from registry

    Args:
        model_name: Name of the model

    Returns:
        Dict containing model information
    """
    MODEL_REGISTRY = {
        "wangchanberta_text_cls": {
            "hf_id": "airesearch/wangchanberta-base-att-spm-uncased",
            "type": "text_classification",
            "labels": ["positive", "negative", "neutral"]
        },
        "wangchanberta_token_cls": {
            "hf_id": "airesearch/wangchanberta-base-att-spm-uncased",
            "type": "token_classification"
        },
        "xlm-roberta-base": {
            "hf_id": "xlm-roberta-base",
            "type": "text_classification"
        },
        "monsoon-nlp/bert-base-thai-squad": {
            "hf_id": "monsoon-nlp/bert-base-thai-squad",
            "type": "question_answering"
        }
    }
    
    return MODEL_REGISTRY.get(model_name, {})
