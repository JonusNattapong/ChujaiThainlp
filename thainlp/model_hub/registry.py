"""
Model Registry for ThaiNLP Model Hub

Defines metadata for available pre-trained models.
"""
from typing import List, Dict, Optional, TypedDict

class ModelInfo(TypedDict):
    """Structure for model metadata."""
    name: str               # Unique name for the model within ThaiNLP
    task: str               # NLP task (e.g., 'pos_tag', 'ner', 'sentiment')
    description: str        # Brief description of the model
    source: str             # Origin (e.g., 'huggingface', 'pythainlp', 'custom')
    hf_id: Optional[str]    # Hugging Face model ID (if applicable)
    framework: str          # Framework (e.g., 'pytorch', 'tensorflow')
    tags: List[str]         # Keywords for filtering

# --- Model Registry ---
# This dictionary holds metadata for all supported models.
# Add new models here.

_MODEL_REGISTRY: Dict[str, ModelInfo] = {
    # --- POS Tagging Models ---
    "hmm_pos": {
        "name": "hmm_pos",
        "task": "pos_tag",
        "description": "Default Hidden Markov Model POS Tagger (built-in).",
        "source": "thainlp",
        "hf_id": None,
        "framework": "custom",
         "tags": ["pos", "hmm", "default"],
     },
    # --- Transformer POS Tagger (Example) ---
    # "wangchanberta_pos": { # Uncomment and adjust if needed later
    #     "name": "wangchanberta_pos",
    #     "task": "pos_tag",
    #     "description": "WangchanBERTa fine-tuned for POS tagging on LST20.",
    #     "source": "huggingface",
    #     "hf_id": "airesearch/wangchanberta-base-att-spm-uncased-pos", # Example ID
    #     "framework": "pytorch",
    #     "tags": ["pos", "transformer", "wangchanberta", "lst20"],
    # },

    # --- NER Models ---
    "rule_based_ner": {
        "name": "rule_based_ner",
        "task": "ner",
        "description": "Default Rule-based NER Tagger (built-in).",
        "source": "thainlp",
        "hf_id": None,
        "framework": "custom",
         "tags": ["ner", "rules", "default"],
     },
    # --- Transformer NER Model ---
     "wangchanberta_ner": {
         "name": "wangchanberta_ner",
         "task": "ner",
         "description": "WangchanBERTa fine-tuned for NER on THAINER.",
         "source": "huggingface",
         "hf_id": "airesearch/wangchanberta-base-att-spm-uncased-ner", # Verified ID
         "framework": "pytorch", # Assuming PyTorch, adjust if TF available/needed
         "tags": ["ner", "transformer", "wangchanberta", "thainer"],
     },

    # --- Sentiment Analysis Models ---
    # --- Transformer Sentiment Model ---
     "wangchanberta_sentiment": {
         "name": "wangchanberta_sentiment",
         "task": "sentiment",
         "description": "WangchanBERTa fine-tuned for Sentiment Analysis on Wisesight-Sentiment.",
         "source": "huggingface",
         "hf_id": "airesearch/wangchanberta-base-att-spm-uncased-sentiment", # Verified ID
         "framework": "pytorch", # Assuming PyTorch
         "tags": ["sentiment", "transformer", "wangchanberta", "wisesight"],
     },

    # --- Other Models (Add as needed) ---
    # "wangchanberta_qa": { ... }
    # "mt5_translation_th_en": { ... }
}

def list_models(task: Optional[str] = None, framework: Optional[str] = None, source: Optional[str] = None) -> List[ModelInfo]:
    """
    List available models, optionally filtering by task, framework, or source.

    Args:
        task (Optional[str]): Filter by NLP task (e.g., 'pos_tag', 'ner').
        framework (Optional[str]): Filter by framework (e.g., 'pytorch').
        source (Optional[str]): Filter by model source (e.g., 'huggingface').

    Returns:
        List[ModelInfo]: A list of model metadata dictionaries matching the filters.
    """
    filtered_models = list(_MODEL_REGISTRY.values())

    if task:
        filtered_models = [m for m in filtered_models if m['task'] == task]
    if framework:
        filtered_models = [m for m in filtered_models if m['framework'] == framework]
    if source:
        filtered_models = [m for m in filtered_models if m['source'] == source]

    return filtered_models

def get_model_info(model_name: str) -> Optional[ModelInfo]:
    """
    Get metadata for a specific model by its name.

    Args:
        model_name (str): The unique name of the model in the registry.

    Returns:
        Optional[ModelInfo]: The model metadata dictionary, or None if not found.
    """
    return _MODEL_REGISTRY.get(model_name)

# Example usage:
if __name__ == "__main__":
    print("--- All Models ---")
    all_models = list_models()
    for model in all_models:
        print(f"- {model['name']} ({model['task']})")

    print("\n--- POS Tagging Models ---")
    pos_models = list_models(task="pos_tag")
    for model in pos_models:
        print(f"- {model['name']} ({model['source']})")

    print("\n--- Hugging Face Models ---")
    hf_models = list_models(source="huggingface")
    if hf_models:
        for model in hf_models:
            print(f"- {model['name']} (HF ID: {model['hf_id']})")
    else:
        print("No Hugging Face models currently registered.")

    print("\n--- Get Specific Model Info ---")
    info = get_model_info("hmm_pos")
    if info:
        print(f"Info for 'hmm_pos': {info}")
    else:
        print("'hmm_pos' not found.")
