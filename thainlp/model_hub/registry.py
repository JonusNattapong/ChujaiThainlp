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

    # --- Speech Models ---
    "thainer-speech": {
        "name": "thainer-speech",
        "task": "tts",
        "description": "Thai Text-to-Speech model based on ESPnet",
        "source": "huggingface",
        "hf_id": "airesearch/thainer-speech-tts",  # Example ID - replace with actual model
        "framework": "pytorch",
        "tags": ["speech", "tts", "audio", "thainer"],
    },

    # --- Speech Models ---
    "facebook/mms-tts-tha": {
        "name": "facebook/mms-tts-tha",
        "task": "tts",
        "description": "Facebook's Massively Multilingual Speech TTS model for Thai",
        "source": "huggingface",
        "hf_id": "facebook/mms-tts-tha",
        "framework": "pytorch",
        "tags": ["speech", "tts", "audio", "mms", "facebook"],
        "sample_rate": 16000,
        "language": "tha",
        "requires": ["transformers>=4.31.0", "torch", "librosa"]  # Add required dependencies
    },

    # --- ASR Models ---
    "openai/whisper-large-v3-turbo": {
        "name": "openai/whisper-large-v3-turbo",
        "task": "asr",
        "description": "OpenAI's Whisper Large V3 Turbo model - high accuracy multilingual ASR",
        "source": "huggingface",
        "hf_id": "openai/whisper-large-v3-turbo",
        "framework": "pytorch",
        "tags": ["speech", "asr", "audio", "whisper", "openai"],
        "sample_rate": 16000,
        "language": "tha",
        "requires": ["transformers>=4.31.0", "torch", "librosa"]
    },

    # --- Speech Summarization ---
    "openai/whisper-large-v3": {
        "name": "openai/whisper-large-v3",
        "task": "speech-summarization",
        "description": "OpenAI's Whisper Large V3 model for speech summarization",
        "source": "huggingface", 
        "hf_id": "openai/whisper-large-v3",
        "framework": "pytorch",
        "tags": ["speech", "summarization", "whisper", "openai"],
        "sample_rate": 16000,
        "language": "multilingual",
        "requires": ["transformers>=4.31.0", "torch", "librosa"]
    },

    # --- Audio Classification ---
    "speechbrain/lang-id-voxlingua107-ecapa": {
        "name": "speechbrain/lang-id-voxlingua107-ecapa",
        "task": "audio-classification",
        "description": "SpeechBrain's ECAPA-TDNN model for language identification",
        "source": "huggingface",
        "hf_id": "speechbrain/lang-id-voxlingua107-ecapa",
        "framework": "pytorch",
        "tags": ["speech", "classification", "language-id", "speechbrain"],
        "sample_rate": 16000,
        "language": "multilingual",
        "requires": ["speechbrain", "torch"]
    },
# --- Voice Activity Detection Models ---
"facebook/wav2vec2-base": {
    "name": "facebook/wav2vec2-base",
    "task": "vad",
    "description": "Facebook's Wav2Vec2 model used for voice activity detection",
    "source": "huggingface",
    "hf_id": "facebook/wav2vec2-base",
    "framework": "pytorch",
    "tags": ["speech", "vad", "audio", "facebook"],
    "sample_rate": 16000,
    "requires": ["transformers>=4.31.0", "torch"]
},

# --- Voice Processing Models ---
"facebook/fastspeech2-en-200": {
    "name": "facebook/fastspeech2-en-200",
    "task": "voice-conversion",
    "description": "Facebook's FastSpeech2 model for voice conversion",
    "source": "huggingface",
    "hf_id": "facebook/fastspeech2-en-200",
    "framework": "pytorch",
    "tags": ["speech", "voice", "conversion", "facebook"],
    "sample_rate": 16000,
    "requires": ["transformers>=4.31.0", "torch"]
},

# --- Other Models (Add as needed) ---
    # --- Text Classification ---
    "wangchanberta_text_cls": {
        "name": "wangchanberta_text_cls",
        "task": "text_classification",
        "description": "WangchanBERTa for Thai text classification",
        "source": "huggingface",
        "hf_id": "airesearch/wangchanberta-base-att-spm-uncased",
        "framework": "pytorch",
        "tags": ["classification", "thai", "wangchanberta"]
    },

    # --- Token Classification ---
    "wangchanberta_token_cls": {
        "name": "wangchanberta_token_cls",
        "task": "token_classification",
        "description": "WangchanBERTa for Thai token classification (POS/NER)",
        "source": "huggingface",
        "hf_id": "airesearch/wangchanberta-base-att-spm-uncased-pos",
        "framework": "pytorch",
        "tags": ["ner", "pos", "thai", "wangchanberta"]
    },

    # --- Question Answering ---
    "wangchanberta_qa": {
        "name": "wangchanberta_qa",
        "task": "question_answering",
        "description": "WangchanBERTa for Thai question answering",
        "source": "huggingface",
        "hf_id": "airesearch/wangchanberta-base-att-spm-uncased-qa",
        "framework": "pytorch",
        "tags": ["qa", "thai", "wangchanberta"]
    },

    # --- Text Generation ---
    "wangchanberta_generation": {
        "name": "wangchanberta_generation",
        "task": "text_generation",
        "description": "WangchanBERTa for Thai text generation",
        "source": "huggingface",
        "hf_id": "airesearch/wangchanberta-base-att-spm-uncased",
        "framework": "pytorch",
        "tags": ["generation", "thai", "wangchanberta"]
    },

    # --- Translation ---
    "mt5_th_en": {
        "name": "mt5_th_en",
        "task": "translation",
        "description": "mT5 for Thai-English translation",
        "source": "huggingface",
        "hf_id": "google/mt5-base",
        "framework": "pytorch",
        "tags": ["translation", "thai", "english", "mt5"]
    }
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
