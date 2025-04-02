"""
Model Manager for ThaiNLP Model Hub

Handles downloading, caching, and loading models.
"""
import os
import logging
from pathlib import Path
from typing import Any, Optional, List, Dict
from huggingface_hub import hf_hub_download, HfFolder
from .registry import get_model_info, list_models, ModelInfo

# Configure logging
logger = logging.getLogger(__name__)

# Default cache directory (can be overridden by environment variable)
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "thainlp"
THAINLP_CACHE_HOME = Path(os.getenv("THAINLP_CACHE_HOME", DEFAULT_CACHE_DIR))

class ModelManager:
    """
    Manages the discovery, download, caching, and loading of models.
    """
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize the ModelManager.

        Args:
            cache_dir (Optional[Path]): Path to the cache directory. 
                                        Defaults to THAINLP_CACHE_HOME.
        """
        self.cache_dir = cache_dir or THAINLP_CACHE_HOME
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using cache directory: {self.cache_dir}")
        # Ensure Hugging Face token is available if needed for private models
        # HfFolder.get_token() # Uncomment if login is strictly required

    def list_available_models(self, **kwargs) -> List[ModelInfo]:
        """
        List models available in the registry, with optional filtering.

        Args:
            **kwargs: Filtering options (e.g., task='ner', source='huggingface').

        Returns:
            List[ModelInfo]: List of matching model metadata.
        """
        return list_models(**kwargs)

    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """
        Get metadata for a specific model.

        Args:
            model_name (str): The unique name of the model.

        Returns:
            Optional[ModelInfo]: Model metadata or None if not found.
        """
        return get_model_info(model_name)

    def get_model_path(self, model_name: str, force_download: bool = False) -> Optional[Path]:
        """
        Get the local path to the model files, downloading if necessary.

        Args:
            model_name (str): The unique name of the model.
            force_download (bool): If True, force redownload even if cached.

        Returns:
            Optional[Path]: Path to the directory containing model files, or None if invalid.
        """
        model_info = self.get_model_info(model_name)
        if not model_info:
            logger.error(f"Model '{model_name}' not found in registry.")
            return None

        if model_info['source'] == 'huggingface' and model_info['hf_id']:
            hf_id = model_info['hf_id']
            try:
                # hf_hub_download downloads specific files, but we often need the whole repo.
                # A common pattern is to just return the expected cache path,
                # and let libraries like transformers handle the actual download on load.
                # However, we can attempt a dummy download to ensure it exists or trigger download.
                # Let's download a common file like config.json to check/trigger download.
                config_path = hf_hub_download(
                    repo_id=hf_id,
                    filename="config.json", # Assuming most HF models have this
                    cache_dir=self.cache_dir / "huggingface",
                    force_download=force_download,
                    # token=HfFolder.get_token(), # Pass token if needed
                )
                # Return the directory containing the config file
                model_dir = Path(config_path).parent
                logger.info(f"Model '{model_name}' (HF: {hf_id}) found/downloaded to: {model_dir}")
                return model_dir
            except Exception as e:
                logger.error(f"Failed to download/locate Hugging Face model '{hf_id}': {e}")
                return None
        elif model_info['source'] == 'thainlp':
            # For built-in models, the path might be relative or handled differently
            logger.info(f"Model '{model_name}' is built-in. Loading handled by specific module.")
            # Returning None as path management is internal for built-ins
            return None
        else:
            logger.warning(f"Model source '{model_info['source']}' not yet fully supported for path management.")
            return None

    def load_model(self, model_name: str, **kwargs) -> Optional[Any]:
        """
        Load a model instance.

        Args:
            model_name (str): The unique name of the model.
            **kwargs: Additional arguments passed to the model loading function
                      (e.g., device='cuda' for transformers).

        Returns:
            Optional[Any]: The loaded model object, or None if loading fails.
        """
        model_info = self.get_model_info(model_name)
        if not model_info:
            logger.error(f"Model '{model_name}' not found in registry.")
            return None

        task = model_info['task']
        source = model_info['source']
        framework = model_info['framework']

        try:
            if source == 'huggingface' and model_info['hf_id']:
                hf_id = model_info['hf_id']
                logger.info(f"Loading Hugging Face model '{hf_id}' for task '{task}'...")
                # Use transformers library to load
                # Ensure transformers is installed: pip install transformers torch or tensorflow
                from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification, pipeline

                # Determine the correct AutoClass based on the task
                if task == 'pos_tag' or task == 'ner':
                    model_class = AutoModelForTokenClassification
                elif task == 'sentiment':
                    model_class = AutoModelForSequenceClassification
                # Add more task mappings as needed (e.g., translation, qa)
                else:
                    # Default or attempt generic load (might need adjustment)
                    logger.warning(f"Task '{task}' not explicitly mapped. Attempting AutoModel.")
                    model_class = AutoModel

                # Load tokenizer and model
                # Let transformers handle caching via its default mechanism or our specified dir
                # Note: Transformers uses its own cache logic, might differ slightly from hf_hub_download path
                cache_path = self.cache_dir / "huggingface"
                tokenizer = AutoTokenizer.from_pretrained(hf_id, cache_dir=cache_path)
                model = model_class.from_pretrained(hf_id, cache_dir=cache_path, **kwargs)

                # Optionally return a pipeline for easier use
                # task_map = {'ner': 'ner', 'pos_tag': 'token-classification', 'sentiment': 'sentiment-analysis'}
                # if task in task_map:
                #     nlp_pipeline = pipeline(task_map[task], model=model, tokenizer=tokenizer, **kwargs)
                #     logger.info(f"Hugging Face pipeline for task '{task}' loaded successfully.")
                #     return nlp_pipeline
                # else:
                logger.info(f"Hugging Face model '{hf_id}' loaded successfully.")
                return model, tokenizer # Return model and tokenizer separately

            elif source == 'thainlp':
                # Load built-in models
                logger.info(f"Loading built-in model '{model_name}' for task '{task}'...")
                if task == 'pos_tag' and model_name == 'hmm_pos':
                    from thainlp.pos_tagging import hmm_tagger
                    return hmm_tagger.HMMPOSTagger()
                elif task == 'ner' and model_name == 'rule_based_ner':
                    from thainlp.ner import rule_based
                    return rule_based.RuleBasedNER()
                # Add other built-in models here
                else:
                    logger.error(f"Built-in model '{model_name}' for task '{task}' not recognized.")
                    return None
            else:
                logger.error(f"Loading not implemented for source '{source}'.")
                return None

        except ImportError as e:
             logger.error(f"Import error loading model '{model_name}'. Is '{e.name}' installed? Try 'pip install {e.name}' or 'pip install thainlp[hf]'")
             return None
        except Exception as e:
            logger.error(f"Failed to load model '{model_name}': {e}")
            return None

# Example usage:
if __name__ == "__main__":
    manager = ModelManager()

    print("--- Listing Models ---")
    models = manager.list_available_models(task='pos_tag')
    print(models)

    print("\n--- Getting Model Path (Example for HF model if registered) ---")
    # Need to register an HF model first for this to work
    # path = manager.get_model_path("wangchanberta_pos")
    # print(f"Path: {path}")
    path_hmm = manager.get_model_path("hmm_pos")
    print(f"Path for hmm_pos: {path_hmm}") # Expected: None (built-in)


    print("\n--- Loading Models ---")
    # Load built-in HMM POS tagger
    hmm_tagger = manager.load_model("hmm_pos")
    if hmm_tagger:
        print(f"Loaded hmm_pos: {type(hmm_tagger)}")
        # Example tag
        # print(hmm_tagger.tag(["ฉัน", "กิน", "ข้าว"]))

    # Load built-in Rule-based NER tagger
    rb_ner = manager.load_model("rule_based_ner")
    if rb_ner:
        print(f"Loaded rule_based_ner: {type(rb_ner)}")
        # Example tag
        # print(rb_ner.tag(["นาย", "สมชาย", "ไป", "กรุงเทพ"]))

    # Example loading a Hugging Face model (requires registration and dependencies)
    # print("\n--- Loading HF Model (Example) ---")
    # Make sure 'transformers' and 'torch'/'tensorflow' are installed
    # And register e.g., "wangchanberta_pos" in registry.py
    # hf_model_tuple = manager.load_model("wangchanberta_pos")
    # if hf_model_tuple:
    #     hf_model, hf_tokenizer = hf_model_tuple
    #     print(f"Loaded HF model: {type(hf_model)}")
    #     print(f"Loaded HF tokenizer: {type(hf_tokenizer)}")
