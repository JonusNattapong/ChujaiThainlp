"""
Advanced text classification for Thai with transformer support and ensemble methods
"""
from typing import Dict, List, Union, Optional, Tuple
import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)
from .model_hub import get_model_info
from .tokenization import word_tokenize
from .thai_preprocessor import ThaiTextPreprocessor

# Initialize preprocessor
_thai_preprocessor = ThaiTextPreprocessor()

class ThaiTextClassifier:
    """Advanced Thai text classifier with transformer support and ensemble methods"""
    
    def __init__(self,
                model_names: Union[str, List[str]] = ["wangchanberta_text_cls", "xlm-roberta-base"],
                device: str = "cuda" if torch.cuda.is_available() else "cpu",
                ensemble_method: str = "weighted_vote"):
        """
        Initialize Thai text classifier with ensemble support
        
        Args:
            model_names: Name or list of model names from registry
            device: Device to run model on
            ensemble_method: Method for combining model predictions ('weighted_vote' or 'average')
        """
        self.models = []
        self.tokenizers = []
        self.device = device
        self.ensemble_method = ensemble_method
        self.labels = []  # Will be populated when loading models

# Rest of the code remains unchanged...
