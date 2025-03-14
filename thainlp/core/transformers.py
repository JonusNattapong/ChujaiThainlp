"""
Core functionality for transformer models in ThaiNLP
"""

from typing import List, Dict, Union, Optional
import torch
from transformers import (
    AutoModel, 
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoModelForMaskedLM,
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer
)

DEFAULT_MODELS = {
    "text-classification": "airesearch/wangchanberta-base-att-spm-uncased",
    "token-classification": "airesearch/wangchanberta-base-att-spm-uncased",
    "question-answering": "airesearch/wangchanberta-base-att-spm-uncased",
    "translation": "airesearch/wmt-thai-translation",
    "summarization": "airesearch/wangchanberta-base-att-spm-uncased",
    "fill-mask": "airesearch/wangchanberta-base-att-spm-uncased",
    "text-generation": "airesearch/gpt-thai-small",
    "sentence-similarity": "airesearch/sbert-base-thai-qa",
}

class TransformerBase:
    """Base class for transformer models"""
    
    def __init__(
        self,
        model_name_or_path: str,
        task_type: str,
        device: Optional[str] = None,
        **kwargs
    ):
        """Initialize transformer model
        
        Args:
            model_name_or_path: Name or path of the model
            task_type: Type of task (classification, token-classification, etc.)
            device: Device to use (cuda/cpu)
            **kwargs: Additional arguments for model initialization
        """
        self.model_name = model_name_or_path
        self.task_type = task_type
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        
        # Load model based on task type
        model_class = self._get_model_class()
        self.model = model_class.from_pretrained(model_name_or_path, **kwargs)
        self.model.to(self.device)
        
    def _get_model_class(self) -> PreTrainedModel:
        """Get appropriate model class based on task type"""
        task_to_class = {
            "text-classification": AutoModelForSequenceClassification,
            "token-classification": AutoModelForTokenClassification,
            "question-answering": AutoModelForQuestionAnswering,
            "translation": AutoModelForSeq2SeqLM,
            "summarization": AutoModelForSeq2SeqLM,
            "fill-mask": AutoModelForMaskedLM,
            "text-generation": AutoModelForCausalLM,
            "sentence-similarity": AutoModel,
        }
        return task_to_class.get(self.task_type, AutoModel)
    
    def encode(
        self,
        texts: Union[str, List[str]],
        padding: bool = True,
        truncation: bool = True,
        max_length: int = 512,
        return_tensors: str = "pt",
        **kwargs
    ) -> Dict:
        """Encode text using tokenizer
        
        Args:
            texts: Input text or list of texts
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences
            max_length: Maximum sequence length
            return_tensors: Type of tensors to return
            **kwargs: Additional arguments for tokenizer
            
        Returns:
            Dictionary containing encoded inputs
        """
        # Handle single text input
        if isinstance(texts, str):
            texts = [texts]
            
        # Encode texts
        encoded = self.tokenizer(
            texts,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
            **kwargs
        )
        
        # Move to device
        return {k: v.to(self.device) for k, v in encoded.items()}
    
    def save(self, path: str):
        """Save model and tokenizer
        
        Args:
            path: Path to save model and tokenizer
        """
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        task_type: str,
        **kwargs
    ) -> "TransformerBase":
        """Load pretrained model
        
        Args:
            model_name_or_path: Name or path of the model
            task_type: Type of task
            **kwargs: Additional arguments
            
        Returns:
            TransformerBase instance
        """
        return cls(model_name_or_path, task_type, **kwargs)
    
    @classmethod
    def from_task(
        cls,
        task_type: str,
        **kwargs
    ) -> "TransformerBase":
        """Load default model for task
        
        Args:
            task_type: Type of task
            **kwargs: Additional arguments
            
        Returns:
            TransformerBase instance
        """
        model_name = DEFAULT_MODELS.get(task_type)
        if not model_name:
            raise ValueError(f"No default model found for task: {task_type}")
        return cls.from_pretrained(model_name, task_type, **kwargs) 