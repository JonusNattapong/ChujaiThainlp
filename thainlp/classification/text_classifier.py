"""
Text Classification for Thai Text using Transformer models
"""

from typing import List, Dict, Union, Optional
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

class ThaiTextClassifier:
    """Text classifier for Thai text using transformer models"""
    
    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        num_labels: Optional[int] = None,
        **kwargs
    ):
        """Initialize text classifier
        
        Args:
            model_name_or_path: Name or path of the model
            num_labels: Number of labels for classification
            **kwargs: Additional arguments for model initialization
        """
        if model_name_or_path is None:
            model_name_or_path = self.get_default_model()
            
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        
    @staticmethod
    def get_default_model() -> str:
        """Get default model for Thai text classification"""
        return "airesearch/wangchanberta-base-att-spm-uncased"
    
    def classify(
        self,
        texts: Union[str, List[str]],
        labels: Optional[List[str]] = None,
        **kwargs
    ) -> List[Dict[str, float]]:
        """Classify text into predefined categories
        
        Args:
            texts: Input text or list of texts
            labels: Optional list of label names
            **kwargs: Additional arguments for encoding
            
        Returns:
            List of dictionaries containing classification probabilities
        """
        # Handle single text input
        if isinstance(texts, str):
            texts = [texts]
            
        # Encode texts
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        
        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
            
        # Convert to list of dictionaries
        results = []
        for prob in probs:
            if labels:
                result = {label: float(p) for label, p in zip(labels, prob)}
            else:
                result = {str(i): float(p) for i, p in enumerate(prob)}
            results.append(result)
            
        return results[0] if isinstance(texts, str) else results
            
    def zero_shot_classify(
        self,
        texts: Union[str, List[str]],
        candidate_labels: List[str],
        hypothesis_template: str = "นี่คือเรื่องเกี่ยวกับ{}",
        multi_label: bool = False,
        **kwargs
    ) -> Union[Dict[str, List], List[Dict[str, List]]]:
        """Zero-shot classification using natural language inference
        
        Args:
            texts: Input text or list of texts
            candidate_labels: List of possible labels
            hypothesis_template: Template for hypothesis generation
            multi_label: Whether to allow multiple labels
            **kwargs: Additional arguments for encoding
            
        Returns:
            Dictionary or list of dictionaries containing classification results
        """
        classifier = pipeline(
            "zero-shot-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
        
        if isinstance(texts, str):
            result = classifier(
                texts,
                candidate_labels,
                hypothesis_template=hypothesis_template,
                multi_label=multi_label
            )
            return {
                "labels": result["labels"],
                "scores": result["scores"]
            }
            
        results = []
        for text in texts:
            result = classifier(
                text,
                candidate_labels,
                hypothesis_template=hypothesis_template,
                multi_label=multi_label
            )
            results.append({
                "labels": result["labels"],
                "scores": result["scores"]
            })
        return results
