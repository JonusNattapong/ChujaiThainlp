"""
Thai text generation using transformer models
"""
from typing import List, Union
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer
)
import torch

class ThaiTextGenerator:
    """Generate Thai text"""
    
    def __init__(self):
        self.model_name = "airesearch/wangchanberta-base-att-spm-uncased"
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def generate_text(self, prompt: str = "", max_length: int = 50, num_return_sequences: int = 1,
                     temperature: float = 1.0, method: str = "auto", **kwargs) -> Union[str, List[str]]:
        """Generate text using different methods
        
        Args:
            prompt: Starting text
            max_length: Maximum length of generated text
            num_return_sequences: Number of sequences to generate
            temperature: Sampling temperature
            method: Generation method ("auto", "template", "pattern")
            
        Returns:
            Generated text or list of texts
        """
        # Simple template-based generation for now
        if method == "template":
            templates = {
                "greeting": ["สวัสดีครับ", "สวัสดีค่ะ", "ยินดีต้อนรับ"],
                "farewell": ["ลาก่อน", "แล้วเจอกัน", "ขอบคุณครับ/ค่ะ"]
            }
            return np.random.choice(templates.get(kwargs.get("template_type", "greeting")))
            
        # Default generation
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            temperature=temperature,
            do_sample=True
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def summarize(self, text: str, max_length: int = 100) -> str:
        """Summarize Thai text
        
        Args:
            text: Input text to summarize
            max_length: Maximum summary length
            
        Returns:
            Summarized text
        """
        inputs = self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
        summary_ids = self.model.generate(inputs["input_ids"], max_length=max_length, min_length=30, length_penalty=2.0)
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)