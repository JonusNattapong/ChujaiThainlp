from typing import List, Dict, Union, Optional
from transformers import (
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline
)
import torch
from torch.nn import functional as F
import numpy as np

class ThaiNamedEntityRecognition:
    """Extract named entities from Thai text"""

    def __init__(self, model_name_or_path: str = "airesearch/wangchanberta-base-att-spm-uncased-ner"):
        """Initialize NER model"""
        self.model = AutoModelForTokenClassification.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.ner_pipeline = pipeline("ner", model=self.model, tokenizer=self.tokenizer, grouped_entities=True)

    def extract_entities(self, text: str) -> List[Dict[str, str]]:
        """Extract named entities
        
        Args:
            text: Input Thai text
            
        Returns:
            List of dictionaries with entity information
        """
        results = self.ner_pipeline(text)
        entities = []
        for result in results:
            entity = {
                "entity": result["entity_group"],
                "word": result["word"],
                "start": result["start"],
                "end": result["end"],
                "score": result["score"]
            }
            entities.append(entity)
        return entities

class ThaiSentimentAnalyzer:
    """Analyze sentiment in Thai text"""
    
    def __init__(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "airesearch/wangchanberta-base-att-spm-uncased-finetuned-sentiment"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "airesearch/wangchanberta-base-att-spm-uncased-finetuned-sentiment"
        )
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=self.model,
            tokenizer=self.tokenizer
        )

    def analyze_sentiment(self, text: str) -> Dict[str, Union[str, float]]:
        """Analyze sentiment of text
        
        Args:
            text: Input Thai text
            
        Returns:
            Dictionary with sentiment label and score
        """
        result = self.sentiment_pipeline(text)[0]
        return {
            "label": result["label"],
            "score": result["score"],
            "sentiment": "positive" if result["label"] == "POS" else "negative"
        }

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

class ThaiTextAnalyzer:
    """Analyze Thai text"""
    
    def __init__(self):
        self.model_name = "airesearch/wangchanberta-base-att-spm-uncased"
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        # Encode texts
        encoding1 = self.tokenizer(text1, return_tensors="pt", padding=True, truncation=True)
        encoding2 = self.tokenizer(text2, return_tensors="pt", padding=True, truncation=True)
        
        # Get embeddings
        with torch.no_grad():
            embedding1 = self.model(**encoding1).logits
            embedding2 = self.model(**encoding2).logits
            
        # Calculate cosine similarity
        similarity = F.cosine_similarity(embedding1, embedding2).item()
        return (similarity + 1) / 2  # Normalize to [0,1]

class TopicModeling:
    """Topic modeling for Thai text"""
    pass

class EmotionDetector:
    """Emotion detection for Thai text"""
    pass

class AdvancedThaiNLP:
    """Advanced Thai NLP functionality"""
    pass
