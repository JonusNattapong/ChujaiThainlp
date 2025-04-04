"""
Thai sentiment analysis using transformer models
"""
from typing import Dict, Union
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline
)

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