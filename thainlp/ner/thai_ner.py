"""
Thai Named Entity Recognition using transformer models
"""
from typing import List, Dict
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    pipeline
)

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