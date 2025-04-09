"""
Named Entity Recognition (NER) Module

Provides access to both rule-based and transformer-based NER models.
"""
# Import from rule-based module
from .rule_based import ThaiNER as RuleBasedThaiNER
from .rule_based import extract_entities as extract_entities_rule_based

# Import from transformer module
from .transformer_ner import tag as tag_transformer
from .transformer_ner import DEFAULT_MODEL as DEFAULT_TRANSFORMER_MODEL

# Define ThaiNamedEntityRecognition class
class ThaiNamedEntityRecognition:
    """
    Thai Named Entity Recognition class for identifying entities in Thai text.
    Wrapper around transformer-based and rule-based NER models.
    """
    
    def __init__(self, model_name=DEFAULT_TRANSFORMER_MODEL, use_rule_based=False):
        """
        Initialize NER model
        
        Args:
            model_name: Name of the transformer model to use
            use_rule_based: Whether to use rule-based NER instead of transformer
        """
        self.model_name = model_name
        self.use_rule_based = use_rule_based
        
    def extract_entities(self, text):
        """
        Extract named entities from text
        
        Args:
            text: Thai text to analyze
            
        Returns:
            List of entities with type and position information
        """
        if self.use_rule_based:
            return extract_entities_rule_based(text)
        else:
            return tag_transformer(text, model_name=self.model_name)
            
    def tag(self, text):
        """Alias for extract_entities"""
        return self.extract_entities(text)

# Define what gets imported with 'from thainlp.ner import *'
__all__ = [
    "ThaiNamedEntityRecognition",
    "RuleBasedThaiNER",
    "extract_entities_rule_based",
    "tag_transformer",
    "DEFAULT_TRANSFORMER_MODEL",
]

# Tag function that selects the model based on an argument
def tag(text: str, model: str = DEFAULT_TRANSFORMER_MODEL) -> list:
    """
    Extract named entities from text
    
    Args:
        text: Thai text to analyze
        model: Model name or 'rule_based'
        
    Returns:
        List of entities with type and position information
    """
    if model == "rule_based":
        return extract_entities_rule_based(text)
    else:
        return tag_transformer(text, model_name=model)
