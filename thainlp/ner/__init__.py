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

# Define what gets imported with 'from thainlp.ner import *'
__all__ = [
    "RuleBasedThaiNER",
    "extract_entities_rule_based",
    "tag_transformer",
    "DEFAULT_TRANSFORMER_MODEL",
]

# You might want a top-level function that selects the model based on an argument
# For example:
# def tag(text: str, model: str = DEFAULT_TRANSFORMER_MODEL) -> list:
#     if model == "rule_based":
#         # Need to adapt output format if necessary
#         return extract_entities_rule_based(text)
#     elif model == DEFAULT_TRANSFORMER_MODEL or model == "wangchanberta_ner":
#         return tag_transformer(text, model_name=model)
#     else:
#         # Try loading via model hub if it's a registered HF model?
#         # Or raise error?
#         raise ValueError(f"Unsupported NER model: {model}")

# For now, keeping them separate is clearer. Users can import specifically.
