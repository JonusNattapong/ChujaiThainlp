"""
Thai Dialect Module

Support for Thai regional dialects including:
- Northern Thai (คำเมือง/ไทยถิ่นเหนือ)
- Northeastern Thai/Isan (ไทยถิ่นอีสาน)
- Southern Thai (ไทยถิ่นใต้/ปักษ์ใต้)
- Central/Standard Thai (ไทยกลาง)
"""

from .dialect_processor import (
    DialectProcessor,
    DialectIdentifier,
    DialectTranslator,
    detect_dialect,
    translate_dialect,
    identify_dialect,
    ThaiDialectProcessor,
    get_dialect_features,
    get_dialect_info
)

from .dialect_tokenizer import DialectTokenizer

# Create standalone functions for dialect translation
def translate_to_standard(text, dialect):
    """
    Translate text from a specific dialect to standard Thai
    
    Args:
        text (str): Text in dialect
        dialect (str): Source dialect name
        
    Returns:
        str: Text translated to standard Thai
    """
    processor = ThaiDialectProcessor()
    return processor.translate_to_standard(text, dialect)

def translate_from_standard(text, dialect):
    """
    Translate text from standard Thai to a specific dialect
    
    Args:
        text (str): Text in standard Thai
        dialect (str): Target dialect name
        
    Returns:
        str: Text translated to specified dialect
    """
    processor = ThaiDialectProcessor()
    return processor.translate_from_standard(text, dialect)

__all__ = [
    'DialectProcessor',
    'DialectIdentifier',
    'DialectTranslator',
    'detect_dialect',
    'translate_dialect',
    'identify_dialect',
    'DialectTokenizer',
    'ThaiDialectProcessor',
    'translate_to_standard',
    'translate_from_standard',
    'get_dialect_features',
    'get_dialect_info'
]