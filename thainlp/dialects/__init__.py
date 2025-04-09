"""
Thai Dialect Module

Support for Thai regional dialects including:
- Northern Thai (คำเมือง/ไทยถิ่นเหนือ)
- Northeastern Thai/Isan (ไทยถิ่นอีสาน)
- Southern Thai (ไทยถิ่นใต้/ปักษ์ใต้)
- Central/Standard Thai (ไทยกลาง)
"""

from .dialect_processor import (
    ThaiDialectProcessor,
    detect_dialect,
    translate_to_standard,
    translate_from_standard,
    get_dialect_features,
    get_dialect_info
)

from .dialect_tokenizer import DialectTokenizer

__all__ = [
    'ThaiDialectProcessor',
    'DialectTokenizer',
    'detect_dialect',
    'translate_to_standard',
    'translate_from_standard',
    'get_dialect_features',
    'get_dialect_info'
]