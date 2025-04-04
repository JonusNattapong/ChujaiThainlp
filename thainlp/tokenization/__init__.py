"""
Thai and English tokenization package
"""
from typing import List, Union, Optional
from .tokenizer import ThaiTokenizer
from .maximum_matching import MaximumMatchingTokenizer

# Default tokenizer instances
_default_tokenizer = None
_default_thai_tokenizer = None

def get_tokenizer(model_name: Optional[str] = None) -> ThaiTokenizer:
    """Get or create default tokenizer
    
    Args:
        model_name: Optional model name for transformer tokenizer
        
    Returns:
        ThaiTokenizer instance
    """
    global _default_tokenizer
    if _default_tokenizer is None:
        _default_tokenizer = ThaiTokenizer(
            model_name=model_name or "xlm-roberta-base"
        )
    return _default_tokenizer

def get_thai_tokenizer() -> MaximumMatchingTokenizer:
    """Get or create default Thai tokenizer
    
    Returns:
        MaximumMatchingTokenizer instance
    """
    global _default_thai_tokenizer
    if _default_thai_tokenizer is None:
        _default_thai_tokenizer = MaximumMatchingTokenizer()
    return _default_thai_tokenizer

def word_tokenize(
    text: Union[str, List[str]],
    engine: str = "default"
) -> Union[List[str], List[List[str]]]:
    """Tokenize text into words
    
    Args:
        text: Input text or list of texts
        engine: Tokenization engine:
               - "default": Use unified tokenizer
               - "thai": Use Thai-specific tokenizer
               - "subword": Use subword tokenization
               
    Returns:
        List of tokens or list of token lists
    """
    if engine == "thai":
        tokenizer = get_thai_tokenizer()
        
        if isinstance(text, str):
            return tokenizer.tokenize(text)
        return [tokenizer.tokenize(t) for t in text]
        
    tokenizer = get_tokenizer()
    
    if engine == "subword":
        return tokenizer.tokenize(text, mode="subword")
    return tokenizer.tokenize(text, mode="word")

def encode(
    text: Union[str, List[str]],
    padding: bool = True,
    truncation: bool = True,
    max_length: Optional[int] = None,
    return_tensors: Optional[str] = None
) -> Union[List[int], dict]:
    """Encode text to model inputs
    
    Args:
        text: Input text or texts
        padding: Whether to pad sequences
        truncation: Whether to truncate sequences
        max_length: Maximum sequence length 
        return_tensors: Return format ('pt' for PyTorch)
        
    Returns:
        Token ids or model inputs dict
    """
    tokenizer = get_tokenizer()
    return tokenizer.encode(
        text,
        padding=padding,
        truncation=truncation,
        max_length=max_length,
        return_tensors=return_tensors
    )

def decode(
    token_ids: List[int],
    skip_special_tokens: bool = True
) -> str:
    """Decode token ids back to text
    
    Args:
        token_ids: List of token ids
        skip_special_tokens: Whether to remove special tokens
        
    Returns:
        Decoded text
    """
    tokenizer = get_tokenizer()
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=skip_special_tokens
    )

__all__ = [
    'ThaiTokenizer',
    'MaximumMatchingTokenizer',
    'get_tokenizer',
    'get_thai_tokenizer',
    'word_tokenize',
    'encode',
    'decode'
]
