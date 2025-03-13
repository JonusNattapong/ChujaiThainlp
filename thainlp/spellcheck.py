"""
Spell checking for Thai text.
"""
from typing import List, Dict, Optional
from thainlp.tokenize import word_tokenize
from thainlp.resources import thai_words, load_custom_dictionary

def spellcheck(text: str, custom_dict_path: Optional[str] = None) -> List[str]:
    """
    Spell check Thai text.

    Args:
        text: Thai text to spell check.
        custom_dict_path: Path to a custom dictionary file.

    Returns:
        List of potentially misspelled words.
    """
    tokens = word_tokenize(text)
    words = thai_words()
    if custom_dict_path:
        custom_dict = load_custom_dictionary(custom_dict_path)
        words.update(custom_dict.keys())

    misspelled = []
    for token in tokens:
        if token not in words and not token.isdigit():  # Ignore numbers
            misspelled.append(token)

    return misspelled
