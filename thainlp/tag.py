"""
Thai part-of-speech tagging
"""
from typing import List, Tuple
from .tokenization import word_tokenize
from .pos_tagging.hmm_tagger import HMMTagger

# Initialize default tagger
_default_tagger = HMMTagger()

def pos_tag(text: str, tokenize: bool = True) -> List[Tuple[str, str]]:
    """Tag parts of speech in Thai text
    
    Args:
        text (str): Input Thai text
        tokenize (bool): Whether to tokenize the text first
        
    Returns:
        List[Tuple[str, str]]: List of (word, tag) tuples
    """
    if tokenize:
        tokens = word_tokenize(text)
    else:
        tokens = text.split()
        
    return _default_tagger.tag(tokens)