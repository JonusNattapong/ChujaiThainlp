"""
Thai text summarization functionality
"""

from typing import List, Dict
import re

def summarize(text: str, num_sentences: int = 3) -> str:
    """
    Summarize Thai text by extracting key sentences.
    
    Args:
        text (str): Thai text to summarize
        num_sentences (int): Number of sentences to include in summary
        
    Returns:
        str: Summarized text
    """
    # Split text into sentences
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return ""
    
    # Score sentences based on length and position
    sentence_scores: Dict[str, float] = {}
    for i, sentence in enumerate(sentences):
        # Score based on position (earlier sentences get higher scores)
        position_score = 1.0 / (i + 1)
        
        # Score based on length (longer sentences might be more important)
        length_score = len(sentence.split()) / 10
        
        # Combined score
        sentence_scores[sentence] = position_score + length_score
    
    # Sort sentences by score
    sorted_sentences = sorted(
        sentence_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Get top sentences
    top_sentences = [s[0] for s in sorted_sentences[:num_sentences]]
    
    # Sort by original position
    top_sentences.sort(key=lambda x: sentences.index(x))
    
    # Join sentences
    return " ".join(top_sentences)
