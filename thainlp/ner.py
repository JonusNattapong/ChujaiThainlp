"""
Thai Named Entity Recognition functionality
"""

from typing import List, Tuple, Dict
import re
from .resources import THAI_NER_PATTERNS

# Common Thai location suffixes
THAI_LOCATION_SUFFIXES = {
    'เมือง': 1.0,
    'จังหวัด': 1.0,
    'อำเภอ': 1.0,
    'ตำบล': 1.0,
    'หมู่บ้าน': 0.8,
    'ซอย': 0.8,
    'ถนน': 0.8,
    'แขวง': 0.8,
    'เขต': 0.8,
    'เกาะ': 0.8,
    'ภูเขา': 0.8,
    'แม่น้ำ': 0.8,
    'ทะเล': 0.8,
    'หาด': 0.8,
    'ป่า': 0.8,
}

# Common Thai organization prefixes
THAI_ORG_PREFIXES = {
    'บริษัท': 1.0,
    'องค์กร': 1.0,
    'สถาบัน': 1.0,
    'มหาวิทยาลัย': 1.0,
    'โรงเรียน': 0.8,
    'โรงพยาบาล': 0.8,
    'ธนาคาร': 0.8,
    'ร้าน': 0.8,
    'ห้าง': 0.8,
    'สหกรณ์': 0.8,
    'สมาคม': 0.8,
    'มูลนิธิ': 0.8,
    'ชมรม': 0.8,
    'สภา': 0.8,
    'กระทรวง': 0.8,
}

def find_entities(text: str) -> List[Tuple[str, str, int, int, float]]:
    """
    Find named entities in Thai text with enhanced accuracy.
    
    Args:
        text (str): Thai text to analyze
        
    Returns:
        List[Tuple[str, str, int, int, float]]: List of (entity text, entity type, start, end, confidence) tuples
    """
    entities = []
    
    for entity_type, pattern in THAI_NER_PATTERNS.items():
        matches = re.finditer(pattern, text)
        for match in matches:
            entity_text = match.group()
            
            # Calculate confidence score based on pattern matching
            confidence = 1.0
            
            # Apply enhanced rules based on entity type
            if entity_type == 'PERSON':
                # Names should be 2-4 words
                words = entity_text.split()
                if not (2 <= len(words) <= 4):
                    confidence *= 0.8
                    
                # Check for common Thai name patterns
                if not any(word.endswith('ชัย') or word.endswith('พร') or word.endswith('ศักดิ์') for word in words):
                    confidence *= 0.9
                    
            elif entity_type == 'LOCATION':
                # Check for location suffixes
                suffix_found = False
                for suffix, weight in THAI_LOCATION_SUFFIXES.items():
                    if entity_text.endswith(suffix):
                        confidence *= weight
                        suffix_found = True
                        break
                if not suffix_found:
                    confidence *= 0.9
                    
            elif entity_type == 'ORGANIZATION':
                # Check for organization prefixes
                prefix_found = False
                for prefix, weight in THAI_ORG_PREFIXES.items():
                    if entity_text.startswith(prefix):
                        confidence *= weight
                        prefix_found = True
                        break
                if not prefix_found:
                    confidence *= 0.9
                    
            elif entity_type == 'DATE':
                # Dates should be in valid format
                if not re.match(r'^\d{1,2}/\d{1,2}/\d{4}$', entity_text):
                    confidence *= 0.9
                    
            elif entity_type == 'TIME':
                # Time should be in valid format
                if not re.match(r'^\d{1,2}:\d{2}$', entity_text):
                    confidence *= 0.9
                    
            elif entity_type == 'MONEY':
                # Money should have valid currency
                if not any(currency in entity_text for currency in ['บาท', 'USD', 'EUR']):
                    confidence *= 0.9
                    
            elif entity_type == 'PHONE':
                # Phone numbers should be in valid format
                if not re.match(r'^\d{3}-\d{3}-\d{4}$|^\d{10}$', entity_text):
                    confidence *= 0.9
                    
            elif entity_type == 'EMAIL':
                # Emails should be in valid format
                if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', entity_text):
                    confidence *= 0.9
                    
            elif entity_type == 'URL':
                # URLs should be in valid format
                if not re.match(r'^https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*)$', entity_text):
                    confidence *= 0.9
                    
            elif entity_type == 'HASHTAG':
                # Hashtags should start with #
                if not entity_text.startswith('#'):
                    confidence *= 0.9
                    
            elif entity_type == 'MENTION':
                # Mentions should start with @
                if not entity_text.startswith('@'):
                    confidence *= 0.9
            
            # Only add entities with confidence above threshold
            if confidence >= 0.7:
                entities.append((
                    entity_text,
                    entity_type,
                    match.start(),
                    match.end(),
                    confidence
                ))
    
    # Sort entities by start position
    entities.sort(key=lambda x: x[2])
    
    return entities