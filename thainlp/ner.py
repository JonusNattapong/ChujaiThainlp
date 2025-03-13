"""
Named Entity Recognition (NER) for Thai text.
"""
from typing import List, Tuple, Dict
from thainlp.tokenize import word_tokenize

def find_entities(text: str, engine: str = "dict") -> List[Tuple[str, str, int, int]]:
    """
    Find named entities in Thai text.
    
    Args:
        text: Thai text to process
        engine: Engine for NER ('dict' for dictionary-based)
        
    Returns:
        List of (entity_text, entity_type, start_char, end_char) tuples
    """
    # This is a simplified implementation
    # In a real system, you would use ML models or more sophisticated rules
    
    tokens = word_tokenize(text)
    entities = []
    position = 0
    
    # Very basic NER using dictionary lookup
    person_names = {"สมชาย", "สมหญิง", "สุชาติ", "วิชัย"}
    locations = {"กรุงเทพ", "เชียงใหม่", "ภูเก็ต", "ประเทศไทย", "ไทย"}
    organizations = {"จุฬา", "มหาวิทยาลัย", "บริษัท", "องค์กร"}
    
    for token in tokens:
        start = text.find(token, position)
        end = start + len(token)
        
        if token in person_names:
            entities.append((token, "PERSON", start, end))
        elif token in locations:
            entities.append((token, "LOCATION", start, end))
        elif token in organizations:
            entities.append((token, "ORGANIZATION", start, end))
            
        position = end
        
    return entities