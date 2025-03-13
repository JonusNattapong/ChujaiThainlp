"""
Rule-based Named Entity Recognition for Thai Text
"""

from typing import List, Dict, Tuple
import re

class ThaiNER:
    def __init__(self):
        """Initialize ThaiNER with patterns and dictionaries"""
        self.patterns = {
            'PERSON': [
                r'คุณ\s+([ก-๛]+)',
                r'นาย\s+([ก-๛]+)',
                r'นาง\s+([ก-๛]+)',
                r'นางสาว\s+([ก-๛]+)',
            ],
            'LOCATION': [
                r'จังหวัด\s+([ก-๛]+)',
                r'อำเภอ\s+([ก-๛]+)',
                r'ตำบล\s+([ก-๛]+)',
                r'ถนน\s+([ก-๛]+)',
            ],
            'ORGANIZATION': [
                r'บริษัท\s+([ก-๛]+)',
                r'มหาวิทยาลัย\s+([ก-๛]+)',
                r'โรงเรียน\s+([ก-๛]+)',
                r'โรงพยาบาล\s+([ก-๛]+)',
            ],
            'DATE': [
                r'\d{1,2}\s+มกราคม',
                r'\d{1,2}\s+กุมภาพันธ์',
                r'\d{1,2}\s+มีนาคม',
                r'\d{1,2}\s+เมษายน',
                r'\d{1,2}\s+พฤษภาคม',
                r'\d{1,2}\s+มิถุนายน',
                r'\d{1,2}\s+กรกฎาคม',
                r'\d{1,2}\s+สิงหาคม',
                r'\d{1,2}\s+กันยายน',
                r'\d{1,2}\s+ตุลาคม',
                r'\d{1,2}\s+พฤศจิกายน',
                r'\d{1,2}\s+ธันวาคม',
            ],
            'TIME': [
                r'\d{1,2}:\d{2}',
                r'\d{1,2}\s+นาฬิกา',
                r'เช้า|กลางวัน|เย็น|กลางคืน',
            ],
            'MONEY': [
                r'\d+(?:,\d+)*\s*บาท',
                r'\d+(?:,\d+)*\s*เหรียญ',
            ],
            'URL': [
                r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*)',
            ],
            'HASHTAG': [
                r'#[\wก-๛]+',
            ],
        }
        
        # Dictionary of known entities
        self.known_entities = {
            'PERSON': {
                'สมชาย', 'สมหญิง', 'วิชัย', 'วิชชุดา',
                'ประยุทธ์', 'ยิ่งลักษณ์', 'ทักษิณ',
            },
            'LOCATION': {
                'กรุงเทพ', 'เชียงใหม่', 'ภูเก็ต', 'พัทยา',
                'หัวหิน', 'เกาะสมุย', 'เกาะพงัน',
            },
            'ORGANIZATION': {
                'จุฬาลงกรณ์มหาวิทยาลัย', 'มหาวิทยาลัยธรรมศาสตร์',
                'มหาวิทยาลัยมหิดล', 'มหาวิทยาลัยเกษตรศาสตร์',
                'บริษัท ปตท. จำกัด', 'บริษัท ทรู คอร์ปอเรชั่น',
            },
        }
        
    def _find_pattern_matches(self, text: str) -> List[Tuple[str, str, int, int]]:
        """
        Find all pattern matches in text
        
        Args:
            text (str): Input text
            
        Returns:
            List[Tuple[str, str, int, int]]: List of (entity_type, text, start, end)
        """
        matches = []
        for entity_type, patterns in self.patterns.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text):
                    matches.append((entity_type, match.group(0), match.start(), match.end()))
        return sorted(matches, key=lambda x: x[2])
        
    def _find_dictionary_matches(self, text: str) -> List[Tuple[str, str, int, int]]:
        """
        Find all dictionary matches in text
        
        Args:
            text (str): Input text
            
        Returns:
            List[Tuple[str, str, int, int]]: List of (entity_type, text, start, end)
        """
        matches = []
        for entity_type, entities in self.known_entities.items():
            for entity in entities:
                start = 0
                while True:
                    start = text.find(entity, start)
                    if start == -1:
                        break
                    matches.append((entity_type, entity, start, start + len(entity)))
                    start += 1
        return sorted(matches, key=lambda x: x[2])
        
    def _merge_overlapping_matches(self, matches: List[Tuple[str, str, int, int]]) -> List[Tuple[str, str, int, int]]:
        """
        Merge overlapping matches, keeping the longer one
        
        Args:
            matches (List[Tuple[str, str, int, int]]): List of matches
            
        Returns:
            List[Tuple[str, str, int, int]]: Merged matches
        """
        if not matches:
            return []
            
        merged = [matches[0]]
        for match in matches[1:]:
            prev = merged[-1]
            if match[2] <= prev[3]:  # Overlapping
                if match[3] - match[2] > prev[3] - prev[2]:  # Current match is longer
                    merged[-1] = match
            else:
                merged.append(match)
        return merged
        
    def extract_entities(self, text: str) -> List[Tuple[str, str, int, int]]:
        """
        Extract named entities from text
        
        Args:
            text (str): Input text
            
        Returns:
            List[Tuple[str, str, int, int]]: List of (entity_type, text, start, end)
        """
        pattern_matches = self._find_pattern_matches(text)
        dict_matches = self._find_dictionary_matches(text)
        all_matches = sorted(pattern_matches + dict_matches, key=lambda x: x[2])
        return self._merge_overlapping_matches(all_matches)

def extract_entities(text: str) -> List[Tuple[str, str, int, int]]:
    """
    Extract named entities from text
    
    Args:
        text (str): Input text
        
    Returns:
        List[Tuple[str, str, int, int]]: List of (entity_type, text, start, end)
    """
    ner = ThaiNER()
    return ner.extract_entities(text) 