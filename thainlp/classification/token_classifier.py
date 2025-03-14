"""
Token Classification for Thai Text
"""

from typing import List, Dict, Tuple, Union, Optional
import re
from collections import defaultdict

class ThaiTokenClassifier:
    def __init__(self):
        """Initialize ThaiTokenClassifier"""
        # Predefined patterns for token classification
        self.patterns = {
            'POS': {  # Part-of-Speech patterns
                'NOUN': [r'[ก-๛]+', r'การ[ก-๛]+', r'ความ[ก-๛]+', r'[ก-๛]+การ', r'[ก-๛]+ภาพ'],
                'VERB': [r'[ก-๛]+', r'[ก-๛]+ไป', r'[ก-๛]+มา', r'[ก-๛]+ขึ้น', r'[ก-๛]+ลง'],
                'ADJ': [r'[ก-๛]+', r'[ก-๛]+มาก', r'[ก-๛]+ที่สุด', r'น่า[ก-๛]+'],
                'ADV': [r'[ก-๛]+ๆ', r'อย่าง[ก-๛]+', r'[ก-๛]+แล้ว'],
                'PRON': [r'ผม', r'ฉัน', r'เขา', r'เธอ', r'มัน', r'เรา', r'พวกเขา', r'พวกเรา'],
                'DET': [r'นี้', r'นั้น', r'โน้น', r'นู้น', r'ทั้งหมด', r'บาง', r'ทุก'],
                'NUM': [r'\d+', r'หนึ่ง', r'สอง', r'สาม', r'สี่', r'ห้า', r'หก', r'เจ็ด', r'แปด', r'เก้า', r'สิบ'],
                'PUNCT': [r'[.,!?;:"\'\(\)\[\]\{\}]']
            },
            'NER': {  # Named Entity Recognition patterns
                'PERSON': [r'นาย[ก-๛]+', r'นาง[ก-๛]+', r'นางสาว[ก-๛]+', r'คุณ[ก-๛]+'],
                'LOCATION': [r'จังหวัด[ก-๛]+', r'อำเภอ[ก-๛]+', r'ตำบล[ก-๛]+', r'ถนน[ก-๛]+'],
                'ORGANIZATION': [r'บริษัท[ก-๛]+', r'มหาวิทยาลัย[ก-๛]+', r'โรงเรียน[ก-๛]+', r'โรงพยาบาล[ก-๛]+'],
                'DATE': [r'\d{1,2}/\d{1,2}/\d{2,4}', r'\d{1,2}-\d{1,2}-\d{2,4}', r'\d{1,2} [ก-๛]+ \d{2,4}'],
                'TIME': [r'\d{1,2}:\d{2}', r'\d{1,2} นาฬิกา', r'\d{1,2} โมง'],
                'MONEY': [r'\d+(\.\d+)? บาท', r'\d+(\.\d+)? ดอลลาร์', r'\d+(\.\d+)? เหรียญ'],
                'PERCENT': [r'\d+(\.\d+)?%', r'\d+(\.\d+)? เปอร์เซ็นต์'],
                'EMAIL': [r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'],
                'PHONE': [r'\d{3}-\d{3}-\d{4}', r'\d{3}-\d{7}', r'\+\d{1,2}-\d{3}-\d{3}-\d{4}'],
                'URL': [r'https?://[^\s]+', r'www\.[^\s]+']
            }
        }
        
        # Known entities for each NER tag
        self.known_entities = {
            'PERSON': [
                'สมชาย', 'สมหญิง', 'วิชัย', 'วิชชุดา', 'ประยุทธ์', 'ยิ่งลักษณ์', 'ทักษิณ',
                'สมคิด', 'สมศักดิ์', 'สมบัติ', 'สมพร', 'สมใจ', 'สมปอง', 'สมหมาย'
            ],
            'LOCATION': [
                'กรุงเทพ', 'เชียงใหม่', 'ภูเก็ต', 'พัทยา', 'หัวหิน', 'เกาะสมุย', 'เกาะพงัน',
                'ขอนแก่น', 'อุดรธานี', 'นครราชสีมา', 'สงขลา', 'ประเทศไทย', 'ลาว', 'กัมพูชา'
            ],
            'ORGANIZATION': [
                'จุฬาลงกรณ์มหาวิทยาลัย', 'มหาวิทยาลัยธรรมศาสตร์', 'มหาวิทยาลัยมหิดล',
                'มหาวิทยาลัยเกษตรศาสตร์', 'บริษัท ปตท. จำกัด', 'บริษัท ทรู คอร์ปอเรชั่น',
                'โรงพยาบาลศิริราช', 'โรงพยาบาลรามาธิบดี', 'โรงเรียนเตรียมอุดมศึกษา'
            ]
        }
        
        # POS tag dictionary
        self.pos_dict = {
            'NOUN': [
                'บ้าน', 'รถ', 'คน', 'หมา', 'แมว', 'โต๊ะ', 'เก้าอี้', 'ต้นไม้', 'ดอกไม้',
                'อาหาร', 'น้ำ', 'ความรัก', 'ความสุข', 'การเรียน', 'การทำงาน'
            ],
            'VERB': [
                'กิน', 'นอน', 'เดิน', 'วิ่ง', 'พูด', 'ฟัง', 'อ่าน', 'เขียน', 'ดู', 'เห็น',
                'ได้', 'เป็น', 'มี', 'ทำ', 'ไป', 'มา', 'ซื้อ', 'ขาย', 'ชอบ', 'รัก'
            ],
            'ADJ': [
                'ดี', 'สวย', 'หล่อ', 'น่ารัก', 'ใหญ่', 'เล็ก', 'สูง', 'ต่ำ', 'หนัก', 'เบา',
                'แข็ง', 'นุ่ม', 'ร้อน', 'เย็น', 'เผ็ด', 'หวาน', 'เปรี้ยว', 'ขม'
            ],
            'ADV': [
                'มาก', 'น้อย', 'เร็ว', 'ช้า', 'ดีมาก', 'เยอะ', 'นิดหน่อย', 'ค่อนข้าง',
                'เกือบ', 'แทบ', 'ทันที', 'บ่อย', 'นานๆ', 'เสมอ', 'ไม่'
            ],
            'PRON': [
                'ผม', 'ฉัน', 'เขา', 'เธอ', 'มัน', 'เรา', 'พวกเขา', 'พวกเรา', 'ที่', 'อัน',
                'ตัว', 'คน', 'ใคร', 'อะไร', 'ไหน', 'เมื่อไร', 'อย่างไร', 'ทำไม'
            ],
            'CONJ': [
                'และ', 'หรือ', 'แต่', 'เพราะ', 'เนื่องจาก', 'ดังนั้น', 'เพราะฉะนั้น',
                'ถ้า', 'ถึงแม้', 'แม้ว่า', 'ก็ตาม', 'จนกระทั่ง', 'ตั้งแต่', 'เมื่อ'
            ],
            'PREP': [
                'ใน', 'นอก', 'บน', 'ล่าง', 'ข้าง', 'หน้า', 'หลัง', 'ระหว่าง', 'ท่ามกลาง',
                'ด้วย', 'โดย', 'แก่', 'สำหรับ', 'กับ', 'แด่', 'ต่อ', 'จาก', 'ถึง'
            ]
        }
        
    def _match_pattern(self, token: str, patterns: List[str]) -> bool:
        """
        Check if token matches any pattern
        
        Args:
            token (str): Input token
            patterns (List[str]): List of regex patterns
            
        Returns:
            bool: True if token matches any pattern
        """
        for pattern in patterns:
            if re.fullmatch(pattern, token):
                return True
        return False
        
    def _is_in_dict(self, token: str, word_list: List[str]) -> bool:
        """
        Check if token is in dictionary
        
        Args:
            token (str): Input token
            word_list (List[str]): List of words
            
        Returns:
            bool: True if token is in dictionary
        """
        return token in word_list
        
    def classify_tokens(self, tokens: List[str], task: str = 'POS') -> List[Tuple[str, str]]:
        """
        Classify tokens based on task
        
        Args:
            tokens (List[str]): List of tokens
            task (str): Classification task ('POS' or 'NER')
            
        Returns:
            List[Tuple[str, str]]: List of (token, tag) pairs
        """
        if task not in ['POS', 'NER']:
            raise ValueError(f"Task '{task}' not supported. Available tasks: ['POS', 'NER']")
            
        results = []
        
        for token in tokens:
            # Default tag
            tag = 'O' if task == 'NER' else 'NOUN'
            
            # Check patterns
            for t, patterns in self.patterns[task].items():
                if self._match_pattern(token, patterns):
                    tag = t
                    break
                    
            # Check dictionaries
            if task == 'POS':
                for t, word_list in self.pos_dict.items():
                    if self._is_in_dict(token, word_list):
                        tag = t
                        break
            elif task == 'NER':
                for t, entities in self.known_entities.items():
                    if self._is_in_dict(token, entities):
                        tag = t
                        break
                        
            results.append((token, tag))
            
        # Post-processing for NER (BIO tagging)
        if task == 'NER':
            bio_results = []
            prev_tag = 'O'
            
            for token, tag in results:
                if tag != 'O':
                    if prev_tag != tag:
                        bio_tag = f'B-{tag}'  # Beginning of entity
                    else:
                        bio_tag = f'I-{tag}'  # Inside entity
                else:
                    bio_tag = 'O'
                    
                bio_results.append((token, bio_tag))
                prev_tag = tag
                
            return bio_results
            
        return results
        
    def find_entities(self, tokens: List[str]) -> List[Dict[str, Union[str, int, float]]]:
        """
        Find named entities in tokens
        
        Args:
            tokens (List[str]): List of tokens
            
        Returns:
            List[Dict[str, Union[str, int, float]]]: List of entities with type, text, start, end, score
        """
        # Get BIO tags
        tagged_tokens = self.classify_tokens(tokens, task='NER')
        
        entities = []
        current_entity = None
        
        for i, (token, tag) in enumerate(tagged_tokens):
            if tag.startswith('B-'):  # Beginning of entity
                if current_entity:
                    entities.append(current_entity)
                    
                entity_type = tag[2:]  # Remove 'B-'
                current_entity = {
                    'entity': entity_type,
                    'text': token,
                    'start': i,
                    'end': i,
                    'score': 1.0  # Confidence score
                }
            elif tag.startswith('I-'):  # Inside entity
                if current_entity and current_entity['entity'] == tag[2:]:
                    current_entity['text'] += ' ' + token
                    current_entity['end'] = i
                    
            elif tag == 'O' and current_entity:  # Outside entity
                entities.append(current_entity)
                current_entity = None
                
        # Add last entity if exists
        if current_entity:
            entities.append(current_entity)
            
        return entities

def classify_tokens(tokens: List[str], task: str = 'POS') -> List[Tuple[str, str]]:
    """
    Classify tokens based on task
    
    Args:
        tokens (List[str]): List of tokens
        task (str): Classification task ('POS' or 'NER')
        
    Returns:
        List[Tuple[str, str]]: List of (token, tag) pairs
    """
    classifier = ThaiTokenClassifier()
    return classifier.classify_tokens(tokens, task)

def find_entities(tokens: List[str]) -> List[Dict[str, Union[str, int, float]]]:
    """
    Find named entities in tokens
    
    Args:
        tokens (List[str]): List of tokens
        
    Returns:
        List[Dict[str, Union[str, int, float]]]: List of entities with type, text, start, end, score
    """
    classifier = ThaiTokenClassifier()
    return classifier.find_entities(tokens) 