"""
Privacy-Preserving NLP for Thai Language
บทคัดย่อ: โมดูลนี้ใช้สำหรับการประมวลผลข้อความภาษาไทยโดยคำนึงถึงความเป็นส่วนตัว
ด้วยเทคนิคต่างๆ เช่น การปกปิดข้อมูลส่วนบุคคล, การเข้ารหัส และ differential privacy
"""

import re
import random
import hashlib
import numpy as np
from typing import List, Dict, Tuple, Union
import thainlp
from thainlp.tokenization import word_tokenize
from thainlp.extensions.advanced_nlp import ThaiNamedEntityRecognition
from thainlp.dialects import (
    ThaiDialectProcessor,
    DialectTokenizer,
    detect_dialect,
    translate_to_standard
)
from .privacy.model_handler import PrivacyModelHandler

class PrivacyPreservingNLP:
    def __init__(self, epsilon=0.1, noise_scale=0.1, handle_dialects=True, validation_threshold=0.95,
                 model_path: str = None, auto_update: bool = True):
        """
        คลาสสำหรับการประมวลผล NLP แบบปกป้องความเป็นส่วนตัว
        
        Parameters:
        -----------
        epsilon: float
            ค่าพารามิเตอร์สำหรับ differential privacy (ค่าน้อย = ความเป็นส่วนตัวมากขึ้น)
        noise_scale: float
            ระดับของ noise ที่เพิ่มเข้าไปใน vector embedding
        """
        self.epsilon = epsilon
        self.noise_scale = noise_scale
        self.ner = ThaiNamedEntityRecognition()
        self.handle_dialects = handle_dialects
        self.dialect_processor = ThaiDialectProcessor() if handle_dialects else None
        self.dialect_tokenizer = DialectTokenizer() if handle_dialects else None
        
        self.model_path = model_path
        self.auto_update = auto_update
        self.validation_threshold = validation_threshold
        self.fuzzy_threshold = 0.85
        
        # โหลดโมเดลถ้ามีการระบุ path
        if model_path:
            self._load_model()
        
        # สร้างตัวนับสำหรับการปรับปรุงโมเดล
        self.update_counter = 0
        self.update_threshold = 1000  # จำนวนตัวอย่างก่อนการปรับปรุง
        self.training_examples = []
        
        # ข้อมูลส่วนบุคคลที่ต้องการปกปิด
        # ข้อมูลส่วนบุคคลที่ต้องการปกปิด
        self.pii_patterns = {
            # พื้นฐาน
            'เลขบัตรประชาชน': r'\b[1-9]\d{12}\b',
            'เบอร์โทรศัพท์': r'(?:\+66|0)\d{1,2}[-\s]?\d{3,4}[-\s]?\d{4}\b',
            'อีเมล': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'บัตรเครดิต': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            'ที่อยู่ IP': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
            'รหัสไปรษณีย์': r'\b[1-9]\d{4}\b',
            
            # ที่อยู่แบบท้องถิ่น
            'ที่อยู่ภาคกลาง': r'(?:บ้านเลขที่|เลขที่)[\s\d/]+(?:หมู่(?:ที่)?[\s\d]+)?(?:ซอย|ถนน|ตำบล|แขวง|อำเภอ|เขต|จังหวัด).*?(?:\d{5})',
            'ที่อยู่ภาคเหนือ': r'(?:เฮือน|บ้าน)(?:เลก|เลข)ที่[\s\d/]+(?:หมู่[\s\d]+)?(?:ซอย|หน|ตำบล|แขวง|อำเภอ|เขต|จังหวัด).*?(?:\d{5})',
            'ที่อยู่อีสาน': r'(?:เฮือน|บ้าน)(?:เลก|เลข)ที่[\s\d/]+(?:หมู่[\s\d]+)?(?:ซอย|ถนน|ตำบล|แขวง|อำเภอ|เขต|จังหวัด).*?(?:\d{5})',
            'ที่อยู่ภาคใต้': r'(?:บ้าน|เรือน)(?:เลก|เลข)ที่[\s\d/]+(?:หมู่[\s\d]+)?(?:ซอย|ถนน|ตำบล|แขวง|อำเภอ|เขต|จังหวัด).*?(?:\d{5})',
            
            # ID และโซเชียล
            'พาสปอร์ต': r'[A-Z]{1,2}\d{6,7}',
            'LINE ID': r'(?:ไลน์|LINE|Line ID|ไลน์ไอดี)[\s:][@\w._-]+',
            'Facebook': r'(?:fb\.com/|facebook\.com/|FB:|เฟซบุ๊ก:?)[\w.-]+',
            
            # ชื่อและตำแหน่งท้องถิ่น
            'ตำแหน่งภาคเหนือ': r'(?:อาจ๋าน|ป้อ|แม่|หนาน|น้อย)[\s\u0E00-\u0E7F]+',
            'ตำแหน่งอีสาน': r'(?:อาจารย์|พ่อ|แม่|หมอ|ไท)[\s\u0E00-\u0E7F]+',
            'ตำแหน่งภาคใต้': r'(?:อาจารย์|ลุง|ป้า|หมอ|ช่าง)[\s\u0E00-\u0E7F]+'
        }
        
    def anonymize_text(self, text: str) -> str:
        """ปกปิดข้อมูลส่วนบุคคลในข้อความ รองรับภาษาท้องถิ่น"""
        
        # ตรวจสอบและแปลงภาษาท้องถิ่นเป็นภาษากลาง
        original_dialect = None
        standardized_text = text
        
        if self.handle_dialects:
            original_dialect = detect_dialect(text)
            if original_dialect and original_dialect != 'standard':
                standardized_text = translate_to_standard(text, original_dialect)
        # ปกปิดข้อมูลจากรูปแบบที่กำหนด
        for label, pattern in self.pii_patterns.items():
            text = re.sub(pattern, f"[{label}]", text)
        
        # ใช้ NER เพื่อระบุและปกปิดข้อมูลส่วนบุคคล
        entities = self.ner.extract_entities(text)
        
        # จัดกลุ่มและรวม entities ที่ซ้อนทับกัน
        merged_entities = []
        current = None
        
        # เรียงลำดับตาม start position
        sorted_entities = sorted(entities, key=lambda x: (x[2], -x[3]))
        
        for entity_type, entity, start, end in sorted_entities:
            if current is None:
                current = [entity_type, entity, start, end]
            else:
                # ถ้า entity ใหม่ซ้อนทับกับ entity ปัจจุบัน
                if start <= current[3]:
                    # ขยาย entity ปัจจุบันถ้าจำเป็น
                    if end > current[3]:
                        current[3] = end
                    # รวมประเภท entity ถ้าแตกต่างกัน
                    if entity_type != current[0]:
                        current[0] = f"{current[0]}+{entity_type}"
                else:
                    merged_entities.append(current)
                    current = [entity_type, entity, start, end]
        
        if current:
            merged_entities.append(current)
        
        # ทำการปกปิดข้อมูลแบบย้อนกลับ
        for entity_type, entity, start, end in sorted(merged_entities, key=lambda x: x[2], reverse=True):
            if any(t in entity_type for t in ['PERSON', 'LOCATION', 'ORGANIZATION']):
                # เพิ่มการตรวจสอบคำนำหน้าชื่อและตำแหน่ง
                prefix_start = max(0, start - 20)
                prefix_text = text[prefix_start:start]
                
                # ถ้ามีคำนำหน้าชื่อหรือตำแหน่ง ขยายการปกปิด
                thai_prefixes = ['นาย', 'นาง', 'นางสาว', 'ดร.', 'ศ.', 'รศ.', 'ผศ.', 'อ.', 'คุณ']
                for prefix in thai_prefixes:
                    if prefix in prefix_text:
                        prefix_idx = prefix_text.rindex(prefix)
                        start = prefix_start + prefix_idx
                
                text = text[:start] + f"[{entity_type}]" + text[end:]
        
        return text
    
    def _normalize_text(self, text: str) -> str:
        """ปรับปรุงข้อความให้อยู่ในรูปแบบมาตรฐาน"""
        # แทนที่เครื่องหมายวรรคตอนที่ไม่จำเป็น
        text = re.sub(r'[\s\xa0]+', ' ', text)
        
        # แก้ไขตัวเลขและเครื่องหมายพิเศษ
        text = re.sub(r'[٠-٩]', lambda x: str('0123456789'[ord(x.group())-ord('٠')]), text)
        text = text.replace('−', '-').replace('‐', '-').replace('‑', '-')
        
        # ทำความสะอาดช่องว่างและขีดเส้น
        text = re.sub(r'[-\s]+', ' ', text)
        
        return text.strip()
    
    def _validate_pii(self, text: str, label: str) -> bool:
        """ตรวจสอบความน่าจะเป็นของการเป็นข้อมูลส่วนบุคคล"""
        if label == 'เลขบัตรประชาชน':
            # ตรวจสอบเลขบัตรประชาชนด้วยการคำนวณ checksum
            if len(text) != 13:
                return False
            try:
                sum = 0
                for i in range(12):
                    sum += int(text[i]) * (13 - i)
                checksum = (11 - (sum % 11)) % 10
                return int(text[-1]) == checksum
            except ValueError:
                return False
                
        elif label == 'เบอร์โทรศัพท์':
            # ตรวจสอบรูปแบบเบอร์โทรที่ถูกต้อง
            if not re.match(r'^(\+66|0)\d{9}$', text.replace(' ', '')):
                return False
            return True
            
        elif label.startswith('ที่อยู่'):
            # ตรวจสอบความยาวขั้นต่ำและการมีคำสำคัญ
            keywords = ['บ้าน', 'เลขที่', 'หมู่', 'ตำบล', 'อำเภอ', 'จังหวัด']
            return len(text) > 20 and any(k in text for k in keywords)
            
        # ค่าเริ่มต้นผ่านการตรวจสอบ
        return True
    
    def _apply_fuzzy_matching(self, text: str) -> str:
        """ใช้ fuzzy matching เพื่อหาข้อความที่ใกล้เคียง"""
        from rapidfuzz import fuzz
        
        # คำศัพท์ที่บ่งชี้ข้อมูลส่วนบุคคล
        pii_indicators = {
            'ชื่อ': ['นาย', 'นาง', 'นางสาว', 'ดร.', 'อาจารย์'],
            'ที่อยู่': ['บ้านเลขที่', 'หมู่', 'ตำบล', 'แขวง', 'เขต', 'อำเภอ', 'จังหวัด'],
            'การติดต่อ': ['โทร', 'มือถือ', 'อีเมล', 'เบอร์', 'ติดต่อ']
        }
        
        # แบ่งข้อความเป็นประโยค
        sentences = re.split(r'[.!?\n]+', text)
        result = []
        
        for sentence in sentences:
            # ตรวจสอบแต่ละประเภทของข้อมูล
            for label, indicators in pii_indicators.items():
                for indicator in indicators:
                    # หาคำที่ใกล้เคียงด้วย fuzzy matching
                    ratio = fuzz.partial_ratio(sentence.lower(), indicator.lower())
                    if ratio > self.fuzzy_threshold * 100:
                        sentence = f"[{label}...]"
                        break
            result.append(sentence)
            
        return ' '.join(result)
    
    def hash_sensitive_terms(self, tokens: List[str], with_probability: bool = True) -> List[str]:
        """แปลงคำที่อ่อนไหวให้เป็น hash พร้อมการตรวจสอบความแม่นยำ"""
        # ประเภทข้อมูลและความน่าเชื่อถือขั้นต่ำ
        sensitive_types = {
            'PERSON': 0.95,      # ชื่อบุคคล (ต้องการความแม่นยำสูง)
            'LOCATION': 0.90,    # สถานที่
            'ORGANIZATION': 0.90, # องค์กร
            'DATE': 0.85,        # วันที่
            'TIME': 0.85,        # เวลา
            'MONEY': 0.95,       # จำนวนเงิน
            'LAW': 0.90,         # ข้อกฎหมาย
            'PERSONAL_ID': 0.98,  # รหัสประจำตัว
            'CONTACT': 0.95      # ข้อมูลการติดต่อ
        }
        
        result = []
        text = " ".join(tokens)
        
        # สร้าง context window สำหรับแต่ละ token
        context_windows = {}
        for i, token in enumerate(tokens):
            # สร้าง context window ขนาด 5 คำ
            start = max(0, i - 2)
            end = min(len(tokens), i + 3)
            context = " ".join(tokens[start:end])
            context_windows[token] = context
        
        # ดึง entities พร้อมความน่าจะเป็น
        if with_probability:
            entities = self.ner.extract_entities(text, with_probability=True)
        else:
            # ถ้าไม่มี probability ให้กำหนดค่าเริ่มต้นเป็น 1.0
            entities = [(e[0], e[1], e[2], e[3], 1.0) for e in self.ner.extract_entities(text)]
        
        for token in tokens:
            is_sensitive = False
            max_prob = 0
            entity_type = None
            
            # ตรวจสอบ NER entities
            for ent_type, entity, start, end, prob in entities:
                if token in entity and ent_type in sensitive_types:
                    if prob >= sensitive_types[ent_type]:
                        if prob > max_prob:
                            max_prob = prob
                            is_sensitive = True
                            entity_type = ent_type
            
            # ตรวจสอบเพิ่มเติมด้วย context
            if not is_sensitive and token in context_windows:
                context = context_windows[token]
                if self._check_sensitive_context(context, token):
                    is_sensitive = True
                    entity_type = 'CONTEXT_SENSITIVE'
            
            if is_sensitive:
                # ใช้ salt แตกต่างกันตามประเภท
                salt = f"{entity_type}_{self.epsilon}"
                hashed = hashlib.sha256((token + salt).encode()).hexdigest()[:8]
                result.append(f"HASH_{entity_type}_{hashed}")
            else:
                result.append(token)
        
        return result
        
    def _check_sensitive_context(self, context: str, token: str) -> bool:
        """ตรวจสอบ context ว่ามีคำบ่งชี้ข้อมูลส่วนบุคคลหรือไม่"""
        # คำที่บ่งชี้ข้อมูลส่วนบุคคล
        indicators = {
            'PERSON': ['ชื่อ', 'นาย', 'นาง', 'นางสาว', 'ดร.', 'อาจารย์', 'คุณ'],
            'CONTACT': ['โทร', 'เบอร์', 'อีเมล', 'ติดต่อ', 'มือถือ'],
            'ID': ['รหัส', 'เลขที่', 'หมายเลข', 'บัตร'],
            'ADDRESS': ['บ้านเลขที่', 'ที่อยู่', 'อาศัยอยู่', 'พักอยู่'],
            'FINANCE': ['บัญชี', 'เงิน', 'จำนวน', 'บาท']
        }
        
        context_lower = context.lower()
        for type_indicators in indicators.values():
            if any(ind.lower() in context_lower for ind in type_indicators):
                if self._validate_token_pattern(token):
                    return True
        return False
    
    def _validate_token_pattern(self, token: str) -> bool:
        """ตรวจสอบรูปแบบของ token ว่าน่าจะเป็นข้อมูลส่วนบุคคลหรือไม่"""
        # ตรวจสอบความยาว
        if len(token) < 3:
            return False
            
        # ตรวจสอบรูปแบบตัวเลขผสมตัวอักษร
        if re.search(r'(?:\d.*[A-Za-z])|(?:[A-Za-z].*\d)', token):
            return True
            
        # ตรวจสอบการมีเครื่องหมายพิเศษ
        if re.search(r'[@#$%&*_\-+=]', token):
            return True
            
        # ตรวจสอบความยาวที่น่าสงสัย
        if len(token) >= 8 and not token.isspace():
            return True
            
        return False
    
    def add_differential_privacy(self, vector: np.ndarray) -> np.ndarray:
        """เพิ่ม differential privacy ใน vector embedding"""
        # สร้าง Laplacian noise โดยใช้ epsilon
        noise = np.random.laplace(0, self.noise_scale/self.epsilon, vector.shape)
        noisy_vector = vector + noise
        
        return noisy_vector
    
    def privacy_preserving_tokenization(self, text: str) -> List[str]:
        """การตัดคำแบบคำนึงถึงความเป็นส่วนตัว รองรับภาษาท้องถิ่น"""
        # ตรวจสอบภาษาท้องถิ่น
        dialect = None
        if self.handle_dialects:
            dialect = detect_dialect(text)
        
        # ปกปิดข้อมูลส่วนบุคคล
        anonymized_text = self.anonymize_text(text)
        
        # เลือกใช้ tokenizer ที่เหมาะสม
        if dialect and self.dialect_tokenizer:
            tokens = self.dialect_tokenizer.tokenize(anonymized_text, dialect=dialect)
        else:
            tokens = word_tokenize(anonymized_text)
        
        return tokens
    
    def privacy_preserving_feature_extraction(self, text: str) -> np.ndarray:
        """สกัดคุณลักษณะแบบคำนึงถึงความเป็นส่วนตัว รองรับภาษาท้องถิ่นด้วยความแม่นยำสูง"""
        # 1. ตรวจสอบและแปลงภาษาท้องถิ่น
        if self.handle_dialects:
            dialect = detect_dialect(text)
            if dialect and dialect != 'standard':
                text = translate_to_standard(text, dialect)
        
        # 2. ทำความสะอาดและปรับปรุงข้อความ
        text = self._normalize_text(text)
        
        # 3. ปกปิดข้อมูลส่วนบุคคลด้วยความแม่นยำสูง
        anonymized_text = self.anonymize_text(text)
        
        # 4. ตรวจสอบเพิ่มเติมด้วย fuzzy matching
        anonymized_text = self._apply_fuzzy_matching(anonymized_text)
        
        # 5. สร้าง document vector ที่ปลอดภัย
        from thainlp.feature_extraction import create_document_vector
        original_vector = create_document_vector(anonymized_text)
        
        # 6. ตรวจสอบและปรับแต่ง vector
        if np.any(np.isnan(original_vector)):
            # แก้ไขค่า NaN ด้วยค่าเฉลี่ยของ vector ที่ไม่ใช่ NaN
            non_nan_mean = np.nanmean(original_vector)
            original_vector = np.nan_to_num(original_vector, nan=non_nan_mean)
        
        # 7. เพิ่ม differential privacy ที่เหมาะสม
        # ปรับ noise scale ตามความยาวข้อความเพื่อรักษาความแม่นยำ
        noise_scale = self.noise_scale * (1.0 / np.sqrt(len(text.split())))
        private_vector = self.add_differential_privacy(original_vector)
        
        # 8. ทำให้ vector มีค่าอยู่ในช่วงที่เหมาะสม
        private_vector = np.clip(private_vector, -1.0, 1.0)
        
        # 9. Normalize vector
        norm = np.linalg.norm(private_vector)
        if norm > 0:
            private_vector = private_vector / norm
            
        # 10. ตรวจสอบความถูกต้องของ vector
        if not self._validate_vector(private_vector):
            # ถ้า vector ไม่ผ่านการตรวจสอบ ให้สร้างใหม่ด้วยค่าเริ่มต้นที่ปลอดภัย
            private_vector = np.zeros_like(private_vector)
            private_vector[0] = 1.0  # ใช้ unit vector ในมิติแรก
            
        return private_vector
        
    def _validate_vector(self, vector: np.ndarray) -> bool:
        """ตรวจสอบความถูกต้องของ vector"""
        # ตรวจสอบ NaN และ Inf
        if np.any(np.isnan(vector)) or np.any(np.isinf(vector)):
            return False
            
        # ตรวจสอบขนาด
        norm = np.linalg.norm(vector)
        if norm < 1e-6 or norm > 1e6:
            return False
            
        # ตรวจสอบค่าที่ผิดปกติ
        mean = np.mean(vector)
        std = np.std(vector)
        if abs(mean) > 10 or std > 10:
            return False
            
        return True
    
    def process_sensitive_document(self, text: str) -> Dict:
        """ประมวลผลเอกสารที่อ่อนไหว"""
        # 1. ปกปิดข้อมูลส่วนบุคคล
        anonymized_text = self.anonymize_text(text)
        
        # 2. ตัดคำที่ปกปิดแล้ว
        tokens = word_tokenize(anonymized_text)
        
        # 3. Hash คำที่อ่อนไหว
        hashed_tokens = self.hash_sensitive_terms(tokens)
        
        # 4. สกัดคุณลักษณะแบบคำนึงถึงความเป็นส่วนตัว
        private_vector = self.privacy_preserving_feature_extraction(anonymized_text)
        
        return {
            'anonymized_text': anonymized_text,
            'tokens': tokens,
            'hashed_tokens': hashed_tokens,
            'private_vector': private_vector
        }
