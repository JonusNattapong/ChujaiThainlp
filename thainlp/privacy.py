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
from thainlp.tokenize import word_tokenize
from thainlp.extensions.advanced_nlp import ThaiNamedEntityRecognition

class PrivacyPreservingNLP:
    def __init__(self, epsilon=0.1, noise_scale=0.1):
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
        
        # ข้อมูลส่วนบุคคลที่ต้องการปกปิด
        self.pii_patterns = {
            'เลขบัตรประชาชน': r'\b[1-9]\d{12}\b',
            'เบอร์โทรศัพท์': r'\b0\d{1,2}[-\s]?\d{3,4}[-\s]?\d{4}\b',
            'อีเมล': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'บัตรเครดิต': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            'ที่อยู่ IP': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
        }
        
    def anonymize_text(self, text: str) -> str:
        """ปกปิดข้อมูลส่วนบุคคลในข้อความ"""
        # ปกปิดข้อมูลจากรูปแบบที่กำหนด
        for label, pattern in self.pii_patterns.items():
            text = re.sub(pattern, f"[{label}]", text)
        
        # ใช้ NER เพื่อระบุและปกปิดชื่อบุคคล สถานที่ และองค์กร
        entities = self.ner.extract_entities(text)
        
        # ทำการปกปิดข้อมูลแบบย้อนกลับ (เพื่อไม่ให้กระทบตำแหน่งของข้อมูลอื่น)
        for entity_type, entity, start, end in sorted(entities, key=lambda x: x[2], reverse=True):
            if entity_type in ['PERSON', 'LOCATION', 'ORGANIZATION']:
                text = text[:start] + f"[{entity_type}]" + text[end:]
        
        return text
    
    def hash_sensitive_terms(self, tokens: List[str]) -> List[str]:
        """แปลงคำที่อ่อนไหวให้เป็น hash"""
        sensitive_types = ['PERSON', 'LOCATION', 'ORGANIZATION', 'DATE', 'TIME', 'MONEY', 'LAW']
        result = []
        
        for token in tokens:
            # ตรวจสอบว่าเป็นข้อมูลที่อ่อนไหวหรือไม่
            is_sensitive = False
            text = " ".join(tokens)
            entities = self.ner.extract_entities(text)
            
            for entity_type, entity, start, end in entities:
                if entity_type in sensitive_types and token in entity:
                    is_sensitive = True
                    break
            
            if is_sensitive:
                # ใช้ SHA-256 เข้ารหัสข้อมูลที่อ่อนไหว
                hashed = hashlib.sha256(token.encode()).hexdigest()[:8]
                result.append(f"HASH_{hashed}")
            else:
                result.append(token)
                
        return result
    
    def add_differential_privacy(self, vector: np.ndarray) -> np.ndarray:
        """เพิ่ม differential privacy ใน vector embedding"""
        # สร้าง Laplacian noise โดยใช้ epsilon
        noise = np.random.laplace(0, self.noise_scale/self.epsilon, vector.shape)
        noisy_vector = vector + noise
        
        return noisy_vector
    
    def privacy_preserving_tokenization(self, text: str) -> List[str]:
        """การตัดคำแบบคำนึงถึงความเป็นส่วนตัว"""
        # ปกปิดข้อมูลส่วนบุคคลก่อน
        anonymized_text = self.anonymize_text(text)
        
        # ทำการตัดคำ
        tokens = word_tokenize(anonymized_text)
        
        return tokens
    
    def privacy_preserving_feature_extraction(self, text: str) -> np.ndarray:
        """สกัดคุณลักษณะแบบคำนึงถึงความเป็นส่วนตัว"""
        # ปกปิดข้อมูลส่วนบุคคลก่อน
        anonymized_text = self.anonymize_text(text)
        
        # สร้าง document vector
        original_vector = thainlp.feature_extraction.create_document_vector(anonymized_text)
        
        # เพิ่ม differential privacy
        private_vector = self.add_differential_privacy(original_vector)
        
        return private_vector
    
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
