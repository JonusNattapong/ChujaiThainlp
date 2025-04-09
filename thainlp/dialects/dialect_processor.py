"""
Thai Dialect Processor

This module provides support for processing various Thai regional dialects:
- Northern Thai (คำเมือง/ไทยถิ่นเหนือ)
- Northeastern Thai/Isan (ไทยถิ่นอีสาน)
- Southern Thai (ไทยถิ่นใต้/ปักษ์ใต้)
- Central/Standard Thai (ไทยกลาง)
- Pattani Malay (ภาษามลายูปัตตานี)
"""

import re
from typing import Dict, List, Optional, Union, Tuple, Any
import json
import os
import torch
import numpy as np
from ..utils.thai_utils import normalize_text, clean_thai_text
from ..core.transformers import TransformerBase

# Define supported dialects
DIALECTS = {
    "northern": {"name": "Northern Thai", "thai_name": "คำเมือง/ไทยถิ่นเหนือ", "code": "nod", 
                 "regions": ["เชียงใหม่", "เชียงราย", "ลำปาง", "ลำพูน", "แพร่", "น่าน", "พะเยา"]},
    "northeastern": {"name": "Northeastern Thai", "thai_name": "ไทยถิ่นอีสาน", "code": "tts", 
                     "regions": ["อุบลราชธานี", "ขอนแก่น", "นครราชสีมา", "ร้อยเอ็ด", "มหาสารคาม"]},
    "southern": {"name": "Southern Thai", "thai_name": "ไทยถิ่นใต้/ปักษ์ใต้", "code": "sou", 
                 "regions": ["นครศรีธรรมราช", "สงขลา", "สุราษฎร์ธานี", "พัทลุง", "ชุมพร", "ภูเก็ต"]},
    "central": {"name": "Central Thai", "thai_name": "ไทยกลาง", "code": "th",
                "regions": ["กรุงเทพมหานคร", "สุพรรณบุรี", "นครปฐม", "พระนครศรีอยุธยา", "อ่างทอง"]},
    "pattani_malay": {"name": "Pattani Malay", "thai_name": "ภาษามลายูปัตตานี", "code": "mfa",
                     "regions": ["ปัตตานี", "ยะลา", "นราธิวาส", "สตูล"]}
}

# Dialect feature maps for lexical characteristics
DIALECT_FEATURES = {
    "northern": {
        "particles": ["เจ้า", "กำ", "ก้อ", "ละ", "เน้อ", "เนาะ", "เนอะ", "จะ", "จ้าว", "เลย", "กา", "กี่"],
        "pronouns": ["อั๋น", "กู", "มึง", "ตั๋ว", "เฮา", "หมู่", "ปี้", "ตู", "สู", "อี่", "ญิ", "เปิ้น"],
        "verb_modifiers": ["บ่", "ใจ้", "หื้อ", "เป๋น", "แอว", "ปั๋น", "จะ", "ละ", "กึด", "สิ"],
        "tones": ["มี 6 เสียงวรรณยุกต์"],
        "script": ["อักษรล้านนา (ตั๋วเมือง)"],
        "vocabulary": {
            "ดี": "งาม",
            "อร่อย": "ลำ",
            "มาก": "หนัก",
            "ไป": "ไป๋",
            "กิน": "ตึ้น",
            "ทำงาน": "เยียะก๋าน",
            "หมู": "หมู",
            "เท่าไหร่": "กี่มากัน",
            "อย่างไร": "จะใด",
            "ทำไม": "เป็นหยัง",
            "ทำอะไร": "เยียะหยัง",
            "มา": "มา",
            "ไม่": "บ่",
            "คน": "คน",
            "ที่ไหน": "ตี้ไหน",
            "ใคร": "ผู้ใด",
            "หลาย": "นัก",
            "ฉัน": "ข้อย",
            "พูด": "อู้",
            "สวย": "งาม",
            "อะไร": "หยัง",
            "นอน": "นอน",
            "มือ": "มือ",
            "เล็ก": "หน้อย",
            "ใหญ่": "ใหญ่"
        }
    },
    "northeastern": {
        "particles": ["เด้อ", "เนาะ", "สิ", "โอ้ย", "หล่ะ", "คือ", "บ้อ", "เวา", "สู", "กะ", "ดอก", "เด", "เนอะ", "กะ"],
        "pronouns": ["ข้อย", "เฮา", "เจ้า", "พวกเฮา", "สู", "เขา", "พู้น", "นั่น", "พุ่น"],
        "verb_modifiers": ["บ่", "กะ", "เบิ่ง", "เฮ็ด", "แน", "สิ", "แล้ว", "คือ", "ได้"],
        "tones": ["มี 6 เสียงวรรณยุกต์"],
        "script": ["ตัวอักษรไทย"],
        "vocabulary": {
            "ดี": "งาม",
            "อร่อย": "แซบ",
            "มาก": "หลาย",
            "ไป": "ไป",
            "กิน": "กิน",
            "ทำงาน": "เฮ็ดวียก",
            "หมู": "ม่วน",
            "เท่าไหร่": "จักได๋",
            "อย่างไร": "จั่งได๋",
            "ทำไม": "เป็นหยัง",
            "ทำอะไร": "เฮ็ดหยัง",
            "มา": "มา",
            "ไม่": "บ่",
            "คน": "คน",
            "ที่ไหน": "ไส",
            "ใคร": "ผู่ได๋",
            "ฉัน": "ข้อย",
            "พูด": "เว้า",
            "เสื้อ": "เสื้อ",
            "หลาย": "หลาย",
            "สวย": "งาม",
            "อะไร": "หยัง",
            "นอน": "นอน",
            "มือ": "มือ",
            "เล็ก": "น้อย",
            "ใหญ่": "ใหญ่"
        }
    },
    "southern": {
        "particles": ["โหล", "เหอ", "เหวย", "เน้อ", "ว่า", "หนิ", "แหละ", "จัง", "เออ", "โว้ย", "วะ", "นิ", "นุ้ย"],
        "pronouns": ["ฉาน", "เรา", "เธอ", "หมด", "เขา", "ไอ้", "อี", "วั่น", "ตู", "ไซ"],
        "verb_modifiers": ["บ่", "ตัง", "ไม่", "ติด", "ค่อย", "แอ", "โหรด", "มุด", "ตอก", "กั้น", "เคย"],
        "tones": ["มี 7 เสียงวรรณยุกต์", "ออกเสียงยาวแบบพิเศษ"],
        "influences": ["มีคำยืมจากภาษามลายู", "มีคำยืมจากภาษาจีน"],
        "vocabulary": {
            "ดี": "ดี",
            "อร่อย": "หรอย",
            "มาก": "นัก",
            "ไป": "ไป",
            "กิน": "กิน",
            "ทำงาน": "ทำงาน",
            "หมู": "หมู",
            "เท่าไหร่": "ท่าไหน",
            "อย่างไร": "อย่างไหร",
            "ทำไม": "ทำไหม",
            "ทำอะไร": "ทำหยัง",
            "มา": "มา",
            "ไม่": "ไม่",
            "คน": "คน",
            "ที่ไหน": "ตรงไหน",
            "ใคร": "ใผ",
            "ฉัน": "ฉาน",
            "พูด": "พูด",
            "เสื้อ": "เสื้อ",
            "สวย": "สวย",
            "อะไร": "หยัง",
            "นอน": "นอน",
            "มือ": "มือ",
            "เล็ก": "แหล็ก",
            "ใหญ่": "ใหญ่"
        },
        "subdialects": {
            "nakhon": {"name": "นครศรีธรรมราช", "thai_name": "สำเนียงนครศรีธรรมราช"},
            "songkhla": {"name": "สงขลา", "thai_name": "สำเนียงสงขลา"},
            "phuket": {"name": "ภูเก็ต", "thai_name": "สำเนียงภูเก็ต", "influences": "มีคำยืมจากภาษาจีนฮกเกี้ยน"},
            "chumphon": {"name": "ชุมพร", "thai_name": "สำเนียงชุมพร"}
        }
    },
    "central": {
        "particles": ["ครับ", "ค่ะ", "ฮะ", "จ้ะ", "จ้า", "นะ", "สิ", "น่ะ", "ล่ะ", "หรอก", "เถอะ", "เลย"],
        "pronouns": ["ผม", "ฉัน", "คุณ", "เธอ", "พวกเรา", "เขา", "ดิฉัน", "ท่าน", "พวกท่าน"],
        "verb_modifiers": ["ไม่", "จะ", "ได้", "ช่วย", "เคย", "กำลัง", "พึ่ง", "ยัง", "ค่อย", "ต้อง"],
        "tones": ["มี 5 เสียงวรรณยุกต์"],
        "script": ["อักษรไทย"],
        "vocabulary": {
            "ดี": "ดี",
            "อร่อย": "อร่อย",
            "มาก": "มาก",
            "ไป": "ไป",
            "กิน": "กิน",
            "ทำงาน": "ทำงาน",
            "หมู": "หมู",
            "เท่าไหร่": "เท่าไหร่",
            "อย่างไร": "อย่างไร",
            "ทำไม": "ทำไม",
            "ทำอะไร": "ทำอะไร",
            "มา": "มา",
            "ไม่": "ไม่",
            "คน": "คน",
            "ที่ไหน": "ที่ไหน",
            "ใคร": "ใคร",
            "ฉัน": "ฉัน",
            "พูด": "พูด",
            "เสื้อ": "เสื้อ", 
            "สวย": "สวย",
            "อะไร": "อะไร",
            "นอน": "นอน",
            "มือ": "มือ",
            "เล็ก": "เล็ก",
            "ใหญ่": "ใหญ่"
        }
    },
    "pattani_malay": {
        "particles": ["มะ", "เลอ", "ยอ", "โต๊ะ", "เดะ"],
        "pronouns": ["อาเกาะ", "เตะ", "ตือ", "กามอ", "ปอเญาะ"],
        "script": ["ตัวอักษรไทย", "อักษรยาวี"],
        "influences": ["ภาษามลายู", "ภาษาอาหรับ", "ภาษาไทย"],
        "regions": ["ปัตตานี", "ยะลา", "นราธิวาส", "ส่วนหนึ่งของสตูล"],
        "vocabulary": {
            "ดี": "บาเอ",
            "อร่อย": "ซาดะ",
            "มาก": "บาญะ",
            "ไป": "เปอกี",
            "กิน": "มากัน",
            "ทำงาน": "เกอยอบูวะ",
            "หมู": "บาบี",
            "เท่าไหร่": "บือราปอ",
            "อย่างไร": "มาเจอมานอ",
            "ทำไม": "กือปอ",
            "มา": "มารี",
            "ไม่": "ตะ",
            "คน": "โอแร",
            "ใคร": "ซาปอ",
            "พูด": "จากะ",
            "สวย": "จันเต",
            "อะไร": "อาปอ",
            "นอน": "ตีโด",
            "มือ": "ตาแง",
            "เล็ก": "เกอเจะ",
            "ใหญ่": "บือซะ"
        }
    }
}

# Additional regional dialect characteristics
DIALECT_REGIONAL_VARIATIONS = {
    "northern": {
        "เชียงใหม่-ลำพูน": {
            "description": "สำเนียงเมืองเชียงใหม่-ลำพูน ออกเสียงสูงชัดเจน มักใช้คำลงท้าย 'เจ้า'",
            "distinctive_words": ["เจ้า", "กำ", "ก้อ", "ละ", "เน้อ", "ขะใจ๋", "เปิ้น"]
        },
        "เชียงราย-พะเยา-ลำปาง": {
            "description": "สำเนียงเชียงราย ออกเสียงแตกต่างจากเชียงใหม่เล็กน้อย",
            "distinctive_words": ["เปิ้น", "จ้าว", "ละ", "ละก่อ"]
        },
        "น่าน-แพร่": {
            "description": "สำเนียงน่าน-แพร่ มีลักษณะเฉพาะ ออกเสียง ร และ ล ชัดเจน",
            "distinctive_words": ["คือ", "กิ๋น"]
        }
    },
    "northeastern": {
        "อีสานเหนือ": {
            "description": "สำเนียงอีสานเหนือ (เลย อุดรฯ หนองคาย บึงกาฬ) คล้ายภาษาลาว", 
            "distinctive_words": ["เด้อ", "เนาะ", "เบิ่ง", "เฮ็ด", "กะ"]
        },
        "อีสานกลาง": {
            "description": "สำเนียงอีสานกลาง (ขอนแก่น มหาสารคาม กาฬสินธุ์ ร้อยเอ็ด)",
            "distinctive_words": ["สิ", "บ่", "กะ", "อีหลี", "อีหยัง"]
        },
        "อีสานใต้": {
            "description": "สำเนียงอีสานใต้ (นครราชสีมา บุรีรัมย์ สุรินทร์ ศรีสะเกษ อุบลฯ)",
            "distinctive_words": ["เด", "สิ", "อ้าย", "อีนาง", "เจ้าคุณ"]
        }
    },
    "southern": {
        "upper_south": {
            "description": "สำเนียงใต้ตอนบน (ชุมพร ระนอง สุราษฎร์ธานี) ออกเสียงใกล้เคียงภาษากลางมากกว่า",
            "distinctive_words": ["ตั๋ว", "หนิ", "วั่น", "นิ", "จุ"]
        },
        "middle_south": {
            "description": "สำเนียงใต้ตอนกลาง (นครศรีธรรมราช พัทลุง ตรัง) มีสำเนียงโดดเด่น",
            "distinctive_words": ["ว่า", "หนิ", "โหล", "แหละ", "เหวย"]
        },
        "lower_south": {
            "description": "สำเนียงใต้ตอนล่าง (สงขลา ปัตตานี) ได้รับอิทธิพลจากภาษามลายู",
            "distinctive_words": ["จัง", "หนิ", "เหอ", "แอ", "ยะ"]
        },
        "phuket_trang": {
            "description": "สำเนียงภูเก็ต-ตรัง (ภูเก็ต ตรัง) มีอิทธิพลจากภาษาจีนฮกเกี้ยน",
            "distinctive_words": ["นุ้ย", "เหอ", "บ่", "ก่อ", "มั่ง"]
        }
    }
}

class ThaiDialectProcessor(TransformerBase):
    """Thai dialect processor supporting multiple Thai dialects"""
    
    def __init__(
        self,
        model_name_or_path: str = "airesearch/wangchanberta-base-att-spm-uncased",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """Initialize Thai dialect processor
        
        Args:
            model_name_or_path: Pretrained model name or path for dialect processing
            device: Device to run model on (cuda/cpu)
        """
        super().__init__(model_name_or_path)
        self.device = device
        self.dialects = DIALECTS
        self.dialect_features = DIALECT_FEATURES
        self.dialect_variations = DIALECT_REGIONAL_VARIATIONS
        
        # Prepare regex patterns for dialect detection
        self.dialect_patterns = {}
        for dialect, features in self.dialect_features.items():
            pattern_parts = []
            for feature_type in ["particles", "pronouns", "verb_modifiers"]:
                if feature_type in features and features[feature_type]:
                    # Escape special characters and join with OR
                    escaped_words = [re.escape(word) for word in features[feature_type]]
                    pattern_parts.append(r'(?:' + '|'.join(escaped_words) + r')')
                    
            # Add vocabulary words specific to this dialect
            if "vocabulary" in features and features["vocabulary"]:
                dialect_words = []
                for standard, dialect_word in features["vocabulary"].items():
                    if dialect_word != standard:  # Only add if different from standard Thai
                        dialect_words.append(re.escape(dialect_word))
                
                if dialect_words:
                    pattern_parts.append(r'(?:' + '|'.join(dialect_words) + r')')
                    
            if pattern_parts:
                # Combine all patterns with word boundaries
                self.dialect_patterns[dialect] = re.compile(
                    r'\b(?:' + '|'.join(pattern_parts) + r')\b', 
                    re.UNICODE
                )

    def detect_dialect(self, text: str, threshold: float = 0.1) -> Dict[str, float]:
        """Detect the likely Thai dialect in the text
        
        Args:
            text: Thai text to analyze
            threshold: Minimum ratio to consider a dialect present
            
        Returns:
            Dictionary mapping dialect codes to confidence scores
        """
        # Normalize and clean the text
        text = clean_thai_text(text)
        
        total_words = len(text.split())
        if total_words == 0:
            return {"central": 1.0}  # Default to standard Thai for empty text
        
        # Count dialect markers
        dialect_scores = {}
        for dialect, pattern in self.dialect_patterns.items():
            matches = pattern.findall(text)
            score = len(matches) / total_words if total_words > 0 else 0
            dialect_scores[dialect] = score
        
        # If no strong dialect markers found, default to central Thai
        if all(score < threshold for score in dialect_scores.values()):
            dialect_scores["central"] = max(0.8, dialect_scores.get("central", 0))
        
        # Normalize scores
        total = sum(dialect_scores.values())
        if total > 0:
            for dialect in dialect_scores:
                dialect_scores[dialect] /= total
                
        return dialect_scores

    def detect_regional_dialect(self, text: str, primary_dialect: Optional[str] = None) -> Dict[str, float]:
        """Detect the regional variation within a primary dialect
        
        Args:
            text: Thai text to analyze
            primary_dialect: The primary dialect to analyze for regional variations
                             (northern, northeastern, southern). If None, will detect first.
            
        Returns:
            Dictionary mapping regional dialect codes to confidence scores
        """
        # Detect primary dialect if not provided
        if primary_dialect is None:
            dialect_scores = self.detect_dialect(text)
            primary_dialect = max(dialect_scores, key=lambda k: dialect_scores[k])
        
        # Check if we have regional variations for this dialect
        if primary_dialect not in self.dialect_variations:
            return {primary_dialect: 1.0}
        
        # Analyze regional variations
        region_scores = {}
        text = clean_thai_text(text)
        word_count = len(text.split())
        
        for region, details in self.dialect_variations[primary_dialect].items():
            score = 0
            if "distinctive_words" in details:
                for word in details["distinctive_words"]:
                    matches = len(re.findall(r'\b' + re.escape(word) + r'\b', text, re.UNICODE))
                    score += matches
            
            region_scores[region] = score / max(1, word_count)
        
        # Normalize scores
        total_score = sum(region_scores.values())
        if total_score > 0:
            for region in region_scores:
                region_scores[region] /= total_score
        else:
            # No specific region detected, assign all to the primary dialect
            region_scores = {primary_dialect: 1.0}
            
        return region_scores

    def translate_to_standard(self, text: str, source_dialect: str) -> str:
        """Translate text from a Thai dialect to standard Thai
        
        Args:
            text: Text in a Thai dialect
            source_dialect: Source dialect code (northern, northeastern, southern)
            
        Returns:
            Text translated to standard Thai
        """
        if source_dialect not in self.dialects or source_dialect == "central":
            return text  # Return as is if no translation needed or unsupported dialect
        
        # Normalize input text
        text = normalize_text(text)
        
        # Simple rule-based translation using vocabulary mappings
        words = text.split()
        translated_words = []
        
        # Get the vocabulary mapping for this dialect (dialect word -> standard word)
        vocab_map = {}
        if source_dialect in self.dialect_features and "vocabulary" in self.dialect_features[source_dialect]:
            # Create reverse mapping: dialect word -> standard word
            for std_word, dialect_word in self.dialect_features[source_dialect]["vocabulary"].items():
                vocab_map[dialect_word] = std_word
        
        # Process each word
        for word in words:
            # Look up in vocabulary map
            if word in vocab_map:
                translated_words.append(vocab_map[word])
            else:
                translated_words.append(word)
                
        return " ".join(translated_words)
        
    def translate_from_standard(self, text: str, target_dialect: str) -> str:
        """Translate text from standard Thai to a Thai dialect
        
        Args:
            text: Text in standard Thai
            target_dialect: Target dialect code (northern, northeastern, southern)
            
        Returns:
            Text translated to target dialect
        """
        if target_dialect not in self.dialects or target_dialect == "central":
            return text  # Return as is if no translation needed or unsupported dialect
        
        # Normalize input text
        text = normalize_text(text)
        
        # Simple rule-based translation using vocabulary mappings
        words = text.split()
        translated_words = []
        
        # Get the vocabulary mapping for the target dialect
        vocab_map = {}
        if target_dialect in self.dialect_features and "vocabulary" in self.dialect_features[target_dialect]:
            vocab_map = self.dialect_features[target_dialect]["vocabulary"]
        
        # Process each word
        for word in words:
            # Look up in vocabulary map
            if word in vocab_map:
                translated_words.append(vocab_map[word])
            else:
                translated_words.append(word)
                
        return " ".join(translated_words)

    def get_dialect_features(self, dialect: str) -> Dict[str, Any]:
        """Get linguistic features of a specific Thai dialect
        
        Args:
            dialect: Dialect code
            
        Returns:
            Dictionary of dialect features
        """
        if dialect not in self.dialect_features:
            return {}
        return self.dialect_features[dialect]
    
    def get_dialect_info(self, dialect: str) -> Dict[str, str]:
        """Get information about a specific Thai dialect
        
        Args:
            dialect: Dialect code
            
        Returns:
            Dictionary with dialect information
        """
        if dialect not in self.dialects:
            return {}
        return self.dialects[dialect]

    def get_all_dialects(self) -> Dict[str, Dict[str, str]]:
        """Get information about all supported dialects
        
        Returns:
            Dictionary mapping dialect codes to dialect information
        """
        return self.dialects
        
    def get_example_phrases(self, dialect: str, num_phrases: int = 5) -> List[Tuple[str, str]]:
        """Get example phrases in the specified dialect with standard Thai translations
        
        Args:
            dialect: Dialect code
            num_phrases: Number of example phrases to return
            
        Returns:
            List of (dialect_phrase, standard_thai) tuples
        """
        if dialect not in self.dialect_features or dialect == "central":
            return []
            
        examples = []
        
        # Common phrases translated to this dialect
        common_phrases = {
            "northern": [
                ("สวัสดีเจ้า", "สวัสดี"),
                ("กิ๋นข้าวแล้วกา", "กินข้าวหรือยัง"),
                ("อั๋นจะไป๋ตลาด", "ผมจะไปตลาด"),
                ("เมื่อคืนนอนหลับงามบ่", "เมื่อคืนนอนหลับดีไหม"),
                ("อากาศฮ้อนเหลือเปิ้น", "อากาศร้อนมาก"),
                ("อู้กำเมืองได้ก่อ", "พูดภาษาเหนือได้ไหม"),
                ("เปิ้นมาตึ้งเจ้า", "เขามากินแล้วนะ"),
                ("ลูกหน้อยจะไปไหนเจ้า", "ลูกจะไปไหน")
            ],
            "northeastern": [
                ("สบายดีบ่", "สบายดีไหม"),
                ("กินเข้าแล้วบ่", "กินข้าวหรือยัง"),
                ("เฮ็ดหยังอยู่", "ทำอะไรอยู่"),
                ("อาหารแซบหลาย", "อาหารอร่อยมาก"),
                ("ข้อยสิไปตลาด", "ฉันจะไปตลาด"),
                ("เว้าภาษาอีสานเป็นบ่", "พูดภาษาอีสานเป็นไหม"),
                ("สิไปไสกะไปเดอ", "จะไปไหนก็ไปเถอะ"),
                ("เฮ็ดงานหนักหลาย", "ทำงานหนักมาก"),
                ("หิวเข้าแล้วบ่", "หิวข้าวหรือยัง")
            ],
            "southern": [
                ("หวัดดีหนิ", "สวัสดี"),
                ("กินข้าวแล้วหรือหนิ", "กินข้าวหรือยัง"),
                ("ฉานจะไปตลาด", "ฉันจะไปตลาด"),
                ("อาหารหรอยนัก", "อาหารอร่อยมาก"),
                ("ไปท่าไหนมา", "ไปไหนมา"),
                ("เธอเป็นคนใต้หรือ", "เธอเป็นคนใต้หรือ"),
                ("บ่เคยไปภูเก็ตเลย", "ไม่เคยไปภูเก็ตเลย"),
                ("ฝนตกนักวันนี้", "ฝนตกมากวันนี้"),
                ("จำได้แอ", "จำได้นะ")
            ],
            "pattani_malay": [
                ("ซาลามัต ดาตัง", "สวัสดี"),
                ("มากัน แล็ฮ กือ บือลูม", "กินข้าวหรือยัง"),
                ("อาเกาะ นะ เปอกี ปาซะ", "ฉันจะไปตลาด"),
                ("มะกัน นิ ซาดะ บาญะ", "อาหารนี้อร่อยมาก"),
                ("เปอกี ดาริ มานอ", "ไปไหนมา"),
                ("อาปอ กาบอ", "คุณสบายดีไหม"),
                ("ตือรีมอ กาซิฮ", "ขอบคุณ"),
                ("กามอ ตืนี ดาริ ปาตตานี", "คุณมาจากปัตตานีใช่ไหม")
            ]
        }
        
        if dialect in common_phrases:
            examples = common_phrases[dialect][:num_phrases]
            
        return examples
        
    def get_regional_dialect_examples(self, dialect: str, region: str, num_phrases: int = 3) -> List[Tuple[str, str]]:
        """Get example phrases for a specific regional dialect variation
        
        Args:
            dialect: Primary dialect code (northern, northeastern, southern)
            region: Regional variation code within the primary dialect
            num_phrases: Number of example phrases to return
            
        Returns:
            List of (regional_dialect_phrase, standard_thai) tuples
        """
        if dialect not in self.dialect_variations or region not in self.dialect_variations.get(dialect, {}):
            return []
            
        # Regional dialect variations with examples
        regional_examples = {
            "northern": {
                "เชียงใหม่-ลำพูน": [
                    ("เปิ้นกำลังมาละเจ้า", "เขากำลังมาแล้วนะ"),
                    ("จะไปก๋าดเจ้า", "จะไปตลาดนะ"),
                    ("เยียะหยังอยู่เจ้า", "ทำอะไรอยู่")
                ],
                "เชียงราย-พะเยา-ลำปาง": [
                    ("เปิ้นจะมาละก่อ", "เขาจะมาแล้วนะ"),
                    ("จะไปตี้ไหนจ้าว", "จะไปไหน"),
                    ("ลำปางกำลังร้อนแต๊ๆ", "ลำปางกำลังร้อนมากๆ")
                ],
                "น่าน-แพร่": [
                    ("กิ๋นเข้าแล้วเลอะ", "กินข้าวแล้วหรือ"),
                    ("บ่เด้อบ่ไปไหน", "ไม่ต้องไปไหน"),
                    ("น่านบ้านเฮา", "น่านบ้านเรา")
                ]
            },
            "northeastern": {
                "อีสานเหนือ": [
                    ("เจ้าสิไปไสมาสิบ่บอก", "คุณจะไปไหนมาไม่บอก"),
                    ("บักหล่า สิไปเด้อ", "น้องชาย จะไปนะ"),
                    ("เฮ็ดกะได้คือกัน", "ทำก็ได้เหมือนกัน")
                ],
                "อีสานกลาง": [
                    ("อีหลีสิไปไส", "น้องสาวจะไปไหน"),
                    ("อยู่ตรงนี่นำกัน", "อยู่ตรงนี้ด้วยกัน"),
                    ("กะสิเอาบ่", "ก็จะเอาไหม")
                ],
                "อีสานใต้": [
                    ("อ้ายเอ้ย ลาวโสเด", "พี่ชาย ชาวโสนะ"),
                    ("พี่เลียบ ไปไสมา", "พี่เลียบ ไปไหนมา"),
                    ("เก็บเงินหลายแท้นิ", "เก็บเงินมากจริงๆ")
                ]
            },
            "southern": {
                "upper_south": [
                    ("ตั๋วกินข้าวกับนิ", "คุณกินข้าวกับอะไร"),
                    ("ไปวั่นมาวั่น", "ไปวันมาวัน"),
                    ("ใจ้ชั่วเหอะ", "ใช่ไหมล่ะ")
                ],
                "middle_south": [
                    ("แหละเราจะไปโหล", "แล้วเราจะไปนะ"),
                    ("หรอยกว่าเหวย", "อร่อยกว่านะ"),
                    ("ว่ามาพร่องเด", "พูดมาบ้างสิ")
                ],
                "lower_south": [
                    ("ฉานจะไปแอจัง", "ฉันจะไปนะ"),
                    ("ยะไหรเหอ", "ทำอะไรอยู่"),
                    ("คนสงขลายะม่าย", "คนสงขลาใช่ไหม")
                ],
                "phuket_trang": [
                    ("กินข้าวมั่งนุ้ย", "กินข้าวด้วยนะ"),
                    ("ใต้ๆ หรอยมาก", "แบบใต้ๆ อร่อยมาก"),
                    ("เหอหนิ", "ใช่ไหม")
                ]
            }
        }
        
        if dialect in regional_examples and region in regional_examples[dialect]:
            return regional_examples[dialect][region][:num_phrases]
        
        return []

# Module-level functions
def detect_dialect(text: str, threshold: float = 0.1) -> Dict[str, float]:
    """Detect the Thai dialect in text
    
    Args:
        text: Thai text to analyze
        threshold: Minimum ratio to consider a dialect present
        
    Returns:
        Dictionary mapping dialect codes to confidence scores
    """
    processor = ThaiDialectProcessor()
    return processor.detect_dialect(text, threshold)

def translate_to_standard(text: str, source_dialect: str) -> str:
    """Translate text from a Thai dialect to standard Thai
    
    Args:
        text: Text in a Thai dialect
        source_dialect: Source dialect code (northern, northeastern, southern)
        
    Returns:
        Text translated to standard Thai
    """
    processor = ThaiDialectProcessor()
    return processor.translate_to_standard(text, source_dialect)

def translate_from_standard(text: str, target_dialect: str) -> str:
    """Translate text from standard Thai to a Thai dialect
    
    Args:
        text: Text in standard Thai
        target_dialect: Target dialect code (northern, northeastern, southern)
        
    Returns:
        Text translated to target dialect
    """
    processor = ThaiDialectProcessor()
    return processor.translate_from_standard(text, target_dialect)

def get_dialect_features(dialect: str) -> Dict[str, Any]:
    """Get linguistic features of a specific Thai dialect
    
    Args:
        dialect: Dialect code
        
    Returns:
        Dictionary of dialect features
    """
    processor = ThaiDialectProcessor()
    return processor.get_dialect_features(dialect)

def get_dialect_info(dialect: str) -> Dict[str, str]:
    """Get information about a specific Thai dialect
    
    Args:
        dialect: Dialect code
        
    Returns:
        Dictionary with dialect information
    """
    processor = ThaiDialectProcessor()
    return processor.get_dialect_info(dialect)

def detect_regional_dialect(text: str, primary_dialect: Optional[str] = None) -> Dict[str, float]:
    """Detect regional dialect variations within a primary dialect
    
    Args:
        text: Thai text to analyze
        primary_dialect: Primary dialect (northern, northeastern, southern).
                        If None, will detect first.
        
    Returns:
        Dictionary mapping regional dialect codes to confidence scores
    """
    processor = ThaiDialectProcessor()
    return processor.detect_regional_dialect(text, primary_dialect)