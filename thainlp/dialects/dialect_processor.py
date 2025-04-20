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
import os
import json
import torch
import numpy as np
import functools
import collections
from typing import Dict, List, Optional, Union, Tuple, Any, Set, Callable
from pathlib import Path
from datetime import datetime
from ..utils.thai_utils import normalize_text, clean_thai_text
from ..core.transformers import TransformerBase
from ..optimization.optimizer import TextProcessor

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
    """Thai dialect processor supporting multiple Thai dialects with enhanced performance and ML capabilities"""
    
    def __init__(
        self,
        model_name_or_path: str = "airesearch/wangchanberta-base-att-spm-uncased",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        cache_size: int = 10000,
        use_ml: bool = True,
        dialect_data_dir: Optional[str] = None
    ):
        """Initialize Thai dialect processor
        
        Args:
            model_name_or_path: Pretrained model name or path for dialect processing
            device: Device to run model on (cuda/cpu)
            cache_size: Size of the result cache for performance optimization
            use_ml: Whether to use ML models for dialect detection when available
            dialect_data_dir: Directory to store/load dialect-specific data
        """
        super().__init__(model_name_or_path)
        self.device = device
        self.dialects = DIALECTS
        self.dialect_features = DIALECT_FEATURES
        self.dialect_variations = DIALECT_REGIONAL_VARIATIONS
        self.cache_size = cache_size
        self.use_ml = use_ml
        
        # Initialize text processor for optimizations
        self.text_processor = TextProcessor()
        
        # Initialize cache
        self._dialect_cache = collections.OrderedDict()
        self._regional_cache = collections.OrderedDict()
        self._translation_cache = collections.OrderedDict()
        
        # Setup data directory
        if dialect_data_dir:
            self.data_dir = Path(dialect_data_dir)
        else:
            self.data_dir = Path(__file__).parent / "data"
        self.data_dir.mkdir(exist_ok=True, parents=True)
        
        # Load user-contributed dialect data if available
        self._load_user_contributions()
        
        # Load acoustic features for dialect recognition if available
        self.acoustic_features = self._load_acoustic_features()
        
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
        
        # Initialize ML-based dialect detector if enabled
        self.ml_detector = None
        if use_ml:
            self._initialize_ml_detector()

    def _initialize_ml_detector(self):
        """Initialize machine learning based dialect detector"""
        try:
            # First try to load fine-tuned dialect classifier if available
            model_path = self.data_dir / "dialect_classifier"
            if model_path.exists():
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                self.ml_detector = {
                    "model": AutoModelForSequenceClassification.from_pretrained(str(model_path)).to(self.device),
                    "tokenizer": AutoTokenizer.from_pretrained(str(model_path)),
                    "labels": list(self.dialects.keys())
                }
            else:
                # Otherwise use base model and adapt for dialect classification
                self.ml_detector = {
                    "model": self.model,
                    "tokenizer": self.tokenizer,
                    "labels": list(self.dialects.keys())
                }
            print("ML-based dialect detector initialized successfully")
        except Exception as e:
            print(f"Could not initialize ML-based dialect detector: {e}")
            self.use_ml = False

    def _load_user_contributions(self):
        """Load user-contributed dialect data"""
        contrib_file = self.data_dir / "user_contributions.json"
        if contrib_file.exists():
            try:
                with open(contrib_file, "r", encoding="utf-8") as f:
                    contributions = json.load(f)
                    
                # Update dialect vocabulary
                for dialect, items in contributions.get("vocabulary", {}).items():
                    if dialect in self.dialect_features:
                        if "vocabulary" not in self.dialect_features[dialect]:
                            self.dialect_features[dialect]["vocabulary"] = {}
                        self.dialect_features[dialect]["vocabulary"].update(items)
                
                # Update particles, pronouns, etc.
                for feature_type in ["particles", "pronouns", "verb_modifiers"]:
                    for dialect, items in contributions.get(feature_type, {}).items():
                        if dialect in self.dialect_features:
                            if feature_type not in self.dialect_features[dialect]:
                                self.dialect_features[dialect][feature_type] = []
                            # Add only new items
                            new_items = [item for item in items if item not in self.dialect_features[dialect][feature_type]]
                            self.dialect_features[dialect][feature_type].extend(new_items)
                
                print(f"Loaded user contributions for dialects: {len(contributions.get('vocabulary', {}))} vocabularies")
            except Exception as e:
                print(f"Error loading user contributions: {e}")

    def _load_acoustic_features(self) -> Dict[str, Any]:
        """Load acoustic features for dialect recognition"""
        features_file = self.data_dir / "acoustic_features.json"
        if features_file.exists():
            try:
                with open(features_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except:
                pass
        return {}

    def add_dialect_vocabulary(self, dialect: str, standard: str, dialect_word: str) -> bool:
        """Add new vocabulary mapping for a dialect
        
        Args:
            dialect: Dialect code
            standard: Standard Thai word
            dialect_word: Word in the specified dialect
            
        Returns:
            True if added successfully, False otherwise
        """
        if dialect not in self.dialect_features:
            return False
            
        if "vocabulary" not in self.dialect_features[dialect]:
            self.dialect_features[dialect]["vocabulary"] = {}
            
        # Add to dialect features
        self.dialect_features[dialect]["vocabulary"][standard] = dialect_word
        
        # Update regex patterns
        if dialect in self.dialect_patterns:
            # Extract existing pattern
            pattern_str = self.dialect_patterns[dialect].pattern
            # Remove closing boundary
            pattern_str = pattern_str[:-2]
            # Add new word with OR if not the first word
            escaped_word = re.escape(dialect_word)
            if pattern_str.endswith("|"):
                pattern_str += escaped_word + r")\b"
            else:
                pattern_str += r"|" + escaped_word + r")\b"
            # Recompile pattern
            self.dialect_patterns[dialect] = re.compile(pattern_str, re.UNICODE)
            
        # Save to user contributions
        self._save_user_contribution("vocabulary", dialect, {standard: dialect_word})
        
        # Clear relevant caches
        self._clear_relevant_caches(dialect)
        
        return True

    def _save_user_contribution(self, category: str, dialect: str, data: Any):
        """Save user contribution to file"""
        contrib_file = self.data_dir / "user_contributions.json"
        
        # Load existing contributions
        if contrib_file.exists():
            try:
                with open(contrib_file, "r", encoding="utf-8") as f:
                    contributions = json.load(f)
            except:
                contributions = {}
        else:
            contributions = {}
            
        # Ensure structure exists
        if category not in contributions:
            contributions[category] = {}
        if dialect not in contributions[category]:
            contributions[category][dialect] = {} if category == "vocabulary" else []
            
        # Add new data
        if category == "vocabulary":
            contributions[category][dialect].update(data)
        else:
            for item in data:
                if item not in contributions[category][dialect]:
                    contributions[category][dialect].append(item)
                    
        # Save back
        with open(contrib_file, "w", encoding="utf-8") as f:
            json.dump(contributions, f, ensure_ascii=False, indent=2)

    def _clear_relevant_caches(self, dialect: str):
        """Clear caches related to a specific dialect"""
        # Clear relevant entries from dialect detection cache
        keys_to_remove = []
        for key in self._dialect_cache:
            keys_to_remove.append(key)
        for key in keys_to_remove:
            if key in self._dialect_cache:
                del self._dialect_cache[key]
                
        # Clear translation caches involving this dialect
        keys_to_remove = []
        for key in self._translation_cache:
            src_dialect, _ = key.split(':', 1)
            if src_dialect == dialect or src_dialect == "central":
                keys_to_remove.append(key)
        for key in keys_to_remove:
            if key in self._translation_cache:
                del self._translation_cache[key]
                
        # Clear regional dialect cache for this dialect
        keys_to_remove = []
        for key in self._regional_cache:
            if key.startswith(dialect + ":"):
                keys_to_remove.append(key)
        for key in keys_to_remove:
            if key in self._regional_cache:
                del self._regional_cache[key]

    @functools.lru_cache(maxsize=1000)
    def _cached_text_cleaning(self, text: str) -> str:
        """Cache text cleaning for performance"""
        return clean_thai_text(text)

    def detect_dialect(self, text: str, threshold: float = 0.1, use_ml: Optional[bool] = None) -> Dict[str, float]:
        """Detect the likely Thai dialect in the text with improved accuracy and performance
        
        Args:
            text: Thai text to analyze
            threshold: Minimum ratio to consider a dialect present
            use_ml: Whether to use ML-based detection if available (overrides instance setting)
            
        Returns:
            Dictionary mapping dialect codes to confidence scores
        """
        # Use the text processor for optimization
        text = self.text_processor.preprocess_text(text)
        
        # Check cache first
        cache_key = f"{text[:100]}:{threshold}"
        if cache_key in self._dialect_cache:
            return self._dialect_cache[cache_key].copy()
        
        # Handle empty text
        if not text.strip():
            result = {"central": 1.0}
            self._update_cache(self._dialect_cache, cache_key, result)
            return result
        
        # Determine whether to use ML
        should_use_ml = use_ml if use_ml is not None else self.use_ml
        
        if should_use_ml and self.ml_detector:
            # Use ML-based approach
            try:
                # Tokenize text
                inputs = self.ml_detector["tokenizer"](
                    text, 
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding="max_length"
                ).to(self.device)
                
                # Get model predictions
                with torch.no_grad():
                    outputs = self.ml_detector["model"](**inputs)
                
                # Convert to probabilities
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                probs = probs.cpu().numpy()[0]
                
                # Create result dictionary
                result = {label: float(prob) for label, prob in zip(self.ml_detector["labels"], probs)}
                
                # Apply threshold filtering
                filtered_result = {k: v for k, v in result.items() if v >= threshold}
                
                # If nothing passes threshold, take the highest probability
                if not filtered_result:
                    max_dialect = max(result.items(), key=lambda x: x[1])[0]
                    filtered_result = {max_dialect: result[max_dialect]}
                
                # Normalize scores
                total = sum(filtered_result.values())
                if total > 0:
                    for dialect in filtered_result:
                        filtered_result[dialect] /= total

                # Update cache and return
                self._update_cache(self._dialect_cache, cache_key, filtered_result)
                return filtered_result
                
            except Exception as e:
                print(f"ML-based dialect detection failed: {e}. Falling back to rule-based.")
        
        # Fallback to rule-based approach or if ML is not enabled
        # Normalize and clean the text for rule-based analysis
        clean_text = self._cached_text_cleaning(text)
        
        words = clean_text.split()
        total_words = len(words)
        
        if total_words == 0:
            result = {"central": 1.0}  # Default to standard Thai for empty text
            self._update_cache(self._dialect_cache, cache_key, result)
            return result
        
        # Count dialect markers with improved pattern matching
        dialect_scores = {}
        
        # First pass: direct dialect marker detection
        for dialect, pattern in self.dialect_patterns.items():
            matches = pattern.findall(clean_text)
            initial_score = len(matches) / total_words if total_words > 0 else 0
            dialect_scores[dialect] = initial_score
        
        # Second pass: context analysis for more accurate detection
        # Look for patterns of words that appear together in certain dialects
        context_bonus = self._analyze_dialect_context(words)
        for dialect, bonus in context_bonus.items():
            if dialect in dialect_scores:
                dialect_scores[dialect] += bonus
        
        # If no strong dialect markers found, default to central Thai
        if all(score < threshold for score in dialect_scores.values()):
            dialect_scores["central"] = max(0.8, dialect_scores.get("central", 0))
        
        # Normalize scores
        total = sum(dialect_scores.values())
        if total > 0:
            for dialect in dialect_scores:
                dialect_scores[dialect] /= total
        
        # Update cache
        self._update_cache(self._dialect_cache, cache_key, dialect_scores)
        
        return dialect_scores

    def _analyze_dialect_context(self, words: List[str]) -> Dict[str, float]:
        """Analyze context patterns for better dialect detection
        
        Args:
            words: List of words from the cleaned text
            
        Returns:
            Dictionary mapping dialect codes to bonus scores
        """
        bonuses = {"northern": 0.0, "northeastern": 0.0, "southern": 0.0, "central": 0.0, "pattani_malay": 0.0}
        
        # Look for sequential patterns specific to dialects
        # Northern Thai patterns
        northern_sequences = [
            ["บ่", "ใจ้"],  # Not true
            ["เจ้า", "กา"],  # Question marker
            ["ปี้", "เปิ้น"],  # They/them
            ["ตึ้น", "ข้าว"],  # Eating rice
            ["งาม", "หนัก"]   # Very beautiful
        ]
        
        # Northeastern Thai patterns
        northeastern_sequences = [
            ["บ่", "แม่น"],   # Not true
            ["สิ", "ไป"],     # Will go
            ["เว้า", "หยัง"],  # Say what
            ["แซบ", "หลาย"],  # Very delicious 
            ["ข้อย", "สิ"]     # I will
        ]
        
        # Southern Thai patterns
        southern_sequences = [
            ["หนิ", "วะ"],     # Question marker
            ["ใผ", "หวา"],     # Who
            ["หรอย", "นัก"],   # Very delicious
            ["แหละ", "ไป"],    # Then go
            ["จัง", "โหล"]     # Emphasis
        ]
        
        # Pattani Malay patterns
        pattani_patterns = [
            ["มากัน", "แล็ฮ"],  # Eat already
            ["อาเกาะ", "นะ"],   # I will
            ["เปอกี", "ปาซะ"],  # Go to market
            ["ซาดะ", "บาญะ"]   # Very delicious
        ]
        
        # Check for sequences in the text
        # Simplified: Just check for pairs of words appearing within 3 words of each other
        for i, word in enumerate(words):
            context_window = words[i:i+4]  # Look at current word and next 3
            
            # Check for northern patterns
            for seq in northern_sequences:
                if all(w in context_window for w in seq):
                    bonuses["northern"] += 0.15
            
            # Check for northeastern patterns
            for seq in northeastern_sequences:
                if all(w in context_window for w in seq):
                    bonuses["northeastern"] += 0.15
                    
            # Check for southern patterns
            for seq in southern_sequences:
                if all(w in context_window for w in seq):
                    bonuses["southern"] += 0.15
                    
            # Check for Pattani Malay patterns
            for seq in pattani_patterns:
                if all(w in context_window for w in seq):
                    bonuses["pattani_malay"] += 0.15
        
        return bonuses

    def _update_cache(self, cache: Dict, key: str, value: Any):
        """Update LRU cache with size limit"""
        cache[key] = value.copy() if isinstance(value, dict) else value
        # If cache is too large, remove oldest items
        if len(cache) > self.cache_size:
            for _ in range(len(cache) - self.cache_size):
                cache.popitem(last=False)  # Remove oldest item

    def detect_regional_dialect(self, text: str, primary_dialect: Optional[str] = None) -> Dict[str, float]:
        """Detect the regional variation within a primary dialect with improved accuracy
        
        Args:
            text: Thai text to analyze
            primary_dialect: The primary dialect to analyze for regional variations
                             (northern, northeastern, southern). If None, will detect first.
            
        Returns:
            Dictionary mapping regional dialect codes to confidence scores
        """
        # Use the text processor for optimization
        text = self.text_processor.preprocess_text(text)
        
        # Detect primary dialect if not provided
        if primary_dialect is None:
            dialect_scores = self.detect_dialect(text)
            primary_dialect = max(dialect_scores, key=lambda k: dialect_scores[k])
        
        # Check cache
        cache_key = f"{primary_dialect}:{text[:100]}"
        if cache_key in self._regional_cache:
            return self._regional_cache[cache_key].copy()
        
        # Check if we have regional variations for this dialect
        if primary_dialect not in self.dialect_variations:
            result = {primary_dialect: 1.0}
            self._update_cache(self._regional_cache, cache_key, result)
            return result
        
        # Analyze regional variations
        region_scores = {}
        clean_text = self._cached_text_cleaning(text)
        word_count = len(clean_text.split())
        
        if word_count == 0:
            result = {primary_dialect: 1.0}
            self._update_cache(self._regional_cache, cache_key, result)
            return result
        
        for region, details in self.dialect_variations[primary_dialect].items():
            score = 0
            if "distinctive_words" in details:
                # Basic word counting with more weighting for rare words
                for word in details["distinctive_words"]:
                    # Count occurrences with word boundaries
                    matches = len(re.findall(r'\b' + re.escape(word) + r'\b', clean_text, re.UNICODE))
                    
                    # Apply higher weight for distinctive words that are unique to this region
                    uniqueness_factor = 1.0
                    for other_region, other_details in self.dialect_variations[primary_dialect].items():
                        if other_region != region:
                            other_words = other_details.get("distinctive_words", [])
                            if word not in other_words:
                                uniqueness_factor = 1.5  # Higher weight for unique words
                    
                    score += matches * uniqueness_factor
                    
            # Add analysis of phrases specific to this region
            region_examples = self.get_regional_dialect_examples(primary_dialect, region)
            for example_phrase, _ in region_examples:
                if example_phrase in clean_text:
                    score += 2.0  # Strong signal if a region-specific phrase is found
            
            region_scores[region] = score / max(1, word_count)
        
        # If scores are all zero, try phonetic pattern matching
        if all(score == 0 for score in region_scores.values()):
            for region, details in self.dialect_variations[primary_dialect].items():
                # Use description to extract phonetic patterns
                if "description" in details:
                    if "ออกเสียงสูง" in details["description"] and any(word.endswith("เจ้า") for word in clean_text.split()):
                        region_scores[region] += 0.5
                    elif "คล้ายภาษาลาว" in details["description"] and any(word in ["เด้อ", "เนาะ"] for word in clean_text.split()):
                        region_scores[region] += 0.5
                    elif "ออกเสียงยาว" in details["description"] and any(word.endswith("โหล") for word in clean_text.split()):
                        region_scores[region] += 0.5
        
        # If still no clear winner, assign score based on acoustic features if available
        if self.acoustic_features and primary_dialect in self.acoustic_features:
            # This would ideally use speech features, but we're simulating with text
            for region in region_scores:
                if region in self.acoustic_features[primary_dialect]:
                    # Look for textual hints of acoustic features
                    acoustic_hints = self.acoustic_features[primary_dialect][region].get("textual_hints", [])
                    for hint in acoustic_hints:
                        hint_pattern = re.compile(r'\b' + re.escape(hint) + r'\b', re.UNICODE)
                        if hint_pattern.search(clean_text):
                            region_scores[region] += 0.3
                    
                    # Check for orthographic representations of tonal patterns
                    if "tonal_patterns" in self.acoustic_features[primary_dialect][region]:
                        tonal_patterns = self.acoustic_features[primary_dialect][region]["tonal_patterns"]
                        for pattern in tonal_patterns:
                            if re.search(pattern, clean_text):
                                region_scores[region] += 0.2
        
        # If still no differentiation, try speech characteristic simulation
        if all(score == 0 for score in region_scores.values()) and primary_dialect in self.dialect_variations:
            # Simulate speech patterns from text using ending particles
            for region, details in self.dialect_variations[primary_dialect].items():
                if "distinctive_words" in details:
                    # Weight more heavily for ending particles which often indicate regional accent
                    for word in details.get("distinctive_words", []):
                        # Look particularly for words at the end of sentences
                        end_pattern = re.compile(r'\b' + re.escape(word) + r'[.,!?]*$', re.MULTILINE | re.UNICODE)
                        end_matches = len(end_pattern.findall(clean_text))
                        if end_matches > 0:
                            region_scores[region] += 0.4 * end_matches
        
        # Normalize scores
        total = sum(region_scores.values())
        if total > 0:
            for region in region_scores:
                region_scores[region] /= total
        else:
            # If still no scores, assign equal probabilities
            equal_score = 1.0 / len(region_scores)
            for region in region_scores:
                region_scores[region] = equal_score
        
        # Cache and return results
        self._update_cache(self._regional_cache, cache_key, region_scores)
        return region_scores

    def translate_from_standard(self, text: str, target_dialect: str) -> str:
        """Translate text from standard Thai to a specific dialect

        Args:
            text (str): Text in standard Thai
            target_dialect (str): Target dialect name

        Returns:
            str: Text translated to specified dialect
        """
        # Use text processor for optimization
        text = self.text_processor.preprocess_text(text)

        # Check if target dialect is valid
        if target_dialect not in self.dialects:
            raise ValueError(f"Unsupported dialect: {target_dialect}")

        # If target is central/standard Thai, return original
        if target_dialect == "central":
            return text

        # Get vocabulary mapping for target dialect
        vocab = self.dialect_features[target_dialect].get("vocabulary", {})

        # Perform word-for-word translation
        words = text.split()
        translated_words = []

        for word in words:
            # Look for standard Thai word in reverse vocab mapping
            translated = None
            for std_word, dialect_word in vocab.items():
                if word == std_word:
                    translated = dialect_word
                    break
            
            # If no translation found, keep original word
            if translated is None:
                translated = word
            
            translated_words.append(translated)

        return " ".join(translated_words)

# Add high-level function for dialect detection
def detect_dialect(text, threshold=0.5):
    """
    Detect dialect in a piece of text
    
    Args:
        text (str): Input text to analyze
        threshold (float): Confidence threshold for detection
        
    Returns:
        dict: Dictionary containing detected dialect and confidence score
    """
    identifier = DialectIdentifier()
    return identifier.identify(text, threshold=threshold)

"""
Thai Dialect Processing Module
"""

from typing import Dict, List, Optional, Union, Any
import os
from ..optimization.optimizer import TextProcessor

class DialectProcessor:
    """Base class for Thai dialect processing"""
    
    def __init__(self):
        self.text_processor = TextProcessor()
        self.supported_dialects = [
            "northern", "northeastern", "southern", "central"
        ]
    
    def process(self, text: str) -> str:
        """Process text with dialect awareness"""
        return self.text_processor.preprocess_text(text)
    
    def is_supported_dialect(self, dialect: str) -> bool:
        """Check if a dialect is supported"""
        return dialect.lower() in self.supported_dialects

class DialectIdentifier(DialectProcessor):
    """Identify Thai dialects in text"""
    
    def __init__(self):
        super().__init__()
        
    def identify(self, text: str, threshold: float = 0.5) -> Dict[str, Any]:
        """Identify the dialect of a text"""
        # Simple placeholder implementation
        # In a real implementation, this would use a machine learning model
        processed_text = self.process(text)
        
        # Placeholder: return central Thai as default
        return {
            "dialect": "central",
            "confidence": 0.95,
            "alternatives": {
                "northeastern": 0.03,
                "northern": 0.01,
                "southern": 0.01
            }
        }

class DialectTranslator(DialectProcessor):
    """Translate between Thai dialects"""
    
    def __init__(self):
        super().__init__()
    
    def translate(self, text: str, source_dialect: str = "auto", 
                 target_dialect: str = "central") -> Dict[str, Any]:
        """Translate text between dialects"""
        if source_dialect == "auto":
            # Detect source dialect
            detection = detect_dialect(text)
            source_dialect = detection["dialect"]
        
        # Check if dialects are supported
        if not self.is_supported_dialect(source_dialect):
            raise ValueError(f"Unsupported source dialect: {source_dialect}")
            
        if not self.is_supported_dialect(target_dialect):
            raise ValueError(f"Unsupported target dialect: {target_dialect}")
            
        # If source and target are the same, return original text
        if source_dialect == target_dialect:
            return {
                "translated_text": text,
                "source_dialect": source_dialect,
                "target_dialect": target_dialect
            }
        
        # Simple placeholder implementation
        # In a real implementation, this would use a translation model
        processed_text = self.process(text)
        
        return {
            "translated_text": processed_text,  # Just return processed text as placeholder
            "source_dialect": source_dialect,
            "target_dialect": target_dialect
        }

# High-level functions for easy access

def detect_dialect(text: str, threshold: float = 0.5) -> Dict[str, Any]:
    """
    Detect dialect in a piece of text
    
    Args:
        text (str): Input text to analyze
        threshold (float): Confidence threshold for detection
        
    Returns:
        dict: Dictionary containing detected dialect and confidence score
    """
    identifier = DialectIdentifier()
    return identifier.identify(text, threshold=threshold)

def identify_dialect(text: str) -> Dict[str, Any]:
    """Alias for detect_dialect"""
    return detect_dialect(text)

def translate_dialect(text: str, source_dialect: str = "auto", 
                      target_dialect: str = "central") -> Dict[str, Any]:
    """
    Translate text between Thai dialects
    
    Args:
        text (str): Input text to translate
        source_dialect (str): Source dialect (or 'auto' to detect)
        target_dialect (str): Target dialect
        
    Returns:
        dict: Dictionary with translated text and metadata
    """
    translator = DialectTranslator()
    return translator.translate(text, source_dialect, target_dialect)

def get_dialect_info() -> Dict[str, Dict]:
    """
    Get information about supported Thai dialects including names, regions, and codes
    
    Returns:
        Dict containing metadata about all supported dialects
    """
    return DIALECTS

def get_dialect_features() -> Dict[str, Dict]:
    """
    Get the dialect features dictionary containing vocabulary and patterns for each dialect
    
    Returns:
        Dict containing dialect features for all supported dialects
    """
    return DIALECT_FEATURES