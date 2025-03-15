"""
เทคนิคการเพิ่มข้อมูลสำหรับภาษาไทย (Thai Data Augmentation)
===========================================================

โมดูลนี้ใช้สำหรับการเพิ่มข้อมูล (Data Augmentation) สำหรับงานประมวลผลภาษาธรรมชาติภาษาไทย
เพื่อช่วยปรับปรุงประสิทธิภาพของโมเดล ML โดยการเพิ่มความหลากหลายของข้อมูลฝึกอบรม
"""

import re
import random
import copy
from typing import List, Dict, Tuple, Union, Optional
from thainlp.tokenize import word_tokenize
from thainlp.utils.thai_utils import get_thai_stopwords
from pythainlp.util import dict_trie
import numpy as np
from transformers import pipeline

class ThaiDataAugmenter:
    def __init__(self, synonym_dict_path: Optional[str] = None, use_built_in: bool = True):
        """
        คลาสสำหรับการเพิ่มข้อมูลภาษาไทย
        
        Parameters:
        -----------
        synonym_dict_path: str หรือ None
            เส้นทางไปยังพจนานุกรมคำพ้องความหมาย
        use_built_in: bool
            ใช้ข้อมูลที่มีในไลบรารีหรือไม่
        """
        self.synonym_dict_path: Optional[str] = synonym_dict_path
        self.use_built_in: bool = use_built_in
        
        # โหลดคำหยุด (stopwords)
        self.stopwords: set = get_thai_stopwords()
        
        # โหลดพจนานุกรมคำพ้องความหมาย
        self.synonyms: Dict[str, List[str]] = self._load_synonyms()
        
        # โหลดโมเดลแปลภาษา (ถ้าระบุ)
        self.translator = self._load_translation_model()
    
    def _load_synonyms(self) -> Dict[str, List[str]]:
        """โหลดพจนานุกรมคำพ้องความหมาย"""
        synonyms: Dict[str, List[str]] = {}
        
        # โหลดจากไฟล์ที่กำหนด
        if self.synonym_dict_path:
            try:
                with open(self.synonym_dict_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        words = line.strip().split(',')
                        if len(words) >= 2:
                            for word in words:
                                synonyms[word] = [w for w in words if w != word]
            except Exception as e:
                print(f"ไม่สามารถโหลดพจนานุกรมคำพ้องความหมายได้: {str(e)}")
        
        # โหลดจากพจนานุกรมที่มีอยู่แล้ว
        if self.use_built_in:
            try:
                # ตัวอย่างคำพ้องความหมายขนาดเล็ก (ในระบบจริงควรใช้พจนานุกรมที่ใหญ่กว่านี้)
                default_synonyms: Dict[str, List[str]] = {
                    "รถ": ["ยานพาหนะ", "รถยนต์", "พาหนะ"],
                    "บ้าน": ["ที่อยู่", "ที่พัก", "เรือน", "คฤหาสน์"],
                    "กิน": ["รับประทาน", "ทาน", "บริโภค"],
                    "พูด": ["คุย", "สนทนา", "เจรจา", "พูดคุย"],
                    "เดิน": ["ย่าง", "ย่างก้าว", "ก้าว"],
                    "เงิน": ["ตังค์", "สตางค์", "ทรัพย์"],
                    "ความสุข": ["ความปลื้มปีติ", "ความยินดี", "ความสำราญ"],
                    "คิด": ["นึก", "คำนึง", "ตรึกตรอง"],
                    "ใหญ่": ["โต", "มหึมา", "ใหญ่โต"],
                    "เล็ก": ["จิ๋ว", "น้อย", "จ้อย"],
                    "สวย": ["งาม", "โสภา", "งดงาม"],
                    "เร็ว": ["ไว", "รวดเร็ว", "ฉับไว", "ด่วน"],
                    "ช้า": ["เชื่องช้า", "เนิบนาบ", "เฉื่อยชา"],
                    "ดี": ["เยี่ยม", "ประเสริฐ", "วิเศษ"],
                    "แย่": ["เลว", "ไม่ดี", "ทราม"],
                    "ชอบ": ["พอใจ", "ถูกใจ", "โปรดปราน"],
                    "รัก": ["เสน่หา", "หลงใหล", "รักใคร่"],
                    "เกลียด": ["ชิงชัง", "รังเกียจ", "เคียดแค้น"],
                    "ร้อน": ["ร้อนระอุ", "ร้อนจัด", "ร้อนผ่าว"],
                    "หนาว": ["เย็น", "เหน็บหนาว", "หนาวเย็น"]
                }
                synonyms.update(default_synonyms)
            except Exception as e:
                print(f"ไม่สามารถโหลดพจนานุกรมคำพ้องความหมายเริ่มต้นได้: {str(e)}")
                
        return synonyms
    
    def _load_translation_model(self):
        """โหลดโมเดลแปลภาษา"""
        if not self.translation_model:
            return None
            
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
            
            # ใช้โมเดลแปลภาษาจาก Hugging Face
            tokenizer = AutoTokenizer.from_pretrained(self.translation_model)
            model = AutoModelForSeq2SeqLM.from_pretrained(self.translation_model)
            translator = pipeline("translation", model=model, tokenizer=tokenizer)
            
            return translator
        except Exception as e:
            print(f"ไม่สามารถโหลดโมเดลแปลภาษาได้: {str(e)}")
            print("การเพิ่มข้อมูลด้วย back-translation จะไม่สามารถใช้งานได้")
            return None
    
    def augment_text(self, text: str, techniques: List[str] = None, n_aug: int = 1) -> List[str]:
        """
        เพิ่มข้อมูลข้อความโดยใช้เทคนิคต่างๆ
        
        Parameters:
        -----------
        text: str
            ข้อความที่ต้องการเพิ่มข้อมูล
        techniques: List[str]
            รายการเทคนิคที่ต้องการใช้ เช่น ["synonym", "random_swap", "back_translation"]
            ถ้าไม่ระบุ จะใช้ทุกเทคนิคที่เป็นไปได้
        n_aug: int
            จำนวนข้อความที่ต้องการสร้าง
            
        Returns:
        --------
        augmented_texts: List[str]
            รายการข้อความที่เพิ่มข้อมูลแล้ว
        """
        available_techniques = ["synonym_replacement", "random_deletion", "random_swap", "random_insertion"]
        
        if self.translator:
            available_techniques.append("back_translation")
            
        # ถ้าไม่ระบุเทคนิค ใช้ทั้งหมดที่มี
        if not techniques:
            techniques = available_techniques
        else:
            # กรองเฉพาะเทคนิคที่มี
            techniques = [t for t in techniques if t in available_techniques]
            
        if not techniques:
            return [text] * n_aug
            
        augmented_texts = []
        for _ in range(n_aug):
            # สุ่มเลือกเทคนิค
            technique = random.choice(techniques)
            
            if technique == "synonym_replacement":
                aug_text = self._synonym_replacement(text)
            elif technique == "random_deletion":
                aug_text = self._random_deletion(text)
            elif technique == "random_swap":
                aug_text = self._random_swap(text)
            elif technique == "random_insertion":
                aug_text = self._random_insertion(text)
            elif technique == "back_translation" and self.translator:
                aug_text = self._back_translation(text)
            else:
                aug_text = text
                
            if aug_text != text:
                augmented_texts.append(aug_text)
        
        # ถ้าไม่มีข้อความที่เพิ่มข้อมูลได้ คืนข้อความเดิม
        if not augmented_texts:
            return [text] * n_aug
            
        # ถ้าได้ข้อความน้อยกว่าที่ต้องการ สุ่มเลือกจากที่มีเพื่อให้ได้จำนวนตามต้องการ
        if len(augmented_texts) < n_aug:
            additional = random.choices(augmented_texts, k=n_aug - len(augmented_texts))
            augmented_texts.extend(additional)
            
        return augmented_texts
    
    def _synonym_replacement(self, text: str, p=0.3) -> str:
        """
        แทนที่คำด้วยคำพ้องความหมาย
        
        Parameters:
        -----------
        text: str
            ข้อความต้นฉบับ
        p: float
            สัดส่วนของคำที่จะถูกแทนที่ (0.0-1.0)
            
        Returns:
        --------
        augmented_text: str
            ข้อความที่แทนที่คำด้วยคำพ้องความหมายแล้ว
        """
        tokens = word_tokenize(text)
        num_to_replace = max(1, int(len(tokens) * p))
        
        # กรองเฉพาะคำที่มีคำพ้องความหมายและไม่ใช่คำหยุด
        candidates = [i for i, token in enumerate(tokens) if token in self.synonyms and token not in self.stopwords]
        
        if not candidates:
            return text
            
        # สุ่มเลือกตำแหน่งที่จะแทนที่
        indexes = random.sample(candidates, min(num_to_replace, len(candidates)))
        
        for idx in indexes:
            token = tokens[idx]
            if token in self.synonyms:
                synonyms = self.synonyms[token]
                if synonyms:
                    tokens[idx] = random.choice(synonyms)
                    
        return ''.join(tokens)
    
    def _random_deletion(self, text: str, p=0.1) -> str:
        """
        ลบคำแบบสุ่ม
        
        Parameters:
        -----------
        text: str
            ข้อความต้นฉบับ
        p: float
            สัดส่วนของคำที่จะถูกลบ (0.0-1.0)
            
        Returns:
        --------
        augmented_text: str
            ข้อความที่ลบคำแล้ว
        """
        tokens = word_tokenize(text)
        
        # ถ้ามีแค่คำเดียว ไม่ลบ
        if len(tokens) == 1:
            return text
            
        # กรองเฉพาะคำที่ไม่ใช่คำหยุด
        candidates = [i for i, token in enumerate(tokens) if token not in self.stopwords]
        
        if not candidates:
            return text
            
        # จำนวนคำที่จะลบ
        num_to_delete = max(1, int(len(candidates) * p))
        
        # สุ่มเลือกตำแหน่งที่จะลบ
        indexes = sorted(random.sample(candidates, min(num_to_delete, len(candidates))), reverse=True)
        
        for idx in indexes:
            tokens.pop(idx)
            
        return ''.join(tokens)
    
    def _random_swap(self, text: str, n=1) -> str:
        """
        สลับตำแหน่งคำแบบสุ่ม
        
        Parameters:
        -----------
        text: str
            ข้อความต้นฉบับ
        n: int
            จำนวนครั้งที่จะสลับ
            
        Returns:
        --------
        augmented_text: str
            ข้อความที่สลับคำแล้ว
        """
        tokens = word_tokenize(text)
        
        # ถ้ามีคำน้อยกว่า 2 คำ ไม่สลับ
        if len(tokens) < 2:
            return text
            
        for _ in range(n):
            # สุ่มเลือกตำแหน่งที่จะสลับ
            idx1, idx2 = random.sample(range(len(tokens)), 2)
            
            # สลับตำแหน่ง
            tokens[idx1], tokens[idx2] = tokens[idx2], tokens[idx1]
            
        return ''.join(tokens)
    
    def _random_insertion(self, text: str, n=1) -> str:
        """
        แทรกคำแบบสุ่ม
        
        Parameters:
        -----------
        text: str
            ข้อความต้นฉบับ
        n: int
            จำนวนคำที่จะแทรก
            
        Returns:
        --------
        augmented_text: str
            ข้อความที่แทรกคำแล้ว
        """
        tokens = word_tokenize(text)
        
        # กรองเฉพาะคำที่มีคำพ้องความหมายและไม่ใช่คำหยุด
        candidates = [token for token in tokens if token in self.synonyms and token not in self.stopwords]
        
        # ถ้าไม่มีคำที่เหมาะสม ไม่แทรก
        if not candidates:
            return text
            
        for _ in range(n):
            # สุ่มเลือกคำ
            token = random.choice(candidates)
            
            # สุ่มเลือกคำพ้องความหมายของคำที่เลือก
            if token in self.synonyms and self.synonyms[token]:
                synonym = random.choice(self.synonyms[token])
                
                # สุ่มเลือกตำแหน่งที่จะแทรก
                insert_position = random.randint(0, len(tokens))
                
                # แทรกคำ
                tokens.insert(insert_position, synonym)
            
        return ''.join(tokens)
    
    def _back_translation(self, text: str, intermediate_lang='en') -> str:
        """
        เพิ่มข้อมูลโดยใช้เทคนิค back-translation
        
        Parameters:
        -----------
        text: str
            ข้อความต้นฉบับ
        intermediate_lang: str
            ภาษาตัวกลาง (เช่น 'en' = ภาษาอังกฤษ)
            
        Returns:
        --------
        augmented_text: str
            ข้อความที่ผ่านการแปลกลับมาแล้ว
        """
        if not self.translator:
            return text
            
        try:
            # แปลจากภาษาไทยเป็นภาษาตัวกลาง
            translated = self.translator(text, src_lang="th", tgt_lang=intermediate_lang)[0]["translation_text"]
            
            # แปลกลับจากภาษาตัวกลางเป็นภาษาไทย
            back_translated = self.translator_back(translated, src_lang=intermediate_lang, tgt_lang="th")[0]["translation_text"]
            
            return back_translated
        except Exception as e:
            print(f"เกิดข้อผิดพลาดในการแปลภาษา: {str(e)}")
            return text
    
    def augment_dataset(self, texts: List[str], labels: List = None, techniques: List[str] = None, 
                        n_aug_per_text: int = 1) -> Tuple[List[str], List]:
        """
        เพิ่มข้อมูลชุดข้อมูล
        
        Parameters:
        -----------
        texts: List[str]
            รายการข้อความต้นฉบับ
        labels: List หรือ None
            รายการป้ายกำกับ (ถ้ามี)
        techniques: List[str]
            รายการเทคนิคที่ต้องการใช้
        n_aug_per_text: int
            จำนวนข้อความที่ต้องการสร้างต่อหนึ่งข้อความต้นฉบับ
            
        Returns:
        --------
        (augmented_texts, augmented_labels): Tuple[List[str], List]
            ข้อความและป้ายกำกับที่เพิ่มข้อมูลแล้ว
        """
        augmented_texts = []
        augmented_labels = []
        
        for i, text in enumerate(texts):
            # เพิ่มข้อความต้นฉบับไปยังผลลัพธ์
            augmented_texts.append(text)
            
            if labels is not None:
                augmented_labels.append(labels[i])
                
            # สร้างข้อความใหม่
            aug_texts = self.augment_text(text, techniques, n_aug=n_aug_per_text)
            
            # เพิ่มข้อความใหม่และป้ายกำกับไปยังผลลัพธ์
            augmented_texts.extend(aug_texts)
            
            if labels is not None:
                augmented_labels.extend([labels[i]] * len(aug_texts))
        
        if labels is None:
            return augmented_texts, None
        else:
            return augmented_texts, augmented_labels

    def eda(self, text: str, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, alpha_rd=0.1, n_aug=4) -> List[str]:
        """
        เพิ่มข้อมูลโดยใช้เทคนิค EDA (Easy Data Augmentation)
        
        Parameters:
        -----------
        text: str
            ข้อความต้นฉบับ
        alpha_sr: float
            สัดส่วนของคำที่จะถูกแทนที่ด้วยคำพ้องความหมาย
        alpha_ri: float
            สัดส่วนของคำที่จะถูกแทรก
        alpha_rs: float
            สัดส่วนของคำที่จะถูกสลับ
        alpha_rd: float
            สัดส่วนของคำที่จะถูกลบ
        n_aug: int
            จำนวนข้อความที่ต้องการสร้าง
            
        Returns:
        --------
        augmented_texts: List[str]
            รายการข้อความที่เพิ่มข้อมูลแล้ว
        """
        augmented_texts = []
        num_words = len(word_tokenize(text))
        
        # สร้างข้อความใหม่
        for _ in range(n_aug):
            a_text = text
            
            # แทนที่คำด้วยคำพ้องความหมาย
            num_sr = max(1, int(alpha_sr * num_words))
            a_text = self._synonym_replacement(a_text, p=alpha_sr)
            
            # แทรกคำ
            num_ri = max(1, int(alpha_ri * num_words))
            a_text = self._random_insertion(a_text, n=num_ri)
            
            # สลับคำ
            num_rs = max(1, int(alpha_rs * num_words))
            a_text = self._random_swap(a_text, n=num_rs)
            
            # ลบคำ
            a_text = self._random_deletion(a_text, p=alpha_rd)
            
            augmented_texts.append(a_text)
            
        return augmented_texts

    def augment_with_templates(self, templates: List[str], slots: Dict[str, List[str]], n_aug: int = 10) -> List[str]:
        """
        เพิ่มข้อมูลโดยใช้แม่แบบ
        
        Parameters:
        -----------
        templates: List[str]
            รายการแม่แบบประโยค เช่น ["ฉัน{กริยา}ไป{สถานที่}", "วันนี้ฉันจะ{กริยา}ที่{สถานที่}"]
        slots: Dict[str, List[str]]
            พจนานุกรมของตำแหน่งและคำที่สามารถใช้ได้ เช่น {"กริยา": ["เดิน", "วิ่ง"], "สถานที่": ["บ้าน", "โรงเรียน"]}
        n_aug: int
            จำนวนข้อความที่ต้องการสร้าง
            
        Returns:
        --------
        augmented_texts: List[str]
            รายการข้อความที่สร้างจากแม่แบบ
        """
        augmented_texts = []
        
        for _ in range(n_aug):
            # สุ่มเลือกแม่แบบ
            template = random.choice(templates)
            
            # แทนที่ทุกตำแหน่งด้วยคำที่สุ่มเลือก
            text = template
            for slot, words in slots.items():
                if f"{{{slot}}}" in text:
                    text = text.replace(f"{{{slot}}}", random.choice(words))
                    
            augmented_texts.append(text)
            
        return augmented_texts
