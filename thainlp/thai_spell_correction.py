"""
ระบบแก้ไขคำผิดภาษาไทย (Thai Spelling Correction System)
=========================================================

โมดูลนี้ใช้สำหรับการตรวจจับและแก้ไขคำผิดในภาษาไทย โดยใช้เทคนิคต่างๆ เช่น:
- การใช้พจนานุกรม (Dictionary-based)
- การใช้กฎ (Rule-based)
- การใช้ Context-aware
- การใช้ Deep learning
"""

import re
import os
import json
import pickle
import numpy as np
import editdistance
from typing import List, Dict, Tuple, Union, Set
from collections import Counter
from thainlp.tokenize import word_tokenize
from thainlp.utils.thai_utils import normalize_text, get_thai_stopwords, spell_correction
from pythainlp.util import dict_trie

class ThaiSpellCorrector:
    def __init__(self, custom_dict_path=None, use_built_in=True, context_window=2, max_edit_distance=2,
                 dialect_support=True, dialect_dict_path=None):
        """
        คลาสสำหรับการตรวจจับและแก้ไขคำผิดในภาษาไทย
        
        Parameters:
        -----------
        custom_dict_path: str หรือ None
            เส้นทางไปยังพจนานุกรมที่กำหนดเอง (ไฟล์ข้อความที่มีคำหนึ่งคำต่อบรรทัด)
        use_built_in: bool
            ใช้พจนานุกรมภาษาไทยที่มีอยู่แล้วหรือไม่
        context_window: int
            จำนวนคำก่อนและหลังที่จะพิจารณาเป็นบริบท
        max_edit_distance: int
            ระยะห่างแก้ไขสูงสุดสำหรับการแนะนำคำที่ถูกต้อง
        dialect_support: bool
            เปิดใช้การรองรับภาษาถิ่นหรือไม่
        dialect_dict_path: str หรือ None
            เส้นทางไปยังพจนานุกรมภาษาถิ่น (JSON หรือข้อความที่มีคำมาตรฐาน[TAB]คำภาษาถิ่น)
        """
        self.custom_dict_path = custom_dict_path
        self.use_built_in = use_built_in
        self.context_window = context_window
        self.max_edit_distance = max_edit_distance
        self.dialect_support = dialect_support
        
        # โหลดพจนานุกรมภาษาไทย
        self.thai_words = set()
        
        if use_built_in:
            try:
                from pythainlp.corpus import thai_words as pythainlp_thai_words
                self.thai_words.update(pythainlp_thai_words())
                print(f"โหลดพจนานุกรมภาษาไทยจาก PyThaiNLP จำนวน {len(self.thai_words)} คำ")
            except:
                print("ไม่สามารถโหลดพจนานุกรมจาก PyThaiNLP ได้")
        
        # โหลดพจนานุกรมกำหนดเอง
        if custom_dict_path and os.path.exists(custom_dict_path):
            with open(custom_dict_path, 'r', encoding='utf-8') as f:
                custom_words = set([line.strip() for line in f.readlines()])
                self.thai_words.update(custom_words)
                print(f"โหลดพจนานุกรมกำหนดเองจำนวน {len(custom_words)} คำ")
        
        # โหลดพจนานุกรมภาษาถิ่น
        self.dialect_to_standard = {}  # คำภาษาถิ่น -> คำมาตรฐาน
        self.standard_to_dialect = {}  # คำมาตรฐาน -> [คำภาษาถิ่น]
        
        if dialect_support:
            if dialect_dict_path and os.path.exists(dialect_dict_path):
                self._load_dialect_dictionary(dialect_dict_path)
            else:
                self._init_default_dialect_dict()
                
        # สร้าง Trie สำหรับการค้นหาคำอย่างรวดเร็ว
        self.word_trie = dict_trie(self.thai_words)
        
        # โหลดโมเดลภาษาเพื่อใช้ในการคำนวณความน่าจะเป็นของคำ
        self.ngram_model = self._load_or_train_ngram_model()
        
        # สร้างแคช
        self._cache = {}
    
    def _load_dialect_dictionary(self, dialect_dict_path: str):
        """โหลดพจนานุกรมภาษาถิ่น"""
        ext = os.path.splitext(dialect_dict_path)[1].lower()
        
        if ext == '.json':
            # โหลดจากไฟล์ JSON
            with open(dialect_dict_path, 'r', encoding='utf-8') as f:
                dialect_dict = json.load(f)
                
            if isinstance(dialect_dict, dict):
                # กรณี format เป็น {"คำภาษาถิ่น": "คำมาตรฐาน", ...}
                self.dialect_to_standard = dialect_dict
                
                # สร้างพจนานุกรมย้อนกลับ
                for dialect, standard in dialect_dict.items():
                    if standard not in self.standard_to_dialect:
                        self.standard_to_dialect[standard] = []
                    self.standard_to_dialect[standard].append(dialect)
            
        else:
            # โหลดจากไฟล์ข้อความ format: "คำมาตรฐาน[TAB]คำภาษาถิ่น"
            with open(dialect_dict_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or '\t' not in line:
                        continue
                        
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        standard = parts[0].strip()
                        dialect = parts[1].strip()
                        
                        self.dialect_to_standard[dialect] = standard
                        
                        if standard not in self.standard_to_dialect:
                            self.standard_to_dialect[standard] = []
                        self.standard_to_dialect[standard].append(dialect)
        
        print(f"โหลดพจนานุกรมภาษาถิ่นจำนวน {len(self.dialect_to_standard)} คำ")
        
        # เพิ่มคำภาษาถิ่นเข้าไปในพจนานุกรมหลักด้วย เพื่อให้ไม่ถูกตรวจว่าเป็นคำผิด
        self.thai_words.update(self.dialect_to_standard.keys())
        
    def _init_default_dialect_dict(self):
        """สร้างพจนานุกรมภาษาถิ่นเริ่มต้น"""
        # ภาษาถิ่นอีสาน
        isan_dict = {
            "เบิ่ง": "ดู",
            "สิเฮ็ด": "จะทำ",
            "บักหล่า": "ตะกร้า",
            "เฮือน": "บ้าน",
            "เจ้า": "คุณ",
            "อีหลี": "นี่ไง",
            "อีหยัง": "อะไร",
            "บ่": "ไม่",
            "แม่น": "ใช่",
            "เด้อ": "นะ",
            "กะ": "ก็",
            "สิ": "จะ",
            "ให้": "ให้",
            "อ้าย": "พี่ชาย",
            "อีสาว": "น้องสาว",
            "ไป": "ไป",
            "มา": "มา",
            "เว้า": "พูด"
        }
        
        # ภาษาถิ่นเหนือ (คำเมือง)
        northern_dict = {
            "กิ๋น": "กิน",
            "เปิ้น": "เขา",
            "อู้": "พูด",
            "บะเดี่ยวนี้": "เดี๋ยวนี้",
            "เตื่อ": "ครั้ง",
            "คัวะ": "โกรธ",
            "เมิน": "นาน",
            "จ๊าง": "สวย",
            "ก่อ": "ก็",
            "ตี้": "ที่",
            "หนาว": "เย็น",
            "หมอก": "เมฆ",
            "บ่าว": "หนุ่ม",
            "แต้": "จริง",
            "ฮ้องไห้": "ร้องไห้",
            "จะอี้": "แบบนี้",
            "กำลังอู้": "กำลังพูด"
        }
        
        # ภาษาถิ่นใต้
        southern_dict = {
            "หวาน": "พูด",
            "ตง": "ตรง",
            "กิน": "กิน",
            "แอ": "เธอ",
            "ไอ้": "เขา",
            "หวัน": "ฉัน",
            "มุง": "เธอ",
            "กูม": "อ่าน",
            "นิ": "นี้",
            "ชิ": "จะ",
            "เหม": "ไม่",
            "บ่": "ไม่",
            "หม่าย": "ไม่",
            "แล": "นะ",
            "เหลียว": "เล่า",
            "นุ้ย": "หนู"
        }
        
        # รวมพจนานุกรม
        self.dialect_to_standard = {**isan_dict, **northern_dict, **southern_dict}
        
        # สร้างพจนานุกรมย้อนกลับ
        for dialect, standard in self.dialect_to_standard.items():
            if standard not in self.standard_to_dialect:
                self.standard_to_dialect[standard] = []
            self.standard_to_dialect[standard].append(dialect)
            
        print(f"สร้างพจนานุกรมภาษาถิ่นเริ่มต้นจำนวน {len(self.dialect_to_standard)} คำ")
    
    def _load_or_train_ngram_model(self, corpus_path=None, n=2):
        """โหลดหรือสร้างโมเดล n-gram"""
        from collections import defaultdict
        
        model_path = "thai_ngram_model.pkl"
        
        # ลองโหลดโมเดลที่มีอยู่
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    return pickle.load(f)
            except:
                print("ไม่สามารถโหลดโมเดล n-gram ได้ จะสร้างโมเดลใหม่")
        
        # สร้างโมเดลใหม่
        model = defaultdict(lambda: defaultdict(int))
        
        # ถ้ามี corpus ให้ฝึกจาก corpus
        if corpus_path and os.path.exists(corpus_path):
            with open(corpus_path, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            # มิฉะนั้นใช้ข้อความตัวอย่างสั้นๆ
            text = """
            ภาษาไทยเป็นภาษาที่มีผู้พูดมากกว่า 60 ล้านคนทั่วโลก ส่วนใหญ่อยู่ในประเทศไทย
            ภาษาไทยเป็นภาษาในกลุ่มตระกูลไท-กะได ซึ่งมีความสัมพันธ์กับภาษาลาว ภาษาไทใหญ่ และภาษาจีนตระกูลย่อยจ้วง-ไต
            
            ภาษาไทยมีระบบเสียงวรรณยุกต์ คือเสียงสูงต่ำที่เปลี่ยนไปตามระดับเสียง โดยมีวรรณยุกต์ 5 เสียง คือ เสียงสามัญ เสียงเอก เสียงโท เสียงตรี และเสียงจัตวา
            
            การแก้ไขคำผิดในภาษาไทยมีความท้าทาย เนื่องจากไม่มีการเว้นวรรคระหว่างคำ และการสะกดคำในภาษาไทยมีความซับซ้อน
            ระบบการแก้ไขคำผิดภาษาไทยอัตโนมัติจะมีประโยชน์อย่างมากในการพัฒนาเทคโนโลยีภาษาไทย
            """
        
        # แบ่งเป็นประโยค
        sentences = re.split(r'[।|।|،|؛|؟|٪|٬|٭|!|"|#|%|&|\'|\(|\)|\*|\+|,|\-|\.|/|:|;|<|=|>|\?|@|\[|\\|\]|\^|_|`|\{|\||\}|~|\n|\t|\r]', text)
        
        # สร้างโมเดล n-gram
        for sentence in sentences:
            tokens = word_tokenize(sentence)
            if len(tokens) < n:
                continue
                
            for i in range(len(tokens) - (n - 1)):
                context = tuple(tokens[i:i+n-1])
                next_word = tokens[i+n-1]
                model[context][next_word] += 1
        
        # บันทึกโมเดล
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(dict(model), f)
        except:
            print("ไม่สามารถบันทึกโมเดล n-gram ได้")
        
        return model
        
    def detect_spelling_errors(self, text: str) -> List[Dict]:
        """
        ตรวจจับคำผิดในข้อความ
        
        Parameters:
        -----------
        text: str
            ข้อความที่ต้องการตรวจสอบ
            
        Returns:
        --------
        errors: List[Dict]
            รายการของคำผิดที่ตรวจพบ พร้อมตำแหน่ง คำแนะนำ และเหตุผล
        """
        # ตัดคำ
        tokens = word_tokenize(text)
        errors = []
        
        for i, token in enumerate(tokens):
            # ข้ามคำที่ไม่ใช่ภาษาไทย (เช่น ตัวเลข, ภาษาอังกฤษ)
            if not re.search(r'[\u0E00-\u0E7F]', token):
                continue
                
            # ข้ามคำที่มีในพจนานุกรม
            if token in self.thai_words:
                continue
            
            # ตรวจสอบว่าเป็นคำภาษาถิ่นหรือไม่
            is_dialect = False
            dialect_standard = None
            
            if self.dialect_support and token in self.dialect_to_standard:
                is_dialect = True
                dialect_standard = self.dialect_to_standard[token]
                
            # หาคำแนะนำ
            suggestions = self._get_suggestions(token, tokens, i)
            
            # ระบุตำแหน่งในข้อความ
            position = text.find(token)
            
            error = {
                "word": token,
                "position": position,
                "suggestions": suggestions,
                "is_dialect": is_dialect
            }
            
            if is_dialect:
                error["dialect_standard"] = dialect_standard
                error["reason"] = f"คำภาษาถิ่น (คำมาตรฐาน: {dialect_standard})"
            else:
                error["reason"] = "คำนี้ไม่พบในพจนานุกรม"
                
            errors.append(error)
            
        return errors
        
    def _get_suggestions(self, misspelled_word: str, tokens: List[str], position: int) -> List[Dict]:
        """
        หาคำแนะนำสำหรับคำผิด
        
        Parameters:
        -----------
        misspelled_word: str
            คำผิดที่ต้องการหาคำแนะนำ
        tokens: List[str]
            รายการคำที่ตัดจากข้อความ
        position: int
            ตำแหน่งของคำผิดในรายการคำ
            
        Returns:
        --------
        suggestions: List[Dict]
            รายการคำแนะนำพร้อมคะแนน
        """
        # ตรวจสอบว่าเคยคำนวณแล้วหรือไม่
        cache_key = (misspelled_word, tuple(tokens[max(0, position-self.context_window):min(len(tokens), position+self.context_window+1)]))
        if cache_key in self._cache:
            return self._cache[cache_key]
            
        suggestions = []
        
        # 1. การแนะนำโดยใช้ระยะห่างแก้ไข (Edit Distance)
        edit_distance_suggestions = self._get_edit_distance_suggestions(misspelled_word)
        
        # 2. การแนะนำโดยใช้โมเดลภาษา (Language Model)
        lm_suggestions = self._get_language_model_suggestions(misspelled_word, tokens, position)
        
        # 3. การแนะนำโดยใช้การแปลงภาษาถิ่น
        dialect_suggestions = []
        if self.dialect_support and misspelled_word in self.dialect_to_standard:
            standard_word = self.dialect_to_standard[misspelled_word]
            dialect_suggestions.append({
                "word": standard_word,
                "score": 0.95,  # ให้คะแนนสูง เพราะเป็นคำมาตรฐานที่ตรงกับคำภาษาถิ่น
                "method": "dialect"
            })
        
        # รวมคำแนะนำทั้งหมด
        all_suggestions = edit_distance_suggestions + lm_suggestions + dialect_suggestions
        
        # กรองคำซ้ำและเรียงลำดับตามคะแนน
        seen = set()
        filtered_suggestions = []
        
        for suggestion in all_suggestions:
            word = suggestion["word"]
            if word not in seen and word != misspelled_word:
                seen.add(word)
                filtered_suggestions.append(suggestion)
        
        filtered_suggestions.sort(key=lambda x: x["score"], reverse=True)
        
        # เก็บในแคช
        self._cache[cache_key] = filtered_suggestions[:5]  # เก็บเฉพาะ 5 อันดับแรก
        
        return filtered_suggestions[:5]
        
    def _get_edit_distance_suggestions(self, misspelled_word: str) -> List[Dict]:
        """หาคำแนะนำโดยใช้ระยะห่างแก้ไข"""
        suggestions = []
        
        for word in self.thai_words:
            # คำนวณระยะห่างแก้ไข
            distance = editdistance.eval(misspelled_word, word)
            
            # พิจารณาเฉพาะคำที่มีระยะห่างไม่เกินที่กำหนด
            if distance <= self.max_edit_distance:
                # คำนวณคะแนน (ยิ่งระยะห่างน้อย ยิ่งมีคะแนนสูง)
                score = 1.0 - (distance / max(len(misspelled_word), len(word)))
                
                suggestions.append({
                    "word": word,
                    "score": score,
                    "method": "edit_distance"
                })
        
        # เรียงลำดับตามคะแนน
        suggestions.sort(key=lambda x: x["score"], reverse=True)
        
        return suggestions[:10]  # ส่งคืนเพียง 10 คำแนะนำที่ดีที่สุด
        
    def _get_language_model_suggestions(self, misspelled_word: str, tokens: List[str], position: int) -> List[Dict]:
        """หาคำแนะนำโดยใช้โมเดลภาษา"""
        suggestions = []
        
        # พิจารณาบริบท (คำก่อนหน้าและหลัง)
        start = max(0, position - self.context_window)
        end = min(len(tokens), position + self.context_window + 1)
        
        context_before = tokens[start:position]
        context_after = tokens[position+1:end]
        
        # คำนวณคะแนนสำหรับแต่ละคำในพจนานุกรม
        candidate_words = set()
        
        # เพิ่มคำที่มีระยะห่างแก้ไขไม่เกินที่กำหนด
        for word in self.thai_words:
            if editdistance.eval(misspelled_word, word) <= self.max_edit_distance:
                candidate_words.add(word)
                
        # เพิ่มคำมาตรฐานจากภาษาถิ่น
        if self.dialect_support and misspelled_word in self.dialect_to_standard:
            candidate_words.add(self.dialect_to_standard[misspelled_word])
            
        for word in candidate_words:
            score = 0.0
            
            # คะแนนจาก bigram ก่อนหน้า
            if context_before and len(context_before) > 0:
                prev_word = context_before[-1]
                if (prev_word,) in self.ngram_model and word in self.ngram_model[(prev_word,)]:
                    bigram_score = self.ngram_model[(prev_word,)][word] / sum(self.ngram_model[(prev_word,)].values())
                    score += bigram_score * 0.4
                    
            # คะแนนจาก bigram หลัง
            if context_after and len(context_after) > 0:
                next_word = context_after[0]
                if (word,) in self.ngram_model and next_word in self.ngram_model[(word,)]:
                    bigram_score = self.ngram_model[(word,)][next_word] / sum(self.ngram_model[(word,)].values())
                    score += bigram_score * 0.4
            
            # คะแนนจากความคล้ายกับคำผิด
            edit_score = 1.0 - (editdistance.eval(misspelled_word, word) / max(len(misspelled_word), len(word)))
            score += edit_score * 0.2
            
            suggestions.append({
                "word": word,
                "score": score,
                "method": "language_model"
            })
        
        # เรียงลำดับตามคะแนน
        suggestions.sort(key=lambda x: x["score"], reverse=True)
        
        return suggestions[:10]  # ส่งคืนเพียง 10 คำแนะนำที่ดีที่สุด
        
    def correct_text(self, text: str, dialect_aware=True) -> str:
        """
        แก้ไขคำผิดในข้อความโดยอัตโนมัติ
        
        Parameters:
        -----------
        text: str
            ข้อความที่ต้องการแก้ไข
        dialect_aware: bool
            พิจารณาภาษาถิ่นหรือไม่
            
        Returns:
        --------
        corrected_text: str
            ข้อความที่แก้ไขแล้ว
        """
        # ตรวจจับคำผิด
        errors = self.detect_spelling_errors(text)
        corrected_text = text
        
        # แก้ไขข้อความจากหลังไปหน้า (เพื่อไม่ให้ตำแหน่งของคำผิดถัดไปเปลี่ยนหลังจากแก้ไข)
        for error in sorted(errors, key=lambda x: x["position"], reverse=True):
            # ถ้าเป็นคำภาษาถิ่นและต้องการรักษาภาษาถิ่นไว้
            if error["is_dialect"] and dialect_aware:
                continue  # ข้ามการแก้ไขคำภาษาถิ่น
                
            # เลือกคำแนะนำที่ดีที่สุด
            if error["suggestions"]:
                best_suggestion = error["suggestions"][0]["word"]
                
                # แทนที่คำผิด
                start_pos = error["position"]
                end_pos = start_pos + len(error["word"])
                corrected_text = corrected_text[:start_pos] + best_suggestion + corrected_text[end_pos:]
        
        return corrected_text
    
    def convert_dialect_to_standard(self, text: str) -> str:
        """
        แปลงคำภาษาถิ่นในข้อความให้เป็นคำมาตรฐาน
        
        Parameters:
        -----------
        text: str
            ข้อความที่มีคำภาษาถิ่น
            
        Returns:
        --------
        standard_text: str
            ข้อความที่แปลงเป็นคำมาตรฐานแล้ว
        """
        if not self.dialect_support:
            return text
            
        # ตัดคำ
        tokens = word_tokenize(text)
        standard_tokens = []
        
        for token in tokens:
            # ถ้าเป็นคำภาษาถิ่น แปลงเป็นคำมาตรฐาน
            if token in self.dialect_to_standard:
                standard_tokens.append(self.dialect_to_standard[token])
            else:
                standard_tokens.append(token)
                
        # รวมคำกลับเป็นข้อความ
        standard_text = "".join(standard_tokens)
        
        return standard_text
    
    def convert_standard_to_dialect(self, text: str, dialect_type: str = "isan") -> str:
        """
        แปลงคำมาตรฐานในข้อความให้เป็นคำภาษาถิ่น
        
        Parameters:
        -----------
        text: str
            ข้อความมาตรฐาน
        dialect_type: str
            ประเภทของภาษาถิ่น ("isan" = อีสาน, "northern" = เหนือ, "southern" = ใต้)
            
        Returns:
        --------
        dialect_text: str
            ข้อความที่แปลงเป็นคำภาษาถิ่นแล้ว
        """
        if not self.dialect_support:
            return text
            
        # กำหนดรูปแบบภาษาถิ่นที่ต้องการ
        dialect_prefixes = {
            "isan": ["อีสาน", "ลาว", "isan"],
            "northern": ["เหนือ", "ล้านนา", "northern"],
            "southern": ["ใต้", "southern"]
        }
        
        selected_prefixes = dialect_prefixes.get(dialect_type.lower(), ["isan"])
        
        # ตัดคำ
        tokens = word_tokenize(text)
        dialect_tokens = []
        
        for token in tokens:
            # ถ้าเป็นคำมาตรฐานที่มีคำภาษาถิ่นตรงกัน
            if token in self.standard_to_dialect:
                dialect_options = self.standard_to_dialect[token]
                
                # พยายามเลือกคำภาษาถิ่นตามประเภทที่กำหนด
                selected_dialect = None
                
                # ถ้ามีหลายตัวเลือก ให้เลือกตามประเภทที่กำหนด
                if len(dialect_options) > 1:
                    for dialect in dialect_options:
                        # ตรวจสอบว่าคำนี้เป็นภาษาถิ่นประเภทที่ต้องการหรือไม่
                        for prefix in selected_prefixes:
                            if prefix in self._get_dialect_metadata(dialect).get("type", ""):
                                selected_dialect = dialect
                                break
                        if selected_dialect:
                            break
                            
                # ถ้าไม่เจอตามประเภทที่ต้องการ ใช้ตัวแรกแทน
                if not selected_dialect and dialect_options:
                    selected_dialect = dialect_options[0]
                    
                if selected_dialect:
                    dialect_tokens.append(selected_dialect)
                else:
                    dialect_tokens.append(token)
            else:
                dialect_tokens.append(token)
                
        # รวมคำกลับเป็นข้อความ
        dialect_text = "".join(dialect_tokens)
        
        return dialect_text
    
    def _get_dialect_metadata(self, word: str) -> Dict:
        """
        ดึงข้อมูลเมตาดาต้าของคำภาษาถิ่น
        
        Parameters:
        -----------
        word: str
            คำภาษาถิ่น
            
        Returns:
        --------
        metadata: Dict
            ข้อมูลเมตาดาต้า เช่น ประเภทของภาษาถิ่น
        """
        # ในอนาคตอาจเพิ่มเมตาดาต้าของคำภาษาถิ่นได้ เช่น ประเภท, ภูมิภาค, ความหมาย
        
        # ทดลองกำหนดประเภทโดยใช้การตรวจสอบง่ายๆ (ในระบบจริงควรมีการเก็บเมตาดาต้าอย่างเหมาะสม)
        if word in ["เบิ่ง", "สิเฮ็ด", "บักหล่า", "เฮือน", "อีหลี", "อีหยัง", "บ่", "แม่น", "เด้อ", "สิ"]:
            return {"type": "isan"}
        elif word in ["กิ๋น", "เปิ้น", "อู้", "บะเดี่ยวนี้", "เตื่อ", "คัวะ", "จ๊าง", "ตี้"]:
            return {"type": "northern"}
        elif word in ["หวาน", "ตง", "แอ", "หวัน", "มุง", "กูม", "นิ", "ชิ", "เหม"]:
            return {"type": "southern"}
        else:
            # ถ้าไม่สามารถระบุได้
            return {"type": "unknown"}

    def add_to_dictionary(self, word: str, is_dialect=False, dialect_standard=None, dialect_type=None) -> bool:
        """
        เพิ่มคำใหม่เข้าไปในพจนานุกรม
        
        Parameters:
        -----------
        word: str
            คำที่ต้องการเพิ่ม
        is_dialect: bool
            เป็นคำภาษาถิ่นหรือไม่
        dialect_standard: str หรือ None
            คำมาตรฐานที่ตรงกับคำภาษาถิ่น (กรณีที่ is_dialect=True)
        dialect_type: str หรือ None
            ประเภทของภาษาถิ่น (เช่น "isan", "northern", "southern")
            
        Returns:
        --------
        success: bool
            สถานะการเพิ่มคำ (สำเร็จหรือไม่)
        """
        if not word:
            return False
            
        # เพิ่มเข้าในพจนานุกรมหลัก
        self.thai_words.add(word)
        
        # ถ้าเป็นคำภาษาถิ่น
        if is_dialect and self.dialect_support and dialect_standard:
            # เพิ่มเข้าในพจนานุกรมภาษาถิ่น
            self.dialect_to_standard[word] = dialect_standard
            
            if dialect_standard not in self.standard_to_dialect:
                self.standard_to_dialect[dialect_standard] = []
                
            if word not in self.standard_to_dialect[dialect_standard]:
                self.standard_to_dialect[dialect_standard].append(word)
                
        # ปรับปรุง Trie
        self.word_trie = dict_trie(self.thai_words)
        
        # บันทึกไปยังไฟล์ (ถ้ามี custom_dict_path)
        if hasattr(self, 'custom_dict_path') and self.custom_dict_path:
            try:
                with open(self.custom_dict_path, 'a', encoding='utf-8') as f:
                    f.write(f"{word}\n")
            except:
                print(f"ไม่สามารถบันทึกคำ '{word}' ไปยังไฟล์พจนานุกรม")
                return False
                
        return True
        
    def analyze_spelling_errors(self, text: str) -> Dict:
        """
        วิเคราะห์คำผิดในข้อความ
        
        Parameters:
        -----------
        text: str
            ข้อความที่ต้องการวิเคราะห์
            
        Returns:
        --------
        analysis: Dict
            ผลการวิเคราะห์
        """
        # ตรวจจับคำผิด
        errors = self.detect_spelling_errors(text)
        
        # ตัดคำทั้งหมด
        tokens = word_tokenize(text)
        
        # นับจำนวนคำภาษาถิ่น
        dialect_count = sum(1 for error in errors if error.get("is_dialect", False))
        
        # จำนวนคำผิดที่ไม่ใช่คำภาษาถิ่น
        non_dialect_errors = len(errors) - dialect_count
        
        # หาคำผิดที่พบบ่อย
        common_errors = Counter([error["word"] for error in errors]).most_common(5)
        
        # จัดประเภทของคำผิด
        error_types = {
            "dialect": dialect_count,
            "non_dialect": non_dialect_errors
        }
        
        return {
            "total_words": len(tokens),
            "total_errors": len(errors),
            "error_percentage": (len(errors) / len(tokens)) * 100 if tokens else 0,
            "dialect_percentage": (dialect_count / len(tokens)) * 100 if tokens else 0,
            "error_types": error_types,
            "common_errors": common_errors
        }

# ตัวอย่างการใช้งาน
if __name__ == "__main__":
    # สร้างตัวอย่างพจนานุกรม
    import tempfile
    
    temp_dict_file = tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8')
    temp_dict_file.write("ทดสอบ\nภาษาไทย\nการแก้ไข\nคำผิด\nระบบ\n")
    temp_dict_file.close()
    
    # สร้างพจนานุกรมภาษาถิ่น
    temp_dialect_file = tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8')
    temp_dialect_file.write("ดู\tเบิ่ง\nไม่\tบ่\nบ้าน\tเฮือน\nพูด\tเว้า\nพูด\tอู้\nกิน\tกิ๋น\nไป\tไป๋\n")
    temp_dialect_file.close()
    
    # สร้าง SpellCorrector
    spell_corrector = ThaiSpellCorrector(
        custom_dict_path=temp_dict_file.name,
        use_built_in=True,
        dialect_support=True,
        dialect_dict_path=temp_dialect_file.name
    )
    
    # ตัวอย่างข้อความที่มีคำผิดและคำภาษาถิ่น
    text = "ทกสอบ ระบบแกไข คำผิิด ภาษาไท เบิ่งนั่นสิ มันบ่ถืก กิ๋นข้าวกับหมู แล้วไป๋ตลาด"
    print(f"ข้อความเดิม: {text}")
    
    # ตรวจจับคำผิด
    errors = spell_corrector.detect_spelling_errors(text)
    print("\nคำผิดที่ตรวจพบ:")
    for error in errors:
        print(f"- คำ: {error['word']}")
        print(f"  ตำแหน่ง: {error['position']}")
        print(f"  เป็นคำภาษาถิ่น: {error['is_dialect']}")
        
        if error["is_dialect"]:
            print(f"  คำมาตรฐาน: {error['dialect_standard']}")
        
        print(f"  คำแนะนำ: {', '.join([s['word'] for s in error['suggestions'][:3]])}")
        print(f"  เหตุผล: {error['reason']}")
        print()
    
    # แก้ไขคำผิด แต่คงคำภาษาถิ่นไว้
    corrected_with_dialect = spell_corrector.correct_text(text, dialect_aware=True)
    print(f"\nข้อความที่แก้ไขแล้ว (คงคำภาษาถิ่นไว้):\n{corrected_with_dialect}")
    
    # แก้ไขคำผิดและแปลงคำภาษาถิ่นเป็นคำมาตรฐาน
    corrected_standard = spell_corrector.correct_text(text, dialect_aware=False)
    print(f"\nข้อความที่แก้ไขแล้ว (แปลงเป็นมาตรฐาน):\n{corrected_standard}")
    
    # แปลงคำภาษาถิ่นเป็นคำมาตรฐาน
    standard_text = spell_corrector.convert_dialect_to_standard(text)
    print(f"\nข้อความที่แปลงเป็นคำมาตรฐาน:\n{standard_text}")
    
    # แปลงคำมาตรฐานเป็นคำภาษาถิ่นอีสาน
    isan_text = spell_corrector.convert_standard_to_dialect("ผมไปดูบ้านเพื่อน แล้วกินข้าวที่ตลาด พูดคุยกับพ่อค้า", dialect_type="isan")
    print(f"\nข้อความที่แปลงเป็นภาษาอีสาน:\n{isan_text}")
    
    # วิเคราะห์คำผิด
    analysis = spell_corrector.analyze_spelling_errors(text)
    print("\nผลการวิเคราะห์คำผิด:")
    print(f"- จำนวนคำทั้งหมด: {analysis['total_words']}")
    print(f"- จำนวนคำผิดทั้งหมด: {analysis['total_errors']}")
    print(f"- เปอร์เซ็นต์คำผิด: {analysis['error_percentage']:.2f}%")
    print(f"- เปอร์เซ็นต์คำภาษาถิ่น: {analysis['dialect_percentage']:.2f}%")
    print(f"- คำผิดที่พบบ่อย: {analysis['common_errors']}")

    # เพิ่มคำใหม่เข้าพจนานุกรม
    spell_corrector.add_to_dictionary("โปรเจกต์", is_dialect=False)
    print("\\nเพิ่มคำ 'โปรเจกต์' เข้าพจนานุกรมแล้ว")

    # เพิ่มคำภาษาถิ่นใหม่
    spell_corrector.add_to_dictionary("เฮ็ด", is_dialect=True, dialect_standard="ทำ", dialect_type="isan")
    print("เพิ่มคำภาษาถิ่น 'เฮ็ด' พร้อมคำมาตรฐาน 'ทำ' เข้าพจนานุกรมแล้ว")

    # ทดสอบหลังเพิ่มคำแล้ว
    test_text = "โปรเจกต์นี้จะเฮ็ดอะไร"
    errors_after = spell_corrector.detect_spelling_errors(test_text)
    print(f"\\nทดสอบคำที่เพิ่ม: '{test_text}'")
    if not errors_after:
        print("ไม่พบคำผิด (การเพิ่มคำสำเร็จ)")
    else:
        print(f"ยังพบคำผิด: {[e['word'] for e in errors_after]}")

    # ลบไฟล์ชั่วคราว
    import os
    os.unlink(temp_dict_file.name)
    os.unlink(temp_dialect_file.name)
