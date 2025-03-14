"""
Thai Text Generation Module
"""

from typing import Dict, List, Union, Optional, Any
import random
import re
from collections import defaultdict, Counter

class ThaiTextGenerator:
    def __init__(self):
        """Initialize ThaiTextGenerator"""
        # Templates for different text types
        self.templates = {
            'greeting': [
                'สวัสดี{time_suffix}{polite_suffix}',
                'สวัสดี{time_suffix} คุณ{name}{polite_suffix}',
                'สวัสดี{polite_suffix} ยินดีต้อนรับ{polite_suffix}',
            ],
            'farewell': [
                'ลาก่อน{polite_suffix}',
                'แล้วพบกันใหม่{polite_suffix}',
                'ขอให้โชคดี{polite_suffix}',
                'ขอบคุณ{polite_suffix} แล้วพบกันใหม่{polite_suffix}',
            ],
            'question': [
                'คุณ{verb}อะไร{question_suffix}',
                '{pronoun}{verb}อะไร{question_suffix}',
                'ทำไม{pronoun}ถึง{verb}{question_suffix}',
                'คุณ{verb}ที่ไหน{question_suffix}',
                'เมื่อไหร่{pronoun}จะ{verb}{question_suffix}',
            ],
            'statement': [
                '{pronoun}{verb}{object}{polite_suffix}',
                '{pronoun}ไม่{verb}{object}{polite_suffix}',
                '{pronoun}กำลัง{verb}{object}{polite_suffix}',
                '{pronoun}จะ{verb}{object}{polite_suffix}',
                '{pronoun}ได้{verb}{object}แล้ว{polite_suffix}',
            ],
            'opinion': [
                '{pronoun}คิดว่า{statement}{polite_suffix}',
                'ในความคิดของ{pronoun} {statement}{polite_suffix}',
                '{pronoun}เห็นว่า{statement}{polite_suffix}',
                '{pronoun}รู้สึกว่า{statement}{polite_suffix}',
            ]
        }
        
        # Template variables
        self.template_vars = {
            'time_suffix': ['', 'ตอนเช้า', 'ตอนสาย', 'ตอนเที่ยง', 'ตอนบ่าย', 'ตอนเย็น', 'ตอนค่ำ'],
            'polite_suffix': ['', 'ครับ', 'คะ', 'ค่ะ', 'นะครับ', 'นะคะ'],
            'question_suffix': ['', 'ครับ', 'คะ', 'หรือ', 'หรือครับ', 'หรือคะ', 'หรือเปล่า'],
            'pronoun': ['ฉัน', 'ผม', 'ดิฉัน', 'เขา', 'เธอ', 'พวกเรา', 'พวกเขา', 'คุณ'],
            'verb': ['ชอบ', 'รัก', 'เกลียด', 'ต้องการ', 'อยาก', 'ไป', 'มา', 'กิน', 'ดื่ม', 'นอน', 'เล่น', 'ทำงาน', 'เรียน', 'อ่าน', 'เขียน', 'พูด', 'ฟัง', 'ดู'],
            'object': ['', 'อาหาร', 'น้ำ', 'กาแฟ', 'หนังสือ', 'ทีวี', 'ภาพยนตร์', 'เพลง', 'เกม', 'กีฬา', 'การเดินทาง', 'การท่องเที่ยว', 'ธรรมชาติ', 'ทะเล', 'ภูเขา'],
            'name': ['สมชาย', 'สมหญิง', 'วิชัย', 'มานี', 'สุดา', 'ประเสริฐ', 'กมลา', 'สมศักดิ์', 'ศิริพร', 'อนุชา'],
            'statement': ['เรื่องนี้น่าสนใจ', 'เรื่องนี้สำคัญมาก', 'สิ่งนี้ดีมาก', 'สิ่งนี้ไม่ดีเลย', 'เรื่องนี้ยากมาก', 'เรื่องนี้ง่ายมาก', 'คนนี้เก่งมาก', 'คนนี้ไม่เก่งเลย']
        }
        
        # N-gram models
        self.unigrams = Counter()
        self.bigrams = defaultdict(Counter)
        self.trigrams = defaultdict(lambda: defaultdict(Counter))
        
        # Vocabulary for different parts of speech
        self.vocabulary = {
            'NOUN': ['บ้าน', 'รถ', 'คน', 'หมา', 'แมว', 'โต๊ะ', 'เก้าอี้', 'ต้นไม้', 'ดอกไม้', 'นก', 'ปลา', 'แม่น้ำ', 'ทะเล', 'ภูเขา', 'เมือง', 'ประเทศ', 'อาหาร', 'น้ำ', 'กาแฟ', 'ชา'],
            'VERB': ['กิน', 'ดื่ม', 'นอน', 'วิ่ง', 'เดิน', 'พูด', 'ฟัง', 'อ่าน', 'เขียน', 'ดู', 'เล่น', 'ทำงาน', 'เรียน', 'สอน', 'ซื้อ', 'ขาย', 'ให้', 'รับ', 'ส่ง', 'ชอบ'],
            'ADJ': ['ดี', 'เลว', 'สวย', 'น่าเกลียด', 'ใหญ่', 'เล็ก', 'สูง', 'ต่ำ', 'หนัก', 'เบา', 'ร้อน', 'เย็น', 'แข็ง', 'นุ่ม', 'เร็ว', 'ช้า', 'ใหม่', 'เก่า', 'แพง', 'ถูก'],
            'ADV': ['มาก', 'น้อย', 'เร็ว', 'ช้า', 'ดี', 'แย่', 'บ่อย', 'นาน', 'ทันที', 'เสมอ', 'บางครั้ง', 'ไม่เคย', 'ค่อนข้าง', 'เกือบ', 'จริงๆ', 'แน่นอน', 'อาจจะ', 'คงจะ', 'น่าจะ', 'ควรจะ'],
            'PRON': ['ฉัน', 'ผม', 'ดิฉัน', 'เขา', 'เธอ', 'มัน', 'พวกเรา', 'พวกเขา', 'คุณ', 'ท่าน', 'นั่น', 'นี่', 'โน่น', 'อันนี้', 'อันนั้น', 'ใคร', 'อะไร', 'ที่ไหน', 'เมื่อไร', 'อย่างไร'],
            'PREP': ['ใน', 'นอก', 'บน', 'ล่าง', 'ข้างใน', 'ข้างนอก', 'ข้างบน', 'ข้างล่าง', 'ระหว่าง', 'รอบ', 'ผ่าน', 'ตาม', 'จาก', 'ถึง', 'สู่', 'กับ', 'โดย', 'ด้วย', 'สำหรับ', 'เพื่อ'],
            'CONJ': ['และ', 'หรือ', 'แต่', 'เพราะ', 'เนื่องจาก', 'ดังนั้น', 'ถ้า', 'เมื่อ', 'ขณะที่', 'ในขณะที่', 'ก่อนที่', 'หลังจาก', 'นอกจาก', 'ส่วน', 'อย่างไรก็ตาม', 'อย่างไรก็ดี', 'ทั้งๆที่', 'แม้ว่า', 'ถึงแม้ว่า', 'เพื่อให้'],
            'DET': ['นี้', 'นั้น', 'โน้น', 'เหล่านี้', 'เหล่านั้น', 'ทั้งหมด', 'ทุก', 'แต่ละ', 'บาง', 'หลาย', 'กี่', 'เท่าไร', 'เท่าไหร่', 'เท่าใด', 'ไหน', 'ใด', 'อะไร', 'ไหน', 'ใด', 'อะไร'],
            'NUM': ['หนึ่ง', 'สอง', 'สาม', 'สี่', 'ห้า', 'หก', 'เจ็ด', 'แปด', 'เก้า', 'สิบ', 'ร้อย', 'พัน', 'หมื่น', 'แสน', 'ล้าน', 'ที่หนึ่ง', 'ที่สอง', 'ที่สาม', 'ครึ่ง', 'หนึ่งในสาม']
        }
        
        # Common sentence patterns
        self.sentence_patterns = [
            ['PRON', 'VERB', 'NOUN'],
            ['PRON', 'VERB', 'ADV'],
            ['PRON', 'ADV', 'VERB', 'NOUN'],
            ['NOUN', 'ADJ'],
            ['NOUN', 'VERB', 'ADV'],
            ['PRON', 'VERB', 'PREP', 'NOUN'],
            ['PRON', 'VERB', 'CONJ', 'PRON', 'VERB'],
            ['DET', 'NOUN', 'ADJ'],
            ['PRON', 'VERB', 'DET', 'NOUN'],
            ['NOUN', 'DET', 'ADJ']
        ]
        
    def _fill_template(self, template_type: str) -> str:
        """
        Fill a template with random variables
        
        Args:
            template_type (str): Type of template to fill
            
        Returns:
            str: Filled template
        """
        if template_type not in self.templates:
            template_type = random.choice(list(self.templates.keys()))
            
        # Select a random template
        template = random.choice(self.templates[template_type])
        
        # Fill in template variables
        for var_name, var_values in self.template_vars.items():
            if '{' + var_name + '}' in template:
                template = template.replace('{' + var_name + '}', random.choice(var_values))
                
        return template
        
    def _train_ngram_model(self, texts: List[str]) -> None:
        """
        Train n-gram models from texts
        
        Args:
            texts (List[str]): List of training texts
        """
        for text in texts:
            # Tokenize (simplified)
            tokens = text.split()
            
            # Update unigrams
            self.unigrams.update(tokens)
            
            # Update bigrams
            for i in range(len(tokens) - 1):
                self.bigrams[tokens[i]][tokens[i+1]] += 1
                
            # Update trigrams
            for i in range(len(tokens) - 2):
                self.trigrams[tokens[i]][tokens[i+1]][tokens[i+2]] += 1
                
    def _generate_from_ngram(self, length: int = 10, start_word: Optional[str] = None) -> str:
        """
        Generate text using n-gram model
        
        Args:
            length (int): Length of text to generate
            start_word (Optional[str]): Word to start with
            
        Returns:
            str: Generated text
        """
        if not self.unigrams:
            return "ไม่มีข้อมูลสำหรับการสร้างข้อความ"
            
        # Start with a random word if not specified
        if start_word is None or start_word not in self.unigrams:
            start_word = random.choices(
                list(self.unigrams.keys()),
                weights=[count for count in self.unigrams.values()],
                k=1
            )[0]
            
        # Generate text
        text = [start_word]
        current_word = start_word
        
        for _ in range(length - 1):
            # Try to use trigram
            if len(text) >= 2 and text[-2] in self.trigrams and text[-1] in self.trigrams[text[-2]]:
                next_word_candidates = self.trigrams[text[-2]][text[-1]]
                if next_word_candidates:
                    next_word = random.choices(
                        list(next_word_candidates.keys()),
                        weights=[count for count in next_word_candidates.values()],
                        k=1
                    )[0]
                    text.append(next_word)
                    continue
                    
            # Try to use bigram
            if current_word in self.bigrams:
                next_word_candidates = self.bigrams[current_word]
                if next_word_candidates:
                    next_word = random.choices(
                        list(next_word_candidates.keys()),
                        weights=[count for count in next_word_candidates.values()],
                        k=1
                    )[0]
                    text.append(next_word)
                    current_word = next_word
                    continue
                    
            # Fallback to unigram
            next_word = random.choices(
                list(self.unigrams.keys()),
                weights=[count for count in self.unigrams.values()],
                k=1
            )[0]
            text.append(next_word)
            current_word = next_word
            
        return ' '.join(text)
        
    def _generate_from_pattern(self, pattern: Optional[List[str]] = None) -> str:
        """
        Generate text using a part-of-speech pattern
        
        Args:
            pattern (Optional[List[str]]): POS pattern to use
            
        Returns:
            str: Generated text
        """
        if pattern is None:
            pattern = random.choice(self.sentence_patterns)
            
        # Generate words based on pattern
        words = []
        for pos in pattern:
            if pos in self.vocabulary and self.vocabulary[pos]:
                words.append(random.choice(self.vocabulary[pos]))
            else:
                words.append("[" + pos + "]")
                
        return ''.join(words)
        
    def generate_template_text(self, template_type: Optional[str] = None) -> str:
        """
        Generate text using templates
        
        Args:
            template_type (Optional[str]): Type of template to use
            
        Returns:
            str: Generated text
        """
        if template_type is None:
            template_type = random.choice(list(self.templates.keys()))
            
        return self._fill_template(template_type)
        
    def generate_ngram_text(self, texts: List[str], length: int = 10, start_word: Optional[str] = None) -> str:
        """
        Generate text using n-gram model
        
        Args:
            texts (List[str]): Training texts
            length (int): Length of text to generate
            start_word (Optional[str]): Word to start with
            
        Returns:
            str: Generated text
        """
        # Train model if needed
        if not self.unigrams:
            self._train_ngram_model(texts)
            
        return self._generate_from_ngram(length, start_word)
        
    def generate_pattern_text(self, pattern: Optional[List[str]] = None) -> str:
        """
        Generate text using a part-of-speech pattern
        
        Args:
            pattern (Optional[List[str]]): POS pattern to use
            
        Returns:
            str: Generated text
        """
        return self._generate_from_pattern(pattern)
        
    def generate_text(self, method: str = 'template', **kwargs) -> str:
        """
        Generate text using specified method
        
        Args:
            method (str): Generation method ('template', 'ngram', or 'pattern')
            **kwargs: Additional arguments for the specific method
            
        Returns:
            str: Generated text
        """
        if method == 'template':
            return self.generate_template_text(kwargs.get('template_type'))
        elif method == 'ngram':
            return self.generate_ngram_text(
                kwargs.get('texts', []),
                kwargs.get('length', 10),
                kwargs.get('start_word')
            )
        elif method == 'pattern':
            return self.generate_pattern_text(kwargs.get('pattern'))
        else:
            return "ไม่รู้จักวิธีการสร้างข้อความ: " + method
            
    def generate_paragraph(self, num_sentences: int = 3, method: str = 'template') -> str:
        """
        Generate a paragraph with multiple sentences
        
        Args:
            num_sentences (int): Number of sentences to generate
            method (str): Generation method
            
        Returns:
            str: Generated paragraph
        """
        sentences = []
        for _ in range(num_sentences):
            if method == 'template':
                sentences.append(self.generate_template_text())
            elif method == 'pattern':
                sentences.append(self.generate_pattern_text())
            else:
                # Default to template
                sentences.append(self.generate_template_text())
                
        return ' '.join(sentences)
        
    def complete_text(self, prefix: str, length: int = 5, method: str = 'template') -> str:
        """
        Complete text given a prefix
        
        Args:
            prefix (str): Text prefix
            length (int): Number of additional words/sentences to generate
            method (str): Generation method
            
        Returns:
            str: Completed text
        """
        if method == 'ngram' and self.unigrams:
            # Get the last word of the prefix
            words = prefix.split()
            if words:
                last_word = words[-1]
                completion = self._generate_from_ngram(length, last_word)
                return prefix + ' ' + ' '.join(completion.split()[1:])  # Avoid repeating the last word
        
        # Default to template-based completion
        if random.random() < 0.5:
            # Complete with a statement
            completion = self.generate_template_text('statement')
        else:
            # Complete with a question
            completion = self.generate_template_text('question')
            
        return prefix + ' ' + completion

def generate_text(method: str = 'template', **kwargs) -> str:
    """
    Generate Thai text using specified method
    
    Args:
        method (str): Generation method ('template', 'ngram', or 'pattern')
        **kwargs: Additional arguments for the specific method
        
    Returns:
        str: Generated text
    """
    generator = ThaiTextGenerator()
    return generator.generate_text(method, **kwargs)

def generate_paragraph(num_sentences: int = 3) -> str:
    """
    Generate a paragraph with multiple sentences
    
    Args:
        num_sentences (int): Number of sentences to generate
        
    Returns:
        str: Generated paragraph
    """
    generator = ThaiTextGenerator()
    return generator.generate_paragraph(num_sentences)

def complete_text(prefix: str, length: int = 5) -> str:
    """
    Complete text given a prefix
    
    Args:
        prefix (str): Text prefix
        length (int): Number of additional words/sentences to generate
        
    Returns:
        str: Completed text
    """
    generator = ThaiTextGenerator()
    return generator.complete_text(prefix, length) 