from typing import List, Dict, Optional
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
import numpy as np
from pythainlp import word_tokenize
from pythainlp.tokenize import Tokenizer
from .caching import cached

class ThaiTextAnalyzer:
    def __init__(self, model_name: str = "airesearch/wangchanberta-base-att-spm-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.sentence_transformer = SentenceTransformer('distiluse-base-multilingual-cased-v2')
        
    @cached(expiration=3600)
    def get_text_embeddings(self, text: str) -> np.ndarray:
        """สร้าง embeddings จากข้อความภาษาไทย"""
        return self.sentence_transformer.encode(text)
    
    @cached(expiration=3600)
    def semantic_similarity(self, text1: str, text2: str) -> float:
        """คำนวณความคล้ายคลึงเชิงความหมายระหว่างข้อความ"""
        emb1 = self.get_text_embeddings(text1)
        emb2 = self.get_text_embeddings(text2)
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))

    @cached(expiration=3600)
    def keyword_extraction(self, text: str, top_k: int = 5) -> List[str]:
        """สกัดคำสำคัญจากข้อความ โดยใช้ความถี่และตำแหน่งของคำ"""
        words = word_tokenize(text)
        word_freq = {}
        
        for i, word in enumerate(words):
            if len(word) > 1:  # ข้ามคำที่สั้นเกินไป
                score = 1.0 * (len(words) - i) / len(words)  # ให้น้ำหนักตามตำแหน่ง
                word_freq[word] = word_freq.get(word, 0) + score
                
        return sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:top_k]

class ThaiTextGenerator:
    def __init__(self, model_name: str = "airesearch/wangchanberta-base-att-spm-uncased"):
        self.generator = pipeline("text-generation", model=model_name)
        
    @cached(expiration=3600)
    def generate_text(self, prompt: str, max_length: int = 100) -> str:
        """สร้างข้อความต่อจาก prompt ที่กำหนด"""
        result = self.generator(prompt, max_length=max_length, num_return_sequences=1)
        return result[0]['generated_text']
    
    @cached(expiration=3600)
    def summarize(self, text: str, max_length: int = 130) -> str:
        """สรุปความข้อความภาษาไทย"""
        summarizer = pipeline("summarization", model="airesearch/wangchanberta-base-att-spm-uncased")
        result = summarizer(text, max_length=max_length, min_length=30, do_sample=False)
        return result[0]['summary_text']

class ThaiNamedEntityRecognition:
    def __init__(self):
        self.custom_tokenizer = Tokenizer(custom_dict=None, engine='newmm')
        
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """สกัดชื่อเฉพาะจากข้อความภาษาไทย"""
        # ตัวอย่างการสกัดชื่อคน สถานที่ องค์กร
        entities = {
            'PERSON': [],
            'LOCATION': [],
            'ORGANIZATION': []
        }
        
        # ใช้ custom rules และ patterns สำหรับการระบุ entities
        # (ในที่นี้เป็นแค่ตัวอย่าง ควรใช้โมเดล NER จริง)
        words = self.custom_tokenizer.word_tokenize(text)
        
        # TODO: เพิ่มการใช้โมเดล NER สำหรับภาษาไทย
        return entities

class ThaiSentimentAnalyzer:
    def __init__(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "airesearch/wangchanberta-base-att-spm-uncased-finetuned-sentiment"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "airesearch/wangchanberta-base-att-spm-uncased-finetuned-sentiment"
        )
    
    @cached(expiration=3600)
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """วิเคราะห์ความรู้สึกจากข้อความภาษาไทย"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        outputs = self.model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        return {
            'positive': float(probs[0][2]),
            'neutral': float(probs[0][1]),
            'negative': float(probs[0][0])
        } 