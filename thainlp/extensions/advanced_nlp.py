from typing import List, Dict, Optional, Tuple
import torch
from transformers import (
    pipeline, 
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoModelForMaskedLM
)
from sentence_transformers import SentenceTransformer
import numpy as np
from pythainlp import word_tokenize
from pythainlp.tokenize import Tokenizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
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

class TopicModeling:
    def __init__(self, num_topics: int = 5, max_features: int = 1000):
        self.num_topics = num_topics
        self.vectorizer = CountVectorizer(
            max_features=max_features,
            tokenizer=word_tokenize,
            stop_words=[]  # ควรเพิ่ม stop words ภาษาไทย
        )
        self.lda = LatentDirichletAllocation(
            n_components=num_topics,
            random_state=42
        )
        
    def fit(self, texts: List[str]) -> None:
        """ฝึกโมเดล topic modeling"""
        dtm = self.vectorizer.fit_transform(texts)
        self.lda.fit(dtm)
        
    def get_topics(self, num_words: int = 10) -> List[List[str]]:
        """ดึงคำสำคัญของแต่ละ topic"""
        feature_names = self.vectorizer.get_feature_names_out()
        topics = []
        
        for topic_idx, topic in enumerate(self.lda.components_):
            top_words_idx = topic.argsort()[:-num_words-1:-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topics.append(top_words)
            
        return topics
    
    @cached(expiration=3600)
    def predict(self, text: str) -> List[float]:
        """ทำนาย topic distribution ของข้อความ"""
        dtm = self.vectorizer.transform([text])
        return self.lda.transform(dtm)[0].tolist()

class EmotionDetector:
    EMOTIONS = [
        "ความสุข", "ความเศร้า", "ความโกรธ", 
        "ความกลัว", "ความประหลาดใจ", "ความรัก"
    ]
    
    def __init__(self, model_name: str = "airesearch/wangchanberta-base-att-spm-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(self.EMOTIONS)
        )
        
    @cached(expiration=3600)
    def detect_emotion(self, text: str) -> Dict[str, float]:
        """ตรวจจับอารมณ์จากข้อความ"""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        outputs = self.model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        return {
            emotion: float(prob)
            for emotion, prob in zip(self.EMOTIONS, probs[0])
        }
    
    def get_dominant_emotion(self, text: str) -> Tuple[str, float]:
        """ดึงอารมณ์ที่โดดเด่นที่สุดจากข้อความ"""
        emotions = self.detect_emotion(text)
        dominant = max(emotions.items(), key=lambda x: x[1])
        return dominant

class AdvancedThaiNLP:
    def __init__(self):
        self.topic_model = TopicModeling()
        self.emotion_detector = EmotionDetector()
        
    def analyze_text(self, text: str) -> Dict:
        """วิเคราะห์ข้อความแบบครบวงจร"""
        return {
            "topics": self.topic_model.predict(text),
            "emotions": self.emotion_detector.detect_emotion(text),
            "dominant_emotion": self.emotion_detector.get_dominant_emotion(text)
        }
    
    def analyze_corpus(self, texts: List[str]) -> Dict:
        """วิเคราะห์คลังข้อความ"""
        # ฝึก topic model ใหม่
        self.topic_model.fit(texts)
        
        results = []
        for text in texts:
            results.append(self.analyze_text(text))
            
        return {
            "individual_analyses": results,
            "topics": self.topic_model.get_topics(),
            "corpus_stats": {
                "num_documents": len(texts),
                "avg_emotions": self._get_average_emotions(results)
            }
        }
        
    def _get_average_emotions(self, analyses: List[Dict]) -> Dict[str, float]:
        """คำนวณค่าเฉลี่ยอารมณ์จากผลการวิเคราะห์หลายข้อความ"""
        emotion_sums = {emotion: 0.0 for emotion in EmotionDetector.EMOTIONS}
        
        for analysis in analyses:
            emotions = analysis["emotions"]
            for emotion, score in emotions.items():
                emotion_sums[emotion] += score
                
        return {
            emotion: score / len(analyses)
            for emotion, score in emotion_sums.items()
        } 