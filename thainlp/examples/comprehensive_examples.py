"""
ตัวอย่างการใช้งานทั้งหมดของ ThaiNLP
"""
import sys
import os
from typing import List, Dict, Any

# Add parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from thainlp.extensions.advanced_nlp import (
    ThaiTextAnalyzer,
    ThaiSentimentAnalyzer,
    ThaiNamedEntityRecognition,
    TopicModeling,
    EmotionDetector,
    ThaiTextGenerator,
    AdvancedThaiNLP
)

def basic_text_analysis():
    """ตัวอย่างการวิเคราะห์ข้อความพื้นฐาน"""
    print("\n=== การวิเคราะห์ข้อความพื้นฐาน ===")
    
    text = "นักวิจัยไทยค้นพบวิธีการรักษาโรคใหม่ที่โรงพยาบาลจุฬาลงกรณ์เมื่อวันที่ 15 มกราคม 2567"
    analyzer = ThaiTextAnalyzer()
    
    # การตัดคำ
    tokens = analyzer.tokenize(text)
    print(f"การตัดคำ: {tokens}")
    
    # การวิเคราะห์ไวยากรณ์
    pos_tags = analyzer.pos_tag(text)
    print(f"การวิเคราะห์ไวยากรณ์: {pos_tags}")
    
    # การแยกประโยค
    sentences = analyzer.segment_sentences(text)
    print(f"การแยกประโยค: {sentences}")
    
    # การนับคำ
    word_freq = analyzer.word_frequency(text)
    print(f"ความถี่ของคำ: {word_freq}")

def named_entity_recognition():
    """ตัวอย่างการรู้จำชื่อเฉพาะ"""
    print("\n=== การรู้จำชื่อเฉพาะ ===")
    
    text = """
    นายสมชาย รักดี เป็นผู้อำนวยการบริษัทไทยเทคโนโลยี จำกัด 
    ที่ตั้งอยู่ในจังหวัดเชียงใหม่ เขาจบการศึกษาจากมหาวิทยาลัยจุฬาลงกรณ์
    และมีความเชี่ยวชาญด้านปัญญาประดิษฐ์
    """
    
    ner = ThaiNamedEntityRecognition()
    entities = ner.extract_entities(text)
    
    print("ชื่อเฉพาะที่พบ:")
    for entity, entity_type, confidence in entities:
        print(f"- {entity} ({entity_type}): ความมั่นใจ {confidence:.2f}")

def sentiment_and_emotion():
    """ตัวอย่างการวิเคราะห์ความรู้สึกและอารมณ์"""
    print("\n=== การวิเคราะห์ความรู้สึกและอารมณ์ ===")
    
    texts = [
        "ร้านอาหารนี้อร่อยมาก บริการดีเยี่ยม ประทับใจสุดๆ",
        "สินค้าคุณภาพแย่มาก ส่งช้า เสียความรู้สึก",
        "วันนี้อากาศดี ท้องฟ้าสดใส ได้ไปเดินเล่นในสวน"
    ]
    
    # วิเคราะห์ความรู้สึก
    sentiment_analyzer = ThaiSentimentAnalyzer()
    print("\nการวิเคราะห์ความรู้สึก:")
    for text in texts:
        sentiment = sentiment_analyzer.analyze_sentiment(text)
        print(f"ข้อความ: {text}")
        print(f"ความรู้สึก: {sentiment['sentiment']}")
        print(f"คะแนน: {sentiment['score']:.2f}")
        print("---")
    
    # วิเคราะห์อารมณ์
    emotion_detector = EmotionDetector()
    print("\nการวิเคราะห์อารมณ์:")
    for text in texts:
        emotions = emotion_detector.detect_emotion(text)
        print(f"ข้อความ: {text}")
        print(f"อารมณ์: {emotions}")
        print("---")

def topic_modeling_and_classification():
    """ตัวอย่างการวิเคราะห์หัวข้อและการจำแนกประเภท"""
    print("\n=== การวิเคราะห์หัวข้อและการจำแนกประเภท ===")
    
    documents = [
        "รัฐบาลประกาศมาตรการช่วยเหลือเกษตรกรรายย่อย",
        "นักวิทยาศาสตร์ค้นพบดาวเคราะห์ดวงใหม่ในระบบสุริยะ",
        "ตลาดหุ้นไทยปรับตัวขึ้นต่อเนื่องจากปัจจัยบวกต่างประเทศ",
        "ทีมฟุตบอลไทยเตรียมลงแข่งขันรายการสำคัญ",
        "นักวิจัยพัฒนาวัคซีนใหม่ต้านไวรัสกลายพันธุ์"
    ]
    
    # วิเคราะห์หัวข้อ
    topic_model = TopicModeling()
    topics = topic_model.extract_topics(documents)
    
    print("\nการวิเคราะห์หัวข้อ:")
    for topic_id, (topic_name, keywords, docs) in enumerate(topics):
        print(f"\nหัวข้อที่ {topic_id + 1}: {topic_name}")
        print(f"คำสำคัญ: {', '.join(keywords)}")
        print(f"ตัวอย่างเอกสาร: {docs[0]}")

def text_generation_and_completion():
    """ตัวอย่างการสร้างข้อความและการเติมข้อความ"""
    print("\n=== การสร้างข้อความและการเติมข้อความ ===")
    
    generator = ThaiTextGenerator()
    
    # สร้างข้อความ
    prompt = "ประเทศไทยเป็นประเทศที่"
    generated_text = generator.generate_text(prompt, max_length=100)
    print(f"\nการสร้างข้อความ:")
    print(f"Prompt: {prompt}")
    print(f"ผลลัพธ์: {generated_text}")
    
    # เติมข้อความ
    incomplete_text = "การพัฒนาเทคโนโลยีปัญญาประดิษฐ์ในประเทศไทย..."
    completed_text = generator.complete_text(incomplete_text)
    print(f"\nการเติมข้อความ:")
    print(f"ข้อความเริ่มต้น: {incomplete_text}")
    print(f"ข้อความที่เติม: {completed_text}")

def advanced_features():
    """ตัวอย่างฟีเจอร์ขั้นสูง"""
    print("\n=== ฟีเจอร์ขั้นสูง ===")
    
    nlp = AdvancedThaiNLP()
    
    # การวิเคราะห์ความคล้ายคลึง
    text1 = "การพัฒนาปัญญาประดิษฐ์ในประเทศไทย"
    text2 = "ประเทศไทยกับการพัฒนา AI"
    similarity = nlp.semantic_similarity(text1, text2)
    print(f"\nความคล้ายคลึงของข้อความ:")
    print(f"ข้อความ 1: {text1}")
    print(f"ข้อความ 2: {text2}")
    print(f"ความคล้ายคลึง: {similarity:.2f}")
    
    # การสรุปความ
    long_text = """
    ปัญญาประดิษฐ์กำลังเปลี่ยนแปลงโลกในหลายด้าน ทั้งการแพทย์ การศึกษา 
    และอุตสาหกรรม ในประเทศไทยมีการนำ AI มาประยุกต์ใช้ในหลายภาคส่วน 
    แต่ยังต้องพัฒนาบุคลากรและโครงสร้างพื้นฐานอีกมาก การสนับสนุนจากภาครัฐ
    และเอกชนจะช่วยผลักดันให้ไทยก้าวสู่ยุค AI ได้อย่างมั่นคง
    """
    summary = nlp.summarize(long_text, ratio=0.3)
    print(f"\nการสรุปความ:")
    print(f"ข้อความต้นฉบับ: {long_text}")
    print(f"สรุป: {summary}")
    
    # การแปลภาษา
    thai_text = "ปัญญาประดิษฐ์กำลังเปลี่ยนแปลงโลก"
    english_translation = nlp.translate(thai_text, target_lang="en")
    print(f"\nการแปลภาษา:")
    print(f"ต้นฉบับ (ไทย): {thai_text}")
    print(f"แปล (อังกฤษ): {english_translation}")

def main():
    """ฟังก์ชันหลักสำหรับรันตัวอย่างทั้งหมด"""
    print("=== ตัวอย่างการใช้งาน ThaiNLP ===")
    
    basic_text_analysis()
    named_entity_recognition()
    sentiment_and_emotion()
    topic_modeling_and_classification()
    text_generation_and_completion()
    advanced_features()

if __name__ == "__main__":
    main() 