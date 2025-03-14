"""
Basic usage examples for ThaiNLP library.
"""
import sys
import os

# Add parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from thainlp.extensions.advanced_nlp import (
    ThaiTextAnalyzer,
    ThaiSentimentAnalyzer,
    ThaiNamedEntityRecognition,
    TopicModeling,
    EmotionDetector
)

def tokenization_example():
    text = "การเดินทางไปท่องเที่ยวที่จังหวัดเชียงใหม่ในฤดูหนาวเป็นประสบการณ์ที่น่าจดจำ"
    print("Original text:", text)

    analyzer = ThaiTextAnalyzer()
    tokens = analyzer.tokenize(text)
    print("Tokenized:", tokens)

def pos_tagging_example():
    text = "นักวิจัยกำลังศึกษาปรากฏการณ์ทางธรรมชาติที่ซับซ้อน"
    print("Original text:", text)

    analyzer = ThaiTextAnalyzer()
    pos_tags = analyzer.pos_tag(text)
    print("POS tags:", pos_tags)

def ner_example():
    text = "นายสมชาย ใจดี เป็นอาจารย์ที่มหาวิทยาลัยเชียงใหม่"
    print("Original text:", text)

    ner = ThaiNamedEntityRecognition()
    entities = ner.extract_entities(text)
    print("Named Entities:", entities)

def sentiment_analysis_example():
    texts = [
        "ฉันรักประเทศไทยมาก สวยงามเสมอ",
        "วันนี้รู้สึกเศร้า เสียใจมาก",
        "วันนี้อากาศดี ฉันไปทำงาน",
    ]

    analyzer = ThaiSentimentAnalyzer()
    for text in texts:
        result = analyzer.analyze_sentiment(text)
        print(f"Text: {text}")
        print(f"Sentiment: {result['sentiment']}, Score: {result['score']:.2f}")
        print("---")

def topic_modeling_example():
    texts = [
        "การพัฒนาเทคโนโลยีปัญญาประดิษฐ์กำลังเปลี่ยนแปลงโลก",
        "นักวิทยาศาสตร์ค้นพบดาวเคราะห์ดวงใหม่",
        "ตลาดหุ้นทั่วโลกปรับตัวขึ้นหลังจากข่าวดีทางเศรษฐกิจ"
    ]
    
    topic_model = TopicModeling()
    topics = topic_model.extract_topics(texts)
    print("Topics:", topics)

def emotion_detection_example():
    texts = [
        "ฉันมีความสุขมากที่ได้เจอเพื่อนเก่า",
        "เสียใจจังที่สอบไม่ผ่าน",
        "โกรธมากที่เขาไม่รักษาสัญญา"
    ]
    
    detector = EmotionDetector()
    for text in texts:
        emotions = detector.detect_emotion(text)
        print(f"Text: {text}")
        print(f"Emotions: {emotions}")
        print("---")

if __name__ == "__main__":
    print("=== Tokenization Example ===")
    tokenization_example()
    print("\n=== POS Tagging Example ===")
    pos_tagging_example()
    print("\n=== NER Example ===")
    ner_example()
    print("\n=== Sentiment Analysis Example ===")
    sentiment_analysis_example()
    print("\n=== Topic Modeling Example ===")
    topic_modeling_example()
    print("\n=== Emotion Detection Example ===")
    emotion_detection_example()