"""
Basic usage examples for ThaiNLP library.
"""
import sys
import os

# Add parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import thainlp

def tokenization_example():
    text = "การเดินทางไปท่องเที่ยวที่จังหวัดเชียงใหม่ในฤดูหนาวเป็นประสบการณ์ที่น่าจดจำ"
    print("Original text:", text)

    tokens = thainlp.word_tokenize(text)
    print("Tokenized (default):", tokens)

    tokens = thainlp.word_tokenize(text, engine="pythainlp:longest")
    print("Tokenized (longest):", tokens)


def pos_tagging_example():
    text = "นักวิจัยกำลังศึกษาปรากฏการณ์ทางธรรมชาติที่ซับซ้อน"
    print("Original text:", text)

    pos_tags = thainlp.pos_tag(text)
    print("POS tags (default):", pos_tags)

    pos_tags = thainlp.pos_tag(text, engine="pythainlp", tagset="orchid", return_tagset="ud")
    print("POS tags (orchid -> UD):", pos_tags)

def summarization_example():
    text = """
การพัฒนาเทคโนโลยีปัญญาประดิษฐ์ (AI) กำลังเปลี่ยนแปลงโลกในหลายด้าน
AI ถูกนำมาใช้ในอุตสาหกรรมต่างๆ เช่น การแพทย์ การเงิน การศึกษา และการขนส่ง
อย่างไรก็ตาม การพัฒนา AI ก็มีความท้าทายหลายประการ เช่น ความเป็นส่วนตัว ความปลอดภัย และจริยธรรม
"""
    print("Original text:", text)
    summary = thainlp.summarize(text, n_sentences=2)
    print("Summary:", summary)

def spellcheck_example():
    text = "ฉันไปเทียวทะเลขาว"
    print("Original text:", text)
    misspelled = thainlp.spellcheck(text)
    print("Misspelled words:", misspelled)

def classification_example():
    text = "ภาพยนตร์เรื่องนี้สนุกมาก"
    print("Original text:", text)
    label = thainlp.classify(text)
    print("Classification label:", label)

def sentiment_analysis_example():
    texts = [
        "ฉันรักประเทศไทยมาก สวยงามเสมอ",
        "วันนี้รู้สึกเศร้า เสียใจมาก",
        "วันนี้อากาศดี ฉันไปทำงาน",
    ]

    for text in texts:
        result = thainlp.analyze_sentiment(text)
        print(f"Text: {text}")
        print(f"Sentiment: {result['sentiment']}, Score: {result['score']:.2f}")
        print("---")

if __name__ == "__main__":
    print("=== Tokenization Example ===")
    tokenization_example()
    print("\n=== POS Tagging Example ===")
    pos_tagging_example()
    print("\n=== Summarization Example ===")
    summarization_example()
    print("\n=== Spellcheck Example ===")
    spellcheck_example()
    print("\n=== Classification Example ===")
    classification_example()
    print("\n=== Sentiment Analysis Example ===")
    sentiment_analysis_example()