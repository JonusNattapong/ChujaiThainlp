"""
Example usage of the ThaiNLP pipeline interface
"""
import pandas as pd
from thainlp import ThaiNLPPipeline

def main():
    # Initialize pipeline
    nlp = ThaiNLPPipeline(load_all=True)
    
    print("=== Basic Text Processing ===")
    text = "นายสมชาย สุดหล่อ เดินทางไปกรุงเทพมหานครเมื่อวานนี้"
    
    # Process multiple tasks at once
    results = nlp.process(text)
    print("\nMulti-task processing:")
    print(f"Preprocessed: {results['preprocessed']}")
    print(f"Tokens: {results['tokens']}")
    print(f"Entities: {results['entities']}")
    print(f"Sentiment: {results['sentiment']}")
    
    print("\n=== Named Entity Recognition ===")
    entities = nlp.get_entities(text)
    for entity in entities:
        print(f"- {entity['word']} ({entity['entity']})")
    
    print("\n=== Sentiment Analysis ===")
    texts = [
        "อาหารอร่อยมาก บริการดีเยี่ยม",
        "รอนานมาก บริการแย่สุดๆ",
        "วันนี้อากาศดี ท้องฟ้าแจ่มใส"
    ]
    for text in texts:
        sentiment = nlp.analyze_sentiment(text)
        print(f"Text: {text}")
        print(f"Sentiment: {sentiment}\n")
    
    print("=== Question Answering ===")
    # Text QA
    context = """
    เมืองไทยมีประชากรประมาณ 70 ล้านคน มีกรุงเทพมหานครเป็นเมืองหลวง 
    ประเทศไทยตั้งอยู่ในภูมิภาคเอเชียตะวันออกเฉียงใต้ มีพื้นที่ประมาณ 513,120 ตารางกิโลเมตร
    """
    questions = [
        "ประเทศไทยมีประชากรเท่าไร?",
        "เมืองหลวงของประเทศไทยคือที่ไหน?",
        "ประเทศไทยตั้งอยู่ในภูมิภาคใด?"
    ]
    
    print("\nText QA:")
    for question in questions:
        answer = nlp.answer_question(question, context)
        print(f"Q: {question}")
        print(f"A: {answer}\n")
    
    # Table QA
    data = {
        'จังหวัด': ['กรุงเทพ', 'เชียงใหม่', 'ภูเก็ต'],
        'ประชากร': ['8.2 ล้าน', '1.7 ล้าน', '0.4 ล้าน'],
        'พื้นที่': ['1,569 ตร.กม.', '20,107 ตร.กม.', '543 ตร.กม.']
    }
    df = pd.DataFrame(data)
    
    print("Table QA:")
    questions = [
        "กรุงเทพมีประชากรเท่าไร?",
        "จังหวัดไหนมีพื้นที่มากที่สุด?",
    ]
    for question in questions:
        answer = nlp.answer_question(question, df)
        print(f"Q: {question}")
        print(f"A: {answer}\n")
    
    print("=== Text Generation ===")
    prompts = [
        "วันนี้อากาศ",
        "ประเทศไทยเป็น"
    ]
    for prompt in prompts:
        generated = nlp.generate(prompt, num_sequences=2)
        print(f"\nPrompt: {prompt}")
        for i, text in enumerate(generated, 1):
            print(f"Generated {i}: {text}")
    
    print("\n=== Text Similarity ===")
    text1 = "วันนี้อากาศดีมาก"
    text2 = "ท้องฟ้าสดใสวันนี้"
    text3 = "ราคาหุ้นลดลง"
    
    print(f"Text 1: {text1}")
    print(f"Text 2: {text2}")
    print(f"Text 3: {text3}")
    
    sim1 = nlp.get_similarity(text1, text2)
    sim2 = nlp.get_similarity(text1, text3)
    
    print(f"\nSimilarity 1-2: {sim1:.3f}")
    print(f"Similarity 1-3: {sim2:.3f}")

if __name__ == "__main__":
    main()
