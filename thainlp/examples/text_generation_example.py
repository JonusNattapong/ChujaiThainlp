#!/usr/bin/env python3
from thainlp.generation import ThaiTextGenerator

def main():
    """ตัวอย่างการใช้งานระบบสร้างข้อความภาษาไทย"""
    print("=== ตัวอย่างการใช้งานระบบสร้างข้อความภาษาไทย ===")
    
    generator = ThaiTextGenerator()
    
    # 1. สร้างข้อความจากเทมเพลต
    print("\n1. สร้างข้อความจากเทมเพลต:")
    template = "สวัสดี{greeting} ฉันชื่อ{name} ฉันเป็น{occupation}"
    params = {
        "greeting": "ครับ",
        "name": "สมชาย",
        "occupation": "วิศวกร"
    }
    text = generator.generate_from_template(template, params)
    print(f"เทมเพลต: {template}")
    print(f"พารามิเตอร์: {params}")
    print(f"ข้อความที่สร้าง: {text}")
    
    # 2. สร้างข้อความตามรูปแบบ n-gram
    print("\n2. สร้างข้อความตามรูปแบบ n-gram:")
    prefix = "ประเทศไทย"
    text = generator.generate_with_ngram(prefix, max_length=50)
    print(f"คำนำ: {prefix}")
    print(f"ข้อความที่สร้าง: {text}")
    
    # 3. สร้างข้อความตามรูปแบบไวยากรณ์
    print("\n3. สร้างข้อความตามรูปแบบไวยากรณ์:")
    pattern = ["PRON", "VERB", "NOUN", "ADP", "NOUN"]
    text = generator.generate_with_pattern(pattern)
    print(f"รูปแบบ: {pattern}")
    print(f"ข้อความที่สร้าง: {text}")
    
    # 4. สร้างข้อความด้วยโมเดลภาษาขั้นสูง
    print("\n4. สร้างข้อความด้วยโมเดลภาษาขั้นสูง:")
    prompt = "อาหารไทยมีเอกลักษณ์คือ"
    text = generator.generate_with_model(prompt, max_length=100)
    print(f"คำนำ: {prompt}")
    print(f"ข้อความที่สร้าง: {text}")
    
    # 5. สร้างย่อหน้า
    print("\n5. สร้างย่อหน้า:")
    topic = "การท่องเที่ยวในประเทศไทย"
    paragraph = generator.generate_paragraph(topic, num_sentences=3)
    print(f"หัวข้อ: {topic}")
    print(f"ย่อหน้าที่สร้าง: {paragraph}")
    
    # 6. ต่อข้อความ
    print("\n6. ต่อข้อความ:")
    prefix = "เมื่อวานนี้ฉันไป"
    completion = generator.complete_text(prefix, max_length=50)
    print(f"ข้อความนำ: {prefix}")
    print(f"ข้อความที่ต่อ: {completion}")
    
    # 7. สร้างบทสนทนา
    print("\n7. สร้างบทสนทนา:")
    context = "การสั่งอาหารในร้านอาหาร"
    conversation = generator.generate_conversation(context, num_turns=3)
    print(f"บริบท: {context}")
    print("\nบทสนทนา:")
    for turn in conversation:
        print(f"{turn['speaker']}: {turn['text']}")

if __name__ == "__main__":
    main()