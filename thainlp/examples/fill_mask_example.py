#!/usr/bin/env python3
from thainlp.generation.fill_mask import ThaiFillMask

def main():
    """ตัวอย่างการใช้งานระบบเติมคำในช่องว่างภาษาไทย"""
    print("=== ตัวอย่างการใช้งานระบบเติมคำในช่องว่างภาษาไทย ===")
    
    # 1. เติมคำในช่องว่างพื้นฐานด้วย WangchanBERTa
    print("\n1. เติมคำในช่องว่างพื้นฐานด้วย WangchanBERTa:")
    fill_mask = ThaiFillMask()
    
    text = "ฉันชอบ<mask>แมว"
    print(f"ข้อความ: {text}")
    predictions = fill_mask.fill_mask(text, top_k=5)
    print("คำทำนาย:")
    for i, pred in enumerate(predictions, 1):
        print(f"{i}. '{pred['token']}' (คะแนน: {pred['score']:.3f})")
    
    # 2. เติมคำในช่องว่างด้วย Gemma
    print("\n2. เติมคำในช่องว่างด้วย Gemma:")
    fill_mask_gemma = ThaiFillMask(model_name_or_path="google/gemma-2b-it")
    
    text = "วันนี้อากาศ<mask>มาก"
    print(f"ข้อความ: {text}")
    predictions = fill_mask_gemma.fill_mask(text, top_k=5)
    print("คำทำนาย:")
    for i, pred in enumerate(predictions, 1):
        print(f"{i}. '{pred['token']}' (คะแนน: {pred['score']:.3f})")
    
    # 3. เติมหลายช่องว่างพร้อมกัน
    print("\n3. เติมหลายช่องว่างพร้อมกัน:")
    text = "ฉัน<mask>ไป<mask>ที่ตลาด"
    print(f"ข้อความ: {text}")
    predictions = fill_mask.fill_multiple_masks(text, top_k=3)
    print("คำทำนาย:")
    for i, pred in enumerate(predictions, 1):
        print(f"{i}. '{pred['text']}' (คะแนน: {pred['score']:.3f})")
    
    # 4. เติมคำในช่องว่างตามบริบท
    print("\n4. เติมคำในช่องว่างตามบริบท:")
    context = """
    ประเทศไทยตั้งอยู่ในภูมิภาค<mask>
    อาหารไทยมีรสชาติ<mask>
    กรุงเทพมหานครเป็น<mask>ของประเทศไทย
    """
    print(f"บริบท:\n{context}")
    predictions = fill_mask.fill_masks_with_context(context, top_k=3)
    print("\nคำทำนาย:")
    for mask_position, preds in predictions.items():
        print(f"\nตำแหน่งที่ {mask_position}:")
        for i, pred in enumerate(preds, 1):
            print(f"{i}. '{pred['token']}' (คะแนน: {pred['score']:.3f})")
    
    # 5. วิเคราะห์บริบทของคำ
    print("\n5. วิเคราะห์บริบทของคำ:")
    text = "อาหารไทยมีรสชาติเผ็ด"
    print(f"ข้อความ: {text}")
    target_word = "เผ็ด"
    print(f"คำเป้าหมาย: {target_word}")
    context_analysis = fill_mask.analyze_word_context(text, target_word)
    print("\nผลการวิเคราะห์บริบท:")
    print(f"คำที่เกี่ยวข้อง: {', '.join(context_analysis['related_words'])}")
    print(f"ความสัมพันธ์: {context_analysis['relationships']}")
    print(f"คะแนนความสัมพันธ์: {context_analysis['relevance_score']:.3f}")

if __name__ == "__main__":
    main()